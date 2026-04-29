import os
import argparse
import numpy as np
import onnxruntime as ort
import random
from itertools import combinations

class GomokuGame:
    def __init__(self, size=20):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0:空, 1:黑, 2:白

    def check_win(self, r, c, player):
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 5):
                nr, nc = r + dr * i, c + dc * i
                if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == player:
                    count += 1
                else: break
            for i in range(1, 5):
                nr, nc = r - dr * i, c - dc * i
                if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr, nc] == player:
                    count += 1
                else: break
            if count >= 5: return True
        return False

    def apply_opening(self, opening_moves):
        """按照开局序列落子，奇数位黑，偶数位白"""
        for i, (r, c) in enumerate(opening_moves):
            player = 1 if i % 2 == 0 else 2
            if 0 <= r < self.size and 0 <= c < self.size:
                self.board[r, c] = player

def load_openings(directory, size=20):
    """
    仿照 C++ 逻辑加载开局
    坐标转换公式：y + size/2, x + size/2
    """
    openings = []
    if not os.path.exists(directory):
        return openings

    for file_name in os.listdir(directory):
        if file_name.endswith(".txt"):
            with open(os.path.join(directory, file_name), 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    
                    # 解析逗号分隔的数字
                    tokens = [t.strip() for t in line.split(',') if t.strip()]
                    coords = []
                    # 每两个一组作为 (x, y)
                    for i in range(0, len(tokens) - 1, 2):
                        try:
                            x = int(tokens[i])
                            y = int(tokens[i+1])
                            # 转换到棋盘坐标 (row, col)
                            r = y + size // 2
                            c = x + size // 2
                            coords.append((r, c))
                        except ValueError:
                            continue
                    if coords:
                        openings.append(coords)
    return openings

def prepare_input_from_board(board, player_color):
    input_data = np.zeros((3, 20, 20), dtype=np.float32)
    input_data[0, :, :] = 1.0  # Mask
    if player_color == 1:
        input_data[1, :, :] = (board == 1).astype(np.float32)
        input_data[2, :, :] = (board == 2).astype(np.float32)
    else:
        input_data[1, :, :] = (board == 2).astype(np.float32)
        input_data[2, :, :] = (board == 1).astype(np.float32)
    return np.expand_dims(input_data, axis=0), np.array([20.0], dtype=np.float32)

def play_match(model1_sess, model2_sess, opening):
    """
    model1 执黑, model2 执白
    """
    game = GomokuGame(size=20)
    # 应用随机开局
    game.apply_opening(opening)
    
    sessions = {1: model1_sess, 2: model2_sess}
    # 从开局后的下一手开始
    start_move = len(opening)
    
    for move_count in range(start_move, 20 * 20):
        current_player = 1 if move_count % 2 == 0 else 2
        sess = sessions[current_player]
        
        input_tensor, board_sizes = prepare_input_from_board(game.board, current_player)
        outputs = sess.run(["policy_logits"], {"input": input_tensor, "board_sizes": board_sizes})
        
        prob_map = outputs[0][0, 0, :20, :20]
        prob_map[game.board != 0] = -np.inf # 禁止落在已有棋子的位置
        
        best_move = np.unravel_index(np.argmax(prob_map), prob_map.shape)
        r, c = best_move
        
        game.board[r, c] = current_player
        if game.check_win(r, c, current_player):
            return current_player
            
    return 0

import os
import argparse
import numpy as np
import onnxruntime as ort
import random

# ... (保留之前的 GomokuGame 类, load_openings 函数 和 prepare_input_from_board 函数) ...

def play_best_of_n(sess1, sess2, openings, n=3):
    """
    两个模型之间的小局对决（三局两胜或五局三胜）
    返回胜者索引 (1 或 2)，平局返回 0
    """
    wins1, wins2 = 0, 0
    for i in range(n):
        opening = random.choice(openings) if openings else []
        # 轮流执黑以示公平
        if i % 2 == 0:
            winner = play_match(sess1, sess2, opening)
            if winner == 1: wins1 += 1
            elif winner == 2: wins2 += 1
        else:
            winner = play_match(sess2, sess1, opening)
            if winner == 1: wins2 += 1
            elif winner == 2: wins1 += 1

    if wins1 > wins2: return 1
    if wins2 > wins1: return 2
    return random.choice([1, 2]) # 若平局则随机选一个晋级，保证淘汰赛进行

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="model")
    parser.add_argument("--opening_dir", default="openings_20")
    args = parser.parse_args()

    # 1. 加载资源
    all_openings = load_openings(args.opening_dir)
    model_files = sorted([f for f in os.listdir(args.model_dir) if f.endswith(".onnx")])
    random.shuffle(model_files) # 随机排布种子选手的初始位置

    print(f"参赛选手数量: {len(model_files)}")

    # 2. 预加载所有 Session (注意：如果 92 个模型太大，内存可能会爆，建议随用随开)
    # 这里我们采用“随用随开”策略以节省内存
    def get_session(name):
        return ort.InferenceSession(os.path.join(args.model_dir, name), providers=['CPUExecutionProvider'])

    # 3. 淘汰赛逻辑
    current_round_models = model_files
    round_num = 1
    rankings = [] # 记录淘汰顺序

    while len(current_round_models) > 1:
        print(f"\n--- 第 {round_num} 轮淘汰赛 (当前选手: {len(current_round_models)}) ---")
        next_round_models = []

        # 处理轮空情况（人数为奇数）
        if len(current_round_models) % 2 != 0:
            bye_player = current_round_models.pop()
            next_round_models.append(bye_player)
            print(f"[轮空] {bye_player} 直接晋级")

        for i in range(0, len(current_round_models), 2):
            m1_name = current_round_models[i]
            m2_name = current_round_models[i+1]

            print(f"对阵: {m1_name} vs {m2_name} ... ", end="", flush=True)

            s1 = get_session(m1_name)
            s2 = get_session(m2_name)

            winner_idx = play_best_of_n(s1, s2, all_openings, n=3) # 三局两胜

            if winner_idx == 1:
                next_round_models.append(m1_name)
                rankings.append((m2_name, f"第 {round_num} 轮淘汰"))
                print(f"胜者: {m1_name}")
            else:
                next_round_models.append(m2_name)
                rankings.append((m1_name, f"第 {round_num} 轮淘汰"))
                print(f"胜者: {m2_name}")

            # 及时释放内存（重要！）
            del s1, s2

        current_round_models = next_round_models
        round_num += 1

    # 最后剩下的就是冠军
    champion = current_round_models[0]

    # 4. 输出最终战果
    print("\n" + "!"*20 + " 比赛结束 " + "!"*20)
    print(f"总冠军: {champion}")
    print("\n淘汰顺序（从后往前排名）:")
    for name, info in reversed(rankings):
        print(f"选手: {name[:30]:<30} | 战绩: {info}")

if __name__ == "__main__":
    main()
