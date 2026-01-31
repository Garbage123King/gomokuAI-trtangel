import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import subprocess
import os
from train import *

# 颜色定义 (ANSI 转义码)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

STATE_LAYER_NUM = 3
size = 8  # 棋盘大小

class Five:
    def __init__(self, size):
        self.size = size
        self.board = [0] * (size * size)  # 0:空, 1:黑, -1:白
        self.current_player = 1
        self.all_moves = []

    def make_move(self, position):
        if self.board[position] != 0:
            return False
        self.board[position] = self.current_player
        self.current_player *= -1
        self.all_moves.append(position)
        return True

def get_state(size, actions):
    cmd = ['./trtangel_release', 'getstate', str(size)] + [str(a) for a in actions]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("C++ getstate error: " + result.stderr)
    state = [float(x) for x in result.stdout.strip().split()]
    return state

# 初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KataNet().to(device)
model.load_state_dict(torch.load(f'model/model.pth', map_location=device))
model.eval()
game = Five(size)

def print_board(policy=None):
    """在终端打印棋盘和策略"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # 打印列标
    header = "    " + " ".join([f"{i:2}" for i in range(size)])
    print(header)
    print("   " + "-" * (size * 3))

    for r in range(size):
        row_str = f"{r:2} |"
        for c in range(size):
            pos = r * size + c
            piece = game.board[pos]
            if piece == 1:
                row_str += " X "  # 黑棋
            elif piece == -1:
                row_str += " O "  # 白棋
            else:
                row_str += " . "  # 空地
        print(row_str)

def show_analysis(policy, reward):
    """显示前 5 个推荐走法和胜率估算"""
    print(f"\n{Colors.HEADER}--- AI 分析 ---{Colors.ENDC}")
    print(f"当前玩家: {'黑 (X)' if game.current_player == 1 else '白 (O)'}")
    print(f"胜率评估 (Value): {reward:+.3f} (范围 -1 到 1)")
    
    available = [p for p in range(size * size) if game.board[p] == 0]
    probs = sorted([(p, policy[p]) for p in available], key=lambda x: x[1], reverse=True)

    print(f"\n推荐走法 (Top 5):")
    for i, (pos, prob) in enumerate(probs[:5]):
        r, c = divmod(pos, size)
        color = Colors.GREEN if i == 0 else Colors.YELLOW
        print(f"{color}#{i+1}: ({r}, {c}) 概率: {prob:.4f}{Colors.ENDC}")

def get_ai_prediction():
    """获取模型输出"""
    state_data = get_state(size, game.all_moves)
    state_tensor = torch.tensor(state_data, dtype=torch.float32, device=device).view(1, STATE_LAYER_NUM, 20, 20)
    
    with torch.no_grad():
        board_sizes = torch.tensor([size], dtype=torch.float32, device=device)
        policy_logits, value_logits = model(state_tensor, board_sizes)

        # 策略处理
        policy_current = policy_logits[0, 0, :size, :size].flatten()
        policy = F.softmax(policy_current, dim=0).cpu().numpy()

        # 价值处理 (Win, Loss, Draw)
        v_probs = F.softmax(value_logits, dim=1).cpu().numpy()[0]
        reward = v_probs[0] - v_probs[1] # Win - Loss
        
    return policy, reward

# 游戏主循环
try:
    while True:
        policy, reward = get_ai_prediction()
        print_board()
        show_analysis(policy, reward)

        try:
            user_input = input(f"\n请输入坐标 (行 列)，例如 '3 4'，或输入 'q' 退出: ").strip()
            if user_input.lower() == 'q':
                break
            
            r, c = map(int, user_input.split())
            if not (0 <= r < size and 0 <= c < size):
                print(f"{Colors.RED}错误: 坐标超出范围!{Colors.ENDC}")
                continue
                
            if not game.make_move(r * size + c):
                print(f"{Colors.RED}错误: 该位置已有棋子!{Colors.ENDC}")
                input("按回车继续...")
                
        except ValueError:
            print(f"{Colors.RED}输入格式错误，请输入两个数字!{Colors.ENDC}")
            continue

except KeyboardInterrupt:
    print("\n游戏结束")