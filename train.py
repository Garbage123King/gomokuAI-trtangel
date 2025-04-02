import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from collections import namedtuple
import argparse
import sys
import numpy as np
import torch.nn.functional as F
import math
import copy
import cProfile
import pstats
import datetime
import pickle
import os
import glob
from tqdm import tqdm
import csv

TOTAL_GAME = 65
PRINT_STEP = 1
START_LEARNING_RATE = 0.0001
LATER_LEARNING_RATE = 0.00001
EXPLORATION_PROBABILITY = 0.1
BATCH_SIZE = 128
MEMORY_CAPACITY = 10000
RANDOM_PLAY_GAMES = 50000
TRAIN_STEP = 1


SCORE_BOOK = {0: 1, 1:200, 2: 400, 3: 2000, 4: 10000, 5: 99999}

class Five:
    name = "five"
    size = 400  # 20x20棋盘
    lr = 0.00001
    test_names=[]
    tests=[] # add later
    expects=[]
    
    # 定义 8 种对称变换类型
    transforms = [
        'identity',         # 不旋转
        'rot90',           # 逆时针90°
        'rot180',          # 180°
        'rot270',          # 逆时针270°
        'flip_horizontal', # 水平翻转
        'flip_vertical',   # 垂直翻转
        'transpose',       # 主对角线翻转
        'transpose_inverse' # 反对角线翻转
    ]

    def __init__(self):
        """初始化游戏状态"""
        self.board = [0] * 400  # 0为空，1为玩家1，-1为玩家2
        self.current_player = 1  # 当前玩家，1或-1
        self.done = False  # 游戏是否结束
        self.winner = None  # 胜利者
        self.win = None  # 玩家1和-1的组合计数
        self.move_history = []  # 记录最近5步落子位置
        self.pos_to_combinations = [[] for _ in range(400)]  # 每个位置对应的组合索引
        self.combination_to_poses = []
        self.score =  [[0] * 400 for _ in range(2)]
        self._init_combinations()  # 初始化所有五连子组合
        self.chongsi_count = [[0] * 400 for _ in range(2)]  # 玩家的“冲四”计数器
        # self.chongsi_history = [[] for _ in range(2)]  # 玩家的“冲四”点历史
        self.chongsi_set = [set() for _ in range(2)]  # 玩家的“冲四”

    def _init_combinations(self):
        """初始化所有可能的五连子组合"""
        combination_idx = 0

        # 横向组合
        for row in range(20):
            for col in range(16):  # 20-5+1=16
                positions = [row * 20 + col + i for i in range(5)]
                self.combination_to_poses.append(tuple(positions))
                for pos in positions:
                    self.pos_to_combinations[pos].append(combination_idx)
                combination_idx += 1

        # 纵向组合
        for col in range(20):
            for row in range(16):
                positions = [(row + i) * 20 + col for i in range(5)]
                self.combination_to_poses.append(tuple(positions))
                for pos in positions:
                    self.pos_to_combinations[pos].append(combination_idx)
                combination_idx += 1

        # 主对角线组合（左上到右下）
        for row in range(16):
            for col in range(16):
                positions = [(row + i) * 20 + (col + i) for i in range(5)]
                self.combination_to_poses.append(tuple(positions))
                for pos in positions:
                    self.pos_to_combinations[pos].append(combination_idx)
                combination_idx += 1

        # 副对角线组合（右上到左下）
        for row in range(16):
            for col in range(4, 20):
                positions = [(row + i) * 20 + (col - i) for i in range(5)]
                self.combination_to_poses.append(tuple(positions))
                for pos in positions:
                    self.pos_to_combinations[pos].append(combination_idx)
                combination_idx += 1

        # 初始化胜负计数器
        self.win = [[0] * combination_idx for _ in range(2)]

        for poses in self.combination_to_poses:
            for each_pos in poses:
                self.score[0][each_pos] += SCORE_BOOK[0]
                self.score[1][each_pos] += SCORE_BOOK[0]

    def available_moves(self):
        """返回当前可用的落子位置"""
        return [i for i, x in enumerate(self.board) if x == 0]

    def make_move(self, position):
        """执行落子并更新游戏状态"""
        if self.current_player == 1:
            making = 0
            defending = 1
        else:
            making = 1
            defending = 0
        isValid = False
        if self.board[position] == 0:
            self.board[position] = self.current_player
            self.move_history.append(position)
            if len(self.move_history) > 5:
                self.move_history.pop(0)
            # 更新组合计数
            for comb_idx in self.pos_to_combinations[position]:
                self.win[making][comb_idx] += 1
                if self.win[making][comb_idx] == 5:
                    self.done = True
                    self.winner = 1
                # 更新making的score
                if self.win[defending][comb_idx] > 0:
                    change_score = 0
                else:
                    new_stones = self.win[making][comb_idx]
                    old_stones = new_stones - 1
                    old_score = SCORE_BOOK[old_stones]
                    new_score = SCORE_BOOK[new_stones]
                    change_score = new_score - old_score
                    for pos in self.combination_to_poses[comb_idx]:
                        self.score[making][pos] += change_score
                        if new_stones == 3 and self.board[pos] == 0:
                            self.chongsi_count[making][pos] += 1
                            self.chongsi_set[making].add(pos)
                # 更新defending的score
                if self.win[making][comb_idx] > 1: #以前就已阻塞
                    change_score = 0
                else:   #新阻塞
                    stones = self.win[defending][comb_idx]
                    old_score = SCORE_BOOK[stones]
                    new_score = 0
                    change_score = new_score - old_score
                    for pos in self.combination_to_poses[comb_idx]:
                        self.score[defending][pos] += change_score
                        if stones == 3 and self.board[pos] == 0:
                            self.chongsi_count[defending][pos] -= 1
                            if self.chongsi_count[defending][pos] == 0:
                                self.chongsi_set[defending].remove(pos)
            isValid = True
        self.current_player = -self.current_player  # 切换玩家
        return isValid

    def unmake_move(self, position):
        """撤销落子并恢复状态"""
        if self.board[position] == 0:
            raise ValueError("Trying to unmake an empty position")
        else:
            if self.current_player == 1:
                unmaking = 1
                profiting = 0
            else:
                unmaking = 0
                profiting = 1
            self.board[position] = 0
            if position in self.move_history:
                self.move_history.remove(position)
            # 更新组合计数
            for comb_idx in self.pos_to_combinations[position]:
                self.win[unmaking][comb_idx] -= 1
                # 更新profiting的score
                if self.win[unmaking][comb_idx] > 0:  #回收完还有子
                        change_score = 0     #那么依然阻塞
                else:
                    stones = self.win[profiting][comb_idx]
                    old_score = 0
                    new_score = SCORE_BOOK[stones]
                    change_score = new_score - old_score
                    for pos in self.combination_to_poses[comb_idx]:
                        self.score[profiting][pos] += change_score
                        if stones == 3 and self.board[pos] == 0:
                            self.chongsi_count[profiting][pos] += 1
                            self.chongsi_set[profiting].add(pos)

                # 更新unmaking的score
                if self.win[profiting][comb_idx] > 0:  #unmake前后都是阻塞状态
                        change_score = 0    
                else:                       #unmake前后都不阻塞
                    new_stones = self.win[unmaking][comb_idx]
                    old_stones = new_stones + 1
                    old_score = SCORE_BOOK[old_stones]
                    new_score = SCORE_BOOK[new_stones]
                    change_score = new_score - old_score
                    for pos in self.combination_to_poses[comb_idx]:
                        self.score[unmaking][pos] += change_score
                        if old_stones == 3 and self.board[pos] == 0:
                            self.chongsi_count[unmaking][pos] -= 1
                            if self.chongsi_count[unmaking][pos] == 0:
                                self.chongsi_set[unmaking].remove(pos)

            self.done = False
            self.winner = None
            self.current_player = -self.current_player

    def is_winner(self, player):
        """检查指定玩家是否获胜"""
        if player == 1:
            return any(count == 5 for count in self.win[0])
        else:
            return any(count == 5 for count in self.win[1])

    def is_draw(self):
        """检查游戏是否平局"""
        return 0 not in self.board

    def is_terminal(self):
        """检查游戏是否结束"""
        return self.is_winner(1) or self.is_winner(-1) or self.is_draw()
    
    def get_state(self, device):
        """获取游戏状态的张量表示，用于深度学习"""
        board_2d = np.array(self.board).reshape(20, 20)
        # 当前玩家的棋子通道
        if self.current_player == 1:
            channel_own = (board_2d == 1).astype(np.float32)
            channel_opp = (board_2d == -1).astype(np.float32)
            own_chongsi_set = self.chongsi_set[0]  # 玩家 1 的冲四点
            opp_chongsi_set = self.chongsi_set[1]  # 玩家 -1 的冲四点
        else:
            channel_own = (board_2d == -1).astype(np.float32)
            channel_opp = (board_2d == 1).astype(np.float32)
            own_chongsi_set = self.chongsi_set[1]  # 玩家 -1 的冲四点
            opp_chongsi_set = self.chongsi_set[0]  # 玩家 1 的冲四点

        # 创建冲四点通道
        channel_own_chongsi = np.zeros((20, 20), dtype=np.float32)
        for pos in own_chongsi_set:
            row, col = divmod(pos, 20)
            channel_own_chongsi[row, col] = 1

        channel_opp_chongsi = np.zeros((20, 20), dtype=np.float32)
        for pos in opp_chongsi_set:
            row, col = divmod(pos, 20)
            channel_opp_chongsi[row, col] = 1

        # 历史落子通道（最近5步）
        history_channels = []
        for i in range(5):
            if i < len(self.move_history):
                move_pos = self.move_history[-(i+1)]
                channel = np.zeros((20, 20), dtype=np.float32)
                row, col = divmod(move_pos, 20)
                channel[row, col] = 1
            else:
                channel = np.zeros((20, 20), dtype=np.float32)
            history_channels.append(channel)

        # 组合成9通道状态
        state = np.stack(
            [channel_own, channel_opp] + history_channels + [channel_own_chongsi, channel_opp_chongsi],
            axis=0
        )
        return torch.tensor(state, dtype=torch.float32, device=device)

    def clone(self):
        """返回游戏状态的深拷贝"""
        return copy.deepcopy(self)

    def get_reward(self, player):
        """获取指定玩家的奖励"""
        if self.is_winner(player):
            return 1
        elif self.is_winner(-player):
            return -1
        return 0
    
    def get_score_policy(self):
        # 获取可用落子位置
        available = self.available_moves()
        if not available:
            return [0] * self.size  # 没有可用位置，返回全 0 列表

        # 计算每个位置的 score，取 score[0][pos] 和 score[1][pos] 中的较大值
        max_scores = [max(self.score[0][pos], self.score[1][pos]) for pos in range(self.size)]

        # 生成策略：只对可用位置分配 score，不可用的位置为 0
        policy = [max_scores[pos] if pos in available else 0 for pos in range(self.size)]

        # 归一化策略
        total = sum(policy)
        if total > 0:
            policy = [p / total for p in policy]  # 总和大于 0 时，归一化
        else:
            # 总和为 0 时，在可用位置上均匀分配
            policy = [1 / len(available) if pos in available else 0 for pos in range(self.size)]

        return policy

Five.test_names.append("1方竖线1步胜")
Five.tests.append([77, 12, 97, 38, 137, 27, 157, 11])
Five.expects.append([117, 1])

Five.test_names.append("1方横线1步胜")
Five.tests.append([56, 157, 59, 97, 57, 137, 58, 133])
Five.expects.append([55, 1])

Five.test_names.append("1方/斜线1步胜")
Five.tests.append([131, 12, 112, 38, 93, 27, 55, 0])
Five.expects.append([74, 1])

Five.test_names.append("1方反斜线1步胜")
Five.tests.append([0, 12, 21, 38, 42, 27, 63, 9])
Five.expects.append([84, 1])

Five.test_names.append("-1方横线1步胜")
Five.tests.append([4, 5, 97, 6, 11, 7, 157, 8, 99])
Five.expects.append([9, 1])

Five.test_names.append("-1方横线1步胜")
Five.tests.append([9, 5, 97, 6, 11, 7, 157, 8, 99])
Five.expects.append([4, 1])

Five.test_names.append("-1方竖线1步胜")
Five.tests.append([190, 210, 170, 230, 169, 250, 168, 270, 189])
Five.expects.append([290, 1])

Five.test_names.append("-1方已活四因此1必输")
Five.tests.append([6, 77, 12, 97, 38, 137, 27, 117])
Five.expects.append([57, -1])

Five.test_names.append("死活题第4题3步胜，但这是挡对方活三所以相对简单吧")
Five.tests.append([210, 211, 230, 231, 270, 250, 228, 208, 188, 207, 189, 169, 168, 147, 192, 249, 173, 172])
Five.expects.append([251, 1])

Five.test_names.append("死活题第8题，1步活三后，207位置1子双杀")
Five.tests.append([210, 191, 229, 208, 248, 269, 270, 251, 289, 266, 227, 226])
Five.expects.append([230, 1])

Five.test_names.append("1方有活三，一步活四")
Five.tests.append([390, 310, 389, 386, 311, 306, 388, 304])
Five.expects.append([391, 1])

Five.test_names.append("1方乱下了一步，-1方大优（必胜），323，262，302都可以")
Five.tests.append([342, 304, 286, 303, 306, 283, 284, 305, 384])
Five.expects.append([[323, 262, 302], 1])

Five.test_names.append("gomocup的一个开局，下67就输了，一定要下107")
Five.tests.append([29, 89, 69, 108, 70, 71, 91, 110, 112, 133, 28, 49, 27, 26, 48, 111, 68])
Five.expects.append([107, 0])

Five.test_names.append("输给蜗牛连珠，千万不能下192，下就输")
Five.tests.append([169,188,187,168,208,229,227,189,210,207,211,190,212,209,213,214])
Five.expects.append([[171, 269], 0])

Five.test_names.append("该局面白必胜，冲四后双活三，随便赢，228也可以")
Five.tests.append([169,188,187,168,208,229,227,189,210,207,211,190,212,209,213,214,192])
Five.expects.append([[249, 228], 1])

Five.test_names.append("欢乐五子棋第25关，有点难，看似很多机会，实则都会被反杀，只能下一个地方")
Five.tests.append([104,125,106,107,108,87,109,88,128,129,148,187,167,186,205,206])
Five.expects.append([127, 1])

class FiveNet(nn.Module):
    def __init__(self):
        super(FiveNet, self).__init__()

        def residual_block(in_channels, out_channels, stride=1):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels)
            ]
            return nn.Sequential(*layers)

        self.conv1 = nn.Conv2d(9, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 增加残差块以适应20x20棋盘
        self.res_block1 = residual_block(64, 64)
        self.res_block2 = residual_block(64, 128)
        self.res_block3 = residual_block(128, 128)
        self.res_block4 = residual_block(128, 256)
        self.res_block5 = residual_block(256, 256)
        self.res_block6 = residual_block(256, 512)
        self.res_block7 = residual_block(512, 512)

        self.skip_conv2 = nn.Conv2d(64, 128, kernel_size=1)
        self.skip_conv4 = nn.Conv2d(128, 256, kernel_size=1)
        self.skip_conv6 = nn.Conv2d(256, 512, kernel_size=1)

        # 策略头
        self.policy_conv = nn.Conv2d(512, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * 20 * 20, 400)

        # 价值头
        self.value_conv = nn.Conv2d(512, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(20 * 20, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        residual = x
        x = self.res_block1(x)
        x = x + residual
        x = self.relu(x)

        residual = self.skip_conv2(x)
        x = self.res_block2(x)
        x = x + residual
        x = self.relu(x)

        residual = x
        x = self.res_block3(x)
        x = x + residual
        x = self.relu(x)

        residual = self.skip_conv4(x)
        x = self.res_block4(x)
        x = x + residual
        x = self.relu(x)

        residual = x
        x = self.res_block5(x)
        x = x + residual
        x = self.relu(x)

        residual = self.skip_conv6(x)
        x = self.res_block6(x)
        x = x + residual
        x = self.relu(x)

        residual = x
        x = self.res_block7(x)
        x = x + residual
        x = self.relu(x)

        # 策略头
        policy = self.policy_conv(x)
        policy = policy.view(-1, 2 * 20 * 20)
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)

        # 价值头
        value = self.value_conv(x)
        value = value.view(-1, 20 * 20)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))

        return policy, value

def middle_test(Game, net, exploration_probability, test_case, expect, device):
    net.eval()
    case_game = Game()
    for move in test_case:
        case_game.make_move(move)
    with torch.no_grad():
        net.eval()
        state = case_game.get_state(device).unsqueeze(0).to(device)  # 添加批次维度并移动到设备
        test_policy, test_value = net(state)
        max_prob_move = torch.argmax(test_policy[0]).item()
        # 处理期望策略
        if isinstance(expect[0], (int, float)):  # 如果 expect[0] 是数字
            print("期望策略：", expect[0])
            print("实际策略：", max_prob_move, "prior：", test_policy[0][max_prob_move].item(), f"动作{expect[0]}的prior：", test_policy[0][expect[0]].item())
        elif isinstance(expect[0], (list, tuple)):  # 如果 expect[0] 是数组
            print("期望策略：", expect[0])
            real_priors=[]
            for exp_move in expect[0]:
                real_priors.append(test_policy[0][exp_move].item())
            print("实际策略：", max_prob_move, "价值：", test_policy[0][max_prob_move].item(),  f"动作{expect[0]}的prior：", real_priors)

        print("期望价值：", expect[1])
        print("实际价值：", test_value.item())  # 网络预测的状态价值
    test_game = Game()
    test_moves = []
    test_steps = 0
    while not (test_game.is_winner(1) or test_game.is_winner(-1) or test_game.is_draw()):
        if random.random() < exploration_probability: # 探索
            use_random = True
            test_move = random.choice(test_game.available_moves())  # Select from available moves
        else:
            use_random = False
            test_state = test_game.get_state(device).unsqueeze(0).to(device)  # Add batch dimension
            with torch.no_grad():
                test_policy, _ = net(test_state)  # Unpack tuple
            test_move = torch.argmax(test_policy, dim=1).item()  # Get move from policy

        test_moves.append((use_random, test_move))
        if test_move not in test_game.available_moves():
            # print("wrong move")
            break
        test_game.make_move(test_move)
        test_steps += 1
        if test_steps > 100: #重复下多次
            print("too many steps")
            break
    #test game end    
    for test_m in test_moves:
        if test_m[0] == True:
            print("(rand)", test_m[1], end = ", ")
        else:
            print(test_m[1], end = ", ")
    if test_game.is_winner(1):
        print("result: x win")
    elif test_game.is_winner(-1):
        print("result: o win")
    elif test_game.is_draw():
        print("result: draw")
    else:
        print("wrong move")

def apply_transform(state, transform_type):
    """
    应用对称变换到状态张量
    :param state: 形状为 [9, 20, 20] 的张量
    :param transform_type: 字符串，表示变换类型
    :return: 变换后的状态张量
    """
    if transform_type == 'identity':
        return state.clone()
    elif transform_type == 'rot90':
        return torch.rot90(state, k=1, dims=(1, 2))
    elif transform_type == 'rot180':
        return torch.rot90(state, k=2, dims=(1, 2))
    elif transform_type == 'rot270':
        return torch.rot90(state, k=3, dims=(1, 2))
    elif transform_type == 'flip_horizontal':
        return torch.flip(state, dims=[2])  # 沿宽度（列）翻转
    elif transform_type == 'flip_vertical':
        return torch.flip(state, dims=[1])  # 沿高度（行）翻转
    elif transform_type == 'transpose':
        return torch.transpose(state, 1, 2)  # 主对角线翻转
    elif transform_type == 'transpose_inverse':
        # 反对角线翻转：先 180 度旋转，再转置
        rotated = torch.rot90(state, k=2, dims=(1, 2))
        return torch.transpose(rotated, 1, 2)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

def transform_policy(policy, transform_type):
    """
    根据变换类型变换 policy
    :param policy: 400 维的列表或张量
    :param transform_type: 字符串，表示变换类型
    :return: 变换后的 policy
    """
    policy_tensor = torch.tensor(policy).reshape(20, 20)
    if transform_type == 'identity':
        transformed_policy = policy_tensor
    elif transform_type == 'rot90':
        transformed_policy = torch.rot90(policy_tensor, k=1)
    elif transform_type == 'rot180':
        transformed_policy = torch.rot90(policy_tensor, k=2)
    elif transform_type == 'rot270':
        transformed_policy = torch.rot90(policy_tensor, k=3)
    elif transform_type == 'flip_horizontal':
        transformed_policy = torch.flip(policy_tensor, dims=[1])
    elif transform_type == 'flip_vertical':
        transformed_policy = torch.flip(policy_tensor, dims=[0])
    elif transform_type == 'transpose':
        transformed_policy = torch.transpose(policy_tensor, 0, 1)
    elif transform_type == 'transpose_inverse':
        # 反对角线翻转：先 180 度旋转，再转置
        rotated = torch.rot90(policy_tensor, k=2)
        transformed_policy = torch.transpose(rotated, 0, 1)
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")
    return transformed_policy.flatten().tolist()

# 训练
def train(Game, GameNet):
    # 检查 GPU 可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练将使用设备: {device}")
    
    # 指定数据目录
    directory = './selfplay'

    # 获取所有 .csv 文件
    csv_files = glob.glob(os.path.join(directory, '*.csv'))
    if not csv_files:
        print(f"目录 {directory} 中没有找到 .csv 文件，请先运行 selfplay 生成数据。")
        return

    # 打乱 csv_files 的顺序
    random.shuffle(csv_files)

    print(f"找到 {len(csv_files)} 个 .csv 文件")

    # 初始化网络和优化器
    real_net = GameNet().to(device)

    # 检查并加载已有模型
    model_path = './model/model.pth'
    if os.path.exists(model_path):
        real_net.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print("No pre-trained model found, starting from scratch.")

    real_net.train()

    optimizer = optim.Adam(real_net.parameters(), lr=Game.lr)

    # 将文件列表分成批次，每批次最多 1000 个文件
    file_batch_size  = 1000
    num_file_batches  = (len(csv_files) + file_batch_size  - 1) // file_batch_size   # 计算总批次数

    # 逐批次处理文件
    for file_batch_idx in range(num_file_batches):
        start = file_batch_idx * file_batch_size
        end = min((file_batch_idx + 1) * file_batch_size, len(csv_files))
        batch_files = csv_files[start:end]
        print(f"处理文件批次 {file_batch_idx + 1}/{num_file_batches}，包含 {len(batch_files)} 个文件")


        # 加载所有数据
        all_data = []
        for file in tqdm(batch_files, desc=f"加载批次 {file_batch_idx + 1}/{num_file_batches}", unit="file"):
            with open(file, 'r') as csvfile:
                reader = csv.reader(csvfile, quotechar='"')
                header = next(reader)  # 跳过表头
                for row in reader:
                    moves = [int(m) for m in row[0].split(',')]  # 解析 moves
                    all_mcts_iterations = [int(it) for it in row[1].split(',')]  # 解析 mcts_iterations
                    policies = []
                    for i in range(2, len(row) - 1):  # 解析每个 policy
                        policies.append([float(p) for p in row[i].split(',')])
                    value_target = float(row[-1])  # 解析 value_target
                    all_data.append((moves, all_mcts_iterations, policies, value_target))
        
        if not all_data:
            print("加载的数据为空，无法进行训练。")
            return
        
        print("成功加载%d条数据" % len(all_data))

        # 还原 state 并准备训练数据
        states = []
        policy_targets = []
        value_targets = []

        game = Game()

        for moves, all_mcts_iterations, policies, game_result in tqdm(all_data, desc="还原 state 进度", unit="item"):
            game = Game()
            for i, (move, mcts_iterations, policy_target) in enumerate(zip(moves, all_mcts_iterations, policies)):
                if mcts_iterations == 600:  # 仅对 mcts_iterations == 600 的步数进行训练
                    state = game.get_state('cpu')  # 获取当前状态
                    value_target = game_result if (i % 2 == 0) else -game_result
                    # 应用数据增强
                    for transform_type in Game.transforms:
                        # 变换 state
                        transformed_state = apply_transform(state, transform_type)
                        # 变换 policy_target
                        transformed_policy = transform_policy(policy_target, transform_type)
                        # 添加到训练数据
                        states.append(transformed_state)
                        policy_targets.append(torch.tensor(transformed_policy, dtype=torch.float32))
                        value_targets.append(value_target)
                
                # 不管mcts_iterations是不是600，都必须执行移动
                game.make_move(move)  # 执行移动

        states = torch.stack(states)  # 在 CPU 上堆叠
        policy_targets = torch.stack(policy_targets)  # 在 CPU 上堆叠
        value_targets = torch.tensor(value_targets, dtype=torch.float32)  # 在 CPU 上堆叠

        
        print(f"游戏局数: {len(all_data)}, 增强后数据量: {len(states)}")

        # 训练循环
        for epoch in range(TRAIN_STEP):
            real_net.train()
            # 随机打乱数据
            indices = torch.randperm(len(states))
            num_batches = (len(states) + BATCH_SIZE - 1) // BATCH_SIZE  # 计算批次总数
            # 使用 tqdm 添加进度条
            for start in tqdm(range(0, len(states), BATCH_SIZE), 
                            total=num_batches, 
                            desc=f"Epoch {epoch} 训练进度", 
                            unit="batch"):
                end = min(start + BATCH_SIZE, len(states))
                batch_indices = indices[start:end]

                # 将当前批次移动到 GPU
                batch_states = states[batch_indices].to(device)
                batch_policy_targets = policy_targets[batch_indices].to(device)
                batch_value_targets = value_targets[batch_indices].to(device)
                
                # 前向传播和损失计算
                optimizer.zero_grad()
                policy_pred, value_pred = real_net(batch_states)
                # Policy 损失：交叉熵
                policy_loss = -torch.mean(torch.sum(batch_policy_targets * torch.log(policy_pred + 1e-8), dim=1))
                # Value 损失：均方误差
                value_loss = F.mse_loss(value_pred.squeeze(), batch_value_targets)
                # 总损失
                loss = policy_loss + value_loss
                loss.backward()
                optimizer.step()
            
            if epoch % PRINT_STEP == 0:
                print(f"Epoch {epoch}: Loss = {loss.item()}")

        # 清空当前批次数据，释放内存
        del all_data, states, policy_targets, value_targets
        torch.cuda.empty_cache()  # 如果使用 GPU，释放显存
        print(f"批次 {file_batch_idx + 1} 处理完成，已清空内存")
            
    
    # 保存模型
    model_pth_path = 'model/candidate.pth'  #训练用
    model_onnx_path = 'model/candidate.onnx'  #selfplay的tensorrt用
    model_pt_path = 'model/candidate.pt'    #导出pt，以后转gomocup ncnn用

    save_dir = os.path.dirname(model_pth_path)

    # 如果目录不存在，则创建它
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    real_net.eval()

    # 保存 PyTorch 模型 (.pth)
    torch.save(real_net.state_dict(), model_pth_path)
    print(f"PyTorch 模型已保存至 {model_pth_path}")

    # 保存 ONNX 模型
    dummy_input = torch.randn(1, 9, 20, 20).to(device)
    torch.onnx.export(
        real_net,
        dummy_input,
        model_onnx_path,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "policy": {0: "batch_size"},
            "value": {0: "batch_size"}
        }
    )
    print(f"ONNX 模型已保存至 {model_onnx_path}")

    # 保存 pt 模型
    # Create a dummy input tensor matching your input shape [1, 9, 20, 20]
    x = torch.rand(1, 9, 20, 20, device=device)

    # Trace the model to export it to TorchScript
    mod = torch.jit.trace(real_net, x)
    mod.save(model_pt_path)  # Save as TorchScript model

    print(f"ncnn 模型已保存至 {model_pt_path}")

    # 测试一下，看看训练是否有效
    for i in range(len(Game.tests)):
        print(f"---------------------------测试{i+1}: {Game.test_names[i]}----------------------------")
        test_case = Game.tests[i]
        expect = Game.expects[i]
        middle_test(Game, real_net, 0.1, test_case, expect, device)
        print("-----------------------------------------------------------")


if __name__ == "__main__":
    Env = Five
    GameNet = FiveNet

    train(Env, GameNet)
    # cProfile.run('train(Five, FiveNet)', 'profile_output')
    # # 解析并排序结果
    # stats = pstats.Stats('profile_output')
    # stats.strip_dirs().sort_stats('tottime').print_stats(10)  # 按函数运行时间排序，显示前 10 个
    # sys.exit(0)

    
