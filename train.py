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
from datetime import datetime
import pickle
import os
import glob
from tqdm import tqdm
import csv
import subprocess
import re

STATE_LAYER_NUM = 3

def random_size(size):
    choices = list(range(size))
    return random.choice(choices)

def random_size_not_number(size, number):
    choices = list(range(size))
    choices.remove(number)
    return random.choice(choices)

def range_numbers_not_poses(size, rowrange, colrange, notposes,num):
    rowes = list(range(rowrange[0], rowrange[1]))
    coles = list(range(colrange[0], colrange[1]))
    choices=[]
    for row in rowes:
        for col in coles:
            pos = row*size + col
            if pos not in notposes:
                choices.append(pos)
    return random.sample(choices, num)


TOTAL_GAME = 65
BATCH_SIZE = 128
TRAIN_STEP = 1

'''
网络深度与宽度：你的 KataNet 有 6 个残差块（OrdiBlock 和 GPoolBlock），参数量不算巨大。
Batch Size 的关系：你目前的 BATCH_SIZE = 256。
在深度学习中有一个线性缩放规则：当你增加 Batch Size 时，通常也需要等比例增加 lr。
对于 256 这样规模的 Batch，通常 lr 的起始值会在 10^{-3} 到 10^{-4} 之间。6 x 10^{-5} 有点像收敛遇到瓶颈后的降频值。
从零训练 vs 微调：
从零开始：建议从 0.001 或 0.0005 开始。
微调（已有基础）：0.0001 到 0.00001 是合理的。
'''

LEARNING_RATE = 0.0005

def get_state(size, actions):
    # Construct the command
    cmd = ['./trtangel_release', 'getstate', str(size)] + [str(a) for a in actions]
    
    # Run the C++ executable
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check for errors
    if result.returncode != 0:
        raise RuntimeError("Error in getstate: " + result.stderr)
    
    # Parse the output into a list of floats
    state_str = result.stdout.strip()
    state = [float(x) for x in state_str.split()]
    
    return state

class Five:
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

Five.test_names.append("1方竖线1步成五")
Five.tests.append((20, [77, 12, 97, 38, 137, 27, 157, 11]))
Five.expects.append([117, 1])

Five.test_names.append("1方横线1步成五")
Five.tests.append((20, [56, 157, 59, 97, 57, 137, 58, 133]))
Five.expects.append([55, 1])

Five.test_names.append("1方/斜线1步成五")
Five.tests.append((20, [131, 12, 112, 38, 93, 27, 55, 0]))
Five.expects.append([74, 1])

Five.test_names.append("1方反斜线1步成五")
Five.tests.append((20, [0, 12, 21, 38, 42, 27, 63, 9]))
Five.expects.append([84, 1])

Five.test_names.append("-1方横线1步成五")
Five.tests.append((20, [4, 5, 97, 6, 11, 7, 157, 8, 99]))
Five.expects.append([9, 1])

Five.test_names.append("-1方横线1步成五")
Five.tests.append((20, [9, 5, 97, 6, 11, 7, 157, 8, 99]))
Five.expects.append([4, 1])

Five.test_names.append("-1方竖线1步成五")
Five.tests.append((20, [190, 210, 170, 230, 169, 250, 168, 270, 189]))
Five.expects.append([290, 1])

Five.test_names.append("-1方已活四因此1必输")
Five.tests.append((20, [6, 77, 12, 97, 38, 137, 27, 117]))
Five.expects.append([57, -1])

Five.test_names.append("死活题第4题3步胜，但这是挡对方活三所以相对简单吧，190也可以胜，但是好像很复杂")
Five.tests.append((20, [210, 211, 230, 231, 270, 250, 228, 208, 188, 207, 189, 169, 168, 147, 192, 249, 173, 172]))
Five.expects.append([[251, 190], 1])

Five.test_names.append("死活题第8题，1步活三后，207位置1子双杀")
Five.tests.append((20, [210, 191, 229, 208, 248, 269, 270, 251, 289, 266, 227, 226]))
Five.expects.append([230, 1])

Five.test_names.append("1方有活三，一步活四")
Five.tests.append((20, [390, 310, 389, 386, 311, 306, 388, 304]))
Five.expects.append([391, 1])

Five.test_names.append("1方乱下了一步，-1方大优（必胜），有至少十个点都必胜")
Five.tests.append((20, [342, 304, 286, 303, 306, 283, 284, 305, 384]))
Five.expects.append([[323, 262, 302], 1])

Five.test_names.append("gomocup的一个开局，下67就输了，一定要下107")
Five.tests.append((20, [29, 89, 69, 108, 70, 71, 91, 110, 112, 133, 28, 49, 27, 26, 48, 111, 68]))
Five.expects.append([107, 0])

Five.test_names.append("输给蜗牛连珠，千万不能下192，下就输")
Five.tests.append((20, [169,188,187,168,208,229,227,189,210,207,211,190,212,209,213,214]))
Five.expects.append([[171, 269], 0])

Five.test_names.append("该局面白必胜，冲四后双活三，随便赢，228也可以")
Five.tests.append((20, [169,188,187,168,208,229,227,189,210,207,211,190,212,209,213,214,192]))
Five.expects.append([[249, 228], 1])

Five.test_names.append("欢乐五子棋第25关，有点难，看似很多机会，实则都会被反杀，只有两个地方能下")
Five.tests.append((20, [104,125,106,107,108,87,109,88,128,129,148,187,167,186,205,206]))
Five.expects.append([[127, 169], 1])

Five.test_names.append("输给蜗牛连珠II，走231就输了啊")
Five.tests.append((20, [169,188,210,168,208,209,230,190,228,187,189,227]))
Five.expects.append([[186, 206], -0.5])

Five.test_names.append("神之大跳，hzy-katagomoku，胜率95.8%，后面几个位置的话胜率最高也才60%")
Five.tests.append((20, [169,188,210,168,208,209,230,190,228,187,189,227,186,206,225,147,207,249,149]))
Five.expects.append([[144, 128, 145, 126], 1])

Five.test_names.append("开局仅6步，千万不能下75，白直接135位置连续活三必胜形了")
Five.tests.append((20, [76, 114, 115, 134, 94, 136]))
Five.expects.append([[135, 113, 73], 1])

Five.test_names.append("下什么不重要，已经输了，问题是value是-1吗，别整个-0.1")
Five.tests.append((20, [76,114,115,134,94,136,75,135,137,156,93,132,133,176,116,198,177,197]))
Five.expects.append([[155, 113], -1])

Five.test_names.append("白已经必赢，它意识到了吗，value是1吗，别整个0.3")
Five.tests.append((20, [76,114,115,134,94,136,75]))
Five.expects.append([135, 1])

Five.test_names.append("白一步胜，这胜率不是1真的是扭曲了")
Five.tests.append((20, [17, 59, 16, 79, 15, 99, 119, 14, 139, 13, 159, 12, 118, 19, 307]))
Five.expects.append([39, 1])

Five.test_names.append("白一步胜，这胜率不是1真的是扭曲了")
Five.tests.append((20, [17, 59, 16, 79, 15, 99, 119, 14, 139, 13, 159, 12, 118, 19, 312]))
Five.expects.append([39, 1])

Five.test_names.append("别人活三都不挡，你是真牛啊")
Five.tests.append((20, [152, 154, 74, 55, 211, 76, 212, 97]))
Five.expects.append([[118, 34], 1])

Five.test_names.append("一步活四你都不下啊，没必要214拦别人")
Five.tests.append((20, [152, 154, 74, 55, 211, 76, 212, 97, 213]))
Five.expects.append([[118, 34], 1])

Five.test_names.append("-1方已活四，1方价值为-1，你却还觉得自己0.15512")
Five.tests.append((20, [152, 154, 74, 55, 211, 76, 212, 97, 213, 34]))
Five.expects.append([118, -1])

Five.test_names.append("-1方已活四，1方价值为-1，你却还觉得自己0.15512")
Five.tests.append((20, [152, 154, 74, 55, 211, 76, 212, 97, 213, 118]))
Five.expects.append([34, -1])

for i in range(0, 11):
    Five.test_names.append("第一排横线一步成五")
    Five.tests.append((15, [i, 154, i+1, 55, i+2, 76, i+3, 97]))
    Five.expects.append([i+4, 1])

Five.test_names.append("第二排横线一步成五")
Five.tests.append((15, [15, 99, 16, 6, 17, 33, 18, 27]))
Five.expects.append([19, 1])

Five.test_names.append("第十五排横线一步成五")
Five.tests.append((15, [210, 77, 211, 123, 212, 69, 213, 33]))
Five.expects.append([214, 1])

# Five.test_names.append("nan错误")
# Five.tests.append((14, []))
# Five.expects.append([0, 0])

Five.test_names.append("对手已活四，价值应为-1")
Five.tests.append((12, [66, 64, 67, 77, 90, 53, 78, 93, 54]))
Five.expects.append([[42, 102], -1])

Five.test_names.append("一子双活三，训3代了这都看不出来")
Five.tests.append((14, [127, 30, 102, 9, 187, 178, 56, 32, 99, 62, 45, 168, 52, 136, 101, 100, 115, 144, 129, 143, 87, 73]))
Five.expects.append([128, 1])

Five.test_names.append("一子冲四活三")
Five.tests.append((15, [59, 176, 132, 193, 208, 204, 1, 34, 190, 140, 11, 63, 167, 55, 172, 171, 144, 142, 118, 141, 139, 126, 156, 110, 94, 95, 155, 125, 80, 66, 127, 111, 33, 96, 81, 44, 79, 78, 158, 157, 173, 186, 109, 124]))
Five.expects.append([49, 1])

一步成五集={}
一步成五集["name"]="一步成五集"
一步成五集["cases"]=[]
for size in range(9, 21):
    for row in range(0, size):
        for col in range(0, size - 3):
            pos = row*size + col
            if col==0:
                avoid_poses = (pos, pos+1, pos+2, pos+3, pos+4)
            elif col==size-4:
                avoid_poses = (pos-1, pos, pos+1, pos+2, pos+3)
            else:
                avoid_poses = (pos-1, pos, pos+1, pos+2, pos+3, pos+4)
            opp_poses = range_numbers_not_poses(size, (size//2-1, size//2+2), (size//2-1, size//2+2), avoid_poses, 4)

            test_case = {}
            test_case["name"]=f"第{i}行（从0计）横线一步成五"
            test_case["size"]=size
            test_case["moves"] = (pos, opp_poses[0], pos+1, opp_poses[1], pos+2, opp_poses[2], pos+3, opp_poses[3])
            if col==0:
                test_case["expects"] = ([[pos+4], 1])
            elif col==size-4:
                test_case["expects"] = ([[pos-1], 1])
            else:
                test_case["expects"] = ([[pos-1, pos+4], 1])
            一步成五集["cases"].append(test_case)

一步活四集={}
一步活四集["name"]="一步活四集"
一步活四集["cases"]=[]
for size in range(9, 21):
    for row in range(0, size):
        for col in range(1, size - 3):
            pos = row*size + col
            if col==1:
                avoid_poses = (pos-1, pos, pos+1, pos+2, pos+3, pos+4)
            elif col==size-4:
                avoid_poses = (pos-2, pos-1, pos, pos+1, pos+2, pos+3)
            else:
                avoid_poses = (pos-2, pos-1, pos, pos+1, pos+2, pos+3, pos+4)
            opp_poses = range_numbers_not_poses(size, (size//2-1, size//2+2), (size//2-1, size//2+2), avoid_poses, 3)

            test_case = {}
            test_case["name"]=f"第{i}行（从0计）横线一步活四"
            test_case["size"]=size
            test_case["moves"] = (pos, opp_poses[0], pos+1, opp_poses[1], pos+2, opp_poses[2])
            if col==0:
                test_case["expects"] = ([[pos+3], 1])
            elif col==size-4:
                test_case["expects"] = ([[pos-1], 1])
            else:
                test_case["expects"] = ([[pos-1, pos+3], 1])
            一步活四集["cases"].append(test_case)

一步冲四活三集={}
一步冲四活三集["name"]="一步冲四活三集"
一步冲四活三集["cases"]=[]
for size in range(9, 21):
    for row in range(0, size - 6):
        for col in range(1, size - 5):
            pos = row*size + col
            my = [(1,1), (2,2), (4,4), (2,4), (4,2)]
            opp = [(0,0)]
            avoid = [(3,3), (5,5), (1,5), (6,0), (5,1)]
            for i,j in enumerate(my):
                my[i] = pos + size * j[0] + j[1]
            for i,j in enumerate(opp):
                opp[i] = pos + size * j[0] + j[1]
            for i,j in enumerate(avoid):
                avoid[i] = pos + size * j[0] + j[1]

            more = len(my) - len(opp)
            all_avoid_poses = my + opp + avoid

            opp_poses = range_numbers_not_poses(size, (row, row+7), (col, col+6), all_avoid_poses, more)
            opp_poses = opp_poses + opp
            moves = []
            myind = 0
            oppind = 0
            for i in range(len(my) + len(opp_poses)):
                if i%2==0:
                    moves.append(my[myind])
                    myind += 1
                else:
                    moves.append(opp_poses[oppind])
                    oppind += 1

            if len(moves) != len(set(moves)):
                print("有重复元素")

            test_case = {}
            test_case["name"]=f"第{i}行（从0计）横线一步冲四活三"
            test_case["size"]=size
            test_case["moves"] = moves
            test_case["expects"] = ([[pos + 3 * size + 3], 1])

            一步冲四活三集["cases"].append(test_case)

all_bundles=[]
all_bundles.append(一步成五集)
# all_bundles.append(阻挡对手成五集)
all_bundles.append(一步活四集)
# all_bundles.append(阻挡对手活四集)
all_bundles.append(一步冲四活三集)
# all_bundles.append(一步双活三集)
# all_bundles.append(困难集)

def test_the_case(Game, net, test_case, expect, device):
    net.eval()
    size, moves = test_case  # 解包 size 和 moves
    state = torch.tensor(get_state(size, moves), dtype=torch.float32, device=device).view(1, STATE_LAYER_NUM, 20, 20).to(device)
    board_sizes = torch.tensor([size], dtype=torch.float32, device=device)
    with torch.no_grad():
        net.eval()

        policy_logits, value_logits = net(state, board_sizes)

        # 提取当前玩家的策略
        policy_logits_current = policy_logits[0, 0, :, :size]  # [20, size]，只取有效区域
        policy_logits_local = policy_logits_current[:size, :].flatten()  # [size*size]
        policy_probs = F.softmax(policy_logits_local, dim=0)  # (1, size * size) 计算策略概率，注意需要转换为局部坐标
        max_prob_move = torch.argmax(policy_probs).item()  # Index in local_pos (0 to size*size-1)

        # 提取下一步对手应对可能的策略
        policy_logits_opp = policy_logits[0, 1, :, :size]  # [20, size]，只取有效区域
        policy_logits_local_opp = policy_logits_opp[:size, :].flatten()  # [size*size]
        policy_probs_opp = F.softmax(policy_logits_local_opp, dim=0)  # (1, size * size) 计算策略概率，注意需要转换为局部坐标
        max_prob_move_opp = torch.argmax(policy_probs_opp[0]).item()  # Index in local_pos (0 to size*size-1)

        # 计算价值 predicted_value = win - lose
        value_probs = F.softmax(value_logits, dim=1)  # 将 logits 转换为概率分布
        win_prob = value_probs[0, 0]  # win 的概率
        lose_prob = value_probs[0, 1]  # lose 的概率
        predicted_value = win_prob - lose_prob

        # # 处理期望策略
        # if isinstance(expect[0], (int, float)):  # 如果 expect[0] 是数字
        #     print("期望策略：", expect[0])
        #     print("实际策略：", max_prob_move, "prior：", policy_probs[max_prob_move].item(), f"动作{expect[0]}的prior：", policy_probs[expect[0]].item())
        # elif isinstance(expect[0], (list, tuple)):  # 如果 expect[0] 是数组
        #     print("期望策略：", expect[0])
        #     real_priors = [policy_probs[exp_move].item() for exp_move in expect[0]]
        #     print("实际策略：", max_prob_move, "价值：", policy_probs[max_prob_move].item(),  f"动作{expect[0]}的prior：", real_priors)

        # print("期望价值：", expect[1])
        # print("实际价值：", predicted_value.item(), "概率分布：", value_probs[0].tolist())  # 网络预测的状态价值

        expected_moves = [int(move) for move in expect[0]]
        sum_policy = sum(policy_probs[move].item() for move in expected_moves)
        return sum_policy, predicted_value.item()

def test_bundle(Env, net, device, bundle):
    net.eval()
    print("开始测试bundle: " + bundle["name"])

    expect_prob_sums=[]
    predict_values=[]

    # 定义区间边界
    policy_intervals = [(-float('inf'), 0.1), (0.1, 0.4), (0.4, 0.6), (0.6, 0.9), (0.9, float('inf'))]
    value_intervals = [(-float('inf'), -0.5), (-0.5, 0), (0, 0.5), (0.5, float('inf'))]
    
    # 初始化计数器
    policy_counts = [0] * len(policy_intervals)
    value_counts = [0] * len(value_intervals)

    total_cases = len(bundle["cases"])
    
    # 遍历所有测试用例
    for branch in tqdm(bundle["cases"], desc="测试用例进度", unit="case"):
        test_case = (branch["size"], branch["moves"])
        expect = branch["expects"]
        expect_prob_sum, predict_value = test_the_case(Env, net, test_case, expect, device)
        expect_prob_sums.append(expect_prob_sum)
        predict_values.append(predict_value)

        # 统计 policy 区间
        for i, (low, high) in enumerate(policy_intervals):
            if low < expect_prob_sum <= high:
                policy_counts[i] += 1
                break
        
        # 统计 value 区间
        for i, (low, high) in enumerate(value_intervals):
            if low < predict_value <= high:
                value_counts[i] += 1
                break

    # 输出expect_prob_sums和predict_values的最小值，平均值，中位数，最大值
    # 计算并输出 expect_prob_sums 的统计信息
    if expect_prob_sums:
        min_expect_prob_sums = min(expect_prob_sums)
        avg_expect_prob_sums = sum(expect_prob_sums) / len(expect_prob_sums)
        median_expect_prob_sums = np.median(expect_prob_sums)
        max_expect_prob_sums = max(expect_prob_sums)
        print("expect_prob_sums 统计:")
        print(f"最小值: {min_expect_prob_sums:.4f}")
        print(f"平均值: {avg_expect_prob_sums:.4f}")
        print(f"中位数: {median_expect_prob_sums:.4f}")
        print(f"最大值: {max_expect_prob_sums:.4f}")

        # 打印 policy 区间百分比
        print("policy 区间百分比:")
        for i, (low, high) in enumerate(policy_intervals):
            percentage = (policy_counts[i] / total_cases) * 100
            if low == -float('inf'):
                label = f"< {high}"
            elif high == float('inf'):
                label = f"> {low}"
            else:
                label = f"{low}~{high}"
            print(f"{label}: {percentage:.2f}%")
    else:
        print("expect_prob_sums 为空")

    # 计算并输出 predict_values 的统计信息
    if predict_values:
        min_predict_values = min(predict_values)
        avg_predict_values = sum(predict_values) / len(predict_values)
        median_predict_values = np.median(predict_values)
        max_predict_values = max(predict_values)
        print("predict_values 统计:")
        print(f"最小值: {min_predict_values:.4f}")
        print(f"平均值: {avg_predict_values:.4f}")
        print(f"中位数: {median_predict_values:.4f}")
        print(f"最大值: {max_predict_values:.4f}")

        # 打印 value 区间百分比
        print("value 区间百分比:")
        for i, (low, high) in enumerate(value_intervals):
            percentage = (value_counts[i] / total_cases) * 100
            if low == -float('inf'):
                label = f"< {high}"
            elif high == float('inf'):
                label = f"> {low}"
            else:
                label = f"{low}~{high}"
            print(f"{label}: {percentage:.2f}%")
    else:
        print("predict_values 为空")
    
    print("测试完成bundle: " + bundle["name"])


def test_all_bundles(Env, net, device):
    net.eval()
    print("开始测试所有bundles...")
    
    # 遍历所有测试用例
    for bundle in all_bundles:
        test_bundle(Env, net, device, bundle)
    
    print("测试完成所有bundles")

def rowsG(input_tensor, is_value_head, board_size):
    """全局池化逻辑封装 - 支持混合 Batch Size"""
    batch_size, channels, h, w = input_tensor.shape
    device = input_tensor.device
    
    # 1. 确保 board_size 是 Tensor 且形状为 [batch, 1, 1]
    if not isinstance(board_size, torch.Tensor):
        board_size = torch.tensor([board_size], device=device).repeat(batch_size)
    b_size_float = board_size.float().view(batch_size, 1, 1)
    
    # 2. 生成 4D Mask: [batch, 1, h, w]
    y = torch.arange(h, device=device).view(1, 1, h, 1)
    x = torch.arange(w, device=device).view(1, 1, 1, w)
    mask = (y < b_size_float.unsqueeze(1)) & (x < b_size_float.unsqueeze(1))
    mask = mask.float() # 1.0 表示有效棋盘区域，0.0 表示填充区

    # 3. 计算 Mean (每个样本有独立的分母)
    # 针对每个样本的棋盘面积计算：size * size
    area = b_size_float.view(batch_size, 1) ** 2 
    masked_input = input_tensor * mask
    mean = torch.sum(masked_input, dim=(2, 3)) / area
    
    # 4. 计算 Max
    # 为了排除填充区的干扰，将填充区的数值设为一个极小值 (如 -1e9)
    # 这样填充区的 0 就不会影响棋盘内真实数据的最大值
    large_negative = -1e9
    temp = torch.where(mask > 0.5, input_tensor, torch.tensor(large_negative, device=device))
    max_val, _ = torch.max(temp.view(batch_size, channels, -1), dim=2)

    # 5. 计算 Scaling Factor (KataGo 特有的缩放逻辑)
    scaling_factor = (b_size_float.view(batch_size, 1) - 14.0)
    
    # 6. 拼接通道
    ch1 = mean
    ch2 = mean * scaling_factor * 0.1
    if is_value_head:
        # Value head 专用的全局特征
        ch3 = mean * (scaling_factor * scaling_factor * 0.01 - 0.1)
    else:
        # Policy head 使用的 Max 特征
        ch3 = max_val
        
    return torch.cat([ch1, ch2, ch3], dim=1)

class OrdiBlock(nn.Module):
    def __init__(self, channels=96):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
    
        
    def apply_mask(self, tensor, board_size):
        """
        tensor: [batch, channels, H, W]
        board_size: [batch] 形状的 Tensor
        """
        B, C, H, W = tensor.shape
        device = tensor.device

        # 1. 生成坐标网格 (0, 1, 2, ..., H-1)
        # y shape: [H, 1], x shape: [1, W]
        y = torch.arange(H, device=device).view(H, 1)
        x = torch.arange(W, device=device).view(1, W)

        # 2. 将 board_size 扩展为 [batch, 1, 1] 方便广播比较
        b_size = board_size.view(B, 1, 1)

        # 3. 生成掩码：只有坐标小于对应 size 的位置才为 True (1)
        # 结果 shape: [batch, H, W]
        mask = (y < b_size) & (x < b_size)

        # 4. 将 mask 扩展到 channel 维度 [batch, 1, H, W] 并乘到原图上
        return tensor * mask.unsqueeze(1).float()

    def forward(self, x, board_size):
        identity = x
        out = self.norm1(x)
        out = self.apply_mask(out, board_size) # board_size外的清零
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.apply_mask(out, board_size) # board_size外的清零
        out = self.relu2(out)
        out = self.conv2(out)
        return out + identity

class GPoolBlock(nn.Module):
    def __init__(self, in_ch=96, mid_ch_c=64, mid_ch_g=32):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv_main = nn.Conv2d(in_ch, mid_ch_c, 3, padding=1, bias=False)
        self.conv_gpool = nn.Conv2d(in_ch, mid_ch_g, 3, padding=1, bias=False)
        
        self.norm_g = nn.BatchNorm2d(mid_ch_g)
        self.relu_g = nn.ReLU(inplace=True)
        
        self.linear_g = nn.Linear(mid_ch_g * 3, mid_ch_c, bias=False)
        
        self.norm2 = nn.BatchNorm2d(mid_ch_c)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv_final = nn.Conv2d(mid_ch_c, in_ch, 3, padding=1, bias=False)


    def apply_mask(self, tensor, board_size):
        """
        tensor: [batch, channels, H, W]
        board_size: [batch] 形状的 Tensor
        """
        B, C, H, W = tensor.shape
        device = tensor.device

        # 1. 生成坐标网格 (0, 1, 2, ..., H-1)
        # y shape: [H, 1], x shape: [1, W]
        y = torch.arange(H, device=device).view(H, 1)
        x = torch.arange(W, device=device).view(1, W)

        # 2. 将 board_size 扩展为 [batch, 1, 1] 方便广播比较
        b_size = board_size.view(B, 1, 1)

        # 3. 生成掩码：只有坐标小于对应 size 的位置才为 True (1)
        # 结果 shape: [batch, H, W]
        mask = (y < b_size) & (x < b_size)

        # 4. 将 mask 扩展到 channel 维度 [batch, 1, H, W] 并乘到原图上
        return tensor * mask.unsqueeze(1).float()

    def forward(self, x, board_size):
        identity = x
        out = self.norm1(x)
        out = self.apply_mask(out, board_size) # board_size外的清零
        out = self.relu1(out)
        
        main_feat = self.conv_main(out)
        
        # GPool 分支
        g = self.conv_gpool(out)
        g = self.norm_g(g)
        g = self.apply_mask(g, board_size) # board_size外的清零
        g = self.relu_g(g)
        g_vec = rowsG(g, False, board_size)
        g_feat = self.linear_g(g_vec)
        
        # 融合
        out = main_feat + g_feat.view(g_feat.size(0), g_feat.size(1), 1, 1)
        out = self.norm2(out)
        out = self.apply_mask(out, board_size) # board_size外的清零
        out = self.relu2(out)
        out = self.conv_final(out)
        return out + identity

class KataNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input Layer
        self.conv0 = nn.Conv2d(3, 96, 3, padding=1, bias=False)
        self.linear0 = nn.Linear(19, 96, bias=False) # 针对 GlobalInput
        
        # Blocks
        self.layer0 = OrdiBlock(96)
        self.layer1 = OrdiBlock(96)
        self.layer2 = GPoolBlock(96, 64, 32)
        self.layer3 = OrdiBlock(96)
        self.layer4 = GPoolBlock(96, 64, 32)
        self.layer5 = OrdiBlock(96)
        
        # Final Norm
        self.final_norm = nn.BatchNorm2d(96)
        
        # Policy Head
        self.p_conv1 = nn.Conv2d(96, 32, 1, bias=False)
        self.p_conv2 = nn.Conv2d(96, 32, 1, bias=False)
        self.p_norm2 = nn.BatchNorm2d(32)
        self.p_linear_g = nn.Linear(96, 32, bias=False)
        self.p_norm_combine = nn.BatchNorm2d(32)
        self.p_conv_final = nn.Conv2d(32, 2, 1, bias=False)
        
        # --- Pass Branch ---
        self.pass_linear1 = nn.Linear(96, 32, bias=False)
        self.pass_adder1 = nn.Parameter(torch.zeros(32))
        self.pass_linear2 = nn.Linear(32, 2, bias=False)
        
        # --- Value & Score Branches ---
        # 共同前置层
        self.v_conv_prep = nn.Conv2d(96, 32, 1, bias=False)
        self.v_norm_prep = nn.BatchNorm2d(32)
        
        # Value/Score 共享的全局融合层
        self.v_linear_g = nn.Linear(96, 64, bias=False)
        self.v_adder_g = nn.Parameter(torch.zeros(64))
        
        # Value 最终输出
        self.v_linear_final = nn.Linear(64, 3, bias=False)
        self.v_adder_final = nn.Parameter(torch.zeros(3))
        
        # Score 最终输出
        self.s_linear_final = nn.Linear(64, 6, bias=False)
        self.s_adder_final = nn.Parameter(torch.zeros(6))
        
        # --- Ownership Branch ---
        self.own_conv_final = nn.Conv2d(32, 1, 1, bias=False)

    def apply_mask(self, tensor, board_size):
        """
        tensor: [batch, channels, H, W]
        board_size: [batch] 形状的 Tensor
        """
        B, C, H, W = tensor.shape
        device = tensor.device

        # 1. 生成坐标网格 (0, 1, 2, ..., H-1)
        # y shape: [H, 1], x shape: [1, W]
        y = torch.arange(H, device=device).view(H, 1)
        x = torch.arange(W, device=device).view(1, W)

        # 2. 将 board_size 扩展为 [batch, 1, 1] 方便广播比较
        b_size = board_size.view(B, 1, 1)

        # 3. 生成掩码：只有坐标小于对应 size 的位置才为 True (1)
        # 结果 shape: [batch, H, W]
        mask = (y < b_size) & (x < b_size)

        # 4. 将 mask 扩展到 channel 维度 [batch, 1, H, W] 并乘到原图上
        return tensor * mask.unsqueeze(1).float()
    
    def forward(self, img_input, board_size):
        # 初始融合
        x = self.conv0(img_input)

        # 特征提取
        x = self.layer0(x, board_size)
        x = self.layer1(x, board_size)
        x = self.layer2(x, board_size)
        x = self.layer3(x, board_size)
        x = self.layer4(x, board_size)
        x = self.layer5(x, board_size)
        
        trunk_out = torch.relu(self.final_norm(x))
        
        # 2. Policy Head
        p1 = self.p_conv1(trunk_out)
        p2 = self.apply_mask(self.p_norm2(self.p_conv2(trunk_out)), board_size)
        p2 = torch.relu(p2)
        p2_g_vec = rowsG(p2, False, board_size)
        p2_feat = self.p_linear_g(p2_g_vec)
        
        policy_hidden = p1 + p2_feat.view(p2_feat.size(0), p2_feat.size(1), 1, 1)
        policy_hidden = torch.relu(self.apply_mask(self.p_norm_combine(policy_hidden), board_size))
        policy = self.p_conv_final(policy_hidden)

        # 4. Value / Score / Ownership Preparation
        v_prep = self.apply_mask(self.v_norm_prep(self.v_conv_prep(trunk_out)), board_size)
        v_prep = torch.relu(v_prep)
        
        # Value Head Global pooling (Value head use is_value_head=True)
        v_g_vec = rowsG(v_prep, True, board_size)
        v_hidden = self.v_linear_g(v_g_vec)
        v_hidden = torch.relu(v_hidden + self.v_adder_g)
        
        # Value Output
        value = self.v_linear_final(v_hidden) + self.v_adder_final
        
        return policy, value


class KataGoWindowSize:
    def __init__(self, min_rows=250_000, exponent=0.7, expand_per_row=0.4):
        self.min_rows = min_rows          # 最小保留量（数据少于这个值时全部保留）
        self.exponent = exponent          # 增长指数 (0.5~0.8)
        self.expand_per_row = expand_per_row  # 扩张系数
        self.S = min_rows                 # 幂律标尺，通常等于 min_rows

    def calculate(self, total_samples):
        """
        输入：你从开始到现在产生的所有样本总数
        返回：当前训练窗口应该包含的样本数
        """
        if total_samples <= self.min_rows:
            return total_samples
        
        # 1. 计算超出的部分
        # 对应源码中的 power_law_x
        x = total_samples - self.min_rows + self.S
        
        # 2. 应用幂律公式 (W(N) = (x^a - S^a) / (a * S^(a-1)))
        # 分子
        numerator = (x ** self.exponent) - (self.S ** self.exponent)
        # 分母 (标准化项，确保导数连续性)
        denominator = self.exponent * (self.S ** (self.exponent - 1))
        
        # 3. 缩放并加上基础保留量
        window_size = (numerator / denominator) * self.expand_per_row + self.min_rows
        
        return int(window_size)


def apply_transform(tensor, transform_type):
    """
    应用对称变换到整个 20x20 的张量（state 或 policy）
    :param tensor: 形状为 [STATE_LAYER_NUM, 20, 20] 的 state 或 [20, 20] 的 policy 张量
    :param transform_type: 字符串，表示变换类型
    :return: 变换后的张量
    """
    if transform_type == 'identity':
        transformed_tensor = tensor.clone()
    elif transform_type == 'rot90':
        transformed_tensor = torch.rot90(tensor, k=1, dims=(-2, -1))
    elif transform_type == 'rot180':
        transformed_tensor = torch.rot90(tensor, k=2, dims=(-2, -1))
    elif transform_type == 'rot270':
        transformed_tensor = torch.rot90(tensor, k=3, dims=(-2, -1))
    elif transform_type == 'flip_horizontal':
        transformed_tensor = torch.flip(tensor, dims=[-1])  # 水平翻转
    elif transform_type == 'flip_vertical':
        transformed_tensor = torch.flip(tensor, dims=[-2])  # 垂直翻转
    elif transform_type == 'transpose':
        transformed_tensor = torch.transpose(tensor, -2, -1)  # 主对角线翻转
    elif transform_type == 'transpose_inverse':
        rotated = torch.rot90(tensor, k=2, dims=(-2, -1))
        transformed_tensor = torch.transpose(rotated, -2, -1)  # 反对角线翻转
    else:
        raise ValueError(f"未知的变换类型: {transform_type}")
    return transformed_tensor

def load_checkpoint(model, optimizer, path='model/checkpoint.pth'):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['total_samples']
    return 0

def softlink(target, link_name):
    # 如果软链接已经存在，需要先删除，否则会报错
    if os.path.lexists(link_name):
        os.remove(link_name)

    try:
        os.symlink(target, link_name)
        print(f"成功创建软链接: {link_name} -> {target}")
    except OSError as e:
        print(f"创建失败: {e}")


def save_3_nets_and_checkpoint(real_net, model_pth_path, model_onnx_path, model_pt_path, device, optimizer=None, total_samples=0):
    save_dir = os.path.dirname(model_pth_path)

    # 如果目录不存在，则创建它
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    real_net.eval()

    # --- 新增：保存完整的训练 Checkpoint ---
    if optimizer is not None:
        checkpoint_path = os.path.join(os.path.dirname(model_pth_path), 'checkpoint.pth')
        torch.save({
            'model_state_dict': real_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_samples': total_samples
        }, checkpoint_path)
        print(f"训练状态(Checkpoint)已保存至 {checkpoint_path}")

    # 保存 PyTorch 模型 (.pth)
    torch.save(real_net.state_dict(), model_pth_path)
    print(f"PyTorch 模型已保存至 {model_pth_path}")

    # 保存 ONNX 模型
    dummy_input = torch.randn(1, STATE_LAYER_NUM, 20, 20).to(device)
    dummy_board_sizes = torch.tensor([20], dtype=torch.float32).to(device)
    torch.onnx.export(
        real_net,
        (dummy_input, dummy_board_sizes),
        model_onnx_path,
        input_names=["input", "board_sizes"],
        output_names=["policy_logits", "value_logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "board_sizes": {0: "batch_size"},
            "policy_logits": {0: "batch_size"},
            "value_logits": {0: "batch_size"}
        }
    )
    print(f"ONNX 模型已保存至 {model_onnx_path}")

    # 保存 pt 模型
    # Trace the model to export it to TorchScript
    mod = torch.jit.trace(real_net, (dummy_input, dummy_board_sizes))
    mod.save(model_pt_path)  # Save as TorchScript model

    print(f"ncnn 模型已保存至 {model_pt_path}")

    softlink(os.path.basename(model_pth_path), "model/candidate.pth")
    softlink(os.path.basename(model_onnx_path), "model/candidate.onnx")
    softlink(os.path.basename(model_pt_path), "model/candidate.pt")


def cross_entropy(pred_logits, target_probs, dim):
    return -torch.sum(target_probs * torch.nn.functional.log_softmax(pred_logits, dim=dim), dim=dim)

def validate(net, val_data, device):
    net.eval()  # 设置模型为评估模式
    total_loss = 0.0
    with torch.no_grad():  # 不计算梯度，节省内存
        for start in range(0, len(val_data), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(val_data))
            batch_data = val_data[start:end]
            
            # 准备批次数据
            sizes = []
            states = []
            policy_targets = []
            opp_policy_targets = []
            value_targets = []
            
            for size, state, policy_target, opp_policy_target, value_target in batch_data:
                sizes.append(size)
                states.append(state)
                policy_targets.append(policy_target)
                opp_policy_targets.append(opp_policy_target)
                value_targets.append(value_target)
            
            batch_sizes = torch.tensor(sizes, dtype=torch.float32).to(device)
            batch_states = torch.stack(states).to(device)
            batch_policy_targets = torch.stack(policy_targets).to(device)
            batch_opp_policy_targets = torch.stack(opp_policy_targets).to(device)
            batch_value_targets = torch.stack(value_targets).to(device)
            
            # 前向传播
            policy_logits, value_logits = net(batch_states, batch_sizes)
            policy_logits_current = policy_logits[:, 0, :, :].view(batch_states.size(0), -1)
            policy_logits_opp = policy_logits[:, 1, :, :].view(batch_states.size(0), -1)
            
            # 计算损失
            policy_loss = torch.sum(cross_entropy(policy_logits_current, batch_policy_targets, dim=1))
            opp_policy_loss = torch.sum(cross_entropy(policy_logits_opp, batch_opp_policy_targets, dim=1))
            value_loss = torch.sum(cross_entropy(value_logits, batch_value_targets, dim=1))
            loss = 1.0 * policy_loss + 0.15 * opp_policy_loss + 1.2 * value_loss
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_data)  # 计算平均损失
    return avg_loss

def get_custom_lr(total_samples):
    """
    根据累积样本数获取对应的学习率
    """
    if total_samples < 1_000_000:
        return 0.000048
    elif total_samples < 2_000_000:
        return 0.00008
    elif total_samples < 4_000_000:
        return 0.00012
    elif total_samples < 6_000_000:
        return 0.000171429
    elif total_samples < 200_000_000:
        return 0.00024 # 6百万到2亿之间，全速前进
    elif total_samples < 400_000_000:
        return 0.00012
    elif total_samples < 500_000_000:
        return 0.00006
    elif total_samples < 550_000_000:
        return 0.00003
    elif total_samples < 600_000_000:
        return 0.000015
    else:
        return 0.0000075

import re

def get_total_rows(directory_path):
    """
    遍历目录，提取文件名末尾数字（如 d34 中的 34）并累加。
    """
    total_rows = 0
    
    # 检查路径是否存在
    if not os.path.exists(directory_path):
        print(f"Error: 路径 {directory_path} 不存在")
        return 0

    # 获取目录下所有文件
    files = os.listdir(directory_path)
    
    for filename in files:
        # 只处理 .npz 文件，避免统计干扰
        if filename.endswith(".npz"):
            try:
                # 逻辑：取文件名去掉后缀后的最后一部分数字
                # 方式 1: 使用 split('_') 配合 re 提取
                # 文件名示例: epoch0_Thread125_..._d34.npz
                
                # 去掉 .npz 后缀
                name_without_ext = os.path.splitext(filename)[0]
                
                # 查找最后一段数字（匹配字母后的数字，如 d34 -> 34）
                match = re.search(r'(\d+)$', name_without_ext)
                
                if match:
                    number = int(match.group(1))
                    total_rows += number
            except Exception as e:
                print(f"解析文件 {filename} 时出错: {e}")
                
    return total_rows

def get_timestamp(filename):
    # 使用正则表达式匹配: 8位日期_6位时间 (例如 20260125_225756)
    match = re.search(r'(\d{8})_(\d{6})', os.path.basename(filename))
    if match:
        time_str = f"{match.group(1)}_{match.group(2)}"
        # 将字符串转换为 datetime 对象
        dt = datetime.strptime(time_str, '%Y%m%d_%H%M%S')
        # 返回秒级时间戳 (float)
        return dt.timestamp()
    return 0  # 如果没匹配到，返回0排在最前面

# 训练
def train(Game, GameNet):
    # 检查 GPU 可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练将使用设备: {device}")
    
    # 指定数据目录
    directory = './selfplay'

    total_rows = get_total_rows(directory)
    win_calc = KataGoWindowSize(min_rows=250_000, exponent=0.7, expand_per_row=0.4)
    desired_rows = win_calc.calculate(total_rows)
    print("扫描发现总数据数", total_rows, "，经过窗口公式计算，实际将训练数据数", desired_rows)

    # 初始化网络和优化器
    real_net = GameNet().to(device)

    print("初始化网络完毕。")

    # 检查并加载已有模型
    model_path = './model/model.pth'
    if os.path.exists(model_path):
        real_net.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print("No pre-trained model found, starting from scratch.")

    real_net.train()
    optimizer = optim.Adam(real_net.parameters(), lr=LEARNING_RATE)

    # 加载持久化的累积样本数
    checkpoint_path = './model/checkpoint.pth'
    total_samples = load_checkpoint(real_net, optimizer, checkpoint_path)
    print(f"当前累积训练样本数: {total_samples}")
    print("学习率：（从0.000048慢慢增长到0.00024（6百万个samples时），然后维持在0.00024直到2亿个samples）")

    # 获取所有 .npz 文件
    npz_files = glob.glob(os.path.join(directory, '*.npz'))
    # 降序（新 -> 旧）
    sorted_files = sorted(npz_files, key=get_timestamp, reverse=True)

    # 加载所有数据
    all_data = []
    for file in tqdm(sorted_files, desc=f"加载npz文件", unit="file"):
        try:
            with np.load(file) as data:
                # 从 npz 中读取数组
                # 此时 states 的形状是 (N, STATE_LAYER_NUM * 400)
                # policy_targets 的形状是 (N, 400)
                states_arr = data['states']
                policy_arr = data['policy_targets']
                opp_policy_arr = data['opp_policy_targets']
                value_arr = data['value_targets']
                size_arr = data['size']
                
                num_entries = states_arr.shape[0]
                
                for i in range(num_entries):
                    # 转换 size
                    size = int(size_arr[i])
                    
                    # 转换 state 并恢复形状 (STATE_LAYER_NUM, 20, 20)
                    state = torch.from_numpy(states_arr[i]).float().view(STATE_LAYER_NUM, 20, 20)
                    
                    # 转换 policy
                    policy_target = torch.from_numpy(policy_arr[i]).float()
                    opp_policy_target = torch.from_numpy(opp_policy_arr[i]).float()
                    
                    # 转换 value
                    value_target = torch.from_numpy(value_arr[i]).float()
                    
                    all_data.append((size, state, policy_target, opp_policy_target, value_target))

                    if len(all_data) >= desired_rows:
                        break
            if len(all_data) >= desired_rows:
                break
        except Exception as e:
            print(f"解析文件 {file} 出错: {e}")
            continue
    print("实际加载训练数据数：", len(all_data))
        
    if not all_data:
        print("加载的数据为空，无法进行训练。")
        return
        
    # 打乱数据并分割为训练集和验证集
    random.shuffle(all_data)
    val_size = len(all_data) // 10  # 验证集大小为 1/10
    train_data = all_data[val_size:]  # 训练集
    val_data = all_data[:val_size]    # 验证集

    # 训练之前看下损失
    val_loss = validate(real_net, val_data, device)
    print(f"训练前的loss: {val_loss}")

    # 训练循环
    for epoch in range(1):
            real_net.train()
            # 随机打乱数据
            random.shuffle(train_data)
            num_batches = (len(train_data) + BATCH_SIZE - 1) // BATCH_SIZE  # 计算批次总数
            # 使用 tqdm 添加进度条
            pbar = tqdm(range(0, len(train_data), BATCH_SIZE), desc="训练中")
            for i, start in enumerate(pbar):
                end = min(start + BATCH_SIZE, len(train_data))
                batch_data = train_data[start:end]

                sizes = []
                states = []
                policy_targets = []
                opp_policy_targets = []
                value_targets = []

                # 对当前批次应用变换
                for size, state, policy_target, opp_policy_target, value_target in batch_data:
                    # 随机选择一种变换类型
                    transform_type = random.choice(Game.transforms)

                    # 应用选定的变换
                    transformed_state = apply_transform(state, transform_type)
                    transformed_policy = apply_transform(policy_target.view(20, 20), transform_type).flatten()
                    transformed_opp_policy = apply_transform(opp_policy_target.view(20, 20), transform_type).flatten()
                    sizes.append(size)
                    states.append(transformed_state)
                    policy_targets.append(transformed_policy)
                    opp_policy_targets.append(transformed_opp_policy)
                    value_targets.append(value_target)

                # 将列表转换为张量并移到设备(GPU)上
                batch_sizes = torch.tensor(sizes, dtype=torch.float32).to(device)
                batch_states = torch.stack(states).to(device)
                batch_policy_targets = torch.stack(policy_targets).to(device)
                batch_opp_policy_targets = torch.stack(opp_policy_targets).to(device)
                batch_value_targets = torch.stack(value_targets).to(device)

                # 动态获取并设置学习率
                current_lr = get_custom_lr(total_samples)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                # 前向传播和损失计算
                optimizer.zero_grad()
                policy_logits, value_logits = real_net(batch_states, batch_sizes)
                policy_logits_current = policy_logits[:, 0, :, :].view(batch_states.size(0), -1)  # [batch, 400]
                policy_logits_opp = policy_logits[:, 1, :, :].view(batch_states.size(0), -1)      # [batch, 400]

                # 策略损失：交叉熵
                policy_loss = torch.sum(cross_entropy(policy_logits_current, batch_policy_targets, dim=1))

                # 对手策略损失：交叉熵
                opp_policy_loss = torch.sum(cross_entropy(policy_logits_opp, batch_opp_policy_targets, dim=1))

                # 价值损失
                value_loss = torch.sum(cross_entropy(value_logits, batch_value_targets, dim=1))
                
                # 加权总损失
                loss = 1.0 * policy_loss + 0.15 * opp_policy_loss + 1.5 * value_loss

                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数

                # 累加已经看过的样本数
                total_samples += len(batch_data)

                # --- 在 Batch 级别实时监控 ---
                if (i + 1) % 20 == 0:  # 每 20 个 Batch 更新一次进度条信息
                    pbar.set_postfix({
                        "Loss": f"{loss.item():.4f}", 
                        "LR": f"{current_lr:.6f}",
                        "Samples": total_samples
                    })
    
    val_loss = validate(real_net, val_data, device)
    print(f"训练结束，验证loss: {val_loss}")
    
            
    
    # 保存模型
    model_pth_path = 'model/candidate-s' + str(total_samples) + '-d' + str(total_rows) +'.pth'  #训练用
    model_onnx_path = 'model/candidate-s' + str(total_samples) + '-d' + str(total_rows) +'.onnx'  #selfplay的tensorrt用
    model_pt_path = 'model/candidate-s' + str(total_samples) + '-d' + str(total_rows) +'.pt'    #导出pt，以后转gomocup ncnn用

    # 训练结束，保存包含进度信息的 checkpoint 和 3个net
    save_3_nets_and_checkpoint(real_net, model_pth_path, model_onnx_path, model_pt_path, device, optimizer, total_samples)

    # 测试一下，看看训练是否有效
    # test_all_bundles(Game, real_net, device)
    


if __name__ == "__main__":
    Env = Five
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 检查是否提供了模型路径
        if len(sys.argv) < 3:
            print("请提供模型路径，例如：python train.py test model/model.pth")
            sys.exit(1)
        
        model_path = sys.argv[2]
        print(f"加载模型：{model_path}")
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备：{device}")
        
        # 初始化网络并加载模型
        net = KataNet().to(device)
        try:
            net.load_state_dict(torch.load(model_path, map_location=device))
            print("模型加载成功")
        except Exception as e:
            print(f"加载模型失败：{e}")
            sys.exit(1)
        
        test_all_bundles(Env, net, device)
    else:
        # 默认执行训练逻辑
        train(Env, KataNet)

    
