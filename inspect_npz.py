import numpy as np
import argparse
import os

# 配置 NumPy 打印选项
np.set_printoptions(linewidth=300, threshold=np.inf, precision=3, suppress=True)

def print_structure(data):
    """打印结构表格"""
    print("\n" + "="*75)
    print(f"{'Array Name':<25} | {'Shape':<25} | {'Dtype':<10}")
    print("-" * 75)
    for key in data.files:
        shape = data[key].shape
        display_shape = str(shape)
        if len(shape) == 2:
            if shape[1] == 1200: display_shape = f"({shape[0]}, 3, 20, 20)"
            elif shape[1] == 400: display_shape = f"({shape[0]}, 20, 20)"
        print(f"{key:<25} | {display_shape:<25} | {str(data[key].dtype):<10}")
    print("="*75 + "\n")

def format_mixed_board(ch1, ch2, policy):
    """
    核心逻辑：
    - 如果 ch1[i,j] == 1 -> ' x '
    - 否则如果 ch2[i,j] == 1 -> ' o '
    - 否则 -> 打印 policy[i,j] 的数值
    """
    output = []
    for r in range(20):
        row_str = []
        for c in range(20):
            if ch1[r, c] == 1:
                row_str.append(f"{'x':^7}")  # 居中对齐，宽度7
            elif ch2[r, c] == 1:
                row_str.append(f"{'o':^7}")
            else:
                # 打印概率值，保留3位小数
                val = float(policy[r, c])
                row_str.append(f"{val:^7.3f}")
        output.append(" ".join(row_str))
    return "\n".join(output)

def print_row_by_row(data):
    keys = data.files
    num_entries = len(data[keys[0]])
    
    for i in range(num_entries):
        print(f"\n🔍 [ Entry #{i} ] " + "="*140)
        
        # 1. 提取并预处理数据
        states = data['states'][i].reshape(3, 20, 20)
        p_target = data['policy_targets'][i].reshape(20, 20)
        opp_p_target = data['opp_policy_targets'][i].reshape(20, 20)
        
        ch0, ch1, ch2 = states[0], states[1], states[2]

        # 2. 打印 Channel 0 (保持原样)
        print("▶ Channel 0 (States):")
        print(ch0)
        print("-" * 40)

        # 3. 打印 Mixed Board 1: (Ch1, Ch2, policy_targets)
        print("▶ Combined View (x=Ch1, o=Ch2, others=Policy):")
        print(format_mixed_board(ch1, ch2, p_target))
        print("-" * 40)

        # 4. 打印 Mixed Board 2: (Ch1, Ch2, opp_policy_targets)
        print("▶ Combined View (x=Ch1, o=Ch2, others=Opp_Policy):")
        print(format_mixed_board(ch1, ch2, opp_p_target))
        print("-" * 40)

        # 5. 打印剩余的小字段
        other_keys = [k for k in keys if k not in ['states', 'policy_targets', 'opp_policy_targets']]
        for key in other_keys:
            print(f"▶ {key}: {data[key][i]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("-s", "--structure", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.file): return

    with np.load(args.file, allow_pickle=True) as data:
        if args.structure:
            print_structure(data)
        else:
            print_row_by_row(data)

if __name__ == "__main__":
    main()
