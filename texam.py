import os
import argparse
import numpy as np
import onnxruntime as ort

def softmax(x):
    """手动实现 softmax，将 logits 转换为概率"""
    # 减去最大值以防止数值溢出
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def parse_board_txt(file_path):
    """解析 txt 文件，提取题目块"""
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    # 按双换行分割题目
    blocks = [b.strip() for b in content.split('\n\n') if b.strip()]
    return blocks

def prepare_input(grid):
    """构建输入 Tensor"""
    target_size = 20
    input_data = np.zeros((3, target_size, target_size), dtype=np.float32)
    rows, cols = grid.shape

    # 找到所有为 '1' 的坐标（目标点）
    targets = [tuple(pos) for pos in np.argwhere(grid == '1')]
    if not targets:
        return None, None, None

    # 填充通道
    input_data[1, :rows, :cols][grid == 'x'] = 1.0
    input_data[2, :rows, :cols][grid == 'o'] = 1.0

    size = rows
    input_data[0, :size, :size] = 1.0

    input_tensor = np.expand_dims(input_data, axis=0) # [1, 3, 20, 20]
    board_sizes_tensor = np.array([float(size)], dtype=np.float32) # [1]

    return input_tensor, board_sizes_tensor, targets

def main():
    parser = argparse.ArgumentParser(description="Evaluate ONNX model with board exams.")
    parser.add_argument("onnx_path", help="Path to the .onnx model file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Output score for each txt and detailed statistics")
    # --- 新增参数 ---
    parser.add_argument("-p", "--print-prob", action="store_true", help="Print policy probability for each cell")
    args = parser.parse_args()

    if not os.path.exists(args.onnx_path):
        print(f"Error: Model file {args.onnx_path} not found.")
        return

    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    session = ort.InferenceSession(args.onnx_path, sess_options)

    exam_dir = "exam"
    if not os.path.exists(exam_dir):
        print(f"Error: Directory '{exam_dir}' not found.")
        return

    txt_files = sorted([f for f in os.listdir(exam_dir) if f.endswith(".txt")])

    total_score = 0
    total_questions = 0

    for file_name in txt_files:
        file_path = os.path.join(exam_dir, file_name)
        blocks = parse_board_txt(file_path)

        file_correct = 0
        file_total = 0
        target_max_values = []

        if args.print_prob:
            print(f"\n" + "="*30)
            print(f"Processing File: {file_name}")
            print("="*30)

        for i, block in enumerate(blocks):
            file_total += 1
            rows_data = block.split('\n')
            grid = np.array([[item.strip() for item in r.split(',') if item.strip()] for r in rows_data])
            r_count, c_count = grid.shape

            input_tensor, board_sizes, target_idx = prepare_input(grid)
            if input_tensor is None:
                continue

            ort_inputs = {"input": input_tensor, "board_sizes": board_sizes}
            outputs = session.run(["policy_logits"], ort_inputs)
            
            # 提取对应的棋盘区域
            pred_map = outputs[0][0, 0, :r_count, :c_count] # 只取实际棋盘大小部分

            # 将 Logits 转换为概率 (0-1)
            prob_map = softmax(pred_map)

            # 1. 判断是否正确
            pred_idx = tuple(np.unravel_index(np.argmax(prob_map), prob_map.shape))
            is_correct = pred_idx in target_idx
            if is_correct:
                file_correct += 1

            # 2. 统计目标点的 Logits 最大值
            vals_at_targets = [pred_map[t[0], t[1]] for t in target_idx]
            target_max_values.append(max(vals_at_targets))

            # --- 如果开启打印概率参数 ---
            if args.print_prob:
                print(f"\n[Question {i+1}] Correct: {is_correct}")
                for r in range(r_count):
                    row_cells = []
                    for c in range(c_count):
                        content = grid[r, c]
                        if content in ['x', 'o']:
                            # 原有棋子，左右各补两个空格对齐 4 位小数的宽度
                            row_cells.append(f"  {content}   ")
                        else:
                            # 空位显示概率
                            row_cells.append(f"{prob_map[r, c]:.4f}")
                    print(" ".join(row_cells))

        # 累加到全局
        total_score += file_correct
        total_questions += file_total

        if args.verbose and file_total > 0:
            v_arr = np.array(target_max_values)
            stats = (f"Min: {v_arr.min():.4f}, Max: {v_arr.max():.4f}, "
                     f"Avg: {v_arr.mean():.4f}, Median: {np.median(v_arr):.4f}")
            print(f"File: {file_name:20} | Score: {file_correct:3}/{file_total:3} | {stats}")

    print(f"\nFinal Score: {total_score} / {total_questions}")

if __name__ == "__main__":
    main()
