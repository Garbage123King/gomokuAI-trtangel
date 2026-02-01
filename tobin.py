import torch
import torch.nn as nn
import numpy as np
import struct
import os
import sys

# 导入你的网络定义
from train import KataNet, STATE_LAYER_NUM

MODEL_PATH = 'model/model.pth' 
OUTPUT_BIN = 'model.bin'
BOARD_SIZE = 20

def get_bn_params(bn_layer):
    """
    导出 BN 参数 (scale, bias)
    注意：在 NHWC 模式下，Scale 和 Bias 仍然是 [C] 的一维数组，不需要转置，
    但在内存中它们将对应于最内层维度的连续块，非常适合 AVX 加载。
    """
    eps = bn_layer.eps
    mu = bn_layer.running_mean
    var = bn_layer.running_var
    gamma = bn_layer.weight
    beta = bn_layer.bias
    
    scale = gamma / torch.sqrt(var + eps)
    bias = beta - (mu * scale)
    
    return scale.detach().cpu().numpy(), bias.detach().cpu().numpy()

def write_tensor(f, tensor):
    data = tensor.detach().cpu().numpy().astype(np.float32)
    f.write(data.tobytes())

def write_layer_conv(f, conv_layer):
    # 原状: [Out, In, H, W]
    # 目标: [H, W, In, Out] -> 对应 C++: kernel[kh][kw][ic][oc]
    print(f"  Exporting Conv (H,W,I,O): {conv_layer.weight.shape}")
    # permute(2, 3, 1, 0) 将 H,W 移到前面, In 放中间, Out 放最后
    w = conv_layer.weight.permute(2, 3, 1, 0).contiguous()
    write_tensor(f, w)

def write_layer_linear(f, linear_layer):
    # 原状: [Out, In]
    # 目标: [In, Out] -> 使得计算 output[0..7] 时可以连续加载权重
    print(f"  Exporting Linear (In, Out): {linear_layer.weight.shape}")
    w = linear_layer.weight.t().contiguous()
    write_tensor(f, w)

def write_layer_bn(f, bn_layer):
    print(f"  Exporting BN (folded): {bn_layer.num_features}")
    scale, bias = get_bn_params(bn_layer)
    f.write(scale.astype(np.float32).tobytes())
    f.write(bias.astype(np.float32).tobytes())

# --- 以下 Block 导出逻辑保持不变，因为它们只是调用上面的原子函数 ---
def write_ordi_block(f, block):
    write_layer_bn(f, block.norm1)
    write_layer_conv(f, block.conv1)
    write_layer_bn(f, block.norm2)
    write_layer_conv(f, block.conv2)

def write_gpool_block(f, block):
    write_layer_bn(f, block.norm1)
    write_layer_conv(f, block.conv_main)
    write_layer_conv(f, block.conv_gpool)
    write_layer_bn(f, block.norm_g)
    write_layer_linear(f, block.linear_g)
    write_layer_bn(f, block.norm2)
    write_layer_conv(f, block.conv_final)

def export():
    device = torch.device("cpu")
    model = KataNet().to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 {MODEL_PATH}")
        return

    print(f"加载模型: {MODEL_PATH}")
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"开始导出到 {OUTPUT_BIN} (Layout: NHWC, Kernel: HWIO) ...")
    
    with open(OUTPUT_BIN, 'wb') as f:
        print("Block: Input")
        write_layer_conv(f, model.conv0)
        
        print("Block: Layer0 (Ordi)")
        write_ordi_block(f, model.layer0)
        print("Block: Layer1 (Ordi)")
        write_ordi_block(f, model.layer1)
        print("Block: Layer2 (GPool)")
        write_gpool_block(f, model.layer2)
        print("Block: Layer3 (Ordi)")
        write_ordi_block(f, model.layer3)
        print("Block: Layer4 (GPool)")
        write_gpool_block(f, model.layer4)
        print("Block: Layer5 (Ordi)")
        write_ordi_block(f, model.layer5)
        
        print("Block: Final Norm")
        write_layer_bn(f, model.final_norm)
        
        print("Block: Policy Head")
        write_layer_conv(f, model.p_conv1)
        write_layer_conv(f, model.p_conv2)
        write_layer_bn(f, model.p_norm2)
        write_layer_linear(f, model.p_linear_g)
        write_layer_bn(f, model.p_norm_combine)
        write_layer_conv(f, model.p_conv_final)
        
        print("Block: Value Head")
        write_layer_conv(f, model.v_conv_prep)
        write_layer_bn(f, model.v_norm_prep)
        write_layer_linear(f, model.v_linear_g)
        print("  Exporting Value Global Bias")
        write_tensor(f, model.v_adder_g)
        write_layer_linear(f, model.v_linear_final)
        print("  Exporting Value Final Bias")
        write_tensor(f, model.v_adder_final)
        
    print("导出完成！请重新将新的 model.bin 与 C++ 引擎 打包。")

if __name__ == "__main__":
    export()