import torch
import torch.nn as nn
import numpy as np
import struct
import os
import sys

# 导入你的网络定义
from train import KataNet, STATE_LAYER_NUM

MODEL_PATH = 'model/model.pth' # 确保这里指向你正确的模型路径
OUTPUT_BIN = 'model.bin'
BOARD_SIZE = 20

def get_bn_params(bn_layer):
    """
    将PyTorch的BatchNorm层转换为推理用的 (scale, bias)
    公式: 
    scale = gamma / sqrt(var + eps)
    bias = beta - mean * scale
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
    """将tensor数据展平并写入文件"""
    # 确保是float32
    data = tensor.detach().cpu().numpy().astype(np.float32)
    f.write(data.tobytes())

def write_layer_conv(f, conv_layer):
    print(f"  Exporting Conv: {conv_layer.weight.shape}")
    write_tensor(f, conv_layer.weight)

def write_layer_linear(f, linear_layer):
    print(f"  Exporting Linear: {linear_layer.weight.shape}")
    write_tensor(f, linear_layer.weight)

def write_layer_bn(f, bn_layer):
    print(f"  Exporting BN (folded): {bn_layer.num_features}")
    scale, bias = get_bn_params(bn_layer)
    f.write(scale.astype(np.float32).tobytes())
    f.write(bias.astype(np.float32).tobytes())

def write_ordi_block(f, block):
    # OrdiBlock: Norm1 -> Conv1 -> Norm2 -> Conv2
    write_layer_bn(f, block.norm1)
    write_layer_conv(f, block.conv1)
    write_layer_bn(f, block.norm2)
    write_layer_conv(f, block.conv2)

def write_gpool_block(f, block):
    # GPoolBlock: 
    # Norm1
    # Main Branch: Conv_main
    # G Branch: Conv_gpool -> Norm_g -> Linear_g
    # Combine
    # Norm2 -> Conv_final
    
    write_layer_bn(f, block.norm1)
    write_layer_conv(f, block.conv_main)
    
    # GPool branch
    write_layer_conv(f, block.conv_gpool)
    write_layer_bn(f, block.norm_g)
    write_layer_linear(f, block.linear_g)
    
    # Post merge
    write_layer_bn(f, block.norm2)
    write_layer_conv(f, block.conv_final)

def export():
    device = torch.device("cpu")
    model = KataNet().to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 找不到模型文件 {MODEL_PATH}")
        return

    print(f"加载模型: {MODEL_PATH}")
    # map_location='cpu' 确保在没显卡时也能跑
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"开始导出到 {OUTPUT_BIN} ...")
    
    with open(OUTPUT_BIN, 'wb') as f:
        # 1. Input Conv
        print("Block: Input")
        write_layer_conv(f, model.conv0)
        # linear0 在 forward 中没有使用，跳过导出
        
        # 2. Trunk Blocks
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
        
        # 3. Final Norm
        print("Block: Final Norm")
        write_layer_bn(f, model.final_norm)
        
        # 4. Policy Head
        print("Block: Policy Head")
        # p1 = conv1
        write_layer_conv(f, model.p_conv1)
        # p2 path = conv2 -> norm2 -> linear_g
        write_layer_conv(f, model.p_conv2)
        write_layer_bn(f, model.p_norm2)
        write_layer_linear(f, model.p_linear_g)
        # combined -> norm -> conv_final
        write_layer_bn(f, model.p_norm_combine)
        write_layer_conv(f, model.p_conv_final)
        
        # 5. Value Head
        print("Block: Value Head")
        # conv_prep -> norm_prep
        write_layer_conv(f, model.v_conv_prep)
        write_layer_bn(f, model.v_norm_prep)
        
        # linear_g (bias=False in definition, but uses adder_g parameter manually)
        # PyTorch: self.v_linear_g(v_g_vec) + self.v_adder_g
        # 我们可以把 adder_g 当作 bias 导出，C语言里 linear 层可能需要支持 bias 或者我们在C里手动加
        write_layer_linear(f, model.v_linear_g)
        print("  Exporting Value Global Bias")
        write_tensor(f, model.v_adder_g)
        
        # linear_final + adder_final
        write_layer_linear(f, model.v_linear_final)
        print("  Exporting Value Final Bias")
        write_tensor(f, model.v_adder_final)
        
    print("导出完成！")

if __name__ == "__main__":
    export()