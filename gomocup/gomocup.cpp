#define NOMINMAX
#include "pisqpipe.h"
#include <windows.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// ==========================================
// Part 1: 网络推理核心 (从 main.c 移植)
// ==========================================

#define TRUE 1
#define FALSE 0
#define NN_BOARD_SIZE 20  // 网络固定的输入尺寸
#define INPUT_CHANNELS 3  // 输入通道数

// --- 数据结构定义 ---
typedef struct { float scale[96]; float bias[96]; } bn_param;
typedef struct { float scale[32]; float bias[32]; } bn_param_32;

typedef struct {
    bn_param norm1;
    float conv1[96][96][3][3];
    bn_param norm2;
    float conv2[96][96][3][3];
} ordi_block;

typedef struct {
    bn_param norm1;
    float conv_main[64][96][3][3];
    float conv_gpool[32][96][3][3];
    bn_param_32 norm_g;
    float linear_g[64][96];
    float scale2[64];
    float bias2[64];
    float conv_final[96][64][3][3];
} gpool_block;

typedef struct {
    float conv0[96][INPUT_CHANNELS][3][3];
    ordi_block layer0; ordi_block layer1;
    gpool_block layer2; ordi_block layer3;
    gpool_block layer4; ordi_block layer5;
    bn_param final_norm;
    float p_conv1[32][96][1][1];
    float p_conv2[32][96][1][1];
    bn_param_32 p_norm2;
    float p_linear_g[32][96];
    bn_param_32 p_norm_combine;
    float p_conv_final[2][32][1][1];
    float v_conv_prep[32][96][1][1];
    bn_param_32 v_norm_prep;
    float v_linear_g[64][96];
    float v_adder_g[64];
    float v_linear_final[3][64];
    float v_adder_final[3];
} NetworkWeights;

// --- 全局网络变量 ---
NetworkWeights net;
float net_input[INPUT_CHANNELS][NN_BOARD_SIZE][NN_BOARD_SIZE]; 
float policy_out[2][NN_BOARD_SIZE][NN_BOARD_SIZE];
float value_out[3];
bool model_loaded = false;

// --- 辅助函数 ---
void read_buffer(FILE *fp, void *target, size_t size) {
    if (fread(target, 1, size, fp) != size) {
        // pipeOut("ERROR reading model file!");
    }
}

void conv3x3(float input[][NN_BOARD_SIZE][NN_BOARD_SIZE], int in_c, 
             float output[][NN_BOARD_SIZE][NN_BOARD_SIZE], int out_c, 
             float kernel[][3][3]) { // Note: kernel dim simplified for pointer logic
             
    // 简化版卷积实现，注意这里需要正确的指针转换，或者为了安全起见，
    // 在C++中最好保持维度一致。下面为了兼容前面的C代码逻辑：
    typedef float (*KernelType)[3][3];
    KernelType k_ptr = (KernelType)kernel;

    memset(output, 0, sizeof(float) * out_c * NN_BOARD_SIZE * NN_BOARD_SIZE);

    for (int oc = 0; oc < out_c; oc++) {
        for (int ic = 0; ic < in_c; ic++) {
            for (int h = 0; h < NN_BOARD_SIZE; h++) {
                for (int w = 0; w < NN_BOARD_SIZE; w++) {
                    float sum = 0.0f;
                    for (int kh = 0; kh < 3; kh++) {
                        for (int kw = 0; kw < 3; kw++) {
                            int oh = h + kh - 1;
                            int ow = w + kw - 1;
                            if (oh >= 0 && oh < NN_BOARD_SIZE && ow >= 0 && ow < NN_BOARD_SIZE) {
                                // kernel indexing: [oc][ic][kh][kw]
                                // flat index logic: oc*(in_c*9) + ic*9 + kh*3 + kw
                                // 这里的 kernel 传参比较 tricky，为了简便，我们直接假设内存连续
                                sum += input[ic][oh][ow] * k_ptr[oc * in_c + ic][kh][kw];
                            }
                        }
                    }
                    output[oc][h][w] += sum;
                }
            }
        }
    }
}

void conv1x1(float input[][NN_BOARD_SIZE][NN_BOARD_SIZE], int in_c, 
             float output[][NN_BOARD_SIZE][NN_BOARD_SIZE], int out_c, 
             void *kernel) {
    
    float *k_ptr = (float*)kernel; // flattened [out_c][in_c]
    
    for (int oc = 0; oc < out_c; oc++) {
        for (int h = 0; h < NN_BOARD_SIZE; h++) {
            for (int w = 0; w < NN_BOARD_SIZE; w++) {
                float sum = 0.0f;
                for (int ic = 0; ic < in_c; ic++) {
                    sum += input[ic][h][w] * k_ptr[oc * in_c + ic];
                }
                output[oc][h][w] = sum;
            }
        }
    }
}

void linear(float *input, int in_dim, float *output, int out_dim, void *weight) {
    float *w_ptr = (float*)weight; // [out_dim][in_dim]
    for (int i = 0; i < out_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < in_dim; j++) {
            sum += input[j] * w_ptr[i * in_dim + j];
        }
        output[i] = sum;
    }
}

void batch_norm_relu(float feature[][NN_BOARD_SIZE][NN_BOARD_SIZE], int channels, 
                     float *scale, float *bias, int use_relu) {
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < NN_BOARD_SIZE; h++) {
            for (int w = 0; w < NN_BOARD_SIZE; w++) {
                float val = feature[c][h][w] * scale[c] + bias[c];
                if (use_relu && val < 0) val = 0.0f;
                feature[c][h][w] = val;
            }
        }
    }
}

void rowsG(float input[][NN_BOARD_SIZE][NN_BOARD_SIZE], int channels, float *output_vec, int is_value_head) {
    float area = (float)(NN_BOARD_SIZE * NN_BOARD_SIZE);
    float scaling_factor = (float)NN_BOARD_SIZE - 14.0f;
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        float max_val = -1e9;
        for (int h = 0; h < NN_BOARD_SIZE; h++) {
            for (int w = 0; w < NN_BOARD_SIZE; w++) {
                float val = input[c][h][w];
                sum += val;
                if (val > max_val) max_val = val;
            }
        }
        float mean = sum / area;
        output_vec[c] = mean;
        output_vec[channels + c] = mean * scaling_factor * 0.1f;
        if (is_value_head) output_vec[channels * 2 + c] = mean * (scaling_factor * scaling_factor * 0.01f - 0.1f);
        else output_vec[channels * 2 + c] = max_val;
    }
}

// 模块应用函数
void apply_ordi_block(float x[96][NN_BOARD_SIZE][NN_BOARD_SIZE], ordi_block *params) {
    float identity[96][NN_BOARD_SIZE][NN_BOARD_SIZE];
    memcpy(identity, x, sizeof(float)*96*NN_BOARD_SIZE*NN_BOARD_SIZE);
    batch_norm_relu(x, 96, params->norm1.scale, params->norm1.bias, TRUE);
    float temp[96][NN_BOARD_SIZE][NN_BOARD_SIZE];
    conv3x3(x, 96, temp, 96, (float(*)[3][3])params->conv1);
    batch_norm_relu(temp, 96, params->norm2.scale, params->norm2.bias, TRUE);
    conv3x3(temp, 96, x, 96, (float(*)[3][3])params->conv2);
    for(int i=0;i<96*NN_BOARD_SIZE*NN_BOARD_SIZE;i++) ((float*)x)[i] += ((float*)identity)[i];
}

void apply_gpool_block(float x[96][NN_BOARD_SIZE][NN_BOARD_SIZE], gpool_block *params) {
    float identity[96][NN_BOARD_SIZE][NN_BOARD_SIZE];
    memcpy(identity, x, sizeof(float)*96*NN_BOARD_SIZE*NN_BOARD_SIZE);
    batch_norm_relu(x, 96, params->norm1.scale, params->norm1.bias, TRUE);
    
    float main_feat[64][NN_BOARD_SIZE][NN_BOARD_SIZE];
    conv3x3(x, 96, main_feat, 64, (float(*)[3][3])params->conv_main);
    
    float g_feat_map[32][NN_BOARD_SIZE][NN_BOARD_SIZE];
    conv3x3(x, 96, g_feat_map, 32, (float(*)[3][3])params->conv_gpool);
    batch_norm_relu(g_feat_map, 32, params->norm_g.scale, params->norm_g.bias, TRUE);
    
    float g_vec[96]; rowsG(g_feat_map, 32, g_vec, FALSE);
    float g_out_vec[64]; linear(g_vec, 96, g_out_vec, 64, params->linear_g);
    
    for (int c=0; c<64; c++)
        for (int h=0; h<NN_BOARD_SIZE; h++)
            for (int w=0; w<NN_BOARD_SIZE; w++)
                main_feat[c][h][w] += g_out_vec[c];
                
    batch_norm_relu(main_feat, 64, params->scale2, params->bias2, TRUE);
    conv3x3(main_feat, 64, x, 96, (float(*)[3][3])params->conv_final);
    for(int i=0;i<96*NN_BOARD_SIZE*NN_BOARD_SIZE;i++) ((float*)x)[i] += ((float*)identity)[i];
}

void load_weights(const char* filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) { pipeOut("ERROR Cannot open model.bin"); return; }
    
    read_buffer(fp, net.conv0, sizeof(net.conv0));
    read_buffer(fp, &net.layer0, sizeof(net.layer0));
    read_buffer(fp, &net.layer1, sizeof(net.layer1));
    
    // GPool Layers struct mapping
    read_buffer(fp, &net.layer2.norm1, sizeof(bn_param));
    read_buffer(fp, net.layer2.conv_main, sizeof(net.layer2.conv_main));
    read_buffer(fp, net.layer2.conv_gpool, sizeof(net.layer2.conv_gpool));
    read_buffer(fp, &net.layer2.norm_g, sizeof(bn_param_32));
    read_buffer(fp, net.layer2.linear_g, sizeof(net.layer2.linear_g));
    read_buffer(fp, net.layer2.scale2, sizeof(float)*64);
    read_buffer(fp, net.layer2.bias2, sizeof(float)*64);
    read_buffer(fp, net.layer2.conv_final, sizeof(net.layer2.conv_final));
    
    read_buffer(fp, &net.layer3, sizeof(net.layer3));
    
    read_buffer(fp, &net.layer4.norm1, sizeof(bn_param));
    read_buffer(fp, net.layer4.conv_main, sizeof(net.layer4.conv_main));
    read_buffer(fp, net.layer4.conv_gpool, sizeof(net.layer4.conv_gpool));
    read_buffer(fp, &net.layer4.norm_g, sizeof(bn_param_32));
    read_buffer(fp, net.layer4.linear_g, sizeof(net.layer4.linear_g));
    read_buffer(fp, net.layer4.scale2, sizeof(float)*64);
    read_buffer(fp, net.layer4.bias2, sizeof(float)*64);
    read_buffer(fp, net.layer4.conv_final, sizeof(net.layer4.conv_final));

    read_buffer(fp, &net.layer5, sizeof(net.layer5));
    read_buffer(fp, &net.final_norm, sizeof(net.final_norm));
    
    read_buffer(fp, net.p_conv1, sizeof(net.p_conv1));
    read_buffer(fp, net.p_conv2, sizeof(net.p_conv2));
    read_buffer(fp, &net.p_norm2, sizeof(net.p_norm2));
    read_buffer(fp, net.p_linear_g, sizeof(net.p_linear_g));
    read_buffer(fp, &net.p_norm_combine, sizeof(net.p_norm_combine));
    read_buffer(fp, net.p_conv_final, sizeof(net.p_conv_final));
    
    read_buffer(fp, net.v_conv_prep, sizeof(net.v_conv_prep));
    read_buffer(fp, &net.v_norm_prep, sizeof(net.v_norm_prep));
    read_buffer(fp, net.v_linear_g, sizeof(net.v_linear_g));
    read_buffer(fp, net.v_adder_g, sizeof(net.v_adder_g));
    read_buffer(fp, net.v_linear_final, sizeof(net.v_linear_final));
    read_buffer(fp, net.v_adder_final, sizeof(net.v_adder_final));
    
    fclose(fp);
    model_loaded = true;
    pipeOut("MESSAGE Model loaded successfully");
}

void forward_net() {
    static float x[96][NN_BOARD_SIZE][NN_BOARD_SIZE];
    conv3x3(net_input, INPUT_CHANNELS, x, 96, (float(*)[3][3])net.conv0);
    apply_ordi_block(x, &net.layer0);
    apply_ordi_block(x, &net.layer1);
    apply_gpool_block(x, &net.layer2);
    apply_ordi_block(x, &net.layer3);
    apply_gpool_block(x, &net.layer4);
    apply_ordi_block(x, &net.layer5);
    batch_norm_relu(x, 96, net.final_norm.scale, net.final_norm.bias, TRUE);
    
    // Policy
    float p1[32][NN_BOARD_SIZE][NN_BOARD_SIZE];
    conv1x1(x, 96, p1, 32, net.p_conv1);
    float p2[32][NN_BOARD_SIZE][NN_BOARD_SIZE];
    conv1x1(x, 96, p2, 32, net.p_conv2);
    batch_norm_relu(p2, 32, net.p_norm2.scale, net.p_norm2.bias, TRUE);
    float p2_vec[96]; rowsG(p2, 32, p2_vec, FALSE);
    float p2_feat[32]; linear(p2_vec, 96, p2_feat, 32, net.p_linear_g);
    
    for(int c=0;c<32;c++)
        for(int h=0;h<NN_BOARD_SIZE;h++)
            for(int w=0;w<NN_BOARD_SIZE;w++) p1[c][h][w] += p2_feat[c];
            
    batch_norm_relu(p1, 32, net.p_norm_combine.scale, net.p_norm_combine.bias, TRUE);
    conv1x1(p1, 32, policy_out, 2, net.p_conv_final);
    
    // Value (Optional output, useful for debug)
    float v_prep[32][NN_BOARD_SIZE][NN_BOARD_SIZE];
    conv1x1(x, 96, v_prep, 32, net.v_conv_prep);
    batch_norm_relu(v_prep, 32, net.v_norm_prep.scale, net.v_norm_prep.bias, TRUE);
    float v_g_vec[96]; rowsG(v_prep, 32, v_g_vec, TRUE);
    float v_hidden[64]; linear(v_g_vec, 96, v_hidden, 64, net.v_linear_g);
    for(int i=0;i<64;i++) { v_hidden[i] += net.v_adder_g[i]; if(v_hidden[i]<0) v_hidden[i]=0.0f; }
    linear(v_hidden, 64, value_out, 3, net.v_linear_final);
    for(int i=0;i<3;i++) value_out[i] += net.v_adder_final[i];
}


std::string getExeDirectory() {
    char buffer[MAX_PATH];
    // 获取 .exe 的完整路径
    GetModuleFileName(NULL, buffer, MAX_PATH);
    std::string path(buffer);
    // 提取目录部分（去掉文件名）
    std::string::size_type pos = path.find_last_of("\\/");
    return path.substr(0, pos);
}

// ==========================================
// Part 2: Piskvork 协议实现
// ==========================================

const char *infotext = "name=\"KataNet_C\", author=\"User\", version=\"1.0\", country=\"CN\", www=\"github.com\"";

#define MAX_BOARD 100
int board[MAX_BOARD][MAX_BOARD];

void brain_init() {
    if (width > MAX_BOARD || height > MAX_BOARD) {
        width = height = 0;
        pipeOut("ERROR Maximal board size is %d", MAX_BOARD);
        return;
    }
    
    if (!model_loaded) {
        std::string exeDir = getExeDirectory();
        std::string paramPath = exeDir + "\\model.bin";

        // 确保 model.bin 与 exe 在同一目录
        load_weights(paramPath.c_str());
    }
    
    pipeOut("OK");
}

void brain_restart() {
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            board[x][y] = 0;
        }
    }
    pipeOut("OK");
}

int isFree(int x, int y) {
    return x >= 0 && y >= 0 && x < width && y < height && board[x][y] == 0;
}

void brain_my(int x, int y) {
    if (isFree(x, y)) {
        board[x][y] = 1; // 1 = My stones
    } else {
        pipeOut("ERROR my move [%d,%d]", x, y);
    }
}

void brain_opponents(int x, int y) {
    if (isFree(x, y)) {
        board[x][y] = 2; // 2 = Opponent stones
    } else {
        pipeOut("ERROR opponents's move [%d,%d]", x, y);
    }
}

void brain_block(int x, int y) {
    if (isFree(x, y)) {
        board[x][y] = 3; 
    } else {
        pipeOut("ERROR winning move [%d,%d]", x, y);
    }
}

int brain_takeback(int x, int y) {
    if (x >= 0 && y >= 0 && x < width && y < height && board[x][y] != 0) {
        board[x][y] = 0;
        return 0;
    }
    return 2;
}

void brain_turn() {
    if (!model_loaded) {
        pipeOut("ERROR Model not loaded");
        return;
    }

    // 1. 准备输入 (将 board 转换为 net_input)
    // 假设网络输入尺寸是 20x20
    // 如果实际 width/height < 20，我们将棋盘放在左上角 (0,0)
    memset(net_input, 0, sizeof(net_input));

    for (int y = 0; y < NN_BOARD_SIZE; y++) {
        for (int x = 0; x < NN_BOARD_SIZE; x++) {
            if (x < width && y < height) {
                // Channel 0: Color/Mask (Always 1 for valid board area)
                net_input[0][y][x] = 1.0f; 
                // Channel 1: My stones (board == 1)
                net_input[1][y][x] = (board[x][y] == 1) ? 1.0f : 0.0f;
                // Channel 2: Opponent stones (board == 2)
                net_input[2][y][x] = (board[x][y] == 2) ? 1.0f : 0.0f;
            }
        }
    }

    // 2. 运行推理
    forward_net();

    // 3. 解析 Output，寻找最大概率的合法移动
    float max_score = -1e9;
    int best_x = -1, best_y = -1;

    // 假设 policy_out[0] 是我们当前走子的logits
    // 注意：train.py 里 policy 是 logits，这里没有做 softmax，
    // 但 logits 越大通过 softmax 后概率也越大，所以直接比 logits 即可。
    
    for (int y = 0; y < height && y < NN_BOARD_SIZE; y++) {
        for (int x = 0; x < width && x < NN_BOARD_SIZE; x++) {
            if (isFree(x, y)) {
                // 读取 Channel 0 (当前玩家策略)
                float score = policy_out[0][y][x]; 
                
                // 加上一点点随机性防止完全确定性循环
                // score += ((float)rand() / RAND_MAX) * 0.001f; 

                if (score > max_score) {
                    max_score = score;
                    best_x = x;
                    best_y = y;
                }
            }
        }
    }
    
    // (可选) 输出 Value 预估值到 Debug 窗口
    // pipeOut("MESSAGE Value: Win=%.2f, Loss=%.2f", value_out[0], value_out[1]);

    if (best_x != -1) {
        do_mymove(best_x, best_y);
    } else {
        pipeOut("ERROR no move found");
    }
}

void brain_end() {
}

#ifdef DEBUG_EVAL
void brain_eval(int x, int y) {
    HDC dc;
    HWND wnd;
    RECR rc;
    char c;
    wnd = GetForegroundWindow();
    dc = GetDC(wnd);
    GetClientRect(wnd, &rc);
    c = (char)(board[x][y] + '0');
    TextOut(dc, rc.right - 15, 3, &c, 1);
    ReleaseDC(wnd, dc);
}
#endif