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
#include <Eigen/Dense> // 引入 Eigen

// ==========================================
// Part 1: 高性能网络推理 (Eigen AVX2 + NHWC)
// ==========================================

#define TRUE 1
#define FALSE 0
#define NN_BOARD_SIZE 20
#define INPUT_CHANNELS 3

// 仍然保留内存对齐，Eigen 依赖对齐来进行最快的 AVX 读写
#define ALIGN32 alignas(32)

// --- 数据结构定义 (NHWC 布局，保持与原版绝对一致以兼容 Python 权重导出) ---
typedef struct { ALIGN32 float scale[96]; ALIGN32 float bias[96]; } bn_param;
typedef struct { ALIGN32 float scale[32]; ALIGN32 float bias[32]; } bn_param_32;

typedef struct {
    bn_param norm1;
    ALIGN32 float conv1[3][3][96][96]; // [KH][KW][IC][OC]
    bn_param norm2;
    ALIGN32 float conv2[3][3][96][96];
} ordi_block;

typedef struct {
    bn_param norm1;
    ALIGN32 float conv_main[3][3][96][64];
    ALIGN32 float conv_gpool[3][3][96][32];
    bn_param_32 norm_g;
    ALIGN32 float linear_g[96][64]; 
    ALIGN32 float scale2[64];
    ALIGN32 float bias2[64];
    ALIGN32 float conv_final[3][3][64][96];
} gpool_block;

typedef struct {
    ALIGN32 float conv0[3][3][3][96]; 
    
    ordi_block layer0; ordi_block layer1;
    gpool_block layer2; ordi_block layer3;
    gpool_block layer4; ordi_block layer5;
    
    bn_param final_norm;
    
    ALIGN32 float p_conv1[1][1][96][32];
    ALIGN32 float p_conv2[1][1][96][32];
    bn_param_32 p_norm2;
    ALIGN32 float p_linear_g[96][32];
    bn_param_32 p_norm_combine;
    ALIGN32 float p_conv_final[1][1][32][2];
    
    ALIGN32 float v_conv_prep[1][1][96][32];
    bn_param_32 v_norm_prep;
    ALIGN32 float v_linear_g[96][64];
    ALIGN32 float v_adder_g[64];
    ALIGN32 float v_linear_final[64][3];
    ALIGN32 float v_adder_final[3];
} NetworkWeights;

// 全局变量
ALIGN32 NetworkWeights net;
ALIGN32 float raw_input[NN_BOARD_SIZE][NN_BOARD_SIZE][3]; 
float policy_out[NN_BOARD_SIZE][NN_BOARD_SIZE][2]; 
float value_out[3];
bool model_loaded = false;

// 静态全局 Im2Col 缓冲区，避免每次 Conv 分配内存 (20*20*3*3*96 = 345600 浮点数 = 1.38MB)
ALIGN32 static float im2col_buf[NN_BOARD_SIZE * NN_BOARD_SIZE * 3 * 3 * 96];

// --- Eigen 加速算子 ---

// 基于 im2col 和 Eigen 矩阵乘法的 3x3 卷积 (自动利用底层 AVX2/FMA)
void conv3x3_eigen(const float* input, int H, int W, int IC, 
                   float* output, int OC, 
                   const float* kernel, const float* bias, bool use_relu) {
    int out_idx = 0;
    int K_SIZE = 9 * IC;
    
    // 1. Im2col 处理 (padding = 1)
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int kh = 0; kh < 3; kh++) {
                int ih = h + kh - 1;
                for (int kw = 0; kw < 3; kw++) {
                    int iw = w + kw - 1;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        memcpy(im2col_buf + out_idx, input + (ih * W + iw) * IC, IC * sizeof(float));
                    } else {
                        memset(im2col_buf + out_idx, 0, IC * sizeof(float));
                    }
                    out_idx += IC;
                }
            }
        }
    }

    // 2. 映射为 Eigen 矩阵进行高性能 GEMM
    // Im2Col: [H*W, 9*IC], Weight: [9*IC, OC], Output: [H*W, OC]
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A(im2col_buf, H * W, K_SIZE);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W_mat(kernel, K_SIZE, OC);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C(output, H * W, OC);

    // noalias() 避免中间变量分配，直接运算
    C.noalias() = A * W_mat;

    if (bias) {
        Eigen::Map<const Eigen::RowVectorXf> b(bias, OC);
        C.rowwise() += b;
    }
    if (use_relu) {
        C = C.cwiseMax(0.0f);
    }
}

// 1x1 卷积本质上就是批量全连接层 (Matrix Multiplication)
void conv1x1_eigen(const float* input, int H, int W, int IC, 
                   float* output, int OC, 
                   const float* kernel, const float* bias, bool use_relu) {
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> A(input, H * W, IC);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W_mat(kernel, IC, OC);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> C(output, H * W, OC);

    C.noalias() = A * W_mat;

    if (bias) {
        Eigen::Map<const Eigen::RowVectorXf> b(bias, OC);
        C.rowwise() += b;
    }
    if (use_relu) {
        C = C.cwiseMax(0.0f);
    }
}

// 全连接层
void linear_eigen(const float* input, int In, float* output, int Out, const float* weight, const float* bias) {
    Eigen::Map<const Eigen::RowVectorXf> A(input, In);
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> W(weight, In, Out);
    Eigen::Map<Eigen::RowVectorXf> C(output, Out);
    
    C.noalias() = A * W;
    
    if (bias) {
        Eigen::Map<const Eigen::RowVectorXf> b(bias, Out);
        C += b;
    }
}

// Batch Norm + ReLU
void batch_norm_relu_eigen(float* data, int H, int W, int C, const float* scale, const float* bias, bool use_relu) {
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m(data, H * W, C);
    Eigen::Map<const Eigen::RowVectorXf> s(scale, C);
    Eigen::Map<const Eigen::RowVectorXf> b(bias, C);
    
    // 利用 Eigen 广播机制
    m.array() = (m.array().rowwise() * s.array()).rowwise() + b.array();
    if (use_relu) m = m.cwiseMax(0.0f);
}

// 逐元素加法 (Residual)
void add_residual_eigen(float* dest, float* src, int size) {
    Eigen::Map<Eigen::VectorXf> d(dest, size);
    Eigen::Map<const Eigen::VectorXf> s(src, size);
    d += s;
}

// --- Block 应用函数 ---

void apply_ordi_block(float* x, ordi_block *params) {
    ALIGN32 float temp[NN_BOARD_SIZE][NN_BOARD_SIZE][96];
    ALIGN32 float residual[NN_BOARD_SIZE][NN_BOARD_SIZE][96];
    
    memcpy(residual, x, sizeof(float)*NN_BOARD_SIZE*NN_BOARD_SIZE*96);
    
    batch_norm_relu_eigen((float*)x, NN_BOARD_SIZE, NN_BOARD_SIZE, 96, params->norm1.scale, params->norm1.bias, TRUE);
    
    conv3x3_eigen((float*)x, NN_BOARD_SIZE, NN_BOARD_SIZE, 96, (float*)temp, 96, (float*)params->conv1, NULL, FALSE);
    
    batch_norm_relu_eigen((float*)temp, NN_BOARD_SIZE, NN_BOARD_SIZE, 96, params->norm2.scale, params->norm2.bias, TRUE);
    
    conv3x3_eigen((float*)temp, NN_BOARD_SIZE, NN_BOARD_SIZE, 96, (float*)x, 96, (float*)params->conv2, NULL, FALSE);
                
    add_residual_eigen((float*)x, (float*)residual, NN_BOARD_SIZE*NN_BOARD_SIZE*96);
}

void apply_gpool_block(float* x, gpool_block *params) {
    ALIGN32 float residual[NN_BOARD_SIZE][NN_BOARD_SIZE][96];
    memcpy(residual, x, sizeof(float)*NN_BOARD_SIZE*NN_BOARD_SIZE*96);
    
    batch_norm_relu_eigen((float*)x, NN_BOARD_SIZE, NN_BOARD_SIZE, 96, params->norm1.scale, params->norm1.bias, TRUE);
    
    // Branch 1: Main Conv
    ALIGN32 float main_feat[NN_BOARD_SIZE][NN_BOARD_SIZE][64];
    conv3x3_eigen((float*)x, NN_BOARD_SIZE, NN_BOARD_SIZE, 96, (float*)main_feat, 64, (float*)params->conv_main, NULL, FALSE);
    
    // Branch 2: GPool path
    ALIGN32 float g_feat[NN_BOARD_SIZE][NN_BOARD_SIZE][32];
    conv3x3_eigen((float*)x, NN_BOARD_SIZE, NN_BOARD_SIZE, 96, (float*)g_feat, 32, (float*)params->conv_gpool, NULL, FALSE);
    batch_norm_relu_eigen((float*)g_feat, NN_BOARD_SIZE, NN_BOARD_SIZE, 32, params->norm_g.scale, params->norm_g.bias, TRUE);
    
    // Global Average/Max Pooling
    Eigen::Map<Eigen::Matrix<float, NN_BOARD_SIZE*NN_BOARD_SIZE, 32, Eigen::RowMajor>> g_mat((float*)g_feat);
    Eigen::RowVectorXf v_mean = g_mat.colwise().mean();
    Eigen::RowVectorXf v_max  = g_mat.colwise().maxCoeff();
    
    ALIGN32 float g_vec_combined[96];
    float scaling_factor = (float)NN_BOARD_SIZE - 14.0f;
    
    for(int c=0; c<32; c++) {
        g_vec_combined[c] = v_mean(c);
        g_vec_combined[32+c] = v_mean(c) * scaling_factor * 0.1f;
        g_vec_combined[64+c] = v_max(c);
    }
    
    ALIGN32 float g_out[64];
    linear_eigen(g_vec_combined, 96, g_out, 64, (float*)params->linear_g, NULL);
    
    // Broadcast Add
    Eigen::Map<Eigen::Matrix<float, NN_BOARD_SIZE*NN_BOARD_SIZE, 64, Eigen::RowMajor>> main_mat((float*)main_feat);
    Eigen::Map<Eigen::RowVectorXf> g_out_vec(g_out, 64);
    main_mat.rowwise() += g_out_vec;
    
    batch_norm_relu_eigen((float*)main_feat, NN_BOARD_SIZE, NN_BOARD_SIZE, 64, params->scale2, params->bias2, TRUE);
    
    // Conv Final
    conv3x3_eigen((float*)main_feat, NN_BOARD_SIZE, NN_BOARD_SIZE, 64, (float*)x, 96, (float*)params->conv_final, NULL, FALSE);
    
    add_residual_eigen((float*)x, (float*)residual, NN_BOARD_SIZE*NN_BOARD_SIZE*96);
}

// --- Loader ---
void load_weights(const char* filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) { pipeOut("ERROR Cannot open model.bin"); return; }
    
    auto read_array = [&](void* ptr, size_t count) {
        fread(ptr, sizeof(float), count, fp);
    };

    read_array(net.conv0, 3*3*3*96);
    
    // 层数据映射和原版保持完全一致 (省略逐层写出的过程，原有的 read 逻辑无损保留)
    auto read_ordi = [&](ordi_block& b) {
        read_array(b.norm1.scale, 96); read_array(b.norm1.bias, 96);
        read_array(b.conv1, 3*3*96*96);
        read_array(b.norm2.scale, 96); read_array(b.norm2.bias, 96);
        read_array(b.conv2, 3*3*96*96);
    };
    
    auto read_gpool = [&](gpool_block& b) {
        read_array(b.norm1.scale, 96); read_array(b.norm1.bias, 96);
        read_array(b.conv_main, 3*3*96*64); read_array(b.conv_gpool, 3*3*96*32);
        read_array(b.norm_g.scale, 32); read_array(b.norm_g.bias, 32);
        read_array(b.linear_g, 96*64); read_array(b.scale2, 64); read_array(b.bias2, 64);
        read_array(b.conv_final, 3*3*64*96);
    };

    read_ordi(net.layer0); read_ordi(net.layer1); read_gpool(net.layer2);
    read_ordi(net.layer3); read_gpool(net.layer4); read_ordi(net.layer5);
    
    read_array(net.final_norm.scale, 96); read_array(net.final_norm.bias, 96);
    
    // Policy
    read_array(net.p_conv1, 1*1*96*32); read_array(net.p_conv2, 1*1*96*32);
    read_array(net.p_norm2.scale, 32); read_array(net.p_norm2.bias, 32);
    read_array(net.p_linear_g, 96*32);
    read_array(net.p_norm_combine.scale, 32); read_array(net.p_norm_combine.bias, 32);
    read_array(net.p_conv_final, 1*1*32*2);
    
    // Value
    read_array(net.v_conv_prep, 1*1*96*32);
    read_array(net.v_norm_prep.scale, 32); read_array(net.v_norm_prep.bias, 32);
    read_array(net.v_linear_g, 96*64); read_array(net.v_adder_g, 64);
    read_array(net.v_linear_final, 64*3); read_array(net.v_adder_final, 3);
    
    fclose(fp);
    model_loaded = true;
    pipeOut("MESSAGE Model loaded successfully");
}

void forward_net() {
    ALIGN32 static float x[NN_BOARD_SIZE][NN_BOARD_SIZE][96];
    
    // 1. Input Conv
    conv3x3_eigen((float*)raw_input, NN_BOARD_SIZE, NN_BOARD_SIZE, 3, (float*)x, 96, (float*)net.conv0, NULL, FALSE);
    
    // 2. Backbone
    apply_ordi_block((float*)x, &net.layer0);
    apply_ordi_block((float*)x, &net.layer1);
    apply_gpool_block((float*)x, &net.layer2);
    apply_ordi_block((float*)x, &net.layer3);
    apply_gpool_block((float*)x, &net.layer4);
    apply_ordi_block((float*)x, &net.layer5);
    
    batch_norm_relu_eigen((float*)x, NN_BOARD_SIZE, NN_BOARD_SIZE, 96, net.final_norm.scale, net.final_norm.bias, TRUE);
    
    // 3. Policy Head
    ALIGN32 static float p1[NN_BOARD_SIZE][NN_BOARD_SIZE][32];
    ALIGN32 static float p2[NN_BOARD_SIZE][NN_BOARD_SIZE][32];
    
    conv1x1_eigen((float*)x, NN_BOARD_SIZE, NN_BOARD_SIZE, 96, (float*)p1, 32, (float*)net.p_conv1, NULL, FALSE);
    conv1x1_eigen((float*)x, NN_BOARD_SIZE, NN_BOARD_SIZE, 96, (float*)p2, 32, (float*)net.p_conv2, NULL, FALSE);
    batch_norm_relu_eigen((float*)p2, NN_BOARD_SIZE, NN_BOARD_SIZE, 32, net.p_norm2.scale, net.p_norm2.bias, TRUE);
    
    // Policy RowsG logic
    Eigen::Map<Eigen::Matrix<float, NN_BOARD_SIZE*NN_BOARD_SIZE, 32, Eigen::RowMajor>> p2_mat((float*)p2);
    Eigen::RowVectorXf p2_mean = p2_mat.colwise().mean();
    Eigen::RowVectorXf p2_max = p2_mat.colwise().maxCoeff();
    
    ALIGN32 float p2_g_vec[96];
    float scale = (float)NN_BOARD_SIZE - 14.0f;
    for(int c=0; c<32; c++) {
        p2_g_vec[c] = p2_mean(c);
        p2_g_vec[32+c] = p2_mean(c) * scale * 0.1f;
        p2_g_vec[64+c] = p2_max(c);
    }
    
    ALIGN32 float p2_feat[32];
    linear_eigen(p2_g_vec, 96, p2_feat, 32, (float*)net.p_linear_g, NULL);
    
    Eigen::Map<Eigen::Matrix<float, NN_BOARD_SIZE*NN_BOARD_SIZE, 32, Eigen::RowMajor>> p1_mat((float*)p1);
    Eigen::Map<Eigen::RowVectorXf> p2_feat_vec(p2_feat, 32);
    p1_mat.rowwise() += p2_feat_vec;
    
    batch_norm_relu_eigen((float*)p1, NN_BOARD_SIZE, NN_BOARD_SIZE, 32, net.p_norm_combine.scale, net.p_norm_combine.bias, TRUE);
    
    // Final Policy Conv
    conv1x1_eigen((float*)p1, NN_BOARD_SIZE, NN_BOARD_SIZE, 32, (float*)policy_out, 2, (float*)net.p_conv_final, NULL, FALSE);
    
    // 4. Value Head 补全逻辑
    ALIGN32 static float v_prep[NN_BOARD_SIZE][NN_BOARD_SIZE][32];
    conv1x1_eigen((float*)x, NN_BOARD_SIZE, NN_BOARD_SIZE, 96, (float*)v_prep, 32, (float*)net.v_conv_prep, NULL, FALSE);
    batch_norm_relu_eigen((float*)v_prep, NN_BOARD_SIZE, NN_BOARD_SIZE, 32, net.v_norm_prep.scale, net.v_norm_prep.bias, TRUE);
    
    Eigen::Map<Eigen::Matrix<float, NN_BOARD_SIZE*NN_BOARD_SIZE, 32, Eigen::RowMajor>> v_mat((float*)v_prep);
    Eigen::RowVectorXf v_mean_vec = v_mat.colwise().mean();
    Eigen::RowVectorXf v_max_vec = v_mat.colwise().maxCoeff();
    
    ALIGN32 float v_g_vec[96];
    for(int c=0; c<32; c++) {
        v_g_vec[c] = v_mean_vec(c);
        v_g_vec[32+c] = v_mean_vec(c) * scale * 0.1f;
        v_g_vec[64+c] = v_max_vec(c);
    }
    
    ALIGN32 float v_feat[64];
    linear_eigen(v_g_vec, 96, v_feat, 64, (float*)net.v_linear_g, (float*)net.v_adder_g);
    
    // ReLU
    Eigen::Map<Eigen::RowVectorXf> v_feat_map(v_feat, 64);
    v_feat_map = v_feat_map.cwiseMax(0.0f);
    
    // Final Value
    linear_eigen(v_feat, 64, value_out, 3, (float*)net.v_linear_final, (float*)net.v_adder_final);
}

// ==========================================
// Part 2: Piskvork 协议实现
// ==========================================

const char *infotext = "name=\"KataNet_C\", author=\"User\", version=\"1.0\", country=\"CN\", www=\"github.com\"";

#define MAX_BOARD 100
int board[MAX_BOARD][MAX_BOARD];

std::string getExeDirectory() {
    char buffer[MAX_PATH];
    // 获取 .exe 的完整路径
    GetModuleFileName(NULL, buffer, MAX_PATH);
    std::string path(buffer);
    // 提取目录部分（去掉文件名）
    std::string::size_type pos = path.find_last_of("\\/");
    return path.substr(0, pos);
}

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

// 注意：在 brain_turn 中填充 raw_input 时，记得它是 HWC [20][20][3]
void brain_turn() {
    if (!model_loaded) { pipeOut("ERROR Model not loaded"); return; }
    
    // 1. 准备输入
    memset(raw_input, 0, sizeof(raw_input));
    for (int y = 0; y < NN_BOARD_SIZE; y++) {
        for (int x = 0; x < NN_BOARD_SIZE; x++) {
            if (x < width && y < height) {
                // HWC
                raw_input[y][x][0] = 1.0f;
                raw_input[y][x][1] = (board[x][y] == 1) ? 1.0f : 0.0f;
                raw_input[y][x][2] = (board[x][y] == 2) ? 1.0f : 0.0f;
            }
        }
    }
    
    // 2. 运行推理
    forward_net();
    
    // 3. 解析 Output，寻找最大概率的合法移动
    float max_score = -1e9;
    int best_x = -1, best_y = -1;

    // policy_out[][][0] 是我们当前走子的logits
    // 注意：train.py 里 policy 是 logits，这里没有做 softmax，
    // 但 logits 越大通过 softmax 后概率也越大，所以直接比 logits 即可。
    
    for (int y = 0; y < height && y < NN_BOARD_SIZE; y++) {
        for (int x = 0; x < width && x < NN_BOARD_SIZE; x++) {
            if (isFree(x, y)) {
                // policy_out is now NHWC [y][x][0]
                float score = policy_out[y][x][0];
                if (score > max_score) {
                    max_score = score;
                    best_x = x; best_y = y;
                }
            }
        }
    }

    // (可选) 输出 Value 预估值到 Debug 窗口
    // pipeOut("MESSAGE Value: Win=%.2f, Loss=%.2f", value_out[0], value_out[1]);
    
    if (best_x != -1) do_mymove(best_x, best_y);
    else pipeOut("ERROR no move found");
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
