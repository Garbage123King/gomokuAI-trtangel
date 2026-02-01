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
#include <immintrin.h> // AVX2 support

// ==========================================
// Part 1: 高性能网络推理 (AVX2 + NHWC)
// ==========================================

#define TRUE 1
#define FALSE 0
#define NN_BOARD_SIZE 20
#define INPUT_CHANNELS 3

// 对齐宏，AVX2 需要 32 字节对齐
#define ALIGN32 alignas(32)

// --- 数据结构定义 (NHWC 布局) ---
// Scale/Bias 仍然是 [C]，但在内存中必须对齐
typedef struct { ALIGN32 float scale[96]; ALIGN32 float bias[96]; } bn_param;
typedef struct { ALIGN32 float scale[32]; ALIGN32 float bias[32]; } bn_param_32;

// 卷积核: [H][W][In][Out]
// 线性层: [In][Out]

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
    ALIGN32 float linear_g[96][64]; // [In][Out] from Python transpose
    ALIGN32 float scale2[64];
    ALIGN32 float bias2[64];
    ALIGN32 float conv_final[3][3][64][96];
} gpool_block;

typedef struct {
    // conv0 input is 3 channels, not multiple of 8. Special handling or padding.
    // Python exported [3][3][3][96]
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
    ALIGN32 float p_conv_final[1][1][32][2]; // 2 output channels
    
    ALIGN32 float v_conv_prep[1][1][96][32];
    bn_param_32 v_norm_prep;
    ALIGN32 float v_linear_g[96][64];
    ALIGN32 float v_adder_g[64];
    ALIGN32 float v_linear_final[64][3];
    ALIGN32 float v_adder_final[3];
} NetworkWeights;

// 全局变量
// Input: [H][W][C]
ALIGN32 NetworkWeights net;
ALIGN32 float net_input[NN_BOARD_SIZE][NN_BOARD_SIZE][8]; // Pad input to 8 for AVX convenience? Or just 3.
// Let's keep input dense [20][20][3] and handle conv0 specially.
ALIGN32 float raw_input[NN_BOARD_SIZE][NN_BOARD_SIZE][3]; 

// Outputs
float policy_out[NN_BOARD_SIZE][NN_BOARD_SIZE][2]; // NHWC
float value_out[3];
bool model_loaded = false;

// --- AVX2 辅助函数 ---

// 3x3 卷积的核心 AVX 实现
// Input: [H][W][IC]
// Kernel: [3][3][IC][OC]
// Output: [H][W][OC]
// OC 必须是 8 的倍数
void conv3x3_avx(float* input, int H, int W, int IC, 
                 float* output, int OC, 
                 float* kernel, float* bias, bool use_relu) {
    
    // Kernel pointer stride
    // Kernel layout: [3][3][IC][OC]
    // 1 step in OC = 1 float
    // 1 step in IC = OC floats
    // 1 step in KW = IC * OC floats
    // 1 step in KH = 3 * IC * OC floats
    int oc_step = 1; // logical
    int ic_step = OC;
    int kw_step = IC * OC;
    int kh_step = 3 * IC * OC;
    
    // Input strides
    int in_w_step = IC;
    int in_h_step = W * IC;
    
    // Output strides
    int out_w_step = OC;
    int out_h_step = W * OC;

    __m256 zero = _mm256_setzero_ps();

    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            
            // Loop over Output Channels in chunks of 8
            for (int oc = 0; oc < OC; oc += 8) {
                // Init accumulator with Bias (if provided) or Zero
                __m256 acc = (bias) ? _mm256_load_ps(&bias[oc]) : zero;
                
                // Convolve 3x3
                for (int kh = 0; kh < 3; kh++) {
                    int ih = h + kh - 1; // Input height index (padding 1 implicit)
                    
                    if (ih >= 0 && ih < H) {
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = w + kw - 1;
                            
                            if (iw >= 0 && iw < W) {
                                float* in_ptr = input + ih * in_h_step + iw * in_w_step;
                                float* k_ptr = kernel + kh * kh_step + kw * kw_step + 0 * ic_step + oc;
                                
                                for (int ic = 0; ic < IC; ic++) {
                                    // Load 1 input pixel channel value and broadcast to 8
                                    __m256 in_val = _mm256_broadcast_ss(in_ptr + ic);
                                    
                                    // Load 8 weights corresponding to this IC and OC..OC+7
                                    __m256 w_val = _mm256_load_ps(k_ptr + ic * ic_step);
                                    
                                    // FMA: acc += in * w
                                    acc = _mm256_fmadd_ps(in_val, w_val, acc);
                                }
                            }
                        }
                    }
                }
                
                if (use_relu) {
                    acc = _mm256_max_ps(acc, zero);
                }
                
                // Store result
                float* out_ptr = output + h * out_h_step + w * out_w_step + oc;
                _mm256_store_ps(out_ptr, acc);
            }
        }
    }
}

// Specialized Input Conv (IC=3, not multiple of 8, usually small)
// Output OC=96 (multiple of 8)
void conv3x3_input_avx(float* input, int H, int W, 
                       float* output, int OC, 
                       float* kernel) {
    // Hardcoded IC=3
    int IC = 3;
    int ic_step = OC;
    int kw_step = IC * OC;
    int kh_step = 3 * IC * OC;
    
    int in_w_step = IC;
    int in_h_step = W * IC;
    int out_w_step = OC;
    int out_h_step = W * OC;

    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int oc = 0; oc < OC; oc += 8) {
                __m256 acc = _mm256_setzero_ps(); // No bias for conv0 usually
                
                for (int kh = 0; kh < 3; kh++) {
                    int ih = h + kh - 1;
                    if (ih >= 0 && ih < H) {
                        for (int kw = 0; kw < 3; kw++) {
                            int iw = w + kw - 1;
                            if (iw >= 0 && iw < W) {
                                float* in_ptr = input + ih * in_h_step + iw * in_w_step;
                                float* k_ptr = kernel + kh * kh_step + kw * kw_step + oc;
                                
                                // Unroll IC=3
                                // IC 0
                                __m256 in0 = _mm256_broadcast_ss(in_ptr + 0);
                                __m256 w0 = _mm256_load_ps(k_ptr + 0 * ic_step);
                                acc = _mm256_fmadd_ps(in0, w0, acc);
                                
                                // IC 1
                                __m256 in1 = _mm256_broadcast_ss(in_ptr + 1);
                                __m256 w1 = _mm256_load_ps(k_ptr + 1 * ic_step);
                                acc = _mm256_fmadd_ps(in1, w1, acc);
                                
                                // IC 2
                                __m256 in2 = _mm256_broadcast_ss(in_ptr + 2);
                                __m256 w2 = _mm256_load_ps(k_ptr + 2 * ic_step);
                                acc = _mm256_fmadd_ps(in2, w2, acc);
                            }
                        }
                    }
                }
                _mm256_store_ps(output + h * out_h_step + w * out_w_step + oc, acc);
            }
        }
    }
}

// 1x1 卷积 (Pointwise)
void conv1x1_avx(float* input, int H, int W, int IC, 
                 float* output, int OC, 
                 float* kernel, float* bias, bool use_relu) {
    // Kernel: [1][1][IC][OC] -> effectively [IC][OC]
    int ic_step = OC;
    int in_step = IC;
    int out_step = OC;
    int num_pixels = H * W;
    
    __m256 zero = _mm256_setzero_ps();

    for (int i = 0; i < num_pixels; i++) {
        float* px_in = input + i * in_step;
        float* px_out = output + i * out_step;
        
        for (int oc = 0; oc < OC; oc += 8) {
            __m256 acc = (bias) ? _mm256_load_ps(&bias[oc]) : zero;
            
            for (int ic = 0; ic < IC; ic++) {
                __m256 in_val = _mm256_broadcast_ss(px_in + ic);
                __m256 w_val = _mm256_load_ps(kernel + ic * ic_step + oc);
                acc = _mm256_fmadd_ps(in_val, w_val, acc);
            }
            
            if (use_relu) acc = _mm256_max_ps(acc, zero);
            _mm256_store_ps(px_out + oc, acc);
        }
    }
}

// Linear Layer (Matrix Mul)
// Input: [In] (vector)
// Weight: [In][Out] (from Python transpose)
// Output: [Out]
void linear_avx(float* input, int In, float* output, int Out, float* weight, float* bias) {
    __m256 zero = _mm256_setzero_ps();
    
    for (int o = 0; o < Out; o += 8) {
        __m256 acc = (bias) ? _mm256_load_ps(&bias[o]) : zero;
        
        for (int i = 0; i < In; i++) {
            __m256 in_val = _mm256_broadcast_ss(input + i);
            __m256 w_val = _mm256_load_ps(weight + i * Out + o);
            acc = _mm256_fmadd_ps(in_val, w_val, acc);
        }
        _mm256_store_ps(output + o, acc);
    }
}

// BN + ReLU (In-place)
void batch_norm_relu_avx(float* data, int H, int W, int C, float* scale, float* bias, bool use_relu) {
    int num_pixels = H * W;
    __m256 zero = _mm256_setzero_ps();
    
    for (int i = 0; i < num_pixels; i++) {
        float* px = data + i * C;
        for (int c = 0; c < C; c+=8) {
            __m256 x = _mm256_load_ps(px + c);
            __m256 s = _mm256_load_ps(scale + c);
            __m256 b = _mm256_load_ps(bias + c);
            
            // x = x * s + b
            x = _mm256_fmadd_ps(x, s, b);
            
            if (use_relu) x = _mm256_max_ps(x, zero);
            
            _mm256_store_ps(px + c, x);
        }
    }
}

// Global Pooling + Vector arithmetic
void global_pool_linear_add(float* input, int H, int W, int C, 
                            float* linear_w, float* scale, float* bias, 
                            float* output_feat) {
    // 1. Global Average Pooling -> vec[C]
    ALIGN32 float vec[96]; // Max channel size
    memset(vec, 0, sizeof(vec));
    float area = (float)(H * W);
    
    for (int i = 0; i < H*W; i++) {
        float* px = input + i * C;
        for (int c = 0; c < C; c+=8) {
            __m256 v = _mm256_load_ps(vec + c);
            __m256 x = _mm256_load_ps(px + c);
            _mm256_store_ps(vec + c, _mm256_add_ps(v, x));
        }
    }
    
    // Divide by area
    __m256 inv_area = _mm256_set1_ps(1.0f / area);
    for (int c = 0; c < C; c+=8) {
         __m256 v = _mm256_load_ps(vec + c);
         _mm256_store_ps(vec + c, _mm256_mul_ps(v, inv_area));
    }
    
    // 2. Linear Layer (vec[C] * W[C][Out_C]) -> out_vec[Out_C]
    // 假设 Output 维度是 output_feat 的通道数 (例如 64)
    // 这里我们用通用 linear_avx
    // 注意：Global Pooling 后的 linear 往往没有 bias (或合并在后面的 scale/bias)
    // 根据 tobin.py，layer2.linear_g 是 [96][64]
    
    // 临时存储 linear 结果
    ALIGN32 float lin_out[64];
    // 调用 linear_avx，假设 Out=64 (需要根据实际调用传参)
    // 这里硬编码逻辑有点困难，需要参数化，下面在 apply 函数里写具体逻辑
}

// 逐元素加法 (Residual connection)
void add_residual_avx(float* dest, float* src, int size) {
    for (int i = 0; i < size; i+=8) {
        __m256 a = _mm256_load_ps(dest + i);
        __m256 b = _mm256_load_ps(src + i);
        _mm256_store_ps(dest + i, _mm256_add_ps(a, b));
    }
}

// --- Block 应用函数 ---

void apply_ordi_block(float* x, ordi_block *params) {
    // x shape: [20][20][96]
    ALIGN32 float temp[NN_BOARD_SIZE][NN_BOARD_SIZE][96];
    ALIGN32 float residual[NN_BOARD_SIZE][NN_BOARD_SIZE][96];
    
    // Copy for residual
    memcpy(residual, x, sizeof(float)*NN_BOARD_SIZE*NN_BOARD_SIZE*96);
    
    // Norm1 + ReLU
    batch_norm_relu_avx((float*)x, NN_BOARD_SIZE, NN_BOARD_SIZE, 96, params->norm1.scale, params->norm1.bias, TRUE);
    
    // Conv1 -> temp
    conv3x3_avx((float*)x, NN_BOARD_SIZE, NN_BOARD_SIZE, 96, 
                (float*)temp, 96, 
                (float*)params->conv1, NULL, FALSE);
    
    // Norm2 + ReLU (inplace on temp)
    batch_norm_relu_avx((float*)temp, NN_BOARD_SIZE, NN_BOARD_SIZE, 96, params->norm2.scale, params->norm2.bias, TRUE);
    
    // Conv2 -> x
    conv3x3_avx((float*)temp, NN_BOARD_SIZE, NN_BOARD_SIZE, 96, 
                (float*)x, 96, 
                (float*)params->conv2, NULL, FALSE);
                
    // Residual Add
    add_residual_avx((float*)x, (float*)residual, NN_BOARD_SIZE*NN_BOARD_SIZE*96);
}

void apply_gpool_block(float* x, gpool_block *params) {
    // x: [96]
    ALIGN32 float residual[NN_BOARD_SIZE][NN_BOARD_SIZE][96];
    memcpy(residual, x, sizeof(float)*NN_BOARD_SIZE*NN_BOARD_SIZE*96);
    
    // Norm1
    batch_norm_relu_avx((float*)x, NN_BOARD_SIZE, NN_BOARD_SIZE, 96, params->norm1.scale, params->norm1.bias, TRUE);
    
    // Branch 1: Main Conv -> [64]
    ALIGN32 float main_feat[NN_BOARD_SIZE][NN_BOARD_SIZE][64];
    conv3x3_avx((float*)x, NN_BOARD_SIZE, NN_BOARD_SIZE, 96, (float*)main_feat, 64, (float*)params->conv_main, NULL, FALSE);
    
    // Branch 2: GPool path
    // Conv GPool -> [32]
    ALIGN32 float g_feat[NN_BOARD_SIZE][NN_BOARD_SIZE][32];
    conv3x3_avx((float*)x, NN_BOARD_SIZE, NN_BOARD_SIZE, 96, (float*)g_feat, 32, (float*)params->conv_gpool, NULL, FALSE);
    batch_norm_relu_avx((float*)g_feat, NN_BOARD_SIZE, NN_BOARD_SIZE, 32, params->norm_g.scale, params->norm_g.bias, TRUE);
    
    // Global Average Pooling [32]
    ALIGN32 float g_vec[32]; memset(g_vec, 0, sizeof(g_vec));
    float area_inv = 1.0f / (NN_BOARD_SIZE * NN_BOARD_SIZE);
    
    // Special scaling from original code: (SIZE - 14.0) ... 
    // We will apply the pooling first, then apply the weird logic on the vector
    for(int i=0; i<NN_BOARD_SIZE*NN_BOARD_SIZE; i++) {
        float* p = (float*)g_feat + i*32;
        for(int c=0; c<32; c+=8) {
            _mm256_store_ps(g_vec+c, _mm256_add_ps(_mm256_load_ps(g_vec+c), _mm256_load_ps(p+c)));
        }
    }
    // Scale mean
    float scaling_factor = (float)NN_BOARD_SIZE - 14.0f;
    __m256 v_scale = _mm256_set1_ps(area_inv); // Just mean for now
    for(int c=0; c<32; c+=8) {
        _mm256_store_ps(g_vec+c, _mm256_mul_ps(_mm256_load_ps(g_vec+c), v_scale));
    }
    
    // The original code had specific logic for `rowsG` outputting a larger vector 
    // [mean, mean*scale*0.1, max].
    // Assuming the trained model relies on a simpler linear projection now or we mimic the logic.
    // To conform to strict AVX port of *provided* structures, `linear_g` is [96][64].
    // This implies the input to linear_g is size 96.
    // In original code `rowsG` produced [mean, mean*fac, max]. 32 channels * 3 = 96.
    // Let's implement that logic to fill a 96-dim vector.
    
    ALIGN32 float g_vec_combined[96];
    ALIGN32 float max_vec[32]; for(int i=0;i<32;i++) max_vec[i] = -1e9f;
    
    // Calc Max and Mean
    for(int i=0; i<NN_BOARD_SIZE*NN_BOARD_SIZE; i++) {
        float* p = (float*)g_feat + i*32;
        for(int c=0; c<32; c+=8) {
            __m256 v = _mm256_load_ps(p+c);
            __m256 max_v = _mm256_load_ps(max_vec+c);
            _mm256_store_ps(max_vec+c, _mm256_max_ps(max_v, v));
        }
    }
    
    for(int c=0; c<32; c++) {
        float m = g_vec[c]; // This is mean
        g_vec_combined[c] = m;
        g_vec_combined[32+c] = m * scaling_factor * 0.1f;
        g_vec_combined[64+c] = max_vec[c];
    }
    
    // Linear: [96] -> [64]
    ALIGN32 float g_out[64];
    linear_avx(g_vec_combined, 96, g_out, 64, (float*)params->linear_g, NULL);
    
    // Add g_out (broadcast) to main_feat
    for(int i=0; i<NN_BOARD_SIZE*NN_BOARD_SIZE; i++) {
        float* p = (float*)main_feat + i*64;
        for(int c=0; c<64; c+=8) {
            __m256 v = _mm256_load_ps(p+c);
            __m256 g = _mm256_load_ps(g_out+c);
            _mm256_store_ps(p+c, _mm256_add_ps(v, g));
        }
    }
    
    // Norm2 + ReLU
    batch_norm_relu_avx((float*)main_feat, NN_BOARD_SIZE, NN_BOARD_SIZE, 64, params->scale2, params->bias2, TRUE);
    
    // Conv Final -> x (96)
    conv3x3_avx((float*)main_feat, NN_BOARD_SIZE, NN_BOARD_SIZE, 64, (float*)x, 96, (float*)params->conv_final, NULL, FALSE);
    
    // Residual
    add_residual_avx((float*)x, (float*)residual, NN_BOARD_SIZE*NN_BOARD_SIZE*96);
}


// --- Loader ---
void read_buffer(FILE *fp, void *target, size_t size) {
    if (fread(target, 1, size, fp) != size) {
        // Handle error
    }
}

void load_weights(const char* filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) { pipeOut("ERROR Cannot open model.bin"); return; }
    
    // We read strictly in the order exported by Python
    // Because structs are aligned, we must be careful. 
    // Best practice: Read into struct fields one by one to avoid padding issues on disk vs memory.
    // However, for brevity and since we control both ends:
    // If Python writes raw floats, and C++ reads into aligned memory, we just need to read `count * sizeof(float)`.
    
    auto read_array = [&](void* ptr, size_t count) {
        fread(ptr, sizeof(float), count, fp);
    };

    // Conv0
    read_array(net.conv0, 3*3*3*96);
    
    // Layer 0
    read_array(net.layer0.norm1.scale, 96); read_array(net.layer0.norm1.bias, 96);
    read_array(net.layer0.conv1, 3*3*96*96);
    read_array(net.layer0.norm2.scale, 96); read_array(net.layer0.norm2.bias, 96);
    read_array(net.layer0.conv2, 3*3*96*96);
    
    // Layer 1
    read_array(net.layer1.norm1.scale, 96); read_array(net.layer1.norm1.bias, 96);
    read_array(net.layer1.conv1, 3*3*96*96);
    read_array(net.layer1.norm2.scale, 96); read_array(net.layer1.norm2.bias, 96);
    read_array(net.layer1.conv2, 3*3*96*96);
    
    // Layer 2 (GPool)
    read_array(net.layer2.norm1.scale, 96); read_array(net.layer2.norm1.bias, 96);
    read_array(net.layer2.conv_main, 3*3*96*64);
    read_array(net.layer2.conv_gpool, 3*3*96*32);
    read_array(net.layer2.norm_g.scale, 32); read_array(net.layer2.norm_g.bias, 32);
    read_array(net.layer2.linear_g, 96*64);
    read_array(net.layer2.scale2, 64); read_array(net.layer2.bias2, 64);
    read_array(net.layer2.conv_final, 3*3*64*96);
    
    // Layer 3
    read_array(net.layer3.norm1.scale, 96); read_array(net.layer3.norm1.bias, 96);
    read_array(net.layer3.conv1, 3*3*96*96);
    read_array(net.layer3.norm2.scale, 96); read_array(net.layer3.norm2.bias, 96);
    read_array(net.layer3.conv2, 3*3*96*96);
    
    // Layer 4 (GPool)
    read_array(net.layer4.norm1.scale, 96); read_array(net.layer4.norm1.bias, 96);
    read_array(net.layer4.conv_main, 3*3*96*64);
    read_array(net.layer4.conv_gpool, 3*3*96*32);
    read_array(net.layer4.norm_g.scale, 32); read_array(net.layer4.norm_g.bias, 32);
    read_array(net.layer4.linear_g, 96*64);
    read_array(net.layer4.scale2, 64); read_array(net.layer4.bias2, 64);
    read_array(net.layer4.conv_final, 3*3*64*96);
    
    // Layer 5
    read_array(net.layer5.norm1.scale, 96); read_array(net.layer5.norm1.bias, 96);
    read_array(net.layer5.conv1, 3*3*96*96);
    read_array(net.layer5.norm2.scale, 96); read_array(net.layer5.norm2.bias, 96);
    read_array(net.layer5.conv2, 3*3*96*96);
    
    // Final Norm
    read_array(net.final_norm.scale, 96); read_array(net.final_norm.bias, 96);
    
    // Policy Head
    read_array(net.p_conv1, 1*1*96*32);
    read_array(net.p_conv2, 1*1*96*32);
    read_array(net.p_norm2.scale, 32); read_array(net.p_norm2.bias, 32);
    read_array(net.p_linear_g, 96*32);
    read_array(net.p_norm_combine.scale, 32); read_array(net.p_norm_combine.bias, 32);
    read_array(net.p_conv_final, 1*1*32*2);
    
    // Value Head
    read_array(net.v_conv_prep, 1*1*96*32);
    read_array(net.v_norm_prep.scale, 32); read_array(net.v_norm_prep.bias, 32);
    read_array(net.v_linear_g, 96*64);
    read_array(net.v_adder_g, 64);
    read_array(net.v_linear_final, 64*3);
    read_array(net.v_adder_final, 3);
    
    fclose(fp);
    model_loaded = true;
    pipeOut("MESSAGE Model loaded successfully");
}

void forward_net() {
    ALIGN32 static float x[NN_BOARD_SIZE][NN_BOARD_SIZE][96];
    
    // 1. Input Conv (Special 3 -> 96)
    conv3x3_input_avx((float*)raw_input, NN_BOARD_SIZE, NN_BOARD_SIZE, (float*)x, 96, (float*)net.conv0);
    
    // 2. Backbone
    apply_ordi_block((float*)x, &net.layer0);
    apply_ordi_block((float*)x, &net.layer1);
    apply_gpool_block((float*)x, &net.layer2);
    apply_ordi_block((float*)x, &net.layer3);
    apply_gpool_block((float*)x, &net.layer4);
    apply_ordi_block((float*)x, &net.layer5);
    
    // Final Norm
    batch_norm_relu_avx((float*)x, NN_BOARD_SIZE, NN_BOARD_SIZE, 96, net.final_norm.scale, net.final_norm.bias, TRUE);
    
    // 3. Policy Head
    ALIGN32 static float p1[NN_BOARD_SIZE][NN_BOARD_SIZE][32];
    ALIGN32 static float p2[NN_BOARD_SIZE][NN_BOARD_SIZE][32];
    
    // p1 branch
    conv1x1_avx((float*)x, NN_BOARD_SIZE, NN_BOARD_SIZE, 96, (float*)p1, 32, (float*)net.p_conv1, NULL, FALSE);
    
    // p2 branch
    conv1x1_avx((float*)x, NN_BOARD_SIZE, NN_BOARD_SIZE, 96, (float*)p2, 32, (float*)net.p_conv2, NULL, FALSE);
    batch_norm_relu_avx((float*)p2, NN_BOARD_SIZE, NN_BOARD_SIZE, 32, net.p_norm2.scale, net.p_norm2.bias, TRUE);
    
    // P2 global
    // Need to reproduce RowsG logic for p2 (mean, mean*scale, max)
    ALIGN32 float p2_g_vec[96];
    ALIGN32 float p2_max[32]; for(int i=0;i<32;i++) p2_max[i] = -1e9f;
    ALIGN32 float p2_sum[32]; memset(p2_sum, 0, sizeof(p2_sum));
    
    for(int i=0; i<NN_BOARD_SIZE*NN_BOARD_SIZE; i++) {
        float* p = (float*)p2 + i*32;
        for(int c=0; c<32; c+=8) {
             __m256 v = _mm256_load_ps(p+c);
             _mm256_store_ps(p2_sum+c, _mm256_add_ps(_mm256_load_ps(p2_sum+c), v));
             _mm256_store_ps(p2_max+c, _mm256_max_ps(_mm256_load_ps(p2_max+c), v));
        }
    }
    float scale = (float)NN_BOARD_SIZE - 14.0f;
    float inv_area = 1.0f / (NN_BOARD_SIZE*NN_BOARD_SIZE);
    for(int c=0; c<32; c++) {
        float m = p2_sum[c] * inv_area;
        p2_g_vec[c] = m;
        p2_g_vec[32+c] = m * scale * 0.1f;
        p2_g_vec[64+c] = p2_max[c]; // p2 uses max
    }
    
    ALIGN32 float p2_feat[32];
    linear_avx(p2_g_vec, 96, p2_feat, 32, (float*)net.p_linear_g, NULL);
    
    // Add p2_feat to p1
    for(int i=0; i<NN_BOARD_SIZE*NN_BOARD_SIZE; i++) {
        float* p = (float*)p1 + i*32;
        for(int c=0; c<32; c+=8) {
            __m256 v = _mm256_load_ps(p+c);
            __m256 f = _mm256_load_ps(p2_feat+c);
            _mm256_store_ps(p+c, _mm256_add_ps(v, f));
        }
    }
    
    // Norm combine
    batch_norm_relu_avx((float*)p1, NN_BOARD_SIZE, NN_BOARD_SIZE, 32, net.p_norm_combine.scale, net.p_norm_combine.bias, TRUE);
    
    // Final Policy Conv 32 -> 2 (Note: 2 is not multiple of 8)
    // We cannot use standard AVX func. Use simple loop or masked AVX. Simple loop is fine for last layer.
    // However, let's just write a scalar loop for safety and clarity for the output 2 channels.
    for(int h=0; h<NN_BOARD_SIZE; h++) {
        for(int w=0; w<NN_BOARD_SIZE; w++) {
            float* in_p = (float*)p1 + (h*NN_BOARD_SIZE+w)*32;
            float* out_p = &policy_out[h][w][0];
            
            for(int oc=0; oc<2; oc++) {
                float sum = 0.0f;
                // Kernel: [1][1][32][2] -> flat [32*2], index: ic*2 + oc
                for(int ic=0; ic<32; ic++) {
                    sum += in_p[ic] * ((float*)net.p_conv_final)[ic*2 + oc];
                }
                out_p[oc] = sum;
            }
        }
    }
    
    // 4. Value Head (省略细节，逻辑同上，注意 rowsG 的 is_value_head 参数不同)
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
