#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <random>
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <future>
#include <memory>
#include <cstring>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <chrono>
#include <string>
#include <ctime>
#include <fstream>
#include <sstream>
#include <numeric>
#include <filesystem> // 添加 filesystem 头文件
#include <stdexcept>
#include <cassert>
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h> // GL headers
#include <cnpy.h>

#define CHECK(status) do { if (status != 0) { std::cerr << "CUDA Error: " << status << std::endl; exit(1); } } while (0)

#define FAST_SEARCH_ITERATION 100
#define FULL_SEARCH_ITERATION 600
#define FIXED_SEED 46

#define STATE_LAYER_NUM 3

std::mutex print_mutex; // 全局互斥锁，用于保护打印操作

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cerr << msg << std::endl;
        }
    }
} gLogger;

class Five {
public:
    int ROWS;
    int COLS;
    int SIZE;
    int nnLen = 20;
    int nnSize = nnLen * nnLen;

    std::vector<int> board;                  // 棋盘：0为空，1为玩家1，-1为玩家2
    int current_player;                      // 当前玩家：1或-1
    bool done;                               // 游戏是否结束
    int winner;                              // 胜利者：0表示无，1或-1表示玩家
    std::vector<std::vector<int>> pos_to_combinations;  // 每个位置所属的五连组合
    std::vector<std::vector<int>> combination_to_poses; // 每个五连组合包含的位置
    std::vector<std::vector<int>> win;       // [2][组合数]，玩家在各组合中的棋子数
    std::vector<int> move_history;           // 最近5次移动
    std::vector<std::vector<int>> score;  // [2][SIZE]，玩家1和-1的得分
    std::vector<std::set<int>> chongsi_set;  // [2]，每个玩家的“冲四”位置集合
    std::vector<std::vector<int>> chongsi_count; // [2][SIZE] - Added for "冲四" counting
    std::vector<std::set<int>> chengwu_set;  // [2]，每个玩家的“成五”位置集合
    std::vector<std::vector<int>> chengwu_count; // [2][SIZE] - Added for "成五" counting

    Five(int size) : ROWS(size), COLS(size), SIZE(size * size), board(SIZE, 0), current_player(1), done(false), winner(0),
             pos_to_combinations(SIZE), win(2), chongsi_set(2), chengwu_set(2), score(2, std::vector<int>(SIZE, 0)), 
             chongsi_count(2, std::vector<int>(SIZE, 0)), chengwu_count(2, std::vector<int>(SIZE, 0)) {
                if (size < 5 || size > 20) {
                    throw std::invalid_argument("Board size must be between 5 and 20.");
                }
        init_combinations();
    }

    void init_combinations() {
        int combination_idx = 0;

        // 水平方向
        for (int row = 0; row < ROWS; ++row) {
            for (int col = 0; col <= COLS - 5; ++col) {
                std::vector<int> poses;
                for (int k = 0; k < 5; ++k) {
                    int pos = row * COLS + col + k;
                    poses.push_back(pos);
                    pos_to_combinations[pos].push_back(combination_idx);
                }
                combination_to_poses.push_back(poses);
                combination_idx++;
            }
        }

        // 垂直方向
        for (int col = 0; col < COLS; ++col) {
            for (int row = 0; row <= ROWS - 5; ++row) {
                std::vector<int> poses;
                for (int k = 0; k < 5; ++k) {
                    int pos = (row + k) * COLS + col;
                    poses.push_back(pos);
                    pos_to_combinations[pos].push_back(combination_idx);
                }
                combination_to_poses.push_back(poses);
                combination_idx++;
            }
        }

        // 主对角线（\）
        for (int row = 0; row <= ROWS - 5; ++row) {
            for (int col = 0; col <= COLS - 5; ++col) {
                std::vector<int> poses;
                for (int k = 0; k < 5; ++k) {
                    int pos = (row + k) * COLS + col + k;
                    poses.push_back(pos);
                    pos_to_combinations[pos].push_back(combination_idx);
                }
                combination_to_poses.push_back(poses);
                combination_idx++;
            }
        }

        // 副对角线（/）
        for (int row = 4; row < ROWS; ++row) {
            for (int col = 0; col <= COLS - 5; ++col) {
                std::vector<int> poses;
                for (int k = 0; k < 5; ++k) {
                    int pos = (row - k) * COLS + col + k;
                    poses.push_back(pos);
                    pos_to_combinations[pos].push_back(combination_idx);
                }
                combination_to_poses.push_back(poses);
                combination_idx++;
            }
        }

        for (auto& w : win) w.resize(combination_idx, 0);

         // 初始化 score
        for (const auto& poses : combination_to_poses) {
            for (int pos : poses) {
                score[0][pos] += 1; // SCORE_BOOK[0] = 1
                score[1][pos] += 1;
            }
        }
    }

    std::vector<int> available_moves() const {
        std::vector<int> moves;
        for (int i = 0; i < SIZE; ++i) {
            if (board[i] == 0) moves.push_back(i);
        }
        return moves;
    }

    void make_move(int position) {
        if (position < 0 || position >= SIZE || board[position] != 0) {
            throw std::runtime_error("invalid make_move() position: " + std::to_string(position));
        }
        board[position] = current_player;
        move_history.push_back(position);
        // cccccccc， 傻逼AI
        // if (move_history.size() > 5) move_history.erase(move_history.begin());
        int making = (current_player == 1) ? 0 : 1; // Player 1: 0, Player -1: 1
        int defending = 1 - making;
        static const int SCORE_BOOK[6] = { 1, 200, 400, 2000, 10000, 99999 };
    
        for (int comb_idx : pos_to_combinations[position]) {
            int old_win_making = win[making][comb_idx];
            win[making][comb_idx]++;
            int new_win_making = win[making][comb_idx];
            if (new_win_making == 5) {
                done = true;
                winner = current_player;
            }
            // Update making player's score
            int change_score = 0;
            if (win[defending][comb_idx] == 0) {
                if(new_win_making == 4)
                {
                    // 下的地方本身也要更新，缺2变缺1了，因此不再是冲四
                    chongsi_count[making][position]--;
                    if (chongsi_count[making][position] == 0) {
                        chongsi_set[making].erase(position);
                    }
                }
                int old_score = (old_win_making >= 0 && old_win_making <= 5) ? SCORE_BOOK[old_win_making] : 0;
                int new_score = (new_win_making >= 0 && new_win_making <= 5) ? SCORE_BOOK[new_win_making] : 0;
                change_score = new_score - old_score;
                for (int pos : combination_to_poses[comb_idx]) {
                    score[making][pos] += change_score;
                    // Update chongsi_count for "冲四" (four stones)
                    if (new_win_making == 3 && board[pos] == 0) {
                        chongsi_count[making][pos]++;
                        chongsi_set[making].insert(pos);
                    }
                    // Update chengwu_count for "成五" (five stones)
                    if (new_win_making == 4 && board[pos] == 0) {
                        chengwu_count[making][pos]++;
                        chengwu_set[making].insert(pos);
                        // 缺2变缺1了，因此不再是冲四
                        chongsi_count[making][pos]--;
                        if (chongsi_count[making][pos] == 0) {
                            chongsi_set[making].erase(pos);
                        }
                    }
                }
            }
            // Update defending player's score
            if (old_win_making == 0 && new_win_making == 1) { // New block
                int stones = win[defending][comb_idx];
                int old_score = (stones >= 0 && stones <= 5) ? SCORE_BOOK[stones] : 0;
                int new_score = 0;
                change_score = new_score - old_score;
                // 下的地方本身就是被block的之一
                if (stones == 3) {
                    chongsi_count[defending][position]--;
                    if (chongsi_count[defending][position] == 0) {
                        chongsi_set[defending].erase(position);
                    }
                }
                if (stones == 4) {
                    chengwu_count[defending][position]--;
                    if (chengwu_count[defending][position] == 0) {
                        chengwu_set[defending].erase(position);
                    }
                }
                for (int pos : combination_to_poses[comb_idx]) {
                    score[defending][pos] += change_score;
                    if (stones == 3 && board[pos] == 0) {
                        chongsi_count[defending][pos]--;
                        if (chongsi_count[defending][pos] == 0) {
                            chongsi_set[defending].erase(pos);
                        }
                    }
                    if (stones == 4 && board[pos] == 0) {
                        chengwu_count[defending][pos]--;
                        if (chengwu_count[defending][pos] == 0) {
                            chengwu_set[defending].erase(pos);
                        }
                    }
                }
            }
        }
        current_player = -current_player;
    }
    
    void unmake_move(int position) {
        if (board[position] == 0) return;
        int unmaking = (board[position] == 1) ? 0 : 1; // Player 1: 0, Player -1: 1
        int profiting = 1 - unmaking;
        static const int SCORE_BOOK[6] = { 1, 200, 400, 2000, 10000, 99999 };
        
        if (!move_history.empty() && move_history.back() == position) move_history.pop_back();
    
        for (int comb_idx : pos_to_combinations[position]) {
            int old_win_unmaking = win[unmaking][comb_idx];
            win[unmaking][comb_idx]--;
            int new_win_unmaking = win[unmaking][comb_idx];
            // Update profiting player's score
            if (new_win_unmaking == 0) { // Block removed
                int stones = win[profiting][comb_idx];
                int old_score = 0;
                int new_score = (stones >= 0 && stones <= 5) ? SCORE_BOOK[stones] : 0;
                int change_score = new_score - old_score;
                // 移走的地方本身也要更新
                if (stones == 3)
                {
                    chongsi_count[profiting][position]++;
                    chongsi_set[profiting].insert(position);
                }
                if (stones == 4)
                {
                    chengwu_count[profiting][position]++;
                    chengwu_set[profiting].insert(position);
                }
                for (int pos : combination_to_poses[comb_idx]) {
                    score[profiting][pos] += change_score;
                    if (stones == 3 && board[pos] == 0) {
                        chongsi_count[profiting][pos]++;
                        chongsi_set[profiting].insert(pos);
                    }
                    if (stones == 4 && board[pos] == 0) {
                        chengwu_count[profiting][pos]++;
                        chengwu_set[profiting].insert(pos);
                    }
                }
            }
            // Update unmaking player's score
            if (win[profiting][comb_idx] == 0) { // No block from profiting player
                int old_score = (old_win_unmaking >= 0 && old_win_unmaking <= 5) ? SCORE_BOOK[old_win_unmaking] : 0;
                int new_score = (new_win_unmaking >= 0 && new_win_unmaking <= 5) ? SCORE_BOOK[new_win_unmaking] : 0;
                int change_score = new_score - old_score;
                // unmake自身点，成五变冲四
                if (old_win_unmaking == 4)
                {
                    chongsi_count[unmaking][position]++;
                    chongsi_set[unmaking].insert(position);
                }
                for (int pos : combination_to_poses[comb_idx]) {
                    score[unmaking][pos] += change_score;
                    if (old_win_unmaking == 3 && board[pos] == 0) {
                        chongsi_count[unmaking][pos]--;
                        if (chongsi_count[unmaking][pos] == 0) {
                            chongsi_set[unmaking].erase(pos);
                        }
                    }
                    if (old_win_unmaking == 4 && board[pos] == 0) {
                        chengwu_count[unmaking][pos]--;
                        if (chengwu_count[unmaking][pos] == 0) {
                            chengwu_set[unmaking].erase(pos);
                        }
                        // 成五变冲四
                        chongsi_count[unmaking][pos]++;
                        chongsi_set[unmaking].insert(pos);
                    }
                }
            }
        }
        board[position] = 0;
        current_player = -current_player;
        done = false;
        winner = 0;
    }

    bool is_winner(int player) const {
        return done && winner == player;
    }

    bool is_draw() const {
        return !done && available_moves().empty();
    }

    bool is_terminal() const {
        return done || is_draw();
    }

    int get_reward(int player) const {
        if (!is_terminal()) return 0;
        if (is_winner(player)) return 1;
        if (is_winner(-player)) return -1;
        return 0;
    }

    std::vector<float> get_state() const {
        // 确保神经网络输入大小不小于棋盘大小
        assert(nnLen >= ROWS && nnLen >= COLS);

        // STATE_LAYER_NUM个平面先初始化为0
        std::vector<float> state(STATE_LAYER_NUM * nnSize, 0.0f);

        int own_idx = (current_player == 1) ? 0 : 1;
        int opp_idx = 1 - own_idx;

        // 第 0 个平面：if is on board (size control)
        for (int row = 0; row < ROWS; ++row) {
            for (int col = 0; col < COLS; ++col) {
                int net_pos = row * nnLen + col;
                state[net_pos] = 1.0f;  // 在棋盘内的位置设为 1.0
            }
        }

        // 第 1、2 个平面：当前玩家和对手的棋子
        for (int row = 0; row < ROWS; ++row) {
            for (int col = 0; col < COLS; ++col) {
                int net_pos = row * nnLen + col;
                int board_pos = row * COLS + col;

                if (board[board_pos] == current_player) {
                    state[1 * nnSize + net_pos] = 1.0f;  // 当前玩家棋子
                } else if (board[board_pos] == -current_player) {
                    state[2 * nnSize + net_pos] = 1.0f;  // 对手棋子
                }
            }
        }

        // // 第 3-7 个平面：最后 5 次移动
        // int history_size = std::min(5, static_cast<int>(move_history.size()));
        // for (int k = 0; k < history_size; ++k) {
        //     int move_pos = move_history.size() - 1 - k;
        //     int pos = move_history[move_pos];
        //     int row = pos / COLS;
        //     int col = pos % COLS;

        //     int nn_pos = row * nnLen + col;
        //     state[(3 + k) * nnSize + nn_pos] = 1.0f;
        // }

        // // 第 8 个平面：当前玩家的冲四位置
        // for (int pos : chongsi_set[own_idx]) {
        //     int row = pos / COLS;
        //     int col = pos % COLS;
            
        //     int nn_pos = row * nnLen + col;
        //     state[8 * nnSize + nn_pos] = 1.0f;
        // }

        // // 第 9 个平面：对手的冲四位置
        // for (int pos : chongsi_set[opp_idx]) {
        //     int row = pos / COLS;
        //     int col = pos % COLS;

        //     int nn_pos = row * nnLen + col;
        //     state[9 * nnSize + nn_pos] = 1.0f;
        // }

        // // 第 10 个平面：当前玩家的成五位置
        // for (int pos : chengwu_set[own_idx]) {
        //     int row = pos / COLS;
        //     int col = pos % COLS;
            
        //     int nn_pos = row * nnLen + col;
        //     state[10 * nnSize + nn_pos] = 1.0f;
        // }

        // // 第 11 个平面：对手的成五位置
        // for (int pos : chengwu_set[opp_idx]) {
        //     int row = pos / COLS;
        //     int col = pos % COLS;

        //     int nn_pos = row * nnLen + col;
        //     state[11 * nnSize + nn_pos] = 1.0f;
        // }

        if (state.size() != STATE_LAYER_NUM * 400) {
            throw std::runtime_error("Invalid state size: " + std::to_string(state.size()));
        }

        return state;
    }

    Five clone() const {
        Five copy = *this;
        return copy;
    }

    std::vector<float> get_score_policy() const {
        auto available = available_moves();
        if (available.empty()) {
            return std::vector<float>(SIZE, 0.0f);
        }
    
        std::vector<int> max_scores(SIZE);
        for (int pos = 0; pos < SIZE; ++pos) {
            max_scores[pos] = std::max(score[0][pos], score[1][pos]);
        }
    
        std::vector<float> policy(SIZE, 0.0f);
        float total = 0.0f;
        for (int pos : available) {
            policy[pos] = static_cast<float>(max_scores[pos]);
            total += policy[pos];
        }
    
        if (total > 0.0f) {
            for (int pos : available) {
                policy[pos] /= total;
            }
        } else {
            float uniform_prob = 1.0f / available.size();
            for (int pos : available) {
                policy[pos] = uniform_prob;
            }
        }
    
        return policy;
    }

    void save_state(const char* filename) const {
        // 获取状态
        std::vector<float> state = get_state();
        
        // 打开文件
        FILE* file = fopen(filename, "w");
        if (file == NULL) {
            printf("无法打开文件: %s\n", filename);
            return;
        }
        
        // 写入状态
        for (size_t i = 0; i < state.size(); ++i) {
            fprintf(file, "%d", (int)state[i]);  // 转换为整数写入
            if (i < state.size() - 1) {
                fprintf(file, " ");
            }
        }
        fprintf(file, "\n");
        fclose(file);
    }
};

nvinfer1::ICudaEngine* buildEngine(const std::string& onnxPath, int maxBatchSize) {
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);

    if (!parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        std::cerr << "Failed to parse ONNX file" << std::endl;
        return nullptr;
    }

    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 30);

    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, STATE_LAYER_NUM, 20, 20});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{maxBatchSize, STATE_LAYER_NUM, 20, 20});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{maxBatchSize, STATE_LAYER_NUM, 20, 20});
    profile->setDimensions("board_sizes", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims{1, {1}});
    profile->setDimensions("board_sizes", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims{1, {maxBatchSize}});
    profile->setDimensions("board_sizes", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims{1, {maxBatchSize}});
    config->addOptimizationProfile(profile);

    nvinfer1::IHostMemory* serializedEngine = builder->buildSerializedNetwork(*network, *config);
    if (!serializedEngine) {
        std::cerr << "Failed to build serialized engine" << std::endl;
        return nullptr;
    }

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size());

    delete parser;
    delete network;
    delete config;
    delete builder;
    delete serializedEngine;
    // delete runtime;

    return engine;
}

struct Request {
    std::vector<float> matrix; // 输入状态 [STATE_LAYER_NUM, 20, 20]
    int board_size;            // 棋盘大小标量
    std::promise<std::vector<float>> promise;
};

// 添加 softmax 函数
std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }
    for (size_t i = 0; i < probs.size(); ++i) {
        probs[i] /= sum_exp;
    }
    return probs;
}

class Evaluate {
public:
    Evaluate(const std::string& onnxPath, int maxBatchSize) : stopFlag(false), sampleCounter(0) {
        engine = buildEngine(onnxPath, maxBatchSize);
        if (!engine) {
            throw std::runtime_error("Failed to build TensorRT engine");
        }
        context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
        evalThread = std::thread(&Evaluate::evaluateThreadFunc, this);
    }

    ~Evaluate() {
        stop();
        if (evalThread.joinable()) evalThread.join();
    }

    std::future<std::vector<float>> submitRequest(const std::vector<float>& matrix, int board_size) {
        Request req;
        req.matrix = matrix;
        req.board_size = board_size;
        auto future = req.promise.get_future();
        {
            std::lock_guard<std::mutex> lock(mutex);
            queue.push(std::move(req));
        }
        cv.notify_one();
        return future;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopFlag = true;
        }
        cv.notify_one();
    }

    // 添加的方法，用于获取 sampleCounter 的值
    int getSampleCounter() const {
        return sampleCounter.load();
    }

private:
    void evaluateThreadFunc() {
        while (true) {
            std::vector<Request> batch;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [this] { return !queue.empty() || stopFlag; });
                if (stopFlag && queue.empty()) break;

                while (batch.size() < 128 && !queue.empty()) {
                    batch.push_back(std::move(queue.front()));
                    queue.pop();
                }
            }
            if (!batch.empty()) {
                auto results = evaluateBatch(batch);
                for (size_t i = 0; i < batch.size(); ++i) {
                    batch[i].promise.set_value(std::move(results[i]));
                }
            }
        }
    }

    std::vector<std::vector<float>> evaluateBatch(const std::vector<Request>& batch) {
        const int batchSize = static_cast<int>(batch.size());
        const int inputSize = STATE_LAYER_NUM * 20 * 20;
        const int boardSizeInputSize = 1; // board_sizes 是标量
        const int policyLogitsSize = 2 * 20 * 20; // [batch, 2, 20, 20]
        const int valueLogitsSize = 3; // [batch, 3]
        const int policySize = 400; // 输出格式仍为 [batch, 400]

        if (batchSize > 128) {
            throw std::runtime_error("Batch size " + std::to_string(batchSize) + " exceeds maxBatchSize 128");
        }

        // 准备输入数据
        std::vector<float> inputData(batchSize * inputSize);
        std::vector<float> boardSizesData(batchSize);
        for (int i = 0; i < batchSize; ++i) {
            if (batch[i].matrix.size() != inputSize) {
                throw std::runtime_error("Invalid input size for batch " + std::to_string(i) + ": " + std::to_string(batch[i].matrix.size()));
            }
            std::copy(batch[i].matrix.begin(), batch[i].matrix.end(), inputData.begin() + i * inputSize);
            boardSizesData[i] = static_cast<float>(batch[i].board_size);
        }

        // CUDA 内存分配
        float *inputDevice, *boardSizesDevice, *policyLogitsDevice, *valueLogitsDevice;
        CHECK(cudaMalloc(&inputDevice, batchSize * inputSize * sizeof(float)));
        CHECK(cudaMalloc(&boardSizesDevice, batchSize * boardSizeInputSize * sizeof(float)));
        CHECK(cudaMalloc(&policyLogitsDevice, batchSize * policyLogitsSize * sizeof(float)));
        CHECK(cudaMalloc(&valueLogitsDevice, batchSize * valueLogitsSize * sizeof(float)));

        // 复制输入数据到设备
        CHECK(cudaMemcpy(inputDevice, inputData.data(), batchSize * inputSize * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(boardSizesDevice, boardSizesData.data(), batchSize * boardSizeInputSize * sizeof(float), cudaMemcpyHostToDevice));

        // 设置输入形状并执行推理
        void* bindings[] = {inputDevice, boardSizesDevice, policyLogitsDevice, valueLogitsDevice};
        context->setInputShape("input", nvinfer1::Dims4{batchSize, STATE_LAYER_NUM, 20, 20});
        context->setInputShape("board_sizes", nvinfer1::Dims{1, {batchSize}});
        if (!context->executeV2(bindings)) {
            throw std::runtime_error("TensorRT execution failed");
        }

        // 获取输出
        std::vector<float> policyLogitsOutput(batchSize * policyLogitsSize);
        std::vector<float> valueLogitsOutput(batchSize * valueLogitsSize);
        CHECK(cudaMemcpy(policyLogitsOutput.data(), policyLogitsDevice, batchSize * policyLogitsSize * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(valueLogitsOutput.data(), valueLogitsDevice, batchSize * valueLogitsSize * sizeof(float), cudaMemcpyDeviceToHost));

        // 释放 CUDA 内存
        CHECK(cudaFree(inputDevice));
        CHECK(cudaFree(boardSizesDevice));
        CHECK(cudaFree(policyLogitsDevice));
        CHECK(cudaFree(valueLogitsDevice));

        // 处理输出
        std::vector<std::vector<float>> results(batchSize);
        for (int i = 0; i < batchSize; ++i) {
            // 处理 policy_logits [2, 20, 20] -> 当前玩家和对手策略 [400]
            std::vector<float> policy_logits(20 * 20);
            std::vector<float> opp_policy_logits(20 * 20);
            for (int pos = 0; pos < 20 * 20; ++pos) {
                policy_logits[pos] = policyLogitsOutput[i * policyLogitsSize + pos];           // 第 0 通道为当前玩家
                opp_policy_logits[pos] = policyLogitsOutput[i * policyLogitsSize + 20 * 20 + pos]; // 第 1 通道为对手
            }
            std::vector<float> policy = softmax(policy_logits);
            std::vector<float> opp_policy = softmax(opp_policy_logits);

            // 处理 value_logits [3] -> 预测胜率
            std::vector<float> value_logits(valueLogitsOutput.begin() + i * valueLogitsSize, 
                                          valueLogitsOutput.begin() + (i + 1) * valueLogitsSize);
            std::vector<float> value_probs = softmax(value_logits); // [胜, 负, 平]

            // 合并结果
            results[i].reserve(policySize * 2 + 1);
            results[i].insert(results[i].end(), policy.begin(), policy.end());
            results[i].insert(results[i].end(), opp_policy.begin(), opp_policy.end());
            results[i].push_back(value_probs[0]); // 胜概率
            results[i].push_back(value_probs[1]); // 负概率
            results[i].push_back(value_probs[2]); // 平概率
        }

        sampleCounter += batchSize;    // 更新统计信息， 累加已处理样本数量
        return results;
    }

    nvinfer1::ICudaEngine* engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<Request> queue;
    std::atomic<bool> stopFlag;
    std::thread evalThread;
    std::atomic<int> sampleCounter;      // 总共处理过的样本数
};

bool areEqual(const Five& a, const Five& b) {
    // 比较基本类型成员变量
    if (a.ROWS != b.ROWS) {
        throw std::runtime_error("ROWS 不相等: a.ROWS=" + std::to_string(a.ROWS) + ", b.ROWS=" + std::to_string(b.ROWS));
    }
    if (a.COLS != b.COLS) {
        throw std::runtime_error("COLS 不相等: a.COLS=" + std::to_string(a.COLS) + ", b.COLS=" + std::to_string(b.COLS));
    }
    if (a.SIZE != b.SIZE) {
        throw std::runtime_error("SIZE 不相等: a.SIZE=" + std::to_string(a.SIZE) + ", b.SIZE=" + std::to_string(b.SIZE));
    }

    // 比较 std::vector<int> 类型成员变量
    if (a.board != b.board) {
        for (size_t i = 0; i < a.board.size(); ++i) {
            if (a.board[i] != b.board[i]) {
                throw std::runtime_error("board 在索引 " + std::to_string(i) + " 处不相等: a.board[" + std::to_string(i) + "]=" + std::to_string(a.board[i]) + ", b.board[" + std::to_string(i) + "]=" + std::to_string(b.board[i]));
            }
        }
    }

    // 比较 bool 类型成员变量
    if (a.done != b.done) {
        throw std::runtime_error("done 不相等: a.done=" + std::to_string(a.done) + ", b.done=" + std::to_string(b.done));
    }

    // 比较嵌套容器 std::vector<std::vector<int>>
    if (a.pos_to_combinations != b.pos_to_combinations) {
        for (size_t i = 0; i < a.pos_to_combinations.size(); ++i) {
            if (a.pos_to_combinations[i] != b.pos_to_combinations[i]) {
                for (size_t j = 0; j < a.pos_to_combinations[i].size(); ++j) {
                    if (a.pos_to_combinations[i][j] != b.pos_to_combinations[i][j]) {
                        throw std::runtime_error("pos_to_combinations 在 [" + std::to_string(i) + "][" + std::to_string(j) + "] 处不相等: a=" + std::to_string(a.pos_to_combinations[i][j]) + ", b=" + std::to_string(b.pos_to_combinations[i][j]));
                    }
                }
            }
        }
    }

    // 比较 std::vector<std::set<int>>
    if (a.chongsi_set != b.chongsi_set) {
        for (size_t i = 0; i < a.chongsi_set.size(); ++i) {
            if (a.chongsi_set[i] != b.chongsi_set[i]) {
                std::ostringstream oss;
                oss << "chongsi_set[" << i << "] 不相等: ";
                for (const auto& elem : a.chongsi_set[i]) {
                    if (b.chongsi_set[i].find(elem) == b.chongsi_set[i].end()) {
                        oss << "a 包含 " << elem << " 但 b 不包含; ";
                    }
                }
                for (const auto& elem : b.chongsi_set[i]) {
                    if (a.chongsi_set[i].find(elem) == a.chongsi_set[i].end()) {
                        oss << "b 包含 " << elem << " 但 a 不包含; ";
                    }
                }
                throw std::runtime_error(oss.str());
            }
        }
    }

    // 比较 std::vector<std::set<int>>
    if (a.chengwu_set != b.chengwu_set) {
        for (size_t i = 0; i < a.chengwu_set.size(); ++i) {
            if (a.chengwu_set[i] != b.chengwu_set[i]) {
                std::ostringstream oss;
                oss << "chengwu_set[" << i << "] 不相等: ";
                for (const auto& elem : a.chengwu_set[i]) {
                    if (b.chengwu_set[i].find(elem) == b.chengwu_set[i].end()) {
                        oss << "a 包含 " << elem << " 但 b 不包含; ";
                    }
                }
                for (const auto& elem : b.chengwu_set[i]) {
                    if (a.chengwu_set[i].find(elem) == a.chengwu_set[i].end()) {
                        oss << "b 包含 " << elem << " 但 a 不包含; ";
                    }
                }
                throw std::runtime_error(oss.str());
            }
        }
    }

    // 如果所有成员变量都相等，返回 true
    return true;
}


class MCTSNode {
public:
    int action_player;
    int action;
    MCTSNode* parent;
    std::vector<MCTSNode*> children;
    int visits; //包括nforced的总visits
    int nforced;
    std::vector<float> value_probs;       // [win, lose, draw] 的累积概率
    // float value;
    float prior;
    std::vector<float> raw_value_probs;
    float total_visited_children_prior;

    MCTSNode(int player, int action = -1, MCTSNode* parent = nullptr, float prior = 0.0f)
        : action_player(player), 
        action(action), 
        parent(parent), 
        visits(0), 
        nforced(0),
        value_probs(3, 0.0f),
        // value(0.0f), 
        prior(prior),
        raw_value_probs(3, 0.0f),
        total_visited_children_prior(0.0f) {}
    ~MCTSNode() {
        for (auto* child : children) delete child;
    }
};

class MCTS {
public:
    MCTSNode* root;
    Evaluate* eval;
    Five* game;
    float c_PUCT;
    int use_mode;           // 0: realnet, 1: scorepolicy, 2: random policy
    bool use_forced_playout;    // Switch for forced playouts
    bool use_dirichlet_noise;   // Switch for Dirichlet noise
    float dirichlet_alpha;      // Dirichlet noise parameter
    float dirichlet_epsilon;    // Dirichlet noise mixing proportion
    std::mt19937& gen;           // Random number generator for Dirichlet noise

    MCTS(Five* game, Evaluate* eval, std::mt19937& gen, float exploration_weight = 1.1f, int use_mode = 0, bool use_forced_playout = false, 
         bool use_dirichlet_noise = false, float dirichlet_alpha = 0.03f, 
         float dirichlet_epsilon = 0.25f)
        : game(game), eval(eval), root(new MCTSNode(-game->current_player/*注意这里是负*/)), c_PUCT(exploration_weight), use_mode(use_mode),
          use_forced_playout(use_forced_playout), use_dirichlet_noise(use_dirichlet_noise),
          dirichlet_alpha(dirichlet_alpha), dirichlet_epsilon(dirichlet_epsilon), gen(gen) {}

    ~MCTS() { delete root; }

    int search(int iterations) {
        Five one_search_game = game->clone();

        for (int i = 0; i < iterations; ++i) {
            std::vector<int> made_moves;
            MCTSNode* leaf = select(one_search_game, made_moves);
#ifdef DEBUG
            // 断言所有元素 >= 0
            if (std::any_of(one_search_game.chongsi_count[0].begin(), one_search_game.chongsi_count[0].end(), [](int x) { return x < 0; }) ||
                std::any_of(one_search_game.chongsi_count[1].begin(), one_search_game.chongsi_count[1].end(), [](int x) { return x < 0; }) ||
                std::any_of(one_search_game.chengwu_count[0].begin(), one_search_game.chengwu_count[0].end(), [](int x) { return x < 0; }) ||
                std::any_of(one_search_game.chengwu_count[1].begin(), one_search_game.chengwu_count[1].end(), [](int x) { return x < 0; }))
            {
                std::cerr << "出错：make可能有bug"<<std::endl;
                std::cerr << "move_history: ";
                for(auto mmm:one_search_game.move_history)
                    std::cerr<<mmm<<", ";
                std::cerr<<std::endl;
                std::cerr << "made_moves(and unmake them): ";
                for(auto mmm:made_moves)
                    std::cerr<<mmm<<", ";
                std::cerr<<std::endl;
                assert(false);
            }
#endif
            MCTSNode* child = expand(one_search_game, leaf);
            std::vector<float> result = simulate(one_search_game, child, made_moves);
            backpropagate(child, result);
            // 撤销所有移动
            for (int j = made_moves.size() - 1; j >= 0; --j) {
                one_search_game.unmake_move(made_moves[j]);
            }
#ifdef DEBUG
            // 断言所有元素 >= 0
            // 断言所有元素 >= 0
            if (std::any_of(one_search_game.chongsi_count[0].begin(), one_search_game.chongsi_count[0].end(), [](int x) { return x < 0; }) ||
                std::any_of(one_search_game.chongsi_count[1].begin(), one_search_game.chongsi_count[1].end(), [](int x) { return x < 0; }) ||
                std::any_of(one_search_game.chengwu_count[0].begin(), one_search_game.chengwu_count[0].end(), [](int x) { return x < 0; }) ||
                std::any_of(one_search_game.chengwu_count[1].begin(), one_search_game.chengwu_count[1].end(), [](int x) { return x < 0; }))
            {
                std::cerr << "出错：make可能有bug"<<std::endl;
                std::cerr << "move_history: ";
                for(auto mmm:one_search_game.move_history)
                    std::cerr<<mmm<<", ";
                std::cerr<<std::endl;
                std::cerr << "made_moves(and unmake them): ";
                for(auto mmm:made_moves)
                    std::cerr<<mmm<<", ";
                std::cerr<<std::endl;
                assert(false);
            }
            const auto &ngs = one_search_game.get_state();
            const auto &gs = game->get_state();
            if(ngs != gs)
            {
                std::cerr << "出错：unmake可能有bug"<<std::endl;
                std::cerr << "move_history: ";
                for(auto mmm:one_search_game.move_history)
                    std::cerr<<mmm<<", ";
                std::cerr<<std::endl;
                std::cerr << "made_moves(and unmake them): ";
                for(auto mmm:made_moves)
                    std::cerr<<mmm<<", ";
                std::cerr<<std::endl;
                areEqual(one_search_game, *game);
                for(auto element:ngs)
                    std::cout<<element<<" ";
                std::cout<<std::endl;
                for(auto element:gs)
                    std::cout<<element<<" ";
                std::cout<<std::endl;
                assert(false);
            }
#endif
        }

        MCTSNode* best_children = find_c_star_with_most_visits_and_puct_max();

        return best_children->action;
    }

    float puct_value_with_visits(const MCTSNode* node, int visits) {
        float q = (visits == 0) ? 0.0f : (node->value_probs[0] - node->value_probs[1]) / visits;
        float u = c_PUCT * node->prior * std::sqrt(static_cast<float>(node->parent->visits)) / (1 + visits);
        return q + u;
    }

    std::vector<float> get_pruned_policy(const Five& game) {
        if (root->children.empty()) {
            // If no children, return a uniform distribution over available moves
            std::vector<float> policy(game.SIZE, 0.0f);
            auto moves = game.available_moves();
            if (!moves.empty()) {
                float uniform_prob = 1.0f / moves.size();
                for (int move : moves) {
                    policy[move] = uniform_prob;
                }
            }
            return policy;
        }
    
        // Step 1: Find the child with the maximum visits (c*)
        const MCTSNode* c_star = find_c_star_with_most_visits_and_puct_max();
    
        if (!c_star) {
            // Should not happen if children exist, but handle gracefully
            std::vector<float> policy(game.SIZE, 0.0f);
            auto moves = game.available_moves();
            float uniform_prob = moves.empty() ? 0.0f : 1.0f / moves.size();
            for (int move : moves) {
                policy[move] = uniform_prob;
            }
            return policy;
        }
    
        // Step 2: Compute PUCT value for c*
        float puct_c_star = puct_value_with_visits(c_star, c_star->visits);
    
        // Step 3: Compute adjusted visit counts with pruning
        std::vector<float> adjusted_visits(game.SIZE, 0.0f);
        float total_adjusted_visits = 0.0f;
    
        for (const auto* child : root->children) {
            if (child == c_star) {
                // Retain all visits for c*
                adjusted_visits[child->action] = static_cast<float>(child->visits);
                total_adjusted_visits += child->visits;
            } else {
                // Calculate forced playouts (nforced)
                // The paper: we subtract up to nforced playouts so long as it does not cause PUCT(c) >= PUCT(c)*
                int adj_visits = child->visits;
                for (int i = 0; i < child->nforced && adj_visits > 1; ++i) {
                    float puct_c = puct_value_with_visits(child, adj_visits - 1);
                    if (puct_c < puct_c_star) {
                        break;
                    }
                    adj_visits--;
                }
                // The paper: Additionally, we outright prune children that are reduced to a single playout.
                if (adj_visits > 1) {
                    adjusted_visits[child->action] = static_cast<float>(adj_visits);
                    total_adjusted_visits += adj_visits;
                }
            }
        }
    
        // Step 4: Normalize to create a probability distribution
        std::vector<float> policy(game.SIZE, 0.0f);
        if (total_adjusted_visits > 0) {
            for (int pos = 0; pos < game.SIZE; ++pos) {
                policy[pos] = adjusted_visits[pos] / total_adjusted_visits;
            }
        } else {
            // Fallback to uniform distribution over available moves
            auto moves = game.available_moves();
            float uniform_prob = moves.empty() ? 0.0f : 1.0f / moves.size();
            for (int move : moves) {
                policy[move] = uniform_prob;
            }
        }
    
        return policy;
    }

    std::vector<float> get_policy(const Five& game) {
        // 如果根节点的子节点为空，返回均匀分布
        if (root->children.empty()) {
            std::vector<float> policy(game.SIZE, 0.0f);
            auto moves = game.available_moves();
            if (!moves.empty()) {
                float uniform_prob = 1.0f / moves.size();
                for (int move : moves) {
                    policy[move] = uniform_prob;
                }
            }
            return policy;
        }
    
        std::vector<float> policy(game.SIZE, 0.0f);
        float total_visits = 0.0f;
    
        // 不修剪，直接使用子节点的访问次数
        for (const auto* child : root->children) {
            policy[child->action] = static_cast<float>(child->visits);
            total_visits += child->visits;
        }
    
        // 归一化策略分布
        if (total_visits > 0) {
            for (int pos = 0; pos < game.SIZE; ++pos) {
                policy[pos] /= total_visits;
            }
        } else {
            // 如果 total_visits == 0，返回均匀分布
            auto moves = game.available_moves();
            float uniform_prob = moves.empty() ? 0.0f : 1.0f / moves.size();
            for (int move : moves) {
                policy[move] = uniform_prob;
            }
        }
    
        return policy;
    }

    
    MCTSNode* find_c_star_with_most_visits_and_puct_max()
    {
        // 找到所有 visits 最大的子节点
        int max_visits = 0;
        std::vector<MCTSNode*> max_visits_children;
        for (auto* child : root->children) {
            if (child->visits > max_visits) {
                max_visits = child->visits;
                max_visits_children = { child }; // 重置集合，只保留当前节点
            }
            else if (child->visits == max_visits) {
                max_visits_children.push_back(child); // 添加到集合
            }
        }

        // 如果只有一个子节点，直接返回
        if (max_visits_children.size() == 1) {
            return max_visits_children[0];
        }

        // 先找到 PUCT 最大值
        float max_puct = -std::numeric_limits<float>::max();
        for (auto* child : max_visits_children) {
            float puct = calculate_puct(child);
            if (puct > max_puct) {
                max_puct = puct;
            }
        }

        // 收集所有 PUCT 最大的子节点
        std::vector<MCTSNode*> best_children;
        for (auto* child : max_visits_children) {
            float puct = calculate_puct(child);
            if (std::abs(puct - max_puct) < 1e-6) { // 避免浮点误差
                best_children.push_back(child);
            }
        }

        // 在 PUCT 最大的子节点中随机选择一个
        int idx = gen() % best_children.size();
        return best_children[idx];
    }

private:
    // Generate Dirichlet noise for exploration
    std::vector<float> generate_dirichlet_noise(size_t size, float alpha) {
        std::vector<float> noise(size);
        std::gamma_distribution<float> gamma(alpha, 1.0f);
        float sum = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            noise[i] = gamma(gen);
            sum += noise[i];
        }
        for (size_t i = 0; i < size; ++i) {
            noise[i] /= sum;
        }
        return noise;
    }
    
    MCTSNode* select(Five& one_search_game, std::vector<int>& made_moves) {
        MCTSNode* node = root;
        while (!node->children.empty()) {
            node = select_best_child(node);
            one_search_game.make_move(node->action);
            made_moves.push_back(node->action);
        }
        return node;
    }

    MCTSNode* select_best_child(MCTSNode* node) {
        // 所有子节点都没有select时，直接选择prior最大的
        if (node->visits == 1)
        {
            float max_prior = -std::numeric_limits<float>::max();
            std::vector<MCTSNode*> max_children;
            for (auto* child : node->children) {
                if (child->prior > max_prior) {
                    max_prior = child->prior;
                    max_children = {child}; // 清空并添加当前子节点
                } else if (child->prior == max_prior) {
                    max_children.push_back(child); // 添加到候选列表
                }
            }
            if (max_children.empty()) {
                throw std::runtime_error("No children found when node->visits == 1");
            }
            std::uniform_int_distribution<> dist(0, max_children.size() - 1);
            return max_children[dist(gen)]; // 随机选择一个
        }

        float max_puct = -std::numeric_limits<float>::max();
        std::vector<MCTSNode*> max_children;
        for (auto* child : node->children) {
            float puct = puct_value(child);
            if (puct > max_puct) {
                max_puct = puct;
                max_children = {child};
            } else if (puct == max_puct) {
                max_children.push_back(child);
            }
        }
        // Ensure max_children is not empty
        if (max_children.empty()) {
            throw std::runtime_error("No children with finite PUCT values");
        }
        std::uniform_int_distribution<> dist(0, max_children.size() - 1);
        return max_children[dist(gen)];
    }

    float first_visit_calc_Q(MCTSNode* node) {
        assert(node->visits == 0);

        // 算Q值，需要特别方法，详细katago issue #1046
        float q;
        float alpha = 1.0f;
        float c = 0.0f;
        float frac = std::pow(node->parent->total_visited_children_prior, alpha);
        // 计算父节点的平均价值
        float parent_avg_value = (node->parent->value_probs[0] - node->parent->value_probs[1]) / node->parent->visits;
        // 这里都要取负号吧 价值取向相反的
        q = (-parent_avg_value) * frac + (-(node->parent->raw_value_probs[0] - node->parent->raw_value_probs[1])) * (1 - frac) - c;

        return q;
    }

    float puct_value(MCTSNode* node) {
        // 对于根节点的子节点，如果曾经visit过，且启用了 forced_playout，实时计算 forced_playout
        if (node->visits >= 1 && node->parent == root && use_forced_playout) {
            int forced_playouts = static_cast<int>(std::sqrt(2.0f * node->prior * static_cast<float>(root->visits - 1)));
            if (node->visits < forced_playouts) {
                node->nforced++;
                return std::numeric_limits<float>::max(); // 设置为极大值，确保优先选择
            }
        }
        // 正常计算 PUCT

        // 算Q值，需要特别方法，详细katago issue #1046
        float q;
        if(node->visits == 0)
        {
            q = first_visit_calc_Q(node);
        }
        else
        {
            q = (node->value_probs[0] - node->value_probs[1]) / node->visits; // Q = (win - lose) / visits
        }
        float u = c_PUCT * node->prior * std::sqrt(static_cast<float>(node->parent->visits - 1)) / (1 + node->visits);
        return q + u;
    }

    MCTSNode* expand(Five& one_search_game, MCTSNode* node) {
        if (one_search_game.is_terminal()) return node;
    
        std::vector<float> raw_policy; // [400]
        std::vector<float> raw_opp_policy; // [400]
        std::vector<float> policy(one_search_game.SIZE, 0.0f);  // 调整后的策略，大小为游戏棋盘的 SIZE
        std::vector<float> value_probs;

        if (use_mode == 1) {
            // Scorepolicy mode
            policy = one_search_game.get_score_policy();
        } else if (use_mode == 2) {
            // Random policy mode: uniform distribution, no GPU, reward = 0
            auto moves = one_search_game.available_moves();
            if (!moves.empty()) {
                float uniform_prob = 1.0f / moves.size();
                for (int move : moves) {
                    policy[move] = uniform_prob;
                }
            }
            value_probs = {0.5f, 0.5f, 0.0f}; // [win, lose, draw]
        } else {// use_mode == 0 (realnet)
            if (eval == nullptr) {
                throw std::runtime_error("eval is nullptr when use_mode == 0 (realnet)");
            }
            auto state = one_search_game.get_state();
            auto future = eval->submitRequest(state, one_search_game.ROWS); // 传入 board_size
            auto result = future.get();
            raw_policy.assign(result.begin(), result.begin() + 400);
            raw_opp_policy.assign(result.begin() + 400, result.begin() + 800);
            value_probs = {result[800], result[801], result[802]}; // [win, lose, draw]

            // 将 raw_policy (20x20) 转换为 policy (game->SIZE)
            float sum_policy = 0.0f;
            for (int row = 0; row < one_search_game.ROWS; ++row) {
                for (int col = 0; col < one_search_game.COLS; ++col) {
                    int local_pos = row * one_search_game.COLS + col;  // 棋盘上的位置
                    int net_pos = row * one_search_game.nnLen + col;   // 神经网络输出中的位置 (20x20)
                    policy[local_pos] = raw_policy[net_pos];
                    sum_policy += policy[local_pos];
                }
            }

            // 归一化 policy
            if (sum_policy > 0.0f) {
                for (int i = 0; i < one_search_game.SIZE; ++i) {
                    policy[i] /= sum_policy;
                }
            }
        }
    
        auto moves = one_search_game.available_moves();

        // Apply Dirichlet noise only at the root node
        // 随机策略也加迪利克雷噪声
        if (node == root && use_dirichlet_noise) {
            // 通过katago的公式动态计算alpha
            float alpha = dirichlet_alpha * static_cast<float>(one_search_game.SIZE) / static_cast<float>(moves.size());
            std::vector<float> dirichlet_noise = generate_dirichlet_noise(moves.size(), alpha);
            float sum_policy = 0.0f;
            for (size_t i = 0; i < moves.size(); ++i) {
                int move = moves[i];
                policy[move] = (1 - dirichlet_epsilon) * policy[move] + dirichlet_epsilon * dirichlet_noise[i];
                sum_policy += policy[move];
            }
            if (sum_policy > 0) {
                for (int move : moves) {
                    policy[move] /= sum_policy;
                }
            }
        }

        for (int move : moves) {
            float prior = policy[move];
            auto* child = new MCTSNode(one_search_game.current_player, move, node, prior);
            node->children.push_back(child);
        }
        // 直接把自身给simulate，防止随机选择一个的时候，value会非常虚假
        // 据说这样也是更接近alphago zero？？
        // 留着这些信息后面计算，详细katago issue #1046
        // 这负号也是坑了老半天，草
        node->raw_value_probs = {value_probs[1], value_probs[0], value_probs[2]}; // 记得win和Lose要颠倒！

        if (node == root)
        {
            // root节点没有父亲
            // 等待后续backpropagate更新visit和value即可
        }
        else
        {
            // 除root外其它节点都有父亲，因此需要计算父亲并汇报累积
            // node->value = first_visit_calc_Q(node);  // NONONO，这个是用来算未expand的东西的，别重复加value了
            node->parent->total_visited_children_prior += node->prior;
        }

        return node;
    }

    std::vector<float> simulate(Five& one_search_game, MCTSNode* node, std::vector<int>& made_moves) {
        // do not delete this
        if (one_search_game.is_terminal()) {
            int reward = one_search_game.get_reward(node->action_player);
            if (reward == 1) return {1.0f, 0.0f, 0.0f};      // win
            else if (reward == -1) return {0.0f, 1.0f, 0.0f}; // lose
            else return {0.0f, 0.0f, 1.0f};                  // draw
        }

        // game is not over
        if (use_mode == 1 || use_mode == 2) {
            return {0.5f, 0.5f, 0.0f}; // 使用 score policy 时不知道价值，干脆设为 0
        } else {
            return node->raw_value_probs; // 非终态，返回神经网络预测的 [win, lose, draw]
        }
    }

    void backpropagate(MCTSNode* node, const std::vector<float>& result) {
        int leaf_player = node->action_player;  // 叶子节点的玩家
        while (node) {
            node->visits++;
            if (node->action_player == leaf_player) {
                // 当前玩家与叶子节点玩家相同，直接累加
                node->value_probs[0] += result[0]; // win
                node->value_probs[1] += result[1]; // lose
                node->value_probs[2] += result[2]; // draw
            } else {
                // 当前玩家与叶子节点玩家相反，win 和 lose 对调
                node->value_probs[0] += result[1]; // 对手的 lose 是自己的 win
                node->value_probs[1] += result[0]; // 对手的 win 是自己的 lose
                node->value_probs[2] += result[2]; // draw 不变
            }
            node = node->parent;
        }
    }

    float calculate_puct(MCTSNode* node) {
        if (node->visits == 0) {
            return 0.0f;
        }
        float q = (node->value_probs[0] - node->value_probs[1]) / node->visits;   // Q = (win - lose) / visits
        float u = c_PUCT * node->prior * std::sqrt(static_cast<float>(node->parent->visits)) / (1 + node->visits);
        return q + u;
    }

};

void save_to_npz(const std::string& filename,
                 const std::vector<int>& all_game_nums,
                 const std::vector<int>& all_game_types,
                 const std::vector<int>& sizes,
                 const std::vector<std::vector<float>>& states,
                 const std::vector<std::vector<float>>& policy_targets,
                 const std::vector<std::vector<float>>& opp_policy_targets,
                 const std::vector<std::vector<float>>& value_targets
                ) {
    
    size_t num_samples = states.size();
    if (num_samples == 0) return;

    assert(all_game_nums.size() == num_samples);
    assert(all_game_types.size() == num_samples);
    assert(sizes.size() == num_samples);
    assert(states.size() == num_samples);
    assert(policy_targets.size() == num_samples);
    assert(opp_policy_targets.size() == num_samples);
    assert(value_targets.size() == num_samples);
    // 辅助 lambda 函数：将嵌套 vector 展平为连续内存的一维 vector
    auto flatten = [](const std::vector<std::vector<float>>& data) {
        std::vector<float> flat;
        for (const auto& v : data) {
            flat.insert(flat.end(), v.begin(), v.end());
        }
        return flat;
    };

    // 准备数据
    std::vector<float> flat_states = flatten(states);
    std::vector<float> flat_policy = flatten(policy_targets);
    std::vector<float> flat_opp_policy = flatten(opp_policy_targets);
    std::vector<float> flat_value = flatten(value_targets);

    // 获取每个数组的维度形状 (Shape)
    // 假设 states[i] 的长度是固定的
    size_t state_dim = states[0].size();
    size_t policy_dim = policy_targets[0].size();

    // 写入 NPZ 文件
    // 参数：文件名, 数组名, 数据指针, 形状(vector<size_t>), 模式 ("w"覆盖, "a"追加)
    cnpy::npz_save(filename, "states", &flat_states[0], {num_samples, state_dim}, "w");
    cnpy::npz_save(filename, "policy_targets", &flat_policy[0], {num_samples, policy_dim}, "a");
    cnpy::npz_save(filename, "opp_policy_targets", &flat_opp_policy[0], {num_samples, policy_dim}, "a");
    cnpy::npz_save(filename, "value_targets", &flat_value[0], {num_samples, value_targets[0].size()}, "a");
    cnpy::npz_save(filename, "size", (int*)&sizes[0], {num_samples}, "a");
    // 元数据
    cnpy::npz_save(filename, "game_num", (int*)&all_game_nums[0], {num_samples}, "a");
    cnpy::npz_save(filename, "game_type", (int*)&all_game_types[0], {num_samples}, "a");
}

std::vector<std::vector<int>> load_openings(const std::string& directory) {
    std::vector<std::vector<int>> openings;
    int size = (directory == "openings_15") ? 15 : 20; // Default to 20 unless 15 is specified
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().extension() == ".txt") {
            std::ifstream file(entry.path());
            std::string line;
            while (std::getline(file, line)) {
                // Remove trailing '\r' if present (e.g., from Windows line endings)
                if (!line.empty() && line.back() == '\r') {
                    line.pop_back();
                }
                std::istringstream iss(line);
                std::string token;
                std::vector<int> opening;
                // Parse comma-separated (x, y) pairs
                while (std::getline(iss, token, ',')) {
                    // Clean token by removing all whitespace characters
                    token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
                    if (token.empty()) continue; // Skip empty tokens

                    try {
                        int x = std::stoi(token);
                        if (!std::getline(iss, token, ',')) break; // Ensure y follows
                        // Clean the next token (y coordinate)
                        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
                        if (token.empty()) break; // Skip if y token is empty

                        int y = std::stoi(token);
                        // Convert (x, y) to position index on a 20x20 board or 15x15 board
                        int pos = (y + size/2) * size + (x + size/2);
                        opening.push_back(pos);
                    } catch (const std::invalid_argument& e) {
                        std::cerr << "Invalid argument in token: " << token 
                                  << " in file: " << entry.path() << std::endl;
                        continue; // Skip this token and proceed
                    } catch (const std::out_of_range& e) {
                        std::cerr << "Out of range in token: " << token 
                                  << " in file: " << entry.path() << std::endl;
                        continue; // Skip this token and proceed
                    }
                }
                if (!opening.empty()) {
                    openings.push_back(opening);
                }
            }
        }
    }
    return openings;
}

std::vector<int> apply_transform(const std::vector<int>& opening, int transform_idx, int size) {
    std::vector<int> transformed_opening;
    for (int pos : opening) {
        int x = pos % size;
        int y = pos / size;
        int new_pos;
        switch (transform_idx) {
            case 0: // Identity
                new_pos = y * size + x;
                break;
            case 1: // 90° clockwise
                new_pos = (size - 1 - x) * size + y;
                break;
            case 2: // 180°
                new_pos = (size - 1 - y) * size + (size - 1 - x);
                break;
            case 3: // 270° clockwise
                new_pos = x * size + (size - 1 - y);
                break;
            case 4: // Horizontal flip
                new_pos = y * size + (size - 1 - x);
                break;
            case 5: // Vertical flip
                new_pos = (size - 1 - y) * size + x;
                break;
            case 6: // Main diagonal flip
                new_pos = x * size + y;
                break;
            case 7: // Anti-diagonal flip
                new_pos = (size - 1 - x) * size + (size - 1 - y);
                break;
            default:
                new_pos = pos;
                break;
        }
        transformed_opening.push_back(new_pos);
    }
    return transformed_opening;
}

// 递归打印节点信息的函数
void printNodeInfo(const MCTSNode* node, const std::string& prefix) {
    if (node->visits >= 1) {
        std::cout << prefix << ": " << node->action << "\t" 
                  << node->visits << "\t" << (node->value_probs[0] - node->value_probs[1]) << "\t" 
                  << node->prior << std::endl;
        // 递归遍历所有子节点
        for (const auto* child : node->children) {
            printNodeInfo(child, prefix + "->child");
        }
    }
}

int select_move_from_raw_policy_by_temperature(const Five& game, Evaluate* eval, std::mt19937& gen, int use_mode, float T = 1.0f) {
        auto state = game.get_state();
        auto future = eval->submitRequest(state, game.ROWS); // 传入 board_size
        auto result = future.get();
        std::vector<float> raw_policy; // [400]
        std::vector<float> policy(game.SIZE, 0.0f);  // 转换为local后的策略
        raw_policy.assign(result.begin(), result.begin() + 400);
        int move;

        // Get available moves from the game
        auto available_moves = game.available_moves();
        if (available_moves.empty()) {
            throw std::runtime_error("No available moves to select from.");
        }

        // Create a policy vector for available moves only
        std::vector<float> avail_policy;
        for (int move : available_moves) {
            int row = move / game.COLS;
            int col = move % game.COLS;
            int net_pos = row * game.nnLen + col; // Map to neural network position (20x20)
            avail_policy.push_back(raw_policy[net_pos]);
        }

        // Normalize the available policy
        float sum_avail = 0.0f;
        for (float p : avail_policy) {
            sum_avail += p;
        }
        if (sum_avail > 0.0f) {
            for (float& p : avail_policy) {
                p /= sum_avail;
            }
        } else {
            // Fallback to uniform distribution if sum is zero
            throw std::runtime_error("Error, sum_avail is 0 after normalization!");
        }

        // Adjust policy based on temperature T
        std::vector<double> adjusted_policy(avail_policy.size(), 0.0);
        double sum_adjusted = 0.0;
        for (size_t i = 0; i < avail_policy.size(); ++i) {
            if (avail_policy[i] > 0) {
                adjusted_policy[i] = std::pow(static_cast<double>(avail_policy[i]), 1.0 / T);
                sum_adjusted += adjusted_policy[i];
            }
        }

        // Normalize the adjusted policy
        if (sum_adjusted > 0) {
            for (size_t i = 0; i < adjusted_policy.size(); ++i) {
                adjusted_policy[i] /= sum_adjusted;
            }
        } else {
            // Fallback to uniform distribution if adjustment fails
            throw std::runtime_error("Error, sum_adjusted is 0 after adjustment!");
        }

        // Sample an index from the adjusted policy
        std::discrete_distribution<int> move_dist(adjusted_policy.begin(), adjusted_policy.end());
        int idx = move_dist(gen);

        // Return the corresponding move from available_moves
        return available_moves[idx];
}

int select_move_uniform(const Five& game, std::mt19937& gen)
{
    auto moves = game.available_moves();
    if (moves.empty()) {
        throw std::runtime_error("No available moves to select from.");
    }
    std::uniform_int_distribution<int> dist(0, moves.size() - 1);
    int move_index = dist(gen);
    return moves[move_index];
}

void forkside_one_pair(Five& forked_game, Evaluate* eval, std::mt19937& gen, int use_mode, int thread_id, unsigned int worker_seed,
                       std::vector<int>& forked_all_moves, std::vector<std::string>& forked_move_labels,
                       std::vector<std::vector<float>>& forkside_states, std::vector<std::vector<float>>& forkside_policy_targets,
                       std::vector<std::vector<float>>& forkside_opp_policy_targets, std::vector<std::vector<float>>& forkside_value_targets,
                       std::vector<int>& forkside_current_players) {

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float rand_val = dist(gen);
    int move;

    if (use_mode == 0){
        if (rand_val < 0.70f) {
            // 70% 概率：温度为 1 的策略
            move = select_move_from_raw_policy_by_temperature(forked_game, eval, gen, use_mode, 1.0f);
        } else if (rand_val < 0.95f) { // 0.70 + 0.25 = 0.95
            // 25% 概率：温度为 2 的策略
            move = select_move_from_raw_policy_by_temperature(forked_game, eval, gen, use_mode, 2.0f);
        } else {
            // 5% 概率：完全随机
            move = select_move_uniform(forked_game, gen);
        }
    }
    else{
        move = select_move_uniform(forked_game, gen);
    }

    forked_game.make_move(move);
    forked_all_moves.push_back(move);
    forked_move_labels.push_back("(forkside)");

    if (!forked_game.is_terminal())// 游戏结束了，那也就没有应对了
    {
        // Record Forkside data
        forkside_states.push_back(forked_game.get_state());
        forkside_current_players.push_back(forked_game.current_player);

        // Perform Forkside MCTS search
        MCTS forked_mcts(&forked_game, (use_mode == 0) ? eval : nullptr, gen, 1.1f, 
                            use_mode, true, true);
        forked_mcts.search(FULL_SEARCH_ITERATION);
        std::vector<float> forked_policy = forked_mcts.get_pruned_policy(forked_game);

        // Convert policy to neural network size (20x20) 用于训练
        std::vector<float> policy_target(forked_game.nnSize, 0.0f);
        for (int row = 0; row < forked_game.ROWS; ++row) {
            for (int col = 0; col < forked_game.COLS; ++col) {
                int local_pos = row * forked_game.COLS + col;
                int net_pos = row * forked_game.nnLen + col;
                policy_target[net_pos] = forked_policy[local_pos];
            }
        }
        forkside_policy_targets.push_back(policy_target);
        // 没有下一步或者下一步也是乱下，因此loss设为0
        std::vector<float> empty_policy(forked_game.nnSize, 0.0f);
        forkside_opp_policy_targets.push_back(empty_policy);

        // 计算 value_target，使用 root 价值的相反数
        std::vector<float> value_target(3, 0.0f);
        if (forked_mcts.root->visits > 0) {
            float win_prob_root = forked_mcts.root->value_probs[0] / forked_mcts.root->visits;  // 对手视角的胜率
            float lose_prob_root = forked_mcts.root->value_probs[1] / forked_mcts.root->visits; // 对手视角的负率
            float draw_prob_root = forked_mcts.root->value_probs[2] / forked_mcts.root->visits; // 平局概率
            // 当前玩家的视角：win = lose_prob_root, lose = win_prob_root, draw = draw_prob_root
            value_target = {lose_prob_root, win_prob_root, draw_prob_root};
        } else {
            throw std::runtime_error("Error, root->visits is 0 after search!");
        }

        forkside_value_targets.push_back(value_target);

        // 计算温度 T
        double N = static_cast<double>(forked_game.ROWS);  // 棋盘大小
        double m = static_cast<double>(forked_all_moves.size()); // 已执行步数
        double T = 0.2 + 0.6 * std::pow(0.5, m / N);

        // 使用温度 T 调整分布
        std::vector<double> adjusted_policy(forked_policy.size(), 0.0);
        double sum_adjusted = 0.0;
        for (size_t i = 0; i < forked_policy.size(); ++i) {
            if (forked_policy[i] > 0) {
                adjusted_policy[i] = std::pow(static_cast<double>(forked_policy[i]), 1.0 / T);
                sum_adjusted += adjusted_policy[i];
            }
        }
        if (sum_adjusted > 0) {
            for (size_t i = 0; i < adjusted_policy.size(); ++i) {
                adjusted_policy[i] /= sum_adjusted;
            }
        } else {
            throw std::runtime_error("Error, sum_adjusted is 0 after adjustment!");
        }

        // 从调整后的分布中采样 move
        std::discrete_distribution<int> move_dist(adjusted_policy.begin(), adjusted_policy.end());
        move = move_dist(gen);
        forked_game.make_move(move);
        forked_all_moves.push_back(move);
        forked_move_labels.push_back("(full)"); // Label full search moves
    }
}

std::string generate_filename(const std::string& prefix, int thread_id, unsigned int worker_seed, int data_size) {
    // 获取当前时间
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_tm = *std::localtime(&now_time_t);

    std::ostringstream oss;

    oss << "selfplay/" << prefix << "_"
        << "Thread" << std::setw(2) << std::setfill('0') << thread_id << "_"
        << std::put_time(&now_tm, "%Y%m%d_%H%M%S") << "_"
        << worker_seed << "_d" << data_size << ".npz";

    return oss.str();
}

void forkside(const Five& game, Evaluate* eval, std::mt19937& gen, int use_mode, int thread_id, unsigned int worker_seed, 
    const std::vector<int>& all_moves, const std::vector<std::string>& move_labels, 
    int thread_game_num,
    std::vector<int>& all_game_nums,
    std::vector<int>& all_game_types,
    std::vector<int>& all_sizes,
    std::vector<std::vector<float>>& all_states,
    std::vector<std::vector<float>>& all_policy_targets,
    std::vector<std::vector<float>>& all_opp_policy_targets,
    std::vector<std::vector<float>>& all_value_targets)
{
    std::vector<std::vector<float>> forkside_states;         // 每个 full search 的状态
    std::vector<std::vector<float>> forkside_policy_targets; // 每个 full search 的策略目标
    std::vector<std::vector<float>> forkside_opp_policy_targets;  // 每个 full search 的预测对手下一步策略
    std::vector<std::vector<float>> forkside_value_targets;
    std::vector<int> forkside_current_players;               // 每个 full search 的当前玩家
    int move;

    // Clone the inherited all_moves and move_labels
    std::vector<int> forked_all_moves = all_moves;
    std::vector<std::string> forked_move_labels = move_labels;

    // Clone the current game state
    Five forked_game = game.clone();

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    forkside_one_pair(forked_game, eval, gen, use_mode, thread_id, worker_seed,
                        forked_all_moves, forked_move_labels,
                        forkside_states, forkside_policy_targets,
                        forkside_opp_policy_targets, forkside_value_targets,
                        forkside_current_players);

    if (!forked_game.is_terminal())
    {
        float rand_val = dist(gen);
        if (rand_val < 0.25f) {
            // 25% 概率：再乱走一次，用来训练对于连续两次乱走的应对
            forkside_one_pair(forked_game, eval, gen, use_mode, thread_id, worker_seed,
                            forked_all_moves, forked_move_labels,
                            forkside_states, forkside_policy_targets,
                            forkside_opp_policy_targets, forkside_value_targets,
                            forkside_current_players);
        }
    }
    
    // 在保存数据之前加入断言
    assert(forkside_current_players.size() == forkside_states.size());
    assert(forkside_current_players.size() == forkside_policy_targets.size());
    assert(forkside_current_players.size() == forkside_opp_policy_targets.size());
    assert(forkside_current_players.size() == forkside_value_targets.size());
    assert(forked_move_labels.size() ==  forked_all_moves.size());

    // 检查并创建目录
    std::string directory = "selfplay";
    if (!std::filesystem::exists(directory)) {
        if (std::filesystem::create_directories(directory)) {
            std::cout << "已创建目录: " << directory << std::endl;
        } else {
            std::cerr << "无法创建目录: " << directory << std::endl;
            return;
        }
    }

    // 保存数据。游戏结束后，不直接存盘，而是 append 到 all_ 容器里
    std::vector<int> game_num_vec(forkside_states.size(), thread_game_num); // 保持维度一致
    all_game_nums.insert(all_game_nums.end(), std::make_move_iterator(game_num_vec.begin()), std::make_move_iterator(game_num_vec.end()));
    std::vector<int> game_type_vec(forkside_states.size(), 2); // 保持维度一致
    all_game_types.insert(all_game_types.end(), std::make_move_iterator(game_type_vec.begin()), std::make_move_iterator(game_type_vec.end()));
    
    std::vector<int> size_vec(forkside_states.size(), forked_game.COLS); // 保持维度一致
    all_sizes.insert(all_sizes.end(), std::make_move_iterator(size_vec.begin()), std::make_move_iterator(size_vec.end()));
    all_states.insert(all_states.end(), std::make_move_iterator(forkside_states.begin()), std::make_move_iterator(forkside_states.end()));
    all_policy_targets.insert(all_policy_targets.end(), std::make_move_iterator(forkside_policy_targets.begin()), std::make_move_iterator(forkside_policy_targets.end()));
    all_opp_policy_targets.insert(all_opp_policy_targets.end(), std::make_move_iterator(forkside_opp_policy_targets.begin()), std::make_move_iterator(forkside_opp_policy_targets.end()));
    all_value_targets.insert(all_value_targets.end(), std::make_move_iterator(forkside_value_targets.begin()), std::make_move_iterator(forkside_value_targets.end()));

    // Print all moves with labels
    int game_result = forked_game.get_reward(1);
    {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cout << "线程 " << thread_id
                << ", gen 种子: " << worker_seed << ", forkside完成, 大小：" << forked_game.COLS << ", 所有动作: [";
        for (size_t i = 0; i < forked_all_moves.size(); ++i) {
            std::cout << forked_all_moves[i];
            if (!forked_move_labels[i].empty()) {
                std::cout << forked_move_labels[i];
            }
            if (i < forked_all_moves.size() - 1) std::cout << ", ";
        }
        std::cout << "], winner: " << game_result << std::endl << std::endl;
    }
}

void earlyfork(Five& game, Evaluate* eval, std::mt19937& gen, int use_mode, int thread_id, unsigned int worker_seed,
               std::vector<int>& all_moves_before, std::vector<std::string>& move_labels_before,
               int thread_game_num,
               std::vector<int>& all_game_nums,
               std::vector<int>& all_game_types,
               std::vector<int>& all_sizes,
               std::vector<std::vector<float>>& all_states,
               std::vector<std::vector<float>>& all_policy_targets,
               std::vector<std::vector<float>>& all_opp_policy_targets,
               std::vector<std::vector<float>>& all_value_targets) {

    Five local_game = game.clone();
    std::vector<int> all_moves = all_moves_before;
    std::vector<std::string> move_labels = move_labels_before;

    std::vector<std::vector<float>> states;         // 每个 full search 的状态
    std::vector<std::vector<float>> policy_targets; // 每个 full search 的策略目标
    std::vector<std::vector<float>> opp_policy_targets;  // 每个 full search 的预测对手下一步策略
    std::vector<std::vector<float>> value_targets;       // 每个 full search 的价值目标
    std::vector<int> current_players;               // 每个 full search 的当前玩家

    bool record_next_opp_policy = false; // 是否记录对手的策略

    // 获取合法动作
    auto moves = local_game.available_moves();
    if (moves.empty()) {
        throw std::runtime_error("No available moves to select from.");
    }

    // 随机决定评估的动作数量（3 到 10）
    std::uniform_int_distribution<> num_actions_dist(3, 10);
    int num_actions = num_actions_dist(gen);
    num_actions = std::min(num_actions, static_cast<int>(moves.size())); // 确保不超过合法动作总数

    // 随机打乱动作列表并选择前 num_actions 个
    std::shuffle(moves.begin(), moves.end(), gen);
    std::vector<int> selected_moves(moves.begin(), moves.begin() + num_actions);

    // 评估每个选中的动作
    std::vector<float> values; // 存储每个动作的价值（对手的胜率）
    for (int move : selected_moves) {
        local_game.make_move(move); // 执行动作，current_player 切换为对手
        if (local_game.is_terminal()) {
            // 如果游戏结束，直接计算价值
            int reward = local_game.get_reward(local_game.current_player);
            float value = (reward == 1) ? 1.0f : (reward == -1) ? -1.0f : 0.0f;
            values.push_back(value);
        } else {
            // 调用神经网络预测价值
            auto state = local_game.get_state();
            auto future = eval->submitRequest(state, local_game.ROWS);
            auto result = future.get();
            float win_prob = result[800] - result[801]; // result[800] 是当前玩家的胜率（此时为对手）
            values.push_back(win_prob);
        }
        local_game.unmake_move(move); // 恢复原始状态
    }

    // 选择价值最小的动作（对手胜率最小，对 current_player 最有利）
    auto min_value_iter = std::min_element(values.begin(), values.end());
    int best_move_index = std::distance(values.begin(), min_value_iter);
    int best_move = selected_moves[best_move_index];

    // 执行最佳动作
    local_game.make_move(best_move);
    all_moves.push_back(best_move);
    move_labels.push_back("(earlyfork)");

    // 如果游戏未结束，执行完整 MCTS 并记录数据
    while (!local_game.is_terminal()) {
        int move;

        // **Forkside Logic Starts Here**
        if (gen() % 10000 < 250) { // 2.5% probability (250 / 10000)
            forkside(local_game, eval, gen, use_mode, thread_id, worker_seed, all_moves, move_labels,
                thread_game_num,
                all_game_nums,
                all_game_types,
                all_sizes,
                all_states,
                all_policy_targets,
                all_opp_policy_targets,
                all_value_targets);
        }

        // Use MCTS for subsequent moves
        std::uniform_int_distribution<> dist(0, 3);
        int mcts_iterations = (dist(gen) == 0) ? FULL_SEARCH_ITERATION : FAST_SEARCH_ITERATION;

        // Set use_forced_playout based on iterations
        bool use_forced_playout = (mcts_iterations == FULL_SEARCH_ITERATION);
        bool use_dirichlet_noise = (mcts_iterations == FULL_SEARCH_ITERATION);
        // Create MCTS instance with switches
        MCTS mcts(&local_game, (use_mode == 0) ? eval : nullptr, gen, 1.1f, 
                use_mode, use_forced_playout, use_dirichlet_noise);

        if (mcts_iterations == FULL_SEARCH_ITERATION){
            // Perform MCTS search
            mcts.search(mcts_iterations);

            // Collect state and current player
            states.push_back(local_game.get_state());
            current_players.push_back(local_game.current_player);

            // Get pruned policy distribution
            std::vector<float> policy = mcts.get_pruned_policy(local_game);

            // Convert policy to neural network size (20x20) 用于训练
            std::vector<float> policy_target(local_game.nnSize, 0.0f);
            for (int row = 0; row < local_game.ROWS; ++row) {
                for (int col = 0; col < local_game.COLS; ++col) {
                    int local_pos = row * local_game.COLS + col;
                    int net_pos = row * local_game.nnLen + col;
                    policy_target[net_pos] = policy[local_pos];
                }
            }
            policy_targets.push_back(policy_target);

            if (record_next_opp_policy == true)
            {
                opp_policy_targets.push_back(policy_target);
            }

            // 计算温度 T
            double N = static_cast<double>(local_game.ROWS);  // 棋盘大小
            double m = static_cast<double>(all_moves.size()); // 已执行步数
            double T = 0.2 + 0.6 * std::pow(0.5, m / N);

            // 使用温度 T 调整分布
            std::vector<double> adjusted_policy(policy.size(), 0.0);
            double sum_adjusted = 0.0;
            for (size_t i = 0; i < policy.size(); ++i) {
                if (policy[i] > 0) {
                    adjusted_policy[i] = std::pow(static_cast<double>(policy[i]), 1.0 / T);
                    sum_adjusted += adjusted_policy[i];
                }
            }
            if (sum_adjusted > 0) {
                for (size_t i = 0; i < adjusted_policy.size(); ++i) {
                    adjusted_policy[i] /= sum_adjusted;
                }
            } else {
                throw std::runtime_error("Error, sum_adjusted is 0 after adjustment!");
            }

            // 从调整后的分布中采样 move
            std::discrete_distribution<int> move_dist(adjusted_policy.begin(), adjusted_policy.end());
            move = move_dist(gen);
            move_labels.push_back("(full)"); // Label full search moves

            record_next_opp_policy = true;
        }
        else{
            // Fast search: use existing search method
            move = mcts.search(mcts_iterations);
            move_labels.push_back(""); // No label for fast search

            if (record_next_opp_policy == true)
            {
                std::vector<float> policy = mcts.get_policy(local_game);

                // Convert policy to neural network size (20x20) 用于训练
                std::vector<float> policy_target(local_game.nnSize, 0.0f);
                for (int row = 0; row < local_game.ROWS; ++row) {
                    for (int col = 0; col < local_game.COLS; ++col) {
                        int local_pos = row * local_game.COLS + col;
                        int net_pos = row * local_game.nnLen + col;
                        policy_target[net_pos] = policy[local_pos];
                    }
                }

                opp_policy_targets.push_back(policy_target);
            }

            record_next_opp_policy = false;
        }
        delete mcts.root;
        mcts.root = new MCTSNode(local_game.current_player);

        local_game.make_move(move);
        all_moves.push_back(move);
    }

    int game_result = local_game.get_reward(1);

    for (int cp : current_players) {
        std::vector<float> value_target(3, 0.0f);
        if (game_result == 1) {
            value_target[(cp == 1) ? 0 : 1] = 1.0f; // cp==1 -> win, cp==-1 -> lose
        } else if (game_result == -1) {
            value_target[(cp == 1) ? 1 : 0] = 1.0f; // cp==1 -> lose, cp==-1 -> win
        } else {
            value_target[2] = 1.0f; // draw
        }
        value_targets.push_back(value_target);
    }

    // 最后一步是full search，但是游戏结束了，导致opp_policy_targets少一个
    if (record_next_opp_policy == true)
    {
        // 如果没有记录对手的策略，填充空值
        // 反正按照交叉熵公式，这样损失是0
        std::vector<float> empty_policy(local_game.nnSize, 0.0f);
        opp_policy_targets.push_back(empty_policy);
    }

    // 在保存数据之前加入断言
    assert(current_players.size() == states.size());
    assert(current_players.size() == policy_targets.size());
    assert(current_players.size() == opp_policy_targets.size());
    assert(current_players.size() == value_targets.size());
    assert(move_labels.size() == all_moves.size());

    // 检查并创建目录
    std::string directory = "selfplay";
    if (!std::filesystem::exists(directory)) {
        if (std::filesystem::create_directories(directory)) {
            std::cout << "已创建目录: " << directory << std::endl;
        } else {
            std::cerr << "无法创建目录: " << directory << std::endl;
            return;
        }
    }

    // 保存数据。游戏结束后，不直接存盘，而是 append 到 all_ 容器里
    std::vector<int> game_num_vec(states.size(), thread_game_num); // 保持维度一致
    all_game_nums.insert(all_game_nums.end(), std::make_move_iterator(game_num_vec.begin()), std::make_move_iterator(game_num_vec.end()));
    std::vector<int> game_type_vec(states.size(), 1); // 保持维度一致
    all_game_types.insert(all_game_types.end(), std::make_move_iterator(game_type_vec.begin()), std::make_move_iterator(game_type_vec.end()));
            
    std::vector<int> size_vec(states.size(),  game.COLS); // 保持维度一致
    all_sizes.insert(all_sizes.end(), std::make_move_iterator(size_vec.begin()), std::make_move_iterator(size_vec.end()));
    all_states.insert(all_states.end(), std::make_move_iterator(states.begin()), std::make_move_iterator(states.end()));
    all_policy_targets.insert(all_policy_targets.end(), std::make_move_iterator(policy_targets.begin()), std::make_move_iterator(policy_targets.end()));
    all_opp_policy_targets.insert(all_opp_policy_targets.end(), std::make_move_iterator(opp_policy_targets.begin()), std::make_move_iterator(opp_policy_targets.end()));
    all_value_targets.insert(all_value_targets.end(), std::make_move_iterator(value_targets.begin()), std::make_move_iterator(value_targets.end()));

    // Print all moves with labels
    {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cout << "线程 " << thread_id << ", gen 种子: " << worker_seed << ", earlyfork局完成, 大小：" << game.COLS << ", 所有动作: [";
        for (size_t i = 0; i < all_moves.size(); ++i) {
            std::cout << all_moves[i];
            if (!move_labels[i].empty()) {
                std::cout << move_labels[i];
            }
            if (i < all_moves.size() - 1) std::cout << ", ";
        }
        std::cout << "], winner: " << game_result << std::endl << std::endl;
    }
}

void selfplay(Evaluate* eval, int num_epochs, int num_threads, int use_mode, const std::string& prefix) {
    // Define board sizes and their weights
    std::vector<int> sizes = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    std::vector<int> weights = {1, 1, 4, 2, 3, 4, 5, 6, 10, 8, 9, 10, 11, 35}; // 权重

    std::cout << "Weights: ";
    for (int w : weights) {
        std::cout << w << " ";
    }
    std::cout << std::endl;

    // Load openings for 15x15 and 20x20
    // std::vector<std::vector<int>> openings_15 = load_openings("openings_15");
    // std::vector<std::vector<int>> openings_20 = load_openings("openings_20");
    // std::cout << "加载 15x15 开局个数: " << openings_15.size() << std::endl;
    // std::cout << "加载 20x20 开局个数: " << openings_20.size() << std::endl;

    auto worker = [&](int thread_id, int num_epochs, int num_threads) {

        // 每个线程对应一个种子，一个文件

#ifdef USE_FIXED_SEED
        std::mt19937 gen(FIXED_SEED + thread_id); // 为每个线程分配不同的固定种子
        unsigned int worker_seed = FIXED_SEED + thread_id;
#else
        std::random_device rd;
        unsigned int worker_seed = rd() + thread_id; // rd() 已经足够随机了，但还是+thread_id以防万一
        // worker_seed = 1580487213;
        std::mt19937 gen(worker_seed);
#endif

        std::vector<int> all_game_nums; // 相对局数序号
        std::vector<int> all_game_types; // 0普通 1earlyfork 2forkside

        std::vector<int> all_sizes;
        std::vector<std::vector<float>> all_states;
        std::vector<std::vector<float>> all_policy_targets;
        std::vector<std::vector<float>> all_opp_policy_targets;
        std::vector<std::vector<float>> all_value_targets;

        int thread_game_num = -1;

        // 线程内部，循环进行游戏
        for (int epoch = thread_id; epoch < num_epochs; epoch += num_threads) { // 每个线程玩多局游戏

            thread_game_num++;
            // Randomly select board size based on weights
            std::discrete_distribution<> dist(weights.begin(), weights.end());
            int size_idx = dist(gen);
            int size = sizes[size_idx];

            Five local_game(size);

            // Vector to store opening moves for this game
            std::vector<int> opening_moves;
            std::vector<int> all_moves;                     // 整局的移动序列
            std::vector<std::string> move_labels;          // Labels for each move

            std::vector<std::vector<float>> states;         // 每个 full search 的状态
            std::vector<std::vector<float>> policy_targets; // 每个 full search 的策略目标
            std::vector<std::vector<float>> opp_policy_targets;  // 每个 full search 的预测对手下一步策略
            std::vector<std::vector<float>> value_targets;       // 每个 full search 的价值目标
            std::vector<int> current_players;               // 每个 full search 的当前玩家

            bool record_next_opp_policy = false; // 是否记录对手的策略

            // Apply opening moves based on board size
            // if (size == 15 && !openings_15.empty()) {
            //     int opening_idx = gen() % openings_15.size();
            //     std::vector<int> selected_opening = openings_15[opening_idx];
            //     int transform_idx = gen() % 8;
            //     opening_moves = apply_transform(selected_opening, transform_idx, size);
            // } else if (size == 20 && !openings_20.empty()) {
            //     int opening_idx = gen() % openings_20.size();
            //     std::vector<int> selected_opening = openings_20[opening_idx];
            //     int transform_idx = gen() % 8;
            //     opening_moves = apply_transform(selected_opening, transform_idx, size);
            // }

            // for (int move : opening_moves) {
            //     local_game.make_move(move);
            //     all_moves.push_back(move);
            //     move_labels.push_back("(opening)"); // Label opening moves
            // }

            // // 假设快下满的平局局面
            // int fill_rows = local_game.ROWS/5 * 5;
            // int fill_cols = local_game.COLS%2==0?local_game.COLS-2:local_game.COLS-1;
            // for (int row = 0; row < fill_rows; ++row) {
            //     for (int col = 0; col < fill_cols; ++col) {
            //         int stone;
            //         if(row%5==4)
            //         {
            //             if(col%2==0)
            //             {
            //                 col += 1;
            //                 local_game.make_move(row*local_game.COLS+col);
            //                 all_moves.push_back(row*local_game.COLS+col);
            //                 move_labels.push_back("(debug)"); // Label opening moves
            //                 col -= 1;
            //             }
            //             else
            //             {
            //                 col -= 1;
            //                 local_game.make_move(row*local_game.COLS+col);
            //                 all_moves.push_back(row*local_game.COLS+col);
            //                 move_labels.push_back("(debug)"); // Label opening moves
            //                 col += 1;
            //             }
                        
            //         }
            //         else{
            //             local_game.make_move(row*local_game.COLS+col);
            //             all_moves.push_back(row*local_game.COLS+col);
            //             move_labels.push_back("(debug)"); // Label opening moves
            //         }
            //     }
            // }
            
            int r;
            // Draw r from exponential distribution with mean 0.04 * b^2
            double mean_r = 0.04 * size * size;
            std::exponential_distribution<double> exp_dist(1.0 / mean_r);
            r = static_cast<int>(std::round(exp_dist(gen)));
            r = std::max(0, r); // Ensure r is non-negative
            
            // 生成 fork_r (early fork)
            double mean_fork_r = 0.025 * size * size;
            std::exponential_distribution<double> exp_dist_fork(1.0 / mean_fork_r);
            int fork_r = static_cast<int>(std::round(exp_dist_fork(gen)));
            fork_r = std::max(0, fork_r);

            std::uniform_real_distribution<float> fork_dist(0.0f, 1.0f);
            bool do_fork = (fork_dist(gen) < 0.05f); // 5% 概率执行分叉


            int move_count = 0;

            while (!local_game.is_terminal()) {
                int move;

                // 更随机的选择，增加极其不寻常的开局
                if (do_fork && move_count == fork_r && use_mode==0) {
                    // 在 fork_r 步调用 earlyfork
                    earlyfork(local_game, eval, gen, use_mode, thread_id, worker_seed,
                              all_moves, move_labels, 
                              thread_game_num,
                              all_game_nums,
                              all_game_types,
                              all_sizes,
                              all_states,
                              all_policy_targets,
                              all_opp_policy_targets,
                              all_value_targets);
                    do_fork = false;
                    continue;
                }
                // 开局动作选择，增加随机性
                else if (move_count < r) {
                    auto moves = local_game.available_moves();
                    if (moves.empty()) {
                        break; // 如果没有可用移动，退出循环
                    }

                    if (use_mode == 0) {
                        // Use raw policy from neural network for the first r moves
                        auto state = local_game.get_state();
                        auto future = eval->submitRequest(state, size);
                        auto result = future.get();
                        std::vector<float> raw_policy(result.begin(), result.begin() + local_game.nnSize);

                        // 如果 raw_policy 中有 nan，直接抛出异常 (训练时size-14做了分母，导致size=14时直接爆了)
                        for (float val : raw_policy) {
                            if (std::isnan(val)) {
                                throw std::runtime_error("raw_policy contains nan");
                            }
                        }

                        // 获取可用移动
                        auto moves = local_game.available_moves();
                        if (moves.empty()) {
                            // 如果没有可用移动，游戏应结束（正常情况下不会发生）
                            throw std::runtime_error("No available moves after raw policy selection.");
                        }

                        // 提取可用移动对应的策略概率
                        std::vector<float> policy_probs;
                        for (int move : moves) {
                            int row = move / local_game.COLS;
                            int col = move % local_game.COLS;
                            int net_pos = row * local_game.nnLen + col;
                            policy_probs.push_back(raw_policy[net_pos]);
                        }

                        // 归一化策略概率
                        float sum_probs = 0.0f;
                        for (float p : policy_probs) sum_probs += p;
                        if (sum_probs > 0.0f) {
                            for (float& p : policy_probs) p /= sum_probs;
                        } else {
                            // 如果概率和为零，回退到均匀分布
                            float uniform_prob = 1.0f / moves.size();
                            for (float& p : policy_probs) p = uniform_prob;
                        }

                        // 从归一化后的概率分布中采样动作
                        std::discrete_distribution<int> move_dist(policy_probs.begin(), policy_probs.end());
                        int idx = move_dist(gen);
                        move = moves[idx];
                    } else {
                        // 直接从可用动作中均匀随机选择
                        std::uniform_int_distribution<> move_dist(0, moves.size() - 1);
                        int idx = move_dist(gen);
                        move = moves[idx];
                    }
                    move_labels.push_back("(raw)"); // Label raw policy moves
                } else {
                    // **Forkside Logic Starts Here**
                    if (gen() % 10000 < 250) { // 2.5% probability (250 / 10000)
                        forkside(local_game, eval, gen, use_mode, thread_id, worker_seed, all_moves, move_labels,
                            thread_game_num,
                            all_game_nums,
                            all_game_types,
                            all_sizes,
                            all_states,
                            all_policy_targets,
                            all_opp_policy_targets,
                            all_value_targets);
                    }

                    // Use MCTS for subsequent moves
                    std::uniform_int_distribution<> dist(0, 3);
                    int mcts_iterations = (dist(gen) == 0) ? FULL_SEARCH_ITERATION : FAST_SEARCH_ITERATION;

                    // Set use_forced_playout based on iterations
                    bool use_forced_playout = (mcts_iterations == FULL_SEARCH_ITERATION);
                    bool use_dirichlet_noise = (mcts_iterations == FULL_SEARCH_ITERATION);
                    // Create MCTS instance with switches
                    MCTS mcts(&local_game, (use_mode == 0) ? eval : nullptr, gen, 1.1f, 
                            use_mode, use_forced_playout, use_dirichlet_noise);
    /*
                    if(all_moves.size()==20)
                    {
                        // 递归打印整棵树
                        printNodeInfo(mcts.root, "root");
                        // Get the state
                        std::vector<float> midstate = local_game.get_state();

                        // Print the state as integers, space-separated
                        for (size_t i = 0; i < midstate.size(); ++i) {
                            std::cout << static_cast<int>(midstate[i]);
                            if (i < midstate.size() - 1) {
                                std::cout << " ";
                            }
                        }
                        std::cout << std::endl;
                    }
    */
                    if (mcts_iterations == FULL_SEARCH_ITERATION){
                        // Perform MCTS search
                        mcts.search(mcts_iterations);

                        // Collect state and current player
                        states.push_back(local_game.get_state());
                        current_players.push_back(local_game.current_player);

                        // Get pruned policy distribution
                        std::vector<float> policy = mcts.get_pruned_policy(local_game);

                        // Convert policy to neural network size (20x20) 用于训练
                        std::vector<float> policy_target(local_game.nnSize, 0.0f);
                        for (int row = 0; row < local_game.ROWS; ++row) {
                            for (int col = 0; col < local_game.COLS; ++col) {
                                int local_pos = row * local_game.COLS + col;
                                int net_pos = row * local_game.nnLen + col;
                                policy_target[net_pos] = policy[local_pos];
                            }
                        }
                        policy_targets.push_back(policy_target);

                        if (record_next_opp_policy == true)
                        {
                            opp_policy_targets.push_back(policy_target);
                        }

                        // 计算温度 T
                        double N = static_cast<double>(local_game.ROWS);  // 棋盘大小
                        double m = static_cast<double>(all_moves.size()); // 已执行步数
                        double T = 0.2 + 0.6 * std::pow(0.5, m / N);

                        // 使用温度 T 调整分布
                        std::vector<double> adjusted_policy(policy.size(), 0.0);
                        double sum_adjusted = 0.0;
                        for (size_t i = 0; i < policy.size(); ++i) {
                            if (policy[i] > 0) {
                                adjusted_policy[i] = std::pow(static_cast<double>(policy[i]), 1.0 / T);
                                sum_adjusted += adjusted_policy[i];
                            }
                        }
                        if (sum_adjusted > 0) {
                            for (size_t i = 0; i < adjusted_policy.size(); ++i) {
                                adjusted_policy[i] /= sum_adjusted;
                            }
                        } else {
                            throw std::runtime_error("Error, sum_adjusted is 0 after adjustment!");
                        }

                        // 从调整后的分布中采样 move
                        std::discrete_distribution<int> move_dist(adjusted_policy.begin(), adjusted_policy.end());
                        move = move_dist(gen);
                        move_labels.push_back("(full)"); // Label full search moves

                        record_next_opp_policy = true;
                    }
                    else{
                        // Fast search: use existing search method
                        move = mcts.search(mcts_iterations);
                        move_labels.push_back(""); // No label for fast search

                        if (record_next_opp_policy == true)
                        {
                            std::vector<float> policy = mcts.get_policy(local_game);

                            // Convert policy to neural network size (20x20) 用于训练
                            std::vector<float> policy_target(local_game.nnSize, 0.0f);
                            for (int row = 0; row < local_game.ROWS; ++row) {
                                for (int col = 0; col < local_game.COLS; ++col) {
                                    int local_pos = row * local_game.COLS + col;
                                    int net_pos = row * local_game.nnLen + col;
                                    policy_target[net_pos] = policy[local_pos];
                                }
                            }

                            opp_policy_targets.push_back(policy_target);
                        }

                        record_next_opp_policy = false;
                    }
                    delete mcts.root;
                    mcts.root = new MCTSNode(local_game.current_player);
                }

                local_game.make_move(move);
                all_moves.push_back(move);
                move_count++;
            }

            int game_result = local_game.get_reward(1);

            for (int cp : current_players) {
                std::vector<float> value_target(3, 0.0f);
                if (game_result == 1) {
                    value_target[(cp == 1) ? 0 : 1] = 1.0f; // cp==1 -> win, cp==-1 -> lose
                } else if (game_result == -1) {
                    value_target[(cp == 1) ? 1 : 0] = 1.0f; // cp==1 -> lose, cp==-1 -> win
                } else {
                    value_target[2] = 1.0f; // draw
                }
                value_targets.push_back(value_target);
            }

            // 最后一步是full search，但是游戏结束了，导致opp_policy_targets少一个
            if (record_next_opp_policy == true)
            {
                // 如果没有记录对手的策略，填充空值
                // 反正按照交叉熵公式，这样损失是0
                std::vector<float> empty_policy(local_game.nnSize, 0.0f);
                opp_policy_targets.push_back(empty_policy);
            }

            // 在保存数据之前加入断言
            assert(current_players.size() == states.size());
            assert(current_players.size() == policy_targets.size());
            assert(current_players.size() == opp_policy_targets.size());
            assert(current_players.size() == value_targets.size());
            assert(move_labels.size() == all_moves.size());

            // 检查并创建目录
            std::string directory = "selfplay";
            if (!std::filesystem::exists(directory)) {
                if (std::filesystem::create_directories(directory)) {
                    std::cout << "已创建目录: " << directory << std::endl;
                } else {
                    std::cerr << "无法创建目录: " << directory << std::endl;
                    return;
                }
            }

            // 游戏结束后，不直接存盘，而是 append 到 all_ 容器里
            // 元数据
            std::vector<int> game_num_vec(states.size(), thread_game_num); // 保持维度一致
            all_game_nums.insert(all_game_nums.end(), std::make_move_iterator(game_num_vec.begin()), std::make_move_iterator(game_num_vec.end()));
            std::vector<int> game_type_vec(states.size(), 0); // 保持维度一致
            all_game_types.insert(all_game_types.end(), std::make_move_iterator(game_type_vec.begin()), std::make_move_iterator(game_type_vec.end()));
            // 训练数据
            std::vector<int> size_vec(states.size(), size); // 保持维度一致
            all_sizes.insert(all_sizes.end(), std::make_move_iterator(size_vec.begin()), std::make_move_iterator(size_vec.end()));
            all_states.insert(all_states.end(), std::make_move_iterator(states.begin()), std::make_move_iterator(states.end()));
            all_policy_targets.insert(all_policy_targets.end(), std::make_move_iterator(policy_targets.begin()), std::make_move_iterator(policy_targets.end()));
            all_opp_policy_targets.insert(all_opp_policy_targets.end(), std::make_move_iterator(opp_policy_targets.begin()), std::make_move_iterator(opp_policy_targets.end()));
            all_value_targets.insert(all_value_targets.end(), std::make_move_iterator(value_targets.begin()), std::make_move_iterator(value_targets.end()));

            // Print all moves with labels
            {
                std::lock_guard<std::mutex> lock(print_mutex);
                std::cout << "线程 " << thread_id << ", gen 种子: " << worker_seed << 
                    ", 本线程的第" << thread_game_num << "局完成，epoch号 " << epoch << " , 大小：" << size << ", 所有动作: [";
                for (size_t i = 0; i < all_moves.size(); ++i) {
                    std::cout << all_moves[i];
                    if (!move_labels[i].empty()) {
                        std::cout << move_labels[i];
                    }
                    if (i < all_moves.size() - 1) std::cout << ", ";
                }
                std::cout << "], winner: " << game_result << std::endl << std::endl;
            }
        }

        //线程所有游戏循环任务结束

        // 线程所有局数跑完后，一次性生成一个文件
        // 这会面临可能的内存压力，但npz的“bug”意味着我们必须这样做
        if (!all_states.empty()) {
            std::string fn = generate_filename(prefix, thread_id, worker_seed, all_states.size());
            save_to_npz(fn, all_game_nums, all_game_types, all_sizes, all_states, all_policy_targets, 
                all_opp_policy_targets, all_value_targets);
            std::cout << "线程 " << thread_id << " 保存数据到 " << fn
                << ", gen 种子: " << worker_seed << std::endl;
        }
    };

    std::vector<std::thread> workers;
    for (int i = 0; i < num_threads; ++i) {
        workers.emplace_back(worker, i, num_epochs, num_threads);
    }

    // 记录 selfplay 开始时间
    auto start_time = std::chrono::high_resolution_clock::now();

    for (auto& t : workers) {
        t.join();
    }

    // 记录 selfplay 结束时间并计算总时间
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_time = end_time - start_time;

    if (use_mode == 0 && eval != nullptr)
    {
        // 获取总样本数并计算平均时间
        int total_samples = eval->getSampleCounter();
        if (total_samples > 0) {
            double average_time = total_time.count() / total_samples;
            std::cout << "Total time: " << total_time.count() << " seconds, Total samples: " << total_samples 
                    << ", Average time per sample: " << average_time << " seconds" << std::endl;
        } else {
            std::cout << "No samples processed." << std::endl;
        }
    }
    
    std::cout << "Selfplay completed." << std::endl;
}

void gatekeeper_play(Evaluate* eval_model, Evaluate* eval_candidate, int num_threads, int num_games) {
    // Define board sizes and their weights
    std::vector<int> sizes = {15, 20};
    std::vector<int> weights = {1, 2};

    // Load openings for 15x15 and 20x20
    std::vector<std::vector<int>> openings_15 = load_openings("openings_15");
    std::vector<std::vector<int>> openings_20 = load_openings("openings_20");
    std::cout << "加载 15x15 开局个数: " << openings_15.size() << std::endl;
    std::cout << "加载 20x20 开局个数: " << openings_20.size() << std::endl;

    std::atomic<int> model_wins(0);
    std::atomic<int> candidate_wins(0);
    std::atomic<int> draws(0);

    // 定义原子变量，用于记录 candidate 玩过的局数
    std::atomic<int> candidate_games(0);

    auto worker = [&](int thread_id) {

        // 每个线程处理的游戏数
        int games_per_thread = num_games / num_threads;
        int start_game = thread_id * games_per_thread;
        int end_game = (thread_id == num_threads - 1) ? num_games : start_game + games_per_thread;

        for (int g = start_game; g < end_game; ++g) {

            // 每局游戏使用不同的种子
#ifdef USE_FIXED_SEED
            std::mt19937 gen(FIXED_SEED + thread_id); // 为每个线程分配不同的固定种子
            unsigned int worker_seed = FIXED_SEED + thread_id;
#else
            std::random_device rd;
            unsigned int worker_seed = rd() + thread_id; // 随机种子加上 thread_id 以区分
            std::mt19937 gen(worker_seed);
#endif
            // Randomly select board size based on weights
            std::discrete_distribution<> dist(weights.begin(), weights.end());
            int size_idx = dist(gen);
            int size = sizes[size_idx];

            Five local_game(size);

            // 为每局游戏选择一个随机开局
            std::vector<int> transformed_opening;
            // Apply opening moves based on board size
            if (size == 15 && !openings_15.empty()) {
                int opening_idx = gen() % openings_15.size();
                std::vector<int> selected_opening = openings_15[opening_idx];
                int transform_idx = gen() % 8;
                transformed_opening = apply_transform(selected_opening, transform_idx, size);
            } else if (size == 20 && !openings_20.empty()) {
                int opening_idx = gen() % openings_20.size();
                std::vector<int> selected_opening = openings_20[opening_idx];
                int transform_idx = gen() % 8;
                transformed_opening = apply_transform(selected_opening, transform_idx, size);
            }

            // 对每种开局，model 和 candidate 各先手一次
            for (int turn = 0; turn < 2; ++turn) {
                local_game = Five(20);
                std::vector<int> moves; // 记录本局非开局移动

                // 应用开局移动
                for (int move : transformed_opening) {
                    local_game.make_move(move);
                }

                // turn=0 时 model 先手，turn=1 时 candidate 先手
                //实际谁开始下棋，由opening决定，也即game.current_player
                int model_player = turn == 0 ? 1 : -1;
                int candidate_player = turn == 1 ? 1 : -1;

                MCTS mcts(&local_game, eval_model, gen, 1.1f, 0 /* mode=0 */, false, false);

                while (!local_game.is_terminal()) {
                    mcts.eval = (local_game.current_player == model_player) ? eval_model : eval_candidate;
                    mcts.search(150);

                    // 搜索完毕，根据结果进行温度采样（Katago论文如此）
                    // 获取原始策略分布
                    std::vector<float> policy = mcts.get_policy(local_game);

                    // 计算当前温度 T
                    int m = moves.size(); // 当前移动次数
                    double N = static_cast<double>(local_game.ROWS); // 棋盘大小，例如 20
                    double T = 0.2 + 0.3 * std::pow(0.5, static_cast<double>(m) / N);

                    // 使用温度调整策略
                    std::vector<double> adjusted_policy(policy.size(), 0.0);
                    double sum_adjusted = 0.0;
                    for (size_t i = 0; i < policy.size(); ++i) {
                        if (policy[i] > 0) {
                            adjusted_policy[i] = std::pow(static_cast<double>(policy[i]), 1.0 / T);
                            sum_adjusted += adjusted_policy[i];
                        }
                    }

                    // 归一化调整后的策略
                    if (sum_adjusted > 0) {
                        for (size_t i = 0; i < adjusted_policy.size(); ++i) {
                            adjusted_policy[i] /= sum_adjusted;
                        }
                    } else {
                        // 如果调整后总和为 0（异常情况），退回到均匀分布
                        auto moves_available = local_game.available_moves();
                        double uniform_prob = 1.0 / moves_available.size();
                        for (int move : moves_available) {
                            adjusted_policy[move] = uniform_prob;
                        }
                    }

                    // 从调整后的策略中采样动作
                    std::discrete_distribution<int> move_dist(adjusted_policy.begin(), adjusted_policy.end());
                    int move = move_dist(gen); // gen 是随机数生成器

                    // 执行动作并更新状态
                    local_game.make_move(move);
                    moves.push_back(move);
                    delete mcts.root;
                    mcts.root = new MCTSNode(local_game.current_player);
                }

                int game_result = local_game.get_reward(model_player);
                if (game_result == 1) {
                    model_wins++;
                } else if (game_result == -1) {
                    candidate_wins++;
                } else {
                    draws++;
                }

                // 游戏结束后，增加 candidate_games
                candidate_games++;

                // 打印每局游戏信息
                {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::cout << "线程 " << thread_id << " 完成第 " << (g + 1) << " 局的第 " << (turn + 1) << " 场对局，"
                              << "开局动作: [";
                    for (size_t i = 0; i < transformed_opening.size(); ++i) {
                        std::cout << transformed_opening[i] << (i < transformed_opening.size() - 1 ? ", " : "");
                    }
                    std::cout << "], 本局步数: " << moves.size() << ", 本局步子: [";
                    for (size_t i = 0; i < moves.size(); ++i) {
                        std::cout << moves[i] << (i < moves.size() - 1 ? ", " : "");
                    }
                    std::cout << "], model是否获胜: " << game_result
                              << ", 目前 candidate 胜利局数: " << candidate_wins.load() << ", candidate 已玩局数: " << candidate_games.load() << std::endl;
                }
            }
        }
    };

    std::vector<std::thread> workers;
    for (int i = 0; i < num_threads; ++i) {
        workers.emplace_back(worker, i);
    }
    for (auto& t : workers) {
        t.join();
    }

    std::cout << "Model wins: " << model_wins << ", Candidate wins: " << candidate_wins << ", Draws: " << draws << std::endl;

    // 生成带时间戳的文件名
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

    if (!std::filesystem::exists("model/defeated")) {
        std::filesystem::create_directory("model/defeated");
    }

    if (candidate_wins > model_wins) {
        // candidate 挑战成功
        // 将当前 model 文件移动到 defeated 目录
        std::string defeated_model_onnx = "model/defeated/model_" + std::to_string(timestamp) + ".onnx";
        std::filesystem::rename("model/model.onnx", defeated_model_onnx);
    
        if (std::filesystem::exists("model/model.pth")) {
            std::string defeated_model_pth = "model/defeated/model_" + std::to_string(timestamp) + ".pth";
            std::filesystem::rename("model/model.pth", defeated_model_pth);
        }
    
        if (std::filesystem::exists("model/model.pt")) {
            std::string defeated_model_pt = "model/defeated/model_" + std::to_string(timestamp) + ".pt";
            std::filesystem::rename("model/model.pt", defeated_model_pt);
        }
    
        // 将 candidate 文件重命名为新的 model 文件
        std::filesystem::rename("model/candidate.onnx", "model/model.onnx");
        if (std::filesystem::exists("model/candidate.pth")) {
            std::filesystem::rename("model/candidate.pth", "model/model.pth");
        }
        if (std::filesystem::exists("model/candidate.pt")) {
            std::filesystem::rename("model/candidate.pt", "model/model.pt");
        }
    
        std::cout << "候选模型获胜，已成为新的 model 文件" << std::endl;
    } else {
        // candidate 挑战失败
        // 将 candidate 文件移动到 defeated 目录
        std::string defeated_candidate_onnx = "model/defeated/candidate_" + std::to_string(timestamp) + ".onnx";
        std::filesystem::rename("model/candidate.onnx", defeated_candidate_onnx);
    
        if (std::filesystem::exists("model/candidate.pth")) {
            std::string defeated_candidate_pth = "model/defeated/candidate_" + std::to_string(timestamp) + ".pth";
            std::filesystem::rename("model/candidate.pth", defeated_candidate_pth);
        }
    
        if (std::filesystem::exists("model/candidate.pt")) {
            std::string defeated_candidate_pt = "model/defeated/candidate_" + std::to_string(timestamp) + ".pt";
            std::filesystem::rename("model/candidate.pt", defeated_candidate_pt);
        }
    
        std::cout << "当前 model 文件保持不变，candidate 文件已移至 defeated 目录" << std::endl;
    }
}

void getstate(int argc, char* argv[]) {
    // Check if enough arguments are provided
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " getstate <size> <action1> <action2> ... <actionN>" << std::endl;
        exit(1);
    }

    // Parse size
    int size;
    try {
        size = std::stoi(argv[2]);
    } catch (const std::exception& e) {
        std::cerr << "Error: invalid size '" << argv[2] << "'. Must be an integer." << std::endl;
        exit(1);
    }

    // Validate size
    if (size < 5 || size > 20) {
        std::cerr << "Error: size must be between 5 and 20." << std::endl;
        exit(1);
    }

    // Create Five game object
    Five game(size);

    // Apply each move
    for (int i = 3; i < argc; ++i) {
        int action;
        try {
            action = std::stoi(argv[i]);
        } catch (const std::exception& e) {
            std::cerr << "Error: invalid action '" << argv[i] << "' at step " << (i - 2) << ". Must be an integer." << std::endl;
            exit(1);
        }

        game.make_move(action);
    }

    // Get the state
    std::vector<float> state = game.get_state();

    // Print the state as integers, space-separated
    for (size_t i = 0; i < state.size(); ++i) {
        std::cout << static_cast<int>(state[i]);
        if (i < state.size() - 1) {
            std::cout << " ";
        }
    }
    std::cout << std::endl;
}

#ifndef RUN_TESTS

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <mode> <num_threads> <num_epochs/games> [use: 0, 1, 2 for selfplay]" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    int num_threads = std::stoi(argv[2]);

    if (mode == "selfplay") {
        if (argc < 4 || argc > 6) {
            std::cerr << "Usage for selfplay: " << argv[0] << " selfplay <threads> <epochs> [use_mode] [prefix]" << std::endl;
            return 1;
        }
        int num_epochs = std::stoi(argv[3]);
        int use_mode = (argc >= 5) ? std::stoi(argv[4]) : 0; // Default to 0 (realnet)
        // --- 获取前缀：如果有则用输入的，否则默认 "undefined" ---
        std::string prefix = (argc == 6) ? argv[5] : "undefined";
        std::cout<<argc<<" "<<argv[5]<<std::endl;

        if (use_mode < 0 || use_mode > 2) {
            std::cerr << "Error: use must be 0 (realnet), 1 (scorepolicy), or 2 (random policy)" << std::endl;
            return 1;
        }
        Evaluate* eval = nullptr;
        if (use_mode == 0) { // Only instantiate eval for realnet
            const std::string onnxPath = "model/model.onnx";
            const int maxBatchSize = 128;
            eval = new Evaluate(onnxPath, maxBatchSize);
        }
        selfplay(eval, num_epochs, num_threads, use_mode, prefix);
        if (eval) delete eval;
    } else if (mode == "gatekeeper") {
        if (argc != 4) {
            std::cerr << "Usage for gatekeeper: " << argv[0] << " gatekeeper <num_threads> <num_games>" << std::endl;
            return 1;
        }
        int num_games = std::stoi(argv[3]);
        const std::string modelPath = "model/model.onnx";
        const std::string candidatePath = "model/candidate.onnx";
        const int maxBatchSize = 128;
        Evaluate* eval_model = new Evaluate(modelPath, maxBatchSize);
        Evaluate* eval_candidate = new Evaluate(candidatePath, maxBatchSize);
        gatekeeper_play(eval_model, eval_candidate, num_threads, num_games);
        delete eval_model;
        delete eval_candidate;
    } else if (mode == "getstate") {
        getstate(argc, argv);
    } else if (mode == "play") {
        // 初始化 GLFW + ImGui
        GLFWwindow* window;
        if (!glfwInit()) return -1;
        window = glfwCreateWindow(800, 800, "MCTS Visualizer", NULL, NULL);
        glfwMakeContextCurrent(window);

        // ImGui 初始化
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        ImGui::StyleColorsDark();

        // 绑定 GLFW + OpenGL3
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 130");

        // 初始化游戏
        int board_size = 11; // 自行设置
        Five game(board_size);

        // 初始化 MCTS 和 Evaluate
        std::mt19937 gen(FIXED_SEED);
        Evaluate eval("model/model.onnx", 128);
        MCTS mcts(&game, &eval, gen);

        // 主循环
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // 搜索若干次
            mcts.search(50);

            // 绘制棋盘
            ImGui::Begin("Gomoku MCTS");
            float cell_size = 30.0f;
            for (int row = 0; row < board_size; ++row) {
                for (int col = 0; col < board_size; ++col) {
                    int pos = row * board_size + col;
                    ImGui::PushID(pos);

                    // 绘制格子
                    ImVec4 col_rect(0.9f, 0.9f, 0.9f, 1.0f);
                    if (game.board[pos] == 1) col_rect = ImVec4(0, 0, 0, 1);   // 黑
                    if (game.board[pos] == -1) col_rect = ImVec4(1, 0, 0, 1);  // 红
                    ImGui::ColorButton("", col_rect, ImGuiColorEditFlags_NoTooltip, ImVec2(cell_size, cell_size));

                    // 绘制 MCTS 节点数字（胜率/visits）
                    for (auto* child : mcts.root->children) {
                        if (child->action == pos) {
                            std::string text = std::to_string(child->visits) + "/" + std::to_string(child->value_probs[0]/(child->visits+1e-6));
                            ImGui::SameLine();
                            ImGui::Text("%s", text.c_str());
                        }
                    }

                    // 鼠标左键落子
                    if (ImGui::IsItemClicked(0) && game.board[pos] == 0) {
                        game.make_move(pos);
                        // 将 MCTS 根节点切换到对应子节点
                        for (auto* child : mcts.root->children) {
                            if (child->action == pos) {
                                MCTSNode* old_root = mcts.root;
                                mcts.root = child;
                                child->parent = nullptr;
                                delete old_root; // 删除旧根节点
                                break;
                            }
                        }
                    }

                    ImGui::PopID();

                    if (col < board_size - 1) ImGui::SameLine(); // 除了最后一列，格子在同一行
                }
            }
            ImGui::End();

            // 鼠标滚轮在棋盘状态树间移动
            float wheel = ImGui::GetIO().MouseWheel;
            if (wheel != 0.0f && !mcts.root->children.empty()) {
                int idx = std::clamp(0, int(wheel > 0 ? 0 : 0), int(mcts.root->children.size()-1));
                mcts.root = mcts.root->children[idx];
                mcts.root->parent = nullptr;
            }

            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window);
        }

        // 清理
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        glfwDestroyWindow(window);
        glfwTerminate();


        return 0;
    } else {
        std::cerr << "Invalid mode: " << mode << ". Must be 'selfplay' or 'gatekeeper'." << std::endl;
        return 1;
    }

    return 0;
}

#endif
