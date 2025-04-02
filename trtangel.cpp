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

#define CHECK(status) do { if (status != 0) { std::cerr << "CUDA Error: " << status << std::endl; exit(1); } } while (0)

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
    static constexpr int ROWS = 20;
    static constexpr int COLS = 20;
    static constexpr int SIZE = ROWS * COLS;

    // 存储 8 种对称变换的静态成员
    static std::vector<std::vector<int>> transforms;

    std::vector<int> board;                  // 棋盘：0为空，1为玩家1，-1为玩家2
    int current_player;                      // 当前玩家：1或-1
    bool done;                               // 游戏是否结束
    int winner;                              // 胜利者：0表示无，1或-1表示玩家
    std::vector<std::vector<int>> pos_to_combinations;  // 每个位置所属的五连组合
    std::vector<std::vector<int>> combination_to_poses; // 每个五连组合包含的位置
    std::vector<std::vector<int>> win;       // [2][组合数]，玩家在各组合中的棋子数
    std::vector<int> move_history;           // 最近5次移动
    std::vector<std::set<int>> chongsi_set;  // [2]，每个玩家的“冲四”位置集合
    std::vector<std::vector<int>> score;  // [2][SIZE]，玩家1和-1的得分
    std::vector<std::vector<int>> chongsi_count; // [2][SIZE] - Added for "冲四" counting

    Five() : board(SIZE, 0), current_player(1), done(false), winner(0),
             pos_to_combinations(SIZE), win(2), chongsi_set(2), score(2, std::vector<int>(SIZE, 0)), 
             chongsi_count(2, std::vector<int>(SIZE, 0)) {
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

    bool make_move(int position) {
        if (position < 0 || position >= SIZE || board[position] != 0) {
            current_player = -current_player;
            return false;
        }
        board[position] = current_player;
        move_history.push_back(position);
        if (move_history.size() > 5) move_history.erase(move_history.begin());
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
                }
            }
            // Update defending player's score
            if (old_win_making == 0 && new_win_making == 1) { // New block
                int stones = win[defending][comb_idx];
                int old_score = (stones >= 0 && stones <= 5) ? SCORE_BOOK[stones] : 0;
                int new_score = 0;
                change_score = new_score - old_score;
                for (int pos : combination_to_poses[comb_idx]) {
                    score[defending][pos] += change_score;
                    if (stones == 3 && board[pos] == 0) {
                        chongsi_count[defending][pos]--;
                        if (chongsi_count[defending][pos] == 0) {
                            chongsi_set[defending].erase(pos);
                        }
                    }
                }
            }
        }
        current_player = -current_player;
        return true;
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
                for (int pos : combination_to_poses[comb_idx]) {
                    score[profiting][pos] += change_score;
                    if (stones == 3 && board[pos] == 0) {
                        chongsi_count[profiting][pos]++;
                        chongsi_set[profiting].insert(pos);
                    }
                }
            }
            // Update unmaking player's score
            if (win[profiting][comb_idx] == 0) { // No block from profiting player
                int old_score = (old_win_unmaking >= 0 && old_win_unmaking <= 5) ? SCORE_BOOK[old_win_unmaking] : 0;
                int new_score = (new_win_unmaking >= 0 && new_win_unmaking <= 5) ? SCORE_BOOK[new_win_unmaking] : 0;
                int change_score = new_score - old_score;
                for (int pos : combination_to_poses[comb_idx]) {
                    score[unmaking][pos] += change_score;
                    if (old_win_unmaking == 3 && board[pos] == 0) {
                        chongsi_count[unmaking][pos]--;
                        if (chongsi_count[unmaking][pos] == 0) {
                            chongsi_set[unmaking].erase(pos);
                        }
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
        std::vector<float> state(9 * SIZE, 0.0f);
        int own_idx = (current_player == 1) ? 0 : 1;
        int opp_idx = 1 - own_idx;

        for (int i = 0; i < SIZE; ++i) {
            if (board[i] == current_player) state[i] = 1.0f;
            if (board[i] == -current_player) state[SIZE + i] = 1.0f;
        }

        int history_size = std::min(5, static_cast<int>(move_history.size()));
        for (int k = 0; k < history_size; ++k) {
            int pos = move_history[move_history.size() - 1 - k];
            state[(2 + k) * SIZE + pos] = 1.0f;
        }

        for (int pos : chongsi_set[own_idx]) {
            state[7 * SIZE + pos] = 1.0f;
        }
        for (int pos : chongsi_set[opp_idx]) {
            state[8 * SIZE + pos] = 1.0f;
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
};

// 在 cpp 文件中初始化静态成员
std::vector<std::vector<int>> Five::transforms(8, std::vector<int>(SIZE));

// 初始化 8 种变换
void init_transforms() {
    for (int i = 0; i < Five::SIZE; ++i) {
        // 不旋转 ('0')
        Five::transforms[0][i] = i;

        // 顺时针90° ('90')
        Five::transforms[1][i] = (i % 20) * 20 + (19 - (i / 20));

        // 180° ('180')
        Five::transforms[2][i] = 399 - i;

        // 顺时针270° ('270')
        Five::transforms[3][i] = (19 - (i % 20)) * 20 + (i / 20);

        // 水平翻转 ('horizontal')
        Five::transforms[4][i] = (19 - (i / 20)) * 20 + (i % 20);

        // 垂直翻转 ('vertical')
        Five::transforms[5][i] = (i / 20) * 20 + (19 - (i % 20));

        // 主对角线翻转 ('diagonal')
        Five::transforms[6][i] = (i % 20) * 20 + (i / 20);

        // 副对角线翻转 ('anti_diagonal')
        Five::transforms[7][i] = (19 - (i % 20)) * 20 + (19 - (i / 20));
    }
}

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
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 9, 20, 20});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{maxBatchSize, 9, 20, 20});
    profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{maxBatchSize, 9, 20, 20});
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
    std::vector<float> matrix;
    std::promise<std::vector<float>> promise;
};

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

    std::future<std::vector<float>> submitRequest(const std::vector<float>& matrix) {
        Request req;
        req.matrix = matrix;
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
        const int inputSize = 9 * 20 * 20;
        const int policySize = 400;
        const int valueSize = 1;
        const int outputSize = policySize + valueSize;

        std::vector<float> inputData(batchSize * inputSize);
        for (int i = 0; i < batchSize; ++i) {
            std::copy(batch[i].matrix.begin(), batch[i].matrix.end(), inputData.begin() + i * inputSize);
        }

        float *inputDevice, *policyDevice, *valueDevice;
        CHECK(cudaMalloc(&inputDevice, batchSize * inputSize * sizeof(float)));
        CHECK(cudaMalloc(&policyDevice, batchSize * policySize * sizeof(float)));
        CHECK(cudaMalloc(&valueDevice, batchSize * valueSize * sizeof(float)));

        CHECK(cudaMemcpy(inputDevice, inputData.data(), batchSize * inputSize * sizeof(float), cudaMemcpyHostToDevice));

        void* bindings[] = {inputDevice, policyDevice, valueDevice};
        context->setInputShape("input", nvinfer1::Dims4{batchSize, 9, 20, 20});
        context->executeV2(bindings);

        std::vector<float> policyOutput(batchSize * policySize);
        std::vector<float> valueOutput(batchSize * valueSize);
        CHECK(cudaMemcpy(policyOutput.data(), policyDevice, batchSize * policySize * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(valueOutput.data(), valueDevice, batchSize * valueSize * sizeof(float), cudaMemcpyDeviceToHost));

        CHECK(cudaFree(inputDevice));
        CHECK(cudaFree(policyDevice));
        CHECK(cudaFree(valueDevice));

        std::vector<std::vector<float>> results(batchSize);
        for (int i = 0; i < batchSize; ++i) {
            results[i].resize(outputSize);
            std::copy(policyOutput.begin() + i * policySize, policyOutput.begin() + (i + 1) * policySize, results[i].begin());
            results[i][policySize] = valueOutput[i];

            float sum = 0.0f;
            for (int j = 0; j < policySize; ++j) sum += results[i][j];
            if (sum > 0.0f) {
                for (int j = 0; j < policySize; ++j) results[i][j] /= sum;
            }
        }

        // 更新统计信息
        sampleCounter += batchSize;          // 累加样本数量

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

    // 如果所有成员变量都相等，返回 true
    return true;
}


class MCTSNode {
public:
    int action_player;
    int action;
    MCTSNode* parent;
    std::vector<MCTSNode*> children;
    int visits;
    float value;
    float prior;

    MCTSNode(int player, int action = -1, MCTSNode* parent = nullptr, float prior = 0.0f)
        : action_player(player), action(action), parent(parent), visits(0), value(0.0f), prior(prior) {}
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
    bool use_score_policy;  // 是否使用基于得分的策略
    bool use_forced_playout;    // Switch for forced playouts
    bool use_dirichlet_noise;   // Switch for Dirichlet noise
    float dirichlet_alpha;      // Dirichlet noise parameter
    float dirichlet_epsilon;    // Dirichlet noise mixing proportion
    std::mt19937 gen;           // Random number generator for Dirichlet noise

    MCTS(Five* game, Evaluate* eval, float exploration_weight = 1.0f, bool use_score_policy = false, bool use_forced_playout = false, 
         bool use_dirichlet_noise = false, float dirichlet_alpha = 0.03f, 
         float dirichlet_epsilon = 0.25f)
        : game(game), eval(eval), root(new MCTSNode(game->current_player)), c_PUCT(exploration_weight), use_score_policy(use_score_policy),
          use_forced_playout(use_forced_playout), use_dirichlet_noise(use_dirichlet_noise),
          dirichlet_alpha(dirichlet_alpha), dirichlet_epsilon(dirichlet_epsilon),
          gen(std::random_device{}()) {}

    ~MCTS() { delete root; }

    int search(int iterations) {
        Five one_search_game = game->clone();
        for (int i = 0; i < iterations; ++i) {
            std::vector<int> made_moves;
            MCTSNode* leaf = select(one_search_game, made_moves);
            MCTSNode* child = expand(one_search_game, leaf);
            float result = simulate(one_search_game, child, made_moves);
            backpropagate(child, result);
            // 撤销所有移动
            for (int j = made_moves.size() - 1; j >= 0; --j) {
                one_search_game.unmake_move(made_moves[j]);
            }
        }

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
            return max_visits_children[0]->action;
        }
        else {
            // 在 visits 相同的子节点中挑 PUCT 最大的
            float max_puct = -std::numeric_limits<float>::max();
            MCTSNode* best_child = nullptr;
            for (auto* child : max_visits_children) {
                float puct = calculate_puct(child);
                if (puct > max_puct) {
                    max_puct = puct;
                    best_child = child;
                }
            }
            return best_child->action;
        }
    }

    float puct_value_with_visits(const MCTSNode* node, int visits) {
        float q = (visits == 0) ? 0.0f : node->value / visits;
        float u = c_PUCT * node->prior * std::sqrt(static_cast<float>(node->parent->visits)) / (1 + visits);
        return q + u;
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
        std::uniform_int_distribution<> dist(0, max_children.size() - 1);
        return max_children[dist(gen)];
    }

    float puct_value(MCTSNode* node) {
        // 对于根节点的子节点，如果曾经visit过，且启用了 forced_playout，实时计算 forced_playout
        if (node->visits >= 1 && node->parent == root && use_forced_playout) {
            int forced_playouts = static_cast<int>(std::sqrt(2.0f * node->prior * static_cast<float>(root->visits)));
            if (node->visits < forced_playouts) {
                return std::numeric_limits<float>::max(); // 设置为极大值，确保优先选择
            }
        }
        // 正常计算 PUCT
        float q = (node->visits == 0) ? 0.0f : node->value / node->visits;
        float u = c_PUCT * node->prior * std::sqrt(static_cast<float>(node->parent->visits)) / (1 + node->visits);
        return q + u;
    }

    MCTSNode* expand(Five& one_search_game, MCTSNode* node) {
        if (one_search_game.is_terminal()) return node;
    
        std::vector<float> policy;
        if (use_score_policy || eval == nullptr) {
            policy = one_search_game.get_score_policy();
        } else {
            auto state = one_search_game.get_state();
            auto future = eval->submitRequest(state);
            auto result = future.get();
            policy.assign(result.begin(), result.begin() + 400);
        }
    
        auto moves = one_search_game.available_moves();
        int total_visits = node->visits;

        // Apply Dirichlet noise only at the root node
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
        return node;
    }

    // result永远是根据总思想者的利益来说的
    float simulate(Five& one_search_game, MCTSNode* node, std::vector<int>& made_moves) {
        if (one_search_game.is_terminal()) {
            return static_cast<float>(one_search_game.get_reward(game->current_player));
        }
        // one_search_game.make_move(node->action);
        // made_moves.push_back(node->action);
        // if (one_search_game.is_terminal()) {
        //     return static_cast<float>(one_search_game.get_reward(game->current_player));
        // }
        float reward;
        if (use_score_policy || eval == nullptr) {
            reward = 0.0f; // Python 中使用 score policy 时设为 0
        } else {
            auto state = one_search_game.get_state();
            auto future = eval->submitRequest(state);
            auto result = future.get();
            reward = result[400];
            // if (node->action == 250)
            // {
            //     Five correct_game;
            //     correct_game.make_move(375);
            //     correct_game.make_move(314);
            //     correct_game.make_move(334);
            //     correct_game.make_move(295);
            //     correct_game.make_move(333);
            //     correct_game.make_move(296);
            //     correct_game.make_move(312);
            //     correct_game.make_move(250);
            //    try {
            //        bool equal = areEqual(correct_game, one_search_game);
            //        std::cout << "两个 Five 对象相等。" << std::endl;
            //    }
            //    catch (const std::runtime_error& e) {
            //        std::cerr << "错误: " << e.what() << std::endl;
            //    }
            // }
        }
        // result永远是根据总思想者的利益来说的
        // 而神经网络结果的result是根据one_search_game.current_player来说的
        return one_search_game.current_player == game->current_player ? reward : -reward;
    }

    // result永远是根据总思想者的利益来说的
    void backpropagate(MCTSNode* node, float result) {
        while (node) {
            node->visits++;
            node->value += (node->action_player == game->current_player) ? result : -result;
            node = node->parent;
        }
    }

    float calculate_puct(MCTSNode* node) {
        if (node->visits == 0) {
            return 0.0f;
        }
        float q = node->value / node->visits;
        float u = c_PUCT * node->prior * std::sqrt(static_cast<float>(node->parent->visits)) / (1 + node->visits);
        return q + u;
    }

};

// 保存到 CSV 文件的函数
void save_to_csv(const std::string& filename,
                 const std::vector<int>& moves,
                 const std::vector<int>& all_mcts_iterations,
                 const std::vector<std::vector<float>>& policies,
                 int value_target) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return;
    }
    // 写入表头
    file << "moves,mcts_iterations,";
    for (size_t i = 0; i < policies.size(); ++i) {
        file << "policy" << (i + 1) << ",";
    }
    file << "value_target\n";

    // 写入数据
    // moves 用双引号包裹，逗号分隔
    std::ostringstream moves_ss;
    moves_ss << "\"";
    for (size_t j = 0; j < moves.size(); ++j) {
        moves_ss << moves[j];
        if (j < moves.size() - 1) moves_ss << ",";
    }
    moves_ss << "\"";
    file << moves_ss.str() << ",";

    // all_mcts_iterations 用双引号包裹，逗号分隔
    std::ostringstream iterations_ss;
    iterations_ss << "\"";
    for (size_t j = 0; j < all_mcts_iterations.size(); ++j) {
        iterations_ss << all_mcts_iterations[j];
        if (j < all_mcts_iterations.size() - 1) iterations_ss << ",";
    }
    iterations_ss << "\"";
    file << iterations_ss.str() << ",";

    // 每个 policy 用双引号包裹，逗号分隔
    for (size_t i = 0; i < policies.size(); ++i) {
        std::ostringstream policy_ss;
        policy_ss << "\"";
        for (size_t j = 0; j < policies[i].size(); ++j) {
            policy_ss << policies[i][j];
            if (j < policies[i].size() - 1) policy_ss << ",";
        }
        policy_ss << "\"";
        file << policy_ss.str() << ",";
    }
    file << value_target << "\n";
    file.close();
}

std::vector<std::vector<int>> load_openings(const std::string& directory) {
    std::vector<std::vector<int>> openings;
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
                        // Convert (x, y) to position index on a 20x20 board
                        int pos = (y + 10) * 20 + (x + 10);
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

// Apply a random transformation to an opening sequence
std::vector<int> apply_transform(const std::vector<int>& opening, int transform_idx) {
    std::vector<int> transformed_opening;
    for (int pos : opening) {
        transformed_opening.push_back(Five::transforms[transform_idx][pos]);
    }
    return transformed_opening;
}

void selfplay(Five& game, Evaluate* eval, int num_epochs, int num_threads, bool use_score_policy) {
    // 初始化变换
    init_transforms();

    // Load all openings at the start
    std::vector<std::vector<int>> openings = load_openings("openings_20");

    // for (auto op : openings) {
    //     for (auto mv : op)
    //         std::cout<< mv%20-10 << "," << mv/20-10  <<",";
    //     std::cout<<std::endl;
    // }

    // 打印加载的开局总数
    std::cout << "加载开局个数: " << openings.size() << std::endl;

    auto worker = [&](int thread_id, int num_epochs, int num_threads) {
        Five local_game = game.clone();
        std::random_device rd;
        std::mt19937 gen(rd() + thread_id);
        
        for (int epoch = thread_id; epoch < num_epochs; epoch += num_threads) { // 假设8个线程
            local_game = game.clone();

            // Vector to store opening moves for this game
            std::vector<int> opening_moves;
            std::vector<int> moves;                     // 整局的移动序列
            std::vector<int> all_mcts_iterations;       // 每一步的 MCTS 迭代次数
            std::vector<std::vector<float>> policies;   // 每一步的 policy_target

            // Randomly select and apply an opening if available
            if (!openings.empty()) {
                int opening_idx = gen() % openings.size();
                std::vector<int> selected_opening = openings[opening_idx];
                int transform_idx = gen() % 8;
                opening_moves = apply_transform(selected_opening, transform_idx);
                
                // Apply opening moves to the game
                for (int move : opening_moves) {
                    local_game.make_move(move);
                    moves.push_back(move);                          // 记录开局移动
                    all_mcts_iterations.push_back(100);             // 开局动作的 mcts_iterations 设为 100
                    policies.push_back(std::vector<float>(Five::SIZE, 0.0f)); // 开局动作的 policy 设为全 0
                }
            }
            
            while (!local_game.is_terminal()) {
                std::uniform_int_distribution<> dist(0, 3);
                int mcts_iterations = (dist(gen) == 0) ? 600 : 100;
                all_mcts_iterations.push_back(mcts_iterations);

                // Set use_forced_playout based on iterations
                bool use_forced_playout = (mcts_iterations == 600);
                bool use_dirichlet_noise = (mcts_iterations == 600);
                // Create MCTS instance with switches
                MCTS mcts(&local_game, use_score_policy ? nullptr : eval, 1.0f, 
                          use_score_policy, use_forced_playout, use_dirichlet_noise);

                int move = mcts.search(mcts_iterations);
                if (!local_game.make_move(move)) {
                    std::cerr << "Invalid move: " << move << std::endl;
                    break;
                }
                moves.push_back(move);

                // 修改后的策略目标计算
                // 步骤1：找到访问次数最多的子节点 c*
                const MCTSNode* c_star = nullptr;
                int max_visits = -1;
                for (const auto* child : mcts.root->children) {
                    if (child->visits > max_visits) {
                        max_visits = child->visits;
                        c_star = child;
                    }
                }

                // 步骤2：计算 c* 的 PUCT 值
                float puct_c_star = mcts.puct_value_with_visits(c_star, c_star->visits);

                // 步骤3：对于每个非 c* 的子节点 c，逐步减去访问次数
                std::vector<float> adjusted_visits(Five::SIZE, 0.0f);
                float total_adjusted_visits = 0.0f;

                for (auto* child : mcts.root->children) {
                    if (child == c_star) {
                        // 对于 c*，保留全部访问次数
                        adjusted_visits[child->action] = static_cast<float>(child->visits);
                        total_adjusted_visits += child->visits;
                    } else {
                        // 计算 forced_playout (nforced)
                        int nforced = static_cast<int>(std::sqrt(2.0f * child->prior * static_cast<float>(mcts.root->visits)));
                        int adj_visits = child->visits;
                        // 最多进行nforced次修剪
                        for (int i = 0; i < nforced; ++i) {
                            float puct_c = mcts.puct_value_with_visits(child, adj_visits - 1);
                            if (puct_c < puct_c_star) {
                                // 当 PUCT(c) < PUCT(c*) 时，停止减去
                                break;
                            }
                            adj_visits--;
                        }
                        // 如果调整后的访问次数大于 1，则保留，否则修剪
                        if (adj_visits > 1) {
                            adjusted_visits[child->action] = static_cast<float>(adj_visits);
                            total_adjusted_visits += adj_visits;
                        }
                    }
                }

                // 步骤4：计算 policy_target 并归一化
                std::vector<float> policy_target(Five::SIZE, 0.0f);
                if (total_adjusted_visits > 0) {
                    for (int i = 0; i < Five::SIZE; ++i) {
                        policy_target[i] = adjusted_visits[i] / total_adjusted_visits;
                    }
                }

                policies.push_back(policy_target);

                delete mcts.root;
                mcts.root = new MCTSNode(local_game.current_player);
            }
            int game_result = local_game.get_reward(1);

            // 在保存数据之前加入断言
            assert(moves.size() == all_mcts_iterations.size());
            assert(moves.size() == policies.size());

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

            // 修改文件名格式：年月日时分秒 + 随机数
            std::time_t now = std::time(nullptr);
            std::tm* now_tm = std::localtime(&now);
            char datetime_str[15]; // YYYYMMDDHHMMSS = 14 位 + 1 位结束符
            std::strftime(datetime_str, sizeof(datetime_str), "%Y%m%d%H%M%S", now_tm);
            int random_suffix = gen() % 900000 + 100000; // 6 位随机数
            std::string filename = "selfplay/memory_" + std::string(datetime_str) + "_" + 
                                  std::to_string(random_suffix) + ".csv";

            // 保存数据
            save_to_csv(filename, moves, all_mcts_iterations, policies, game_result);
            std::cout << "线程 " << thread_id << " 保存数据到 " << filename << std::endl;
    
            // 计算开局动作的数量
            size_t opening_moves_count = opening_moves.size();

            // 创建仅包含后续动作的向量
            std::vector<int> subsequent_moves(moves.begin() + opening_moves_count, moves.end());

            // 输出格式
            // Print opening moves and game moves
            std::cout << "第 " << (epoch + 1) << " 局完成，开局动作: [";
            for (size_t i = 0; i < opening_moves.size(); ++i) {
                std::cout << opening_moves[i];
                if (i < opening_moves.size() - 1) std::cout << ", ";
            }
            std::cout << "], 本局步数: " << subsequent_moves.size() << ", 本局步子: [";
            for (size_t i = 0; i < subsequent_moves.size(); ++i) {
                std::cout << subsequent_moves[i];
                if (all_mcts_iterations[opening_moves_count + i] == 600) {
                    std::cout << "(full)";
                }
                if (i < subsequent_moves.size() - 1) std::cout << ", ";
            }
            std::cout << "], winner: " << game_result << std::endl;
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

    if (!use_score_policy && eval != nullptr)
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
    // 初始化变换
    init_transforms();

    // 加载开局库
    std::vector<std::vector<int>> openings = load_openings("openings_20");
    if (openings.empty()) {
        std::cerr << "警告：未加载到任何开局，将不使用随机开局" << std::endl;
    } else {
        std::cout << "加载开局个数: " << openings.size() << std::endl;
    }

    std::atomic<int> model_wins(0);
    std::atomic<int> candidate_wins(0);
    std::atomic<int> draws(0);

    // 定义原子变量，用于记录 candidate 玩过的局数
    std::atomic<int> candidate_games(0);

    auto worker = [&](int thread_id) {
        Five local_game;
        std::random_device rd;
        std::mt19937 gen(rd() + thread_id);

        // 每个线程处理的游戏数
        int games_per_thread = num_games / num_threads;
        int start_game = thread_id * games_per_thread;
        int end_game = (thread_id == num_threads - 1) ? num_games : start_game + games_per_thread;

        for (int g = start_game; g < end_game; ++g) {
            // 为每局游戏选择一个随机开局
            std::vector<int> transformed_opening;
            if (!openings.empty()) {
                int opening_idx = gen() % openings.size();
                std::vector<int> selected_opening = openings[opening_idx];
                int transform_idx = gen() % 8;
                transformed_opening = apply_transform(selected_opening, transform_idx);
            }

            // 对每种开局，model 和 candidate 各先手一次
            for (int turn = 0; turn < 2; ++turn) {
                local_game = Five();
                std::vector<int> moves; // 记录本局非开局移动

                // 应用开局移动
                for (int move : transformed_opening) {
                    local_game.make_move(move);
                }

                int k = transformed_opening.size();
                // turn=0 时 model 先手，turn=1 时 candidate 先手
                //实际谁开始下棋，由opening决定，也即game.current_player
                int model_player = turn == 0 ? 1 : -1;
                int candidate_player = turn == 1 ? 1 : -1;

                MCTS mcts(&local_game, eval_model, 1.0f, false, false, false);

                while (!local_game.is_terminal()) {
                    mcts.eval = (local_game.current_player == model_player) ? eval_model : eval_candidate;
                    int move = mcts.search(150);
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
                    std::cout << "], 胜利者: " << game_result
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

#ifndef RUN_TESTS

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <mode> <num_threads> <num_epochs/games> [use_score_policy: 0 or 1 for selfplay]" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    int num_threads = std::stoi(argv[2]);

    if (mode == "selfplay") {
        if (argc < 4 || argc > 5) {
            std::cerr << "Usage for selfplay: " << argv[0] << " selfplay <num_threads> <num_epochs> [use_score_policy: 0 or 1]" << std::endl;
            return 1;
        }
        int num_epochs = std::stoi(argv[3]);
        bool use_score_policy = (argc == 5 && std::stoi(argv[4]) == 1);
        Evaluate* eval = nullptr;
        if (!use_score_policy) {
            const std::string onnxPath = "model/model.onnx";
            const int maxBatchSize = 128;
            eval = new Evaluate(onnxPath, maxBatchSize);
        }
        Five game;
        selfplay(game, eval, num_epochs, num_threads, use_score_policy);
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
    } else if (mode == "test") {
        Evaluate* eval = nullptr;
        eval = new Evaluate("model/model.onnx", 128);
        Five game;
        game.make_move(390);
        game.make_move(310);
        game.make_move(389);
        game.make_move(386);
        game.make_move(311);
        game.make_move(306);
        game.make_move(388);
        game.make_move(304);
        // game.make_move(250);
        auto state = game.get_state();
        auto future = eval->submitRequest(state);
        auto result = future.get();
        std::cout << result[304] << std::endl;
        std::cout << result[387] << std::endl;
        std::cout << result[391] << std::endl;
        std::cout << result[392] << std::endl;
        std::cout << " 价值： "<< result[400] << std::endl;
        // 创建索引数组
        std::vector<size_t> indices(result.size());
        for (size_t i = 0; i < indices.size(); ++i) {
           indices[i] = i;
        }

        // 按值排序索引（降序）
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
           return result[a] > result[b];
           });

        // 打印前 10 名的下标和值
        std::cout << "前十名的下标和值：" << std::endl;
        for (size_t i = 0; i < std::min<size_t>(10, result.size()); ++i) {
           std::cout << "第 " << (i + 1) << " 名: 下标 " << indices[i] << ", 值 " << result[indices[i]] << std::endl;
        }
        MCTS mcts(&game, eval, 1.0f, 0);
        int move = mcts.search(100);
        for (auto child : mcts.root->children)
        {
          //if (child->action == 41 || child->action == 248)
              std::cout << child->action << " " << child->visits << " " << child->value << " " << child->prior << std::endl;
        }
        std::cout << move << std::endl;
    }else {
        std::cerr << "Invalid mode: " << mode << ". Must be 'selfplay' or 'gatekeeper'." << std::endl;
        return 1;
    }

    return 0;
}

#endif