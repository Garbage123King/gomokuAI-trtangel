# gomokuAI-trtangel
use NVIDIA TensorRT to accelerate multi-thread selfplay. Basically I'm reproducing the paper of AlphaGO zero and Katago, but it is a gomoku project.


I have fixed a lot of bugs of it, and it has runned well on my machine for several days. It won't become a very strong AI because I only reproduce few part of Katago. But as I said, I have fixed a lot of bugs, since I got most of it from AI. Anyway, thanks to Grok3, I finished my dream to build a MCTS ML AI.

If you want to run it yourself:

1、Compile.
```
g++ -g -I<YOURDIR>/TensorRT-10.2.0.19/include -I/usr/local/cuda-12.5/targets/x86_64-linux/include trtangel.cpp -L<YOURDIR>/TensorRT-10.2.0.19/lib -L/usr/local/cuda-12.5/targets/x86_64-linux/lib -lnvinfer -lnvonnxparser -lcudart -o trtangel_debug

g++ -O2 -I<YOURDIR>/TensorRT-10.2.0.19/include -I/usr/local/cuda-12.5/targets/x86_64-linux/include trtangel.cpp -L<YOURDIR>/TensorRT-10.2.0.19/lib -L/usr/local/cuda-12.5/targets/x86_64-linux/lib -lnvinfer -lnvonnxparser -lcudart -o trtangel_release
```
2、Run selfplay.

<thread_num><total_games><use_scorenet>
```
./trtangel_release selfplay 128 500 1     #It starts a selfplay with "ScoreNet", which is a CPU simple algorithm
./trtangel_release selfplay 128 500 0     #It starts a selfplay with a real neural net, maybe you should run train.py to get model.onnx
```
3、Run train.

```
python3 train.py
```

train produces three files, `model.pth` is used to train net, `model.onnx` is used to selfplay, `model.pt` is used to convert to ncnn model so I can use it to participate the gomocup championship.

4、Run gatekeeper.

<thread_num><total_games>
```
./trtangel_release gatekeeper 100 100
```
