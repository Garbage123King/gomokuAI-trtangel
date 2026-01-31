# gomokuAI-trtangel
This project uses NVIDIA TensorRT to play multi-thread, high performace selfplay. 

After 3000 games selfplay, we use pytorch to train.

Everything is inspired from Katago paper and code.




1. Compile

```
./make.sh
```
2. Run

```
./start.sh
```

3. If you want to compile Gomocup version:

 ```
python tobin.py
cp model.bin gomocup/
cd gomocup
g++ -O2 gomocup.cpp pisqpipe.cpp -o pbrain-angel2026 and model.bin --static
// Then zip pbrain-angel2026 and model.bin
 ```

 

This is the result after 5 days training.

| -           | SLOWRENJU19 | angel-3e | angel-1e | angel-2000w | angel-5000w | angel-2e | angel-1000w | ANGEL   |
| ----------- | ----------- | -------- | -------- | ----------- | ----------- | -------- | ----------- | ------- |
| SLOWRENJU19 | -           | 2 : 8    | 2 : 8    | 1 : 9       | 2 : 8       | 3 : 7    | 1 : 9       | 2 : 8   |
| angel-3e    | 8 : 2       | -        | 6 : 4    | 3 : 7       | 4 : 6       | 4 : 6    | 1 : 9       | 0 : 10  |
| angel-1e    | 8 : 2       | 4 : 6    | -        | 7 : 3       | 5 : 5       | 1 : 9    | 3 : 7       | 2 : 8   |
| angel-2000w | 9 : 1       | 7 : 3    | 3 : 7    | -           | 8 : 2       | 4 : 6    | 2 : 8       | 3 : 7   |
| angel-5000w | 8 : 2       | 6 : 4    | 5 : 5    | 2 : 8       | -           | 5 : 5    | 2 : 8       | 3 : 7   |
| angel-2e    | 7 : 3       | 6 : 4    | 9 : 1    | 6 : 4       | 5 : 5       | -        | 3 : 7       | 0 : 10  |
| angel-1000w | 9 : 1       | 9 : 1    | 7 : 3    | 8 : 2       | 8 : 2       | 7 : 3    | -           | 5 : 5   |
| ANGEL       | 8 : 2       | 10 : 0   | 8 : 2    | 7 : 3       | 7 : 3       | 10 : 0   | 5 : 5       | -       |
| 总比分      | 57 : 13     | 44 : 26  | 40 : 30  | 34 : 36     | 39 : 31     | 34 : 36  | 17 : 53     | 15 : 55 |
| 胜负比      | 4.385       | 1.692    | 1.333    | 0.944       | 1.258       | 0.944    | 0.321       | 0.273   |
| 得分        | 21          | 15       | 13       | 12          | 11          | 7        | 1           | 1       |


2026/1/31 12:15 - 13:57