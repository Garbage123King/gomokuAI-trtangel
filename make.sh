g++ -g -DDEBUG -I/home/easyai/TensorRT-10.16.1.11/include -I/usr/local/cuda/targets/x86_64-linux/include trtangel.cpp -L/home/easyai/TensorRT-10.16.1.11/lib -L/usr/local/cuda/targets/x86_64-linux/lib -lnvinfer -lnvonnxparser -lcudart \
    -I/home/easyai/imgui-1.92.3 -I/home/easyai/imgui-1.92.3/backends \
    /home/easyai/imgui-1.92.3/imgui.cpp \
    /home/easyai/imgui-1.92.3/imgui_draw.cpp \
    /home/easyai/imgui-1.92.3/imgui_widgets.cpp \
    /home/easyai/imgui-1.92.3/imgui_tables.cpp \
    /home/easyai/imgui-1.92.3/backends/imgui_impl_glfw.cpp \
    /home/easyai/imgui-1.92.3/backends/imgui_impl_opengl3.cpp \
    -lglfw -lGL -ldl -pthread -lcnpy -lz -o trtangel_debug

g++ -O2 -I/home/easyai/TensorRT-10.16.1.11/include -I/usr/local/cuda/targets/x86_64-linux/include trtangel.cpp -L/home/easyai/TensorRT-10.16.1.11/lib -L/usr/local/cuda/targets/x86_64-linux/lib -lnvinfer -lnvonnxparser -lcudart \
    -I/home/easyai/imgui-1.92.3 -I/home/easyai/imgui-1.92.3/backends \
    /home/easyai/imgui-1.92.3/imgui.cpp \
    /home/easyai/imgui-1.92.3/imgui_draw.cpp \
    /home/easyai/imgui-1.92.3/imgui_widgets.cpp \
    /home/easyai/imgui-1.92.3/imgui_tables.cpp \
    /home/easyai/imgui-1.92.3/backends/imgui_impl_glfw.cpp \
    /home/easyai/imgui-1.92.3/backends/imgui_impl_opengl3.cpp \
    -lglfw -lGL -ldl -pthread -lz -lcnpy -o trtangel_release

# export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:/home/easyai/TensorRT-10.16.1.11/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
