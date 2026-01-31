g++ -g -DDEBUG -I/home/dashi/Downloads/TensorRT-10.2.0.19/include -I/usr/local/cuda-12.5/targets/x86_64-linux/include trtangel.cpp -L/home/dashi/Downloads/TensorRT-10.2.0.19/lib -L/usr/local/cuda-12.5/targets/x86_64-linux/lib -lnvinfer -lnvonnxparser -lcudart \
    -I/home/dashi/Downloads/imgui-1.92.3 -I/home/dashi/Downloads/imgui-1.92.3/backends \
    /home/dashi/Downloads/imgui-1.92.3/imgui.cpp \
    /home/dashi/Downloads/imgui-1.92.3/imgui_draw.cpp \
    /home/dashi/Downloads/imgui-1.92.3/imgui_widgets.cpp \
    /home/dashi/Downloads/imgui-1.92.3/imgui_tables.cpp \
    /home/dashi/Downloads/imgui-1.92.3/backends/imgui_impl_glfw.cpp \
    /home/dashi/Downloads/imgui-1.92.3/backends/imgui_impl_opengl3.cpp \
    -lglfw -lGL -ldl -pthread -lcnpy -lz -o trtangel_debug

g++ -O2 -I/home/dashi/Downloads/TensorRT-10.2.0.19/include -I/usr/local/cuda-12.5/targets/x86_64-linux/include trtangel.cpp -L/home/dashi/Downloads/TensorRT-10.2.0.19/lib -L/usr/local/cuda-12.5/targets/x86_64-linux/lib -lnvinfer -lnvonnxparser -lcudart \
    -I/home/dashi/Downloads/imgui-1.92.3 -I/home/dashi/Downloads/imgui-1.92.3/backends \
    /home/dashi/Downloads/imgui-1.92.3/imgui.cpp \
    /home/dashi/Downloads/imgui-1.92.3/imgui_draw.cpp \
    /home/dashi/Downloads/imgui-1.92.3/imgui_widgets.cpp \
    /home/dashi/Downloads/imgui-1.92.3/imgui_tables.cpp \
    /home/dashi/Downloads/imgui-1.92.3/backends/imgui_impl_glfw.cpp \
    /home/dashi/Downloads/imgui-1.92.3/backends/imgui_impl_opengl3.cpp \
    -lglfw -lGL -ldl -pthread -lcnpy -lz -o trtangel_release

# g++ -g -DDEBUG -DRUN_TESTS -I/home/dashi/Downloads/TensorRT-10.2.0.19/include -I/usr/local/cuda-12.5/targets/x86_64-linux/include unittest.cpp -L/home/dashi/Downloads/TensorRT-10.2.0.19/lib -L/usr/local/cuda-12.5/targets/x86_64-linux/lib -lnvinfer -lnvonnxparser -lcudart -o unittest_debug

# g++ -g -DDEBUG -DUSE_FIXED_SEED -I/home/dashi/Downloads/TensorRT-10.2.0.19/include -I/usr/local/cuda-12.5/targets/x86_64-linux/include trtangel.cpp -L/home/dashi/Downloads/TensorRT-10.2.0.19/lib -L/usr/local/cuda-12.5/targets/x86_64-linux/lib -lnvinfer -lnvonnxparser -lcudart -o trtangel_fixed_seed
