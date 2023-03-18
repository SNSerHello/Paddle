rm -rf build
mkdir build
cd build
ulimit -n 4096
export PADDLE_VERSION=2.4.99
cmake .. \
	-DPY_VERSION=`python --version | cut -d ' ' -f 2 | cut -d '.' -f -2` \
	-DWITH_GPU=ON \
	-DWITH_TESTING=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX \
	-DCUDA_SDK_ROOT_DIR=$CONDA_PREFIX \
	-DCUDNN_ROOT=$CONDA_PREFIX \
	-DNCCL_ROOT=$CONDA_PREFIX \
	-DCUPTI_ROOT=$CONDA_PREFIX/pkgs/cuda-toolkit/extras/CUPTI \
	-DWITH_LITE=ON \
	-DCMAKE_CXX_FLAGS=-I/usr/include/mkl \
	-DWITH_ONEMKL=ON \
	-DWITH_PROFILER=OFF \
	-DWITH_ONNXRUNTIME=ON \
	-DON_INFER=ON && make -j6
