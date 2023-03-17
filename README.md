<p align="center">
<img align="center" src="doc/imgs/logo.png", width=1600>
<p>

--------------------------------------------------------------------------------

# Ubuntu20.04LTS

## 标准CUDA环境搭建

### CPU版本

```bash
$ mkdir build
$ cd build
$ cmake .. \
	-DPY_VERSION=`python --version | cut -d ' ' -f 2 | cut -d '.' -f -2` \
	-DWITH_GPU=OFF \
	-DWITH_TESTING=OFF \
	-DCMAKE_BUILD_TYPE=Release
$ make -j
```

### GPU版本

```bash
$ mkdir build
$ cd build
$ cmake .. \
	-DPY_VERSION=`python --version | cut -d ' ' -f 2 | cut -d '.' -f -2` \
	-DWITH_GPU=ON \
	-DWITH_TESTING=OFF \
	-DCMAKE_BUILD_TYPE=Release
$ make -j
```

## Anaconda3环境搭建

根据[py37-paddle-dev](https://github.com/SNSerHello/MyNotes/blob/main/paddle/py37-paddle-dev.yaml)搭建CUDA10.1+CUDNN7.6.5后，可以用它来编译PaddlePaddle。

```bash
$ conda activate py37-paddle-dev
(py37-paddle-dev) $ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
(py37-paddle-dev) $ nano $CONDA_PREFIX/etc/conda/activate.d/env_vars.h
文件内容如下：
CUDA_ROOT=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CUDA_ROOT/lib:$LD_LIBRARY_PATH
(py37-paddle-dev) $ mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
(py37-paddle-dev) $ nano $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.h
文件内容如下：
export LD_LIBRARY_PATH=`echo $LD_LIBRARY_PATH | cut -d : -f 2-`
(py37-paddle-dev) $ ln -s $CONDA_PREFIX/pkgs/cuda-toolkit/lib64/libcudadevrt.a  $CONDA_PREFIX/lib/libcudadevrt.a
(py37-paddle-dev) $ conda deactivate
```

### CPU版本

```bash
$ conda activate py37-paddle-dev
(py37-paddle-dev) $ mkdir build
(py37-paddle-dev) $ cd build
(py37-paddle-dev) $ cmake .. \
	-DPY_VERSION=`python --version | cut -d ' ' -f 2 | cut -d '.' -f -2` \
	-DWITH_GPU=OFF \
	-DWITH_TESTING=OFF \
	-DCMAKE_BUILD_TYPE=Release
(py37-paddle-dev) $ make -j
```

### GPU版本

```bash
$ conda activate py37-paddle-dev
(py37-paddle-dev) $ mkdir build
(py37-paddle-dev) $ cd build
(py37-paddle-dev) $ cmake .. \
	-DPY_VERSION=`python --version | cut -d ' ' -f 2 | cut -d '.' -f -2` \
	-DWITH_GPU=ON \
	-DWITH_TESTING=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX \
	-DCUDA_SDK_ROOT_DIR=$CONDA_PREFIX \
	-DCUDNN_ROOT=$CONDA_PREFIX \
	-DNCCL_ROOT=$CONDA_PREFIX \
	-DCUPTI_ROOT=$CONDA_PREFIX/pkgs/cuda-toolkit/extras/CUPTI
(py37-paddle-dev) $ make -j
```

对于NVIDIA 3070/3080/3090显卡的编译，CUDA10.x不支持`compute_86`的算力，所以可以降低到`compute_75`的算力来编译，如下所示

```bash
(py37-paddle-dev) $ cmake .. \
	-DPY_VERSION=`python --version | cut -d ' ' -f 2 | cut -d '.' -f -2` \
	-DWITH_GPU=ON \
	-DWITH_TESTING=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX \
	-DCUDA_SDK_ROOT_DIR=$CONDA_PREFIX \
	-DCUDNN_ROOT=$CONDA_PREFIX \
	-DNCCL_ROOT=$CONDA_PREFIX \
	-DCUPTI_ROOT=$CONDA_PREFIX/pkgs/cuda-toolkit/extras/CUPTI \
	-DCUDA_ARCH_NAME=Ampere \
	-DCMAKE_CUDA_ARCHITECTURES=75 \
	-DCMAKE_MATCH_1=75 \
	-DCMAKE_MATCH_2=75
```

**备注**：

- `CUDA10.x+CUDA7.6`需要NVIDIA driver release 396, 384.111+, 410, 418.xx or 440.30，在高版本的情况下，初始化时间很长，可用运行一下如下程序：

    ```python
    In [1]: import paddle
    In [2]: paddle.utils.run_check()
    Running verify PaddlePaddle program ...
    PaddlePaddle works well on 1 GPU.
    PaddlePaddle works well on 1 GPUs.
    PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.
    In [3]: paddle.fluid.install_check.run_check()
    Running Verify Fluid Program ...
    Your Paddle Fluid works well on SINGLE GPU or CPU.
    Your Paddle Fluid works well on MUTIPLE GPU or CPU.
    Your Paddle Fluid is installed successfully! Let's start deep Learning with Paddle Fluid now
    ```

### Python3.8+CUDA11.3+CUDNN8.2

`py38-paddle-dev`环境搭建，请参考：[py38-paddle-dev.yaml](https://github.com/SNSerHello/MyNotes/blob/main/paddle/py38-paddle-dev.yaml)。CUDA11.3支持`compute_86`，`CUDNN≥8.0.2`时候支持`CUDNN_FMA_MATH`，可参考：[cudnnMathType_t](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnMathType_t)

#### Training版本

```bash
$ conda env create --file py38-paddle-dev.yaml
$ conda activate py38-paddle-dev
(py38-paddle-dev) $ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
(py38-paddle-dev) $ nano $CONDA_PREFIX/etc/conda/activate.d/env_vars.h
文件内容如下：
CUDA_ROOT=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CUDA_ROOT/lib:$LD_LIBRARY_PATH
(py38-paddle-dev) $ mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
(py38-paddle-dev) $ nano $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.h
文件内容如下：
export LD_LIBRARY_PATH=`echo $LD_LIBRARY_PATH | cut -d : -f 2-`
(py38-paddle-dev) $ conda deactivate
# 使用新配置的环境
$ ulimit -n 4096
$ conda activate py38-paddle-dev
(py38-paddle-dev) $ mkdir build
(py38-paddle-dev) $ cd build
(py38-paddle-dev) $ cmake .. \
	-DPY_VERSION=`python --version | cut -d ' ' -f 2 | cut -d '.' -f -2` \
	-DWITH_GPU=ON \
	-DWITH_TESTING=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX \
	-DCUDA_SDK_ROOT_DIR=$CONDA_PREFIX \
	-DCUDNN_ROOT=$CONDA_PREFIX \
	-DNCCL_ROOT=$CONDA_PREFIX \
	-DCUPTI_ROOT=$CONDA_PREFIX/pkgs/cuda-toolkit/extras/CUPTI
(py38-paddle-dev) $ make -j
```

**注**： 默认Ubuntu20.04LTS版本打开的文件数为1024个，如果觉得每次设置`ulimit -n 4096`比较麻烦的话，那么请在`~/.bashrc`中加入此设置。这样每次打开bash的时候都自动设置为你想要的，否则可以考虑写一个PaddlePaddle的编译脚本，一次性搞定。

#### Inference版本

```bash
$ conda activate py38-paddle-dev
(py38-paddle-dev) $ mkdir build
(py38-paddle-dev) $ cd build
(py38-paddle-dev) $ cmake .. \
	-DPY_VERSION=`python --version | cut -d ' ' -f 2 | cut -d '.' -f -2` \
	-DWITH_GPU=ON \
	-DWITH_TESTING=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX \
	-DCUDA_SDK_ROOT_DIR=$CONDA_PREFIX \
	-DCUDNN_ROOT=$CONDA_PREFIX \
	-DNCCL_ROOT=$CONDA_PREFIX \
	-DCUPTI_ROOT=$CONDA_PREFIX/pkgs/cuda-toolkit/extras/CUPTI \
	-DON_INFER=ON
(py38-paddle-dev) $ ulimit -n 4096
(py38-paddle-dev) $ make -j
```

在完成编译后，在`build`目录中会多出两个文件夹，`paddle_inference_c_install_dir`和`paddle_inference_install_dir`

##### paddle_inference_c_install_dir

```bash
paddle_inference_c_install_dir
├── paddle
│   ├── include
│   │   ├── pd_common.h
│   │   ├── pd_config.h
│   │   ├── pd_inference_api.h
│   │   ├── pd_predictor.h
│   │   ├── pd_tensor.h
│   │   ├── pd_types.h
│   │   └── pd_utils.h
│   └── lib
│       ├── libpaddle_inference_c.a
│       └── libpaddle_inference_c.so
├── third_party
│   └── install
│       ├── cryptopp
│       │   ├── include
│       │   └── lib
│       ├── gflags
│       │   ├── include
│       │   └── lib
│       ├── glog
│       │   ├── include
│       │   └── lib
│       ├── mkldnn
│       │   ├── include
│       │   └── lib
│       ├── mklml
│       │   ├── include
│       │   └── lib
│       ├── onnxruntime
│       │   ├── include
│       │   └── lib
│       ├── paddle2onnx
│       │   ├── include
│       │   └── lib
│       ├── protobuf
│       │   ├── include
│       │   └── lib
│       ├── utf8proc
│       │   ├── include
│       │   └── lib
│       └── xxhash
│           ├── include
│           └── lib
└── version.txt
```

##### paddle_inference_install_dir

```bash
paddle_inference_install_dir
├── CMakeCache.txt
├── paddle
│   ├── include
│   │   ├── crypto
│   │   │   └── cipher.h
│   │   ├── experimental
│   │   │   ├── ext_all.h
│   │   │   ├── phi
│   │   │   └── utils
│   │   ├── internal
│   │   │   └── framework.pb.h
│   │   ├── paddle_analysis_config.h
│   │   ├── paddle_api.h
│   │   ├── paddle_infer_contrib.h
│   │   ├── paddle_infer_declare.h
│   │   ├── paddle_inference_api.h
│   │   ├── paddle_mkldnn_quantizer_config.h
│   │   ├── paddle_pass_builder.h
│   │   └── paddle_tensor.h
│   └── lib
│       ├── libpaddle_inference.a
│       └── libpaddle_inference.so
├── third_party
│   ├── externalError
│   │   └── data
│   │       └── data
│   ├── install
│   │   ├── cryptopp
│   │   │   ├── include
│   │   │   └── lib
│   │   ├── gflags
│   │   │   ├── include
│   │   │   └── lib
│   │   ├── glog
│   │   │   ├── include
│   │   │   └── lib
│   │   ├── mkldnn
│   │   │   ├── include
│   │   │   └── lib
│   │   ├── mklml
│   │   │   ├── include
│   │   │   └── lib
│   │   ├── onnxruntime
│   │   │   ├── include
│   │   │   └── lib
│   │   ├── paddle2onnx
│   │   │   ├── include
│   │   │   └── lib
│   │   ├── protobuf
│   │   │   ├── include
│   │   │   └── lib
│   │   ├── utf8proc
│   │   │   ├── include
│   │   │   └── lib
│   │   └── xxhash
│   │       ├── include
│   │       └── lib
│   └── threadpool
│       └── ThreadPool.h
└── version.txt
```



### 编译PaddlePaddle v2.3.1版本

#### Training编译

```bash
$ conda activate py38-paddle-dev
(py38-paddle-dev) $ git checkout v2.3.1
(py38-paddle-dev) $ mkdir build
(py38-paddle-dev) $ cd build
(py38-paddle-dev) $ export PADDLE_VERSION=2.3.1
(py38-paddle-dev) $ cmake .. \
	-DPY_VERSION=`python --version | cut -d ' ' -f 2 | cut -d '.' -f -2` \
	-DWITH_GPU=ON \
	-DWITH_TESTING=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX \
	-DCUDA_SDK_ROOT_DIR=$CONDA_PREFIX \
	-DCUDNN_ROOT=$CONDA_PREFIX \
	-DNCCL_ROOT=$CONDA_PREFIX \
	-DCUPTI_ROOT=$CONDA_PREFIX/pkgs/cuda-toolkit/extras/CUPTI
(py38-paddle-dev) $ ulimit -n 4096
(py38-paddle-dev) $ make -j
```

#### Training编译(NVIDIA3070/3080/3090)

```bash
$ conda activate py38-paddle-dev
(py38-paddle-dev) $ git checkout v2.3.1
(py38-paddle-dev) $ mkdir build
(py38-paddle-dev) $ cd build
(py38-paddle-dev) $ export PADDLE_VERSION=2.3.1
(py38-paddle-dev) $ cmake .. \
	-DPY_VERSION=`python --version | cut -d ' ' -f 2 | cut -d '.' -f -2` \
	-DWITH_GPU=ON \
	-DWITH_TESTING=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX \
	-DCUDA_SDK_ROOT_DIR=$CONDA_PREFIX \
	-DCUDNN_ROOT=$CONDA_PREFIX \
	-DNCCL_ROOT=$CONDA_PREFIX \
	-DCUPTI_ROOT=$CONDA_PREFIX/pkgs/cuda-toolkit/extras/CUPTI \
	-DCMAKE_CUDA_ARCHITECTURES=86
(py38-paddle-dev) $ ulimit -n 4096
(py38-paddle-dev) $ make -j
```

#### Inference编译

```bash
$ conda activate py38-paddle-dev
(py38-paddle-dev) $ git checkout v2.3.1
(py38-paddle-dev) $ mkdir build
(py38-paddle-dev) $ cd build
(py38-paddle-dev) $ export PADDLE_VERSION=2.3.1
(py38-paddle-dev) $ cmake .. \
	-DPY_VERSION=`python --version | cut -d ' ' -f 2 | cut -d '.' -f -2` \
	-DWITH_GPU=ON \
	-DWITH_TESTING=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX \
	-DCUDA_SDK_ROOT_DIR=$CONDA_PREFIX \
	-DCUDNN_ROOT=$CONDA_PREFIX \
	-DNCCL_ROOT=$CONDA_PREFIX \
	-DCUPTI_ROOT=$CONDA_PREFIX/pkgs/cuda-toolkit/extras/CUPTI \
	-DWITH_ONNXRUNTIME=ON \
	-DON_INFER=ON
(py38-paddle-dev) $ ulimit -n 4096
(py38-paddle-dev) $ make -j
```

#### Inference编译(NVIDIA3070/3080/3090)

```bash
$ conda activate py38-paddle-dev
(py38-paddle-dev) $ git checkout v2.3.1
(py38-paddle-dev) $ mkdir build
(py38-paddle-dev) $ cd build
(py38-paddle-dev) $ export PADDLE_VERSION=2.3.1
(py38-paddle-dev) $ cmake .. \
	-DPY_VERSION=`python --version | cut -d ' ' -f 2 | cut -d '.' -f -2` \
	-DWITH_GPU=ON \
	-DWITH_TESTING=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX \
	-DCUDA_SDK_ROOT_DIR=$CONDA_PREFIX \
	-DCUDNN_ROOT=$CONDA_PREFIX \
	-DNCCL_ROOT=$CONDA_PREFIX \
	-DCUPTI_ROOT=$CONDA_PREFIX/pkgs/cuda-toolkit/extras/CUPTI \
	-DWITH_ONNXRUNTIME=ON \
	-DON_INFER=ON \
	-DCMAKE_CUDA_ARCHITECTURES=86
(py38-paddle-dev) $ ulimit -n 4096
(py38-paddle-dev) $ make -j
```

#### Inference编译2(NVIDIA3070/3080/3090)

```bash
$ sudo apt install libmkl-full-dev libgoogle-perftools-dev google-perftools
$ conda activate py38-paddle-dev
(py38-paddle-dev) $ git checkout v2.3.1
(py38-paddle-dev) $ mkdir build
(py38-paddle-dev) $ cd build
(py38-paddle-dev) $ export PADDLE_VERSION=2.3.1
(py38-paddle-dev) $ cmake .. \
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
	-DWITH_PROFILER=ON \
	-DWITH_ONNXRUNTIME=ON \
	-DON_INFER=ON \
	-DCMAKE_CUDA_ARCHITECTURES=86
(py38-paddle-dev) $ ulimit -n 4096
(py38-paddle-dev) $ make -j
```



## Docker环境搭建

### GPU版本(Python3.7+CUDA10.2+CUDNN7.6)

```bash
$ sudo docker pull paddlepaddle/paddle:2.3.1-gpu-cuda10.2-cudnn7
$ sudo nvidia-docker run --rm -itv your_path/Paddle:/workspace -w /workspace paddlepaddle/paddle:2.3.1-gpu-cuda10.2-cudnn7 /bin/bash
$ mkdir build
$ cd build
$ export PADDLE_VERSION=2.3.1
$ cmake .. \
	-DPY_VERSION=`python --version | cut -d ' ' -f 2 | cut -d '.' -f -2` \
	-DWITH_GPU=ON \
	-DWITH_TESTING=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DWITH_LITE=ON \
	-DWITH_ONEMKL=OFF \
	-DWITH_ONNXRUNTIME=ON \
	-DON_INFER=ON
$ make -j
```

### GPU版本(Python3.7+CUDA11.2+CUDNN8.1)

```bash
$ sudo docker pull paddlepaddle/paddle:2.3.1-gpu-cuda11.2-cudnn8
$ sudo nvidia-docker run --rm -itv your_path/Paddle:/workspace -w /workspace paddlepaddle/paddle:2.3.1-gpu-cuda11.2-cudnn8 /bin/bash
$ mkdir build
$ cd build
$ export PADDLE_VERSION=2.3.1
$ cmake .. \
	-DPY_VERSION=`python --version | cut -d ' ' -f 2 | cut -d '.' -f -2` \
	-DWITH_GPU=ON \
	-DWITH_TESTING=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DWITH_LITE=ON \
	-DWITH_ONEMKL=OFF \
	-DWITH_ONNXRUNTIME=ON \
	-DON_INFER=ON
$ make -j
```

## FAQ

**问题1**：`unsupported GNU version! gcc versions later than 8 are not supported!`

**解决方法**：请参考[GCC环境搭建](https://github.com/SNSerHello/MyNotes/tree/main/gcc)，建议安装GCCv8.4.0，可以同时解决AVX intrinsic的编译问题。



**问题2**：`Too many open files`

**解决方法**：默认是1024个文件，可以设置成一个较大的数，比如说4096个，如下所示

```bash
$ ulimit -n
$ ulimit -n 4096
```



**问题3**：`nvcc fatal   : Unsupported gpu architecture 'compute_86'`

**解决方法**：

1、检查当前支持的`gpu architecture`，可以知道是`compute_75`

```bash
$ nvcc --help | egrep -i compute
        specified with this option must be a 'virtual' architecture (such as compute_50).
        --gpu-architecture=sm_50' is equivalent to 'nvcc --gpu-architecture=compute_50
        --gpu-code=sm_50,compute_50'.
        Allowed values for this option:  'compute_30','compute_32','compute_35',
        'compute_37','compute_50','compute_52','compute_53','compute_60','compute_61',
        'compute_62','compute_70','compute_72','compute_75','sm_30','sm_32','sm_35',
        (such as sm_50), and PTX code for the 'virtual' architecture (such as compute_50).
        For instance, '--gpu-architecture=compute_35' is not compatible with '--gpu-code=sm_30',
        because the earlier compilation stages will assume the availability of 'compute_35'
        Allowed values for this option:  'compute_30','compute_32','compute_35',
        'compute_37','compute_50','compute_52','compute_53','compute_60','compute_61',
        'compute_62','compute_70','compute_72','compute_75','sm_30','sm_32','sm_35',
```

2、`cmake/cuda.cmake`里面相关部分成`-gencode arch=compute_75,code=sm_75`。**注**：如果在cmake中指定了`-DCUDA_ARCH_NAME=Ampere`或者其他的，比如Turing等，那么这步修改可以跳过。

3、增加编译选项，支持`compute_75`

NVIDIA 3070/3080/3090是Ampere结构，CUDA10.2在NCCL仅仅支持`compute_75`，下方演示对应的配置，其他显卡可以参看NVIDIA官网的[Compare Geforce Graphics Cards](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/?section=compare-specs)来确定对应结构。

| sm_35, and sm_37        | Basic features+ Kepler support+ Unified memory programming+ Dynamic parallelism support |
| ----------------------- | ------------------------------------------------------------ |
| sm_50, sm_52 and sm_53  | + Maxwell support                                            |
| sm_60, sm_61, and sm_62 | + Pascal support                                             |
| sm_70 and sm_72         | + Volta support                                              |
| sm_75                   | + Turing support                                             |
| sm_80, sm_86 and sm_87  | + NVIDIA Ampere GPU architecture support                     |

**source**: https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list



|                     |                                                              |       RTX 30 Series        |        RTX 20 Series        |        GTX 16 Series         |        GTX 10 Series        |       GTX 9 Series        |
| :-----------------: | :----------------------------------------------------------- | :------------------------: | :-------------------------: | :--------------------------: | :-------------------------: | :-----------------------: |
| NVIDIA Architecture | Architecture Name                                            |           Ampere           |           Turing            |            Turing            |           Pascal            |          Maxwell          |
|                     | Streaming Multiprocessors                                    |          2x FP32           |           1x FP32           |           1x FP32            |           1x FP32           |          1x FP32          |
|                     | Ray Tracing Cores                                            |           Gen 2            |            Gen 1            |              -               |              -              |             -             |
|                     | Tensor Cores (AI)                                            |           Gen 3            |            Gen 2            |              -               |              -              |             -             |
|                     | Memory                                                       |     Up to 24 GB GDDR6X     |      Up to 11 GB GDDR6      |       Up to 6GB GDDR6        |     Up to 11 GB GDDR5X      |     Up to 6 GB GDDR5      |
|                     |                                                              |                            |                             |                              |                             |                           |
|      Platform       | [NVIDIA DLSS](https://www.nvidia.com/en-us/geforce/technologies/dlss/) |            Yes             |             Yes             |              -               |              -              |             -             |
|                     | [NVIDIA Reflex](https://www.nvidia.com/en-us/geforce/technologies/reflex/) |            Yes             |             Yes             |             Yes              |             Yes             |            Yes            |
|                     | [NVIDIA Broadcast](https://www.nvidia.com/en-us/geforce/broadcasting/) |            Yes             |             Yes             | GTX 1650 Super or 1660 Super |              -              |             -             |
|                     | [NVIDIA GeForce Experience](https://www.nvidia.com/en-us/geforce/geforce-experience/) |            Yes             |             Yes             |             Yes              |             Yes             |            Yes            |
|                     | [Game Ready Drivers](https://www.nvidia.com/en-us/geforce/game-ready-drivers/) |            Yes             |             Yes             |             Yes              |             Yes             |            Yes            |
|                     | [NVIDIA Studio Drivers](https://www.nvidia.com/en-us/studio/#studio-drivers) |            Yes             |             Yes             |             Yes              |             Yes             |             -             |
|                     | [NVIDIA ShadowPlay](https://www.nvidia.com/en-us/geforce/geforce-experience/shadowplay/) |            Yes             |             Yes             |             Yes              |             Yes             |            Yes            |
|                     | NVIDIA Highlights                                            |            Yes             |             Yes             |             Yes              |             Yes             |            Yes            |
|                     | [NVIDIA Ansel](https://www.nvidia.com/en-us/geforce/geforce-experience/ansel/) |            Yes             |             Yes             |             Yes              |             Yes             |            Yes            |
|                     | NVIDIA Freestyle                                             |            Yes             |             Yes             |             Yes              |             Yes             |            Yes            |
|                     | [VR Ready](https://www.nvidia.com/en-us/geforce/technologies/vr/) |            Yes             |             Yes             |   GTX 1650 Super or higher   |     GTX 1060 or higher      |     GTX 970 or higher     |
|                     | [NVIDIA Omniverse](https://www.nvidia.com/en-us/omniverse/creators/) |            Yes             |             Yes             |              -               |              -              |             -             |
|                     |                                                              |                            |                             |                              |                             |                           |
| Additional Features | PCIe                                                         |           Gen 4            |            Gen 3            |            Gen 3             |            Gen 3            |           Gen 3           |
|                     | [NVIDIA Encoder (NVENC)](https://developer.nvidia.com/video-encode-decode-gpu-support-matrix) |           Gen 7            |            Gen 7            |            Gen 6             |            Gen 6            |           Gen 5           |
|                     | [NVIDIA Decoder (NVDEC)](https://developer.nvidia.com/video-encode-decode-gpu-support-matrix) |           Gen 5            |            Gen 4            |            Gen 4             |            Gen 3            |           Gen 2           |
|                     | [CUDA Capability](https://developer.nvidia.com/cuda-gpus)    |            8.6             |             7.5             |             7.5              |             6.1             |            5.2            |
|                     | [DX12 Ultimate](https://www.nvidia.com/en-us/geforce/technologies/directx-12-ultimate/) |            Yes             |             Yes             |              -               |              -              |             -             |
|                     | Video Outputs                                                | HDMI 2.1, DisplayPort 1.4a | HDMI 2.0b, DisplayPort 1.4a | HDMI 2.0b, DisplayPort 1.4a  | HDMI 2.0b, DisplayPort 1.4a | HDMI 2.0, DisplayPort 1.2 |

**source**: https://www.nvidia.com/en-us/geforce/graphics-cards/compare/?section=compare-specs



| cuDNN Package                                                | Supported NVIDIA Hardware                                    | [CUDA Toolkit Version](https://developer.nvidia.com/cuda-toolkit-archive) | [CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus) | [Supports static linking?](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html#fntarg_1) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [cuDNN 8.4.1 for CUDA 11.x](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html#fntarg_2) | NVIDIA Ampere Architecture<br />NVIDIA Turing™<br />NVIDIA Volta™<br />NVIDIA Pascal™<br />NVIDIA Maxwell®<br />NVIDIA Kepler™ | [11.7](https://developer.nvidia.com/cuda-toolkit-archive)<br />[11.6](https://developer.nvidia.com/cuda-toolkit-archive)<br />[11.5](https://developer.nvidia.com/cuda-toolkit-archive)<br />[11.4](https://developer.nvidia.com/cuda-toolkit-archive)<br />[11.3](https://developer.nvidia.com/cuda-toolkit-archive) | SM 3.5 and later                                             | Yes                                                          |
|                                                              |                                                              | [11.2](https://developer.nvidia.com/cuda-toolkit-archive)<br />[11.1](https://developer.nvidia.com/cuda-toolkit-archive)<br />[11.0](https://developer.nvidia.com/cuda-toolkit-archive) |                                                              | No                                                           |
| [cuDNN 8.4.1 for CUDA 10.2](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html#cudnn-cuda-hardware-versions) | NVIDIA Turing<br />NVIDIA Volta<br />Xavier™<br />NVIDIA Pascal<br />NVIDIA Maxwell<br />NVIDIA Kepler | [10.2](https://developer.nvidia.com/cuda-toolkit-archive)    | SM 3.0 and later                                             | Yes                                                          |

**source**: https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html



```bash
$ cmake .. \
	-DPY_VERSION=`python --version | cut -d ' ' -f 2 | cut -d '.' -f -2` \
	-DWITH_GPU=ON \
	-DWITH_TESTING=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX \
	-DCUDA_SDK_ROOT_DIR=$CONDA_PREFIX \
	-DCUDNN_ROOT=$CONDA_PREFIX \
	-DNCCL_ROOT=$CONDA_PREFIX \
	-DCUPTI_ROOT=$CONDA_PREFIX/pkgs/cuda-toolkit/extras/CUPTI \
	-DCUDA_ARCH_NAME=Ampere \
	-DCMAKE_CUDA_ARCHITECTURES=75 \
	-DCMAKE_MATCH_1=75 \
	-DCMAKE_MATCH_2=75
```



**问题4**：`error: identifier "__builtin_ia32_sqrtsd_round" is undefined`

**解决方法1**：升级GCC版本到v8.4.0，请参考：[GCC v8.4.0安装](https://github.com/SNSerHello/MyNotes/tree/main/gcc)

**解决方法2**：修改`CMakeLists.txt`

```cmake
option(WITH_AVX "Compile PaddlePaddle with AVX intrinsics" ${AVX_FOUND}) # 第243行
修改成
option(WITH_AVX "Compile PaddlePaddle with AVX intrinsics" OFF)
```



**问题5**：`Policy CMP0104 is not set: CMAKE_CUDA_ARCHITECTURES now detected for NVCC,
  empty CUDA_ARCHITECTURES not allowed.  Run "cmake --help-policy CMP0104"
  for policy details.  Use the cmake_policy command to set the policy and
  suppress this warning.`

**解决方法**：在cmake的时候，增加`CMAKE_CUDA_ARCHITECTURES`的定义，假设我们使用`NVIDIA 3070/3080/3090`显卡，那么

```bash
$ cmake .. \
	-DPY_VERSION=`python --version | cut -d ' ' -f 2 | cut -d '.' -f -2` \
	-DWITH_GPU=ON \
	-DWITH_TESTING=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX \
	-DCUDA_SDK_ROOT_DIR=$CONDA_PREFIX \
	-DCUDNN_ROOT=$CONDA_PREFIX \
	-DNCCL_ROOT=$CONDA_PREFIX \
	-DCUPTI_ROOT=$CONDA_PREFIX/pkgs/cuda-toolkit/extras/CUPTI \
	-DCMAKE_CUDA_ARCHITECTURES=86
```

对于其他类的显卡设置的值，可以参考**问题3**中的`nvcc --help`部分内容。

**问题6**：如何自动设置编译的python版本？

```bash
$ cmake .. \
	-DPY_VERSION=`python --version | cut -d ' ' -f 2 | cut -d '.' -f -2` \
	-DWITH_GPU=ON \
	-DWITH_TESTING=OFF \
	-DCMAKE_BUILD_TYPE=Release \
	-DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX \
	-DCUDA_SDK_ROOT_DIR=$CONDA_PREFIX \
	-DCUDNN_ROOT=$CONDA_PREFIX \
	-DNCCL_ROOT=$CONDA_PREFIX \
	-DCUPTI_ROOT=$CONDA_PREFIX/pkgs/cuda-toolkit/extras/CUPTI
```



## 参考

- [Linux下从源码编译](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/compile/linux-compile.html)
- [anaconda3](https://github.com/SNSerHello/MyNotes/tree/main/anaconda3)
- [nvcc fatal : Unsupported gpu architecture ‘compute_75‘](https://blog.csdn.net/m0_46429066/article/details/116307051)
- [GCC安装](https://github.com/SNSerHello/MyNotes/tree/main/gcc)
- [GPU Feature List](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list)
- [cuDNN Support Matrix](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html#abstract)
- [py38-paddle-dev.yaml](https://github.com/SNSerHello/MyNotes/blob/main/paddle/py38-paddle-dev.yaml)
- [cudnnMathType_t](https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnMathType_t)
- [CMAKE_CUDA_ARCHITECTURES)](https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html#variable:CMAKE_CUDA_ARCHITECTURES)
- [CMake CMP0104](https://cmake.org/cmake/help/latest/policy/CMP0104.html)
- [GPU, CUDA Toolkit, and CUDA Driver Requirements](https://docs.nvidia.com/deeplearning/cudnn/support-matrix/index.html#cudnn-cuda-hardware-versions)



English | [简体中文](./README_cn.md) | [日本語](./README_ja.md)

[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html)
[![Documentation Status](https://img.shields.io/badge/中文文档-最新-brightgreen.svg)](https://paddlepaddle.org.cn/documentation/docs/zh/guides/index_cn.html)
[![Release](https://img.shields.io/github/release/PaddlePaddle/Paddle.svg)](https://github.com/PaddlePaddle/Paddle/releases)
[![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE)
[![Twitter](https://img.shields.io/badge/Twitter-1ca0f1.svg?logo=twitter&logoColor=white)](https://twitter.com/PaddlePaddle_)

Welcome to the PaddlePaddle GitHub.

PaddlePaddle, as the first independent R&D deep learning platform in China, has been officially open-sourced to professional communities since 2016. It is an industrial platform with advanced technologies and rich features that cover core deep learning frameworks, basic model libraries, end-to-end development kits, tools & components as well as service platforms.
PaddlePaddle is originated from industrial practices with dedication and commitments to industrialization. It has been widely adopted by a wide range of sectors including manufacturing, agriculture, enterprise service, and so on while serving more than 5.35 million developers, 200,000 companies and generating 670,000 models. With such advantages, PaddlePaddle has helped an increasing number of partners commercialize AI.


## Installation

### Latest PaddlePaddle Release: [v2.4](https://github.com/PaddlePaddle/Paddle/tree/release/2.4)

Our vision is to enable deep learning for everyone via PaddlePaddle.
Please refer to our [release announcement](https://github.com/PaddlePaddle/Paddle/releases) to track the latest features of PaddlePaddle.
### Install Latest Stable Release:
```
# CPU
pip install paddlepaddle
# GPU
pip install paddlepaddle-gpu

```
For more information about installation, please view [Quick Install](https://www.paddlepaddle.org.cn/install/quick)

Now our developers can acquire Tesla V100 online computing resources for free. If you create a program by AI Studio, you will obtain 8 hours to train models online per day. [Click here to start](https://aistudio.baidu.com/aistudio/index).

## FOUR LEADING TECHNOLOGIES

- **Agile Framework for Industrial Development of Deep Neural Networks**

    The PaddlePaddle deep learning framework facilitates the development while lowering the technical burden, through leveraging a programmable scheme to architect the neural networks. It supports both declarative programming and imperative programming with both development flexibility and high runtime performance preserved.  The neural architectures could be automatically designed by algorithms with better performance than the ones designed by human experts.


-  **Support Ultra-Large-Scale Training of Deep Neural Networks**

    PaddlePaddle has made breakthroughs in ultra-large-scale deep neural networks training. It launched the world's first large-scale open-source training platform that supports the training of deep networks with 100 billion features and trillions of parameters using data sources distributed over hundreds of nodes. PaddlePaddle overcomes the online deep learning challenges for ultra-large-scale deep learning models, and further achieved real-time model updating with more than 1 trillion parameters.
     [Click here to learn more](https://github.com/PaddlePaddle/Fleet)


- **High-Performance Inference Engines for Comprehensive Deployment Environments**

   PaddlePaddle is not only compatible with models trained in 3rd party open-source frameworks , but also offers complete inference products for various production scenarios. Our inference product line includes [Paddle Inference](https://paddle-inference.readthedocs.io/en/master/guides/introduction/index_intro.html): Native inference library for high-performance server and cloud inference; [Paddle Serving](https://github.com/PaddlePaddle/Serving): A service-oriented framework suitable for distributed and pipeline productions; [Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite): Ultra-Lightweight inference engine for mobile and IoT environments; [Paddle.js](https://www.paddlepaddle.org.cn/paddle/paddlejs): A frontend inference engine for browser and mini-apps. Furthermore, by great amounts of optimization with leading hardware in each scenario, Paddle inference engines outperform most of the other mainstream frameworks.


- **Industry-Oriented Models and Libraries with Open Source Repositories**

     PaddlePaddle includes and maintains more than 100 mainstream models that have been practiced and polished for a long time in the industry. Some of these models have won major prizes from key international competitions. In the meanwhile, PaddlePaddle has further more than 200 pre-training models (some of them with source codes) to facilitate the rapid development of industrial applications.
     [Click here to learn more](https://github.com/PaddlePaddle/models)


## Documentation

We provide [English](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html) and
[Chinese](https://www.paddlepaddle.org.cn/documentation/docs/zh/guide/index_cn.html) documentation.

- [Guides](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/index_en.html)

  You might want to start from how to implement deep learning basics with PaddlePaddle.

- [Practice](https://www.paddlepaddle.org.cn/documentation/docs/zh/tutorial/index_cn.html)

  So far you have already been familiar with Fluid. And the next step should be building a more efficient model or inventing your original Operator.

- [API Reference](https://www.paddlepaddle.org.cn/documentation/docs/en/api/index_en.html)

   Our new API enables much shorter programs.

- [How to Contribute](https://www.paddlepaddle.org.cn/documentation/docs/en/guides/08_contribution/index_en.html)

   We appreciate your contributions!

## Communication

- [Github Issues](https://github.com/PaddlePaddle/Paddle/issues): bug reports, feature requests, install issues, usage issues, etc.
- QQ discussion group: 441226485 (PaddlePaddle).
- [Forums](https://aistudio.baidu.com/paddle/forum): discuss implementations, research, etc.

## Courses

- [Server Deployments](https://aistudio.baidu.com/aistudio/course/introduce/19084): Courses introducing high performance server deployments via local and remote services.
- [Edge Deployments](https://aistudio.baidu.com/aistudio/course/introduce/22690): Courses introducing edge deployments from mobile, IoT to web and applets.

## Copyright and License
PaddlePaddle is provided under the [Apache-2.0 license](LICENSE).
