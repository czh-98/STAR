# 安装

- [安装](#安装)
- [依赖](#依赖)
- [准备环境](#准备环境)
- [安装 MMHuman3D](#安装-mmhuman3d)
- [从头开始安装的脚本](#从头开始安装的脚本)

## 依赖

- Linux
- ffmpeg
- Python 3.7+
- PyTorch 1.6.0, 1.7.0, 1.7.1, 1.8.0, 1.8.1, 1.9.0 或 1.9.1.
- CUDA 9.2+
- GCC 5+
- PyTorch3D 0.4+
- [MMCV](https://github.com/open-mmlab/mmcv) (请安装用于GPU的mmcv-full, 要求版本>=1.3.17,<1.6.0)

Optional:
- [MMPose](https://github.com/open-mmlab/mmpose) (只用于演示)
- [MMDetection](https://github.com/open-mmlab/mmdetection) (只用于演示)
- [MMTracking](https://github.com/open-mmlab/mmtracking) (只用于多人的演示。请安装 mmcls<0.23.1, mmcv-full>=1.3.17,<1.6.0)

## 准备环境

a. 创建conda虚拟环境并激活.

```shell
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab
```

b. 安装 ffmpeg

直接使用conda安装ffmpeg

```shell
conda install ffmpeg
```

c. 根据 [官方指导](https://pytorch.org/) ，安装PyTorch和torchvision .
```shell
conda install pytorch={torch_version} torchvision cudatoolkit={cu_version} -c pytorch
```

以安装 PyTorch 1.8.0 和 CUDA 10.2为例.
```shell
conda install pytorch=1.8.0 torchvision cudatoolkit=10.2 -c pytorch
```

**注意:** 请确保compilation CUDA version和runtime CUDA version相匹配。
另外， 对于RTX 30系列的显卡， 需要cudatoolkit>=11.0.

d. 在Linux或者Windows上安装PyTorch3D.

对于Linux：

```shell
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y

conda install pytorch3d -c pytorch3d
```

用户也可以参考 [PyTorch3D-install](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) 的更多细节。
然而我们近期的测试显示使用 conda 安装会遇到依赖冲突的问题。
因此，用户也可选择按照下列步骤从源码安装Pytorch3D。

```shell
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install .
cd ..
```

对于Windows：

请参考 [官方安装文档](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) 。 这里我们提供一个 [例子](https://github.com/open-mmlab/mmhuman3d/pull/199#issue-1274739041) 供用户参考。
**注意:** 这部分是针对想在Windows平台安装MMHuman3D的用户。

运行如下命令测试PyTorch3D是否安装成功:

```shell
echo "import pytorch3d;print(pytorch3d.__version__); \
    from pytorch3d.renderer import MeshRenderer;print(MeshRenderer);\
    from pytorch3d.structures import Meshes;print(Meshes);\
    from pytorch3d.renderer import cameras;print(cameras);\
    from pytorch3d.transforms import Transform3d;print(Transform3d);"|python

echo "import torch;device=torch.device('cuda');\
    from pytorch3d.utils import torus;\
    Torus = torus(r=10, R=20, sides=100, rings=100, device=device);\
    print(Torus.verts_padded());"|python
```

## 安装 MMHuman3D

a. 安装 mmcv-full 、 mmpose 、 mmdet 和 mmtrack

- mmcv-full

推荐使用如下命令安装mmcv-full.

对于 CPU:
```shell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cpu/{torch_version}/index.html
```
使用您的pytorch版本号替换`{torch_version}`

对于 GPU:
 ```shell
 pip install "mmcv-full>=1.3.17,<=1.5.3" -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
 ```
使用您的cuda版本号和pytorch版本号替换`{cu_version}`和`{torch_version}`

以在CUDA 10.2和PyTorch 1.8.0的环境下, 安装mmcv-full为例:
```shell
pip install "mmcv-full>=1.3.17,<=1.5.3" -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html
```
您可以从 [这里](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) 查看适配于不同CUDA版本和PyTorch版本的MMCV.
更多版本的下载信息，请参考 [openmmlab-download](https://download.openmmlab.com/mmcv/dist/index.html) 。

您也可以使用如下命令，从源码编译mmcv

```shell
git clone https://github.com/open-mmlab/mmcv.git -b v1.5.3
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # 执行此步骤会安装包含cuda算子的mmcv-full
# 或者运行 pip install -e .  # 执行此步骤会安装不包含cuda算子的mmcv
cd ..
```

**注意** 如果已经安装了mmcv，您需要先运行`pip uninstall mmcv`。 同时安装了mmcv和mmcv-full，可能会出现`ModuleNotFoundError`。

- mmdetection (可选)

```shell
pip install "mmdet<=2.25.1"
```

如果想要修改mmdet的代码，您也可以使用如下命令，从源码构建mmdet：
```shell
git clone https://github.com/open-mmlab/mmdetection.git -b v2.25.1
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
```

- mmpose (可选)
```shell
pip install "mmpose<=0.28.1"
```

如果您想要修改mmdet的代码，您也可以使用如下命令，从源码构建mmpose：

```shell
git clone https://github.com/open-mmlab/mmpose.git -b v0.28.1
cd mmpose
pip install -r requirements.txt
pip install -v -e .
```

- mmtracking (可选)

```shell
pip install "mmcls<=0.23.2" "mmtrack<=0.13.0"
```

如果您想要修改mmdet的代码，您也可以使用如下命令，从源码构建mmtracking：

```shell
git clone https://github.com/open-mmlab/mmtracking.git -b v0.13.0
cd mmtracking
pip install -r requirements/build.txt
pip install -v -e .  # 或者 "python setup.py develop"
```
b. 克隆mmhuman3d仓库.

```shell
git clone https://github.com/open-mmlab/mmhuman3d.git
cd mmhuman3d
```


c. 安装依赖项并安装mmhuman3d.

```shell
pip install -v -e .  # 或者 "python setup.py develop"
```

## 从头开始安装的脚本

```shell
# 安装 conda 环境
conda create -n open-mmlab python=3.8 -y
conda activate open-mmlab

# 安装 ffmpeg
conda install ffmpeg

# 安装 PyTorch
conda install pytorch==1.8.0 torchvision cudatoolkit=10.2 -c pytorch -y

# 安装 PyTorch3D
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d -y
# 在遇到依赖冲突的情况下也可使用从源码中安装
# git clone https://github.com/facebookresearch/pytorch3d.git
# cd pytorch3d
# pip install .
# cd ..

# 安装 mmcv-full
pip install "mmcv-full>=1.3.17,<1.6.0" -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.0/index.html

# 可选: 安装 mmdetection，mmpose 及 mmtracking
pip install "mmdet<=2.25.1"
pip install "mmpose<=0.28.1"
pip install "mmcls<=0.23.2" "mmtrack<=0.13.0"

# 安装 mmhuman3d
git clone https://github.com/open-mmlab/mmhuman3d.git
cd mmhuman3d
pip install -v -e .
```
