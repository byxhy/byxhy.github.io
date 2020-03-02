---

layout: post
title: "Deep learning with PyTorch"
author: "Xhy"
categories: DeepLearning
tags: [improve]
image: ./PyTorch/pytorch.jpg
---

Image from [Google images](https://images.app.goo.gl/bqZGCzKDWBSw39qL9)

> An open source machine learning [framework](https://pytorch.org/) that accelerates the path from research prototyping to production deployment.

<br />



## `Table of Contents`

* [INSTALLATION][1]
* [Standard installation method][2]
* [Regression problems][3]
* [TensorFlow 2.0 basic operations][4]
* [Compilation and training][5]
* [Evaluate the model][6]

[1]: #1
[2]: #2
[3]: #3
[4]: #4
[5]: #5
[6]: #6



<br />



<h2 id="1">INSTALLATION</h2>

Take my Ubuntu as an example, and refer to the [official website](https://pytorch.org/get-started/locally/#macos-version) for other cases

### 1. CUDA Toolkit 10.1 Archive

- a. [Download CUDA 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal)

- b. Installation Instructions
```
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-<version>/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda
```
- c. Use **nvidia-smi** command to test if the installation is normal
![png](/assets/img/PyTorch/nvidia-smi.png)

- d. Add nvcc environment to PATH
```
vi ~/.bashrc
```
add the following to the bashrc file
```
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:"$LD_LIBRARY_PATH:/usr/loacl/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
export PATH=/usr/local/cuda/bin:$PATH
```
```
source ~/.bashrc
echo $PATH
nvcc -V
```
![png](/assets/img/PyTorch/nvcc-V.png)

### 2. Anaconda

- a. [Download Anaconda](https://www.anaconda.com/distribution/)

- b. Installation Instructions
```
sha256sum Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh
conda config --set auto_activate_base false
```

### 3. PyTorch

- a. Create a torch virtual environment
```
conda create -n torch
conda activate torch
```

- b. [Copy PyTorch installation command](https://pytorch.org/get-started/locally/)

![png](/assets/img/PyTorch/pytorch-install-command.png)

```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

- c. Test if the PyTorch installation is normal
![png](/assets/img/PyTorch/pytorch-test.png)

<br />


<h2 id="2">[ 2. Standard installation method ]</h2>



<br />


<h2 id="3">[ 3. Regression problems ]</h2>

[Anaconda Download](https://www.anaconda.com/distribution/)
![png](/assets/img/PyTorch/pytorch-test.png)
