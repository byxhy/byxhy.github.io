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
* [GET STARTED][2]
* [BASIC TENSOR OPERATION][3]
* [TensorFlow 2.0 basic operations][4]
* [Compilation and training][5]
* [Evaluate the model][6]

[1]: #1
[2]: #2
[3]: #3
[4]: #4
[5]: #5
[6]: #6

---


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

- b. Copy PyTorch installation [command](https://pytorch.org/get-started/locally/)
![png](/assets/img/PyTorch/pytorch-install-command.png)
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

- c. Test if the PyTorch installation is normal
![png](/assets/img/PyTorch/pytorch-test.png)

---

<br />


<h2 id="2">GET STARTED</h2>

### 1. Linear regression

- Core of [gradient descent](https://www.bilibili.com/video/av15997678?p=3)
![png](/assets/img/PyTorch/gradient-descent.jpg)

- a. Question
  - How to figure out the parameters w and b to fit the function y = wx + b ?


- b. Framework
  - create a data array by y = 0.72x + 3.19
  - compute the loss = MSE(y_pre - p_true)
  - update the parameters w and b by gradient descent method (Get the gradient of loss)


- c. Action

``` python
import numpy as np
np.random.seed(1234)

# 1. Create a data array

num = 100

w_true, b_true = 0.727, 3.193
# esp = np.random.random(100)
# esp = np.random.normal(0.1, 1, num)
# print(np.mean(esp), np.std(esp))

esp = 0

x = np.linspace(-50, 50, num)

y_true = w_true * x + b_true + esp


# 2. loss
def gradient_descent(w_init, b_init, lr, x, y):
    N = len(x)

    # You need to re-understand loss function, don't lost the sum
    # grad_w = 2 / N * (w_init * x + b_init - y) * x
    # grad_b = 2 / N * (w_init * x + b_init - y)

    grad_w = 2 / N * np.sum((w_init * x + b_init - y) * x)
    grad_b = 2 / N * np.sum((w_init * x + b_init - y))

    w_new = w_init - lr * grad_w
    b_new = b_init - lr * grad_b

    return w_new, b_new


def train_epoch(epochs):
    lr = 1e-3

    w, b = 0, 0

    for e in range(epochs):
        w, b = gradient_descent(w, b, lr, x, y_true) # PA: update w, b

        y_pre = w * x + b

        loss = np.sum((y_pre - y_true) ** 2) / len(x)

        print('Epoch:', e, ' loss:', loss, ' w =', w, ' b =', b)


def main():
    train_epoch(3500)

    print('\nw_true:', w_true, ' b_true:', b_true)


if __name__ == '__main__':
    main()
```

- d. Review
  - Don't forget the sum action when we compute the gradient of loss
  - You should do some practice of numpy, like np.random.normal()...

- e. Futrue
  - The example of gradient descent helps us better understand deep learning algorithms

### 2. MNIST

xxxxxxxxxxx

---

<br />


<h2 id="3">BASIC TENSOR OPERATION</h2>

### 1. Basic data type

- Type check


```python
import torch
```


```python
a = torch.randn(2, 3)
a
```




    tensor([[-1.3514, -0.5171, -0.3782],
            [ 0.2258, -1.0002,  0.9651]])




```python
a.type()
```




    'torch.FloatTensor'




```python
type(a)
```




    torch.Tensor




```python
isinstance(a, torch.FloatTensor)
```




    True



- Dimension 0 / rank 0


```python
torch.tensor(1.)
```




    tensor(1.)




```python
torch.tensor(1.3)
```




    tensor(1.3000)



 - loss


```python
a = torch.tensor(2.2)
a
```




    tensor(2.2000)




```python
a.shape
```




    torch.Size([])




```python
len(a.shape)
```




    0




```python
a.size()
```




    torch.Size([])



- Dim1 / rank 1


```python
b = torch.tensor([1.1])
b
```




    tensor([1.1000])




```python
b.shape
```




    torch.Size([1])




```python
len(b.shape)
```




    1




```python
len(b.size())
```




    1



 - bias, linear


```python
torch.tensor([1.1])
```




    tensor([1.1000])




```python
torch.tensor([1.1, 2.2])
```




    tensor([1.1000, 2.2000])




```python
torch.FloatTensor(1)
```




    tensor([2.1985])




```python
torch.FloatTensor(2)
```




    tensor([1.0043e-11, 4.5776e-41])




```python
import numpy as np
data = np.ones(2)
data
```




    array([1., 1.])




```python
torch.from_numpy(data)
```




    tensor([1., 1.], dtype=torch.float64)




```python
a = torch.ones(2)
a
```




    tensor([1., 1.])




```python
a.shape
```




    torch.Size([2])



 - Size vs size


```python
torch.size([2])
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-24-db1c029f8b12> in <module>
    ----> 1 torch.size([2])


    AttributeError: module 'torch' has no attribute 'size'



```python
torch.Size([2])
```




    torch.Size([2])




```python
a.Size()
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-26-b19ed1e6466a> in <module>
    ----> 1 a.Size()


    AttributeError: 'Tensor' object has no attribute 'Size'



```python
a.size()
```




    torch.Size([2])



 - Dim 2


```python
a = torch.randn(2, 3)
a
```




    tensor([[ 0.9040, -0.8217,  0.9151],
            [ 0.0539, -0.1903, -0.2225]])




```python
a.shape
```




    torch.Size([2, 3])




```python
a.size(0)
```




    2




```python
a.size(1)
```




    3




```python
a.shape[1]
```




    3



 - Dim 3


```python
a = torch.randn(1, 2, 3)
a
```




    tensor([[[-0.3924,  0.3791, -0.0067],
             [-0.3150, -0.5330, -0.1551]]])




```python
a.shape
```




    torch.Size([1, 2, 3])




```python
a[0]
```




    tensor([[-0.3924,  0.3791, -0.0067],
            [-0.3150, -0.5330, -0.1551]])




```python
a[1]
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-36-8bc71255a22e> in <module>
    ----> 1 a[1]


    IndexError: index 1 is out of bounds for dimension 0 with size 1



```python
list(a.shape)
```




    [1, 2, 3]



 - Dim 4 ( CNN [b, c, h, w] )


```python
a = torch.rand(2, 3, 28, 28)
```


```python
a.shape
```




    torch.Size([2, 3, 28, 28])



 - Mixed


```python
a.numel() # Returns the total number of elements in the input tensor
```




    4704




```python
a.dim()
```




    4




```python
a = torch.tensor(1)
a
```




    tensor(1)




```python
a.dim()
```




    0




```python
a.size()
```




    torch.Size([])






```python

```
