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




### 2. Create Tensor

- Import from numpy


```python
import torch
import numpy as np
```


```python
a = np.array([2, 3.3])
a
```




    array([2. , 3.3])




```python
torch.from_numpy(a)
```




    tensor([2.0000, 3.3000], dtype=torch.float64)




```python
b = np.ones([2, 3])
b
```




    array([[1., 1., 1.],
           [1., 1., 1.]])




```python
torch.from_numpy(b)
```




    tensor([[1., 1., 1.],
            [1., 1., 1.]], dtype=torch.float64)



- Import from list


```python
torch.tensor([2., 3.2])
```




    tensor([2.0000, 3.2000])




```python
torch.tensor(2, 3)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-7-348b4305455b> in <module>
    ----> 1 torch.tensor(2, 3)


    TypeError: tensor() takes 1 positional argument but 2 were given



```python
torch.FloatTensor([2., 3.2])  # we don't recommend this usage, Tensor(shape), like Tensor(2, 3)
```




    tensor([2.0000, 3.2000])




```python
torch.Tensor(2, 3)
```




    tensor([[1.3340e+25, 3.0844e-41, 1.3372e+25],
            [3.0844e-41, 1.6297e-28, 4.5573e-41]])




```python
torch.tensor([[2., 3.2], [1., 22.3]])
```




    tensor([[ 2.0000,  3.2000],
            [ 1.0000, 22.3000]])



- set default type


```python
torch.tensor([1.2, 3]).type()
```




    'torch.FloatTensor'




```python
torch.set_default_tensor_type(torch.DoubleTensor)
```


```python
torch.tensor([1.2, 3]).type()
```




    'torch.DoubleTensor'



- uninitialized (Infinity or infinitesimal, we don't recommend it)


```python
torch.empty(1)
```




    tensor([6.9013e-310])




```python
torch.Tensor(2, 3)
```




    tensor([[6.9013e-310, 6.9013e-310, 1.5810e-322],
            [3.1620e-322, 4.6708e-310, 3.5975e+252]])




```python
torch.IntTensor(2, 3)
```




    tensor([[-1941471968,       32522,  1764817520],
            [      22011,           0,           0]], dtype=torch.int32)




```python
torch.FloatTensor(2, 3)
```




    tensor([[-1.5368e-31,  4.5573e-41,  1.3518e+25],
            [ 3.0844e-41,  4.4842e-44,  0.0000e+00]], dtype=torch.float32)



- rand/rand_like, randint (werecommend it)


```python
torch.rand(3, 3)
```




    tensor([[0.6695, 0.1251, 0.9080],
            [0.7135, 0.7052, 0.6049],
            [0.7605, 0.7369, 0.4836]])




```python
a = torch.rand(3, 3)
a
```




    tensor([[0.3567, 0.8419, 0.9836],
            [0.8136, 0.3909, 0.6134],
            [0.4892, 0.9364, 0.7614]])




```python
torch.rand_like(a)
```




    tensor([[0.9906, 0.6952, 0.0366],
            [0.1730, 0.5296, 0.1647],
            [0.0849, 0.3927, 0.2205]])




```python
torch.randint(1, 10, (3, 3))
```




    tensor([[5, 9, 5],
            [6, 6, 4],
            [2, 7, 9]])



- randn vs rand


```python
torch.randn(3, 3)   # N(0, 1)
```




    tensor([[ 1.0855,  0.4058, -1.2779],
            [ 0.7032,  0.3273, -0.2740],
            [-0.9013, -1.3954,  0.7301]])




```python
torch.normal(mean=torch.full([10], 0), std=torch.arange(1, 0, -0.1)) # N(u, std)
```




    tensor([-0.4578, -0.8805,  0.5602, -0.1000, -0.3663,  0.2051,  0.1285,  0.1521,
            -0.0412, -0.1021])




```python
torch.normal(mean=torch.full([10], 0), std=torch.arange(1, 0, -0.1))
```




    tensor([-0.0983, -0.3923, -1.0584, -0.0477, -0.2888,  0.0473, -0.1540,  0.7578,
             0.1673, -0.0741])



- full


```python
torch.full([2, 3], 7)
```




    tensor([[7., 7., 7.],
            [7., 7., 7.]])




```python
torch.full([], 7)
```




    tensor(7.)




```python
torch.full([1], 7)
```




    tensor([7.])



- arange/range


```python
torch.arange(0, 10) # [start; end)
```




    tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
torch.arange(0, 10, 2)
```




    tensor([0, 2, 4, 6, 8])




```python
torch.range(0, 10)
```

    <ipython-input-30-cf0dcdb325c8>:1: UserWarning: torch.range is deprecated in favor of torch.arange and will be removed in 0.5. Note that arange generates values in [start; end), not [start; end].
      torch.range(0, 10)





    tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])



- linspace vs arange


```python
torch.linspace(0, 10) #[start; end]
```




    tensor([ 0.0000,  0.1010,  0.2020,  0.3030,  0.4040,  0.5051,  0.6061,  0.7071,
             0.8081,  0.9091,  1.0101,  1.1111,  1.2121,  1.3131,  1.4141,  1.5152,
             1.6162,  1.7172,  1.8182,  1.9192,  2.0202,  2.1212,  2.2222,  2.3232,
             2.4242,  2.5253,  2.6263,  2.7273,  2.8283,  2.9293,  3.0303,  3.1313,
             3.2323,  3.3333,  3.4343,  3.5354,  3.6364,  3.7374,  3.8384,  3.9394,
             4.0404,  4.1414,  4.2424,  4.3434,  4.4444,  4.5455,  4.6465,  4.7475,
             4.8485,  4.9495,  5.0505,  5.1515,  5.2525,  5.3535,  5.4545,  5.5556,
             5.6566,  5.7576,  5.8586,  5.9596,  6.0606,  6.1616,  6.2626,  6.3636,
             6.4646,  6.5657,  6.6667,  6.7677,  6.8687,  6.9697,  7.0707,  7.1717,
             7.2727,  7.3737,  7.4747,  7.5758,  7.6768,  7.7778,  7.8788,  7.9798,
             8.0808,  8.1818,  8.2828,  8.3838,  8.4848,  8.5859,  8.6869,  8.7879,
             8.8889,  8.9899,  9.0909,  9.1919,  9.2929,  9.3939,  9.4949,  9.5960,
             9.6970,  9.7980,  9.8990, 10.0000])




```python
torch.arange(0, 10)
```




    tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
torch.linspace(0, 10, 2)
```




    tensor([ 0., 10.])




```python
torch.arange(0, 10, 2)
```




    tensor([0, 2, 4, 6, 8])




```python
torch.linspace(0, 10, 10)
```




    tensor([ 0.0000,  1.1111,  2.2222,  3.3333,  4.4444,  5.5556,  6.6667,  7.7778,
             8.8889, 10.0000])




```python
torch.linspace(0, 10, 11)
```




    tensor([ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])




```python
torch.logspace(0, -1, 10)
```




    tensor([1.0000, 0.7743, 0.5995, 0.4642, 0.3594, 0.2783, 0.2154, 0.1668, 0.1292,
            0.1000])




```python
torch.logspace(0, 1, 10)
```




    tensor([ 1.0000,  1.2915,  1.6681,  2.1544,  2.7826,  3.5938,  4.6416,  5.9948,
             7.7426, 10.0000])



- ones / zeros / eys


```python
torch.eye(3)
```




    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])




```python
torch.eye(3, 3)
```




    tensor([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])




```python
a = torch.ones(3, 3)
a
```




    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]])




```python
b = torch.zeros_like(a)
b
```




    tensor([[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]])



- randperm (Random permutation of integers)


```python
torch.randperm(10)
```




    tensor([9, 7, 8, 5, 1, 0, 4, 6, 3, 2])




```python
np.random.permutation(10)
```




    array([2, 4, 9, 8, 0, 3, 7, 1, 6, 5])




```python
a = torch.rand(2, 3)
b = torch.rand(2, 2)

print(a)
print("\n")
print(b)
```

    tensor([[0.7973, 0.5826, 0.7557],
            [0.9783, 0.9409, 0.6193]])


​    

    tensor([[0.0902, 0.7453],
            [0.0874, 0.7084]])



```python
idx = torch.randperm(a.shape[0])
idx
```




    tensor([1, 0])




```python
idx
```




    tensor([1, 0])




```python
a[idx]
```




    tensor([[0.9783, 0.9409, 0.6193],
            [0.7973, 0.5826, 0.7557]])




```python
b[idx]
```




    tensor([[0.0874, 0.7084],
            [0.0902, 0.7453]])




```python
a, b
```




    (tensor([[0.7973, 0.5826, 0.7557],
             [0.9783, 0.9409, 0.6193]]),
     tensor([[0.0902, 0.7453],
             [0.0874, 0.7084]]))



- Summary
  - Tensor(shape) vs tensor(list): torch.Tensor(3, 3) / torch.tensor([3., 2.])
  - empty/Tensor vs rand/randn: recommend the last one usage, if you meet 'nan'or 'inf', you should check if you have initialized the variable
  - arange[start, end) vs linspace[start, end]: arange(low, high, step) / linspace(low, high, nums)


  
