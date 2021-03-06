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
* [STOCHASTIC GRADIENT DESCENT][4]
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
          w, b = gradient_descent(w, b, lr, x, y_true) # update w, b

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

---

<br />


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



*Summary*
- Tensor(shape) vs tensor(list): torch.Tensor(3, 3) / torch.tensor([3., 2.])
- empty/Tensor vs rand/randn: recommend the last one usage, if you meet 'nan'or 'inf', you should check if you have initialized the variable
- arange[start, end) vs linspace[start, end]: arange(low, high, step) / linspace(low, high, nums)

---

<br />


### 3. Index and slice

- Indexing


```python
import torch
import numpy as np
```


```python
a = torch.rand(4, 3, 28, 28)
```


```python
a[0].shape
```




    torch.Size([3, 28, 28])




```python
a[0, 0].shape
```




    torch.Size([28, 28])




```python
a[0, 0, 2, 24] # dimension = 0
```




    tensor(0.8909)



- Select first/last N


```python
a.shape
```




    torch.Size([4, 3, 28, 28])




```python
a[:2].shape
```




    torch.Size([2, 3, 28, 28])




```python
a[:2, :1, :, :].shape  # [start:end)
```




    torch.Size([2, 1, 28, 28])




```python
a[:2, 1:, :, :].shape
```




    torch.Size([2, 2, 28, 28])




```python
a[:2, -1:, :, :].shape  #[0, 1, 2] --> [-3, -2, -1]
```




    torch.Size([2, 1, 28, 28])




```python
a[:2, :-1, :, :].shape
```




    torch.Size([2, 2, 28, 28])



- select by setps


```python
a[:, :, 0:28:2, 0:28:2].shape
```




    torch.Size([4, 3, 14, 14])




```python
a[:, :, ::2, ::2].shape
```




    torch.Size([4, 3, 14, 14])




```python
torch.tensor([1.2, 3]).type()
```

- selct by specific index


```python
a.shape
```




    torch.Size([4, 3, 28, 28])




```python
a.index_select(0, [1, 2]).shape
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-26-99f26bb9aaf7> in <module>
    ----> 1 a.index_select(0, [1, 2]).shape


    TypeError: index_select() received an invalid combination of arguments - got (int, list), but expected one of:
     * (name dim, Tensor index)
          didn't match because some of the arguments have invalid types: (int, list)
     * (int dim, Tensor index)
          didn't match because some of the arguments have invalid types: (int, list)




```python
a.index_select(0, (1, 2)).shape
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-28-191f52239e29> in <module>
    ----> 1 a.index_select(0, (1, 2)).shape


    TypeError: index_select() received an invalid combination of arguments - got (int, tuple), but expected one of:
     * (name dim, Tensor index)
          didn't match because some of the arguments have invalid types: (int, tuple)
     * (int dim, Tensor index)
          didn't match because some of the arguments have invalid types: (int, tuple)




```python
a.index_select(0, torch.tensor([1, 2])).shape # Tensor index
```




    torch.Size([2, 3, 28, 28])




```python
a.index_select(3, torch.arange(28)).shape
```




    torch.Size([4, 3, 28, 28])




```python
a.index_select(3, torch.arange(8)).shape
```




    torch.Size([4, 3, 28, 8])




```python
torch.arange(8)
```




    tensor([0, 1, 2, 3, 4, 5, 6, 7])



- ...


```python
a.shape
```




    torch.Size([4, 3, 28, 28])




```python
a[..., :2].shape
```




    torch.Size([4, 3, 28, 2])




```python
a[:, 1, ...].shape
```




    torch.Size([4, 28, 28])




```python
torch.randint(1, 10, (3, 3))
```

- select by mask


```python
x = torch.randn(3, 4)
x
```




    tensor([[ 1.3120, -0.4552, -0.9988, -1.0441],
            [-0.8049, -0.2233,  0.9265, -1.0436],
            [ 1.8328, -0.8679, -1.3924,  0.7992]])




```python
mask = x.ge(0.5)
mask
```




    tensor([[ True, False, False, False],
            [False, False,  True, False],
            [ True, False, False,  True]])




```python
torch.masked_select(x, mask)
```




    tensor([1.3120, 0.9265, 1.8328, 0.7992])




```python
torch.masked_select(x, mask).shape
```




    torch.Size([4])



- select by flatten index


```python
src = torch.tensor([[3, 4, 5], [6, 7, 8]])
src
```




    tensor([[3, 4, 5],
            [6, 7, 8]])




```python
torch.take(src, torch.tensor([0, 2, 5]))
```




    tensor([3, 5, 8])



*Summary*

- index_select(name dim, Tensor index)

---

<br />


### 4. Tensor dimension transformation

- View reshape (Lost dim infomation)


```python
import torch
import numpy as np
```


```python
x = torch.rand(4, 1, 28, 28)
```


```python
a = x
```


```python
a.shape
```




    torch.Size([4, 1, 28, 28])




```python
a.view(4, 28*28)
```




    tensor([[0.7975, 0.5761, 0.8444,  ..., 0.9069, 0.5710, 0.7413],
            [0.8978, 0.9502, 0.2907,  ..., 0.4465, 0.4286, 0.7255],
            [0.3094, 0.0622, 0.5224,  ..., 0.1396, 0.4282, 0.8647],
            [0.7251, 0.6360, 0.3513,  ..., 0.9942, 0.3114, 0.3732]])




```python
a.view(4, 28*28).shape
```




    torch.Size([4, 784])




```python
a.view(4*28, 28).shape
```




    torch.Size([112, 28])




```python
a.view(4*1, 28, 28).shape
```




    torch.Size([4, 28, 28])




```python
b = a.view(4, 784)
b.shape
```




    torch.Size([4, 784])




```python
c = b.view(4, 28, 28, 1)
c.shape
```




    torch.Size([4, 28, 28, 1])




```python
d = b.view(4, 1, 28, 28)
d.shape
```




    torch.Size([4, 1, 28, 28])




```python
torch.equal(c, x)
```




    False




```python
torch.equal(d, x)
```




    True




```python
a.view(4, 783)
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-14-0404691742fb> in <module>
    ----> 1 a.view(4, 783)


    RuntimeError: shape '[4, 783]' is invalid for input of size 3136


- unsqueeze  [-input.dim() - 1, input.dim() + 1)

![png](/assets/img/PyTorch/unsqueeze.png)


```python
a = torch.rand(4, 1, 28, 28)
a.shape
```




    torch.Size([4, 1, 28, 28])




```python
a.unsqueeze(0).shape # vs a.unsqueeze(-5).shape
```




    torch.Size([1, 4, 1, 28, 28])




```python
a.shape
```




    torch.Size([4, 1, 28, 28])




```python
a.unsqueeze(-5).shape
```




    torch.Size([1, 4, 1, 28, 28])




```python
a.unsqueeze(3).shape # vs a.unsqueeze(-2).shape
```




    torch.Size([4, 1, 28, 1, 28])




```python
a.shape
```




    torch.Size([4, 1, 28, 28])




```python
a.unsqueeze(-2).shape
```




    torch.Size([4, 1, 28, 1, 28])




```python
b = torch.tensor([1.2, 2.3])
b
```




    tensor([1.2000, 2.3000])




```python
b.shape
```




    torch.Size([2])




```python
b.unsqueeze(-1)
```




    tensor([[1.2000],
            [2.3000]])




```python
b.unsqueeze(0)
```




    tensor([[1.2000, 2.3000]])



- unsqueeze example


```python
c = torch.rand(32)
c.shape
```




    torch.Size([32])




```python
target = torch.rand(4, 32, 14, 14)
target.shape
```




    torch.Size([4, 32, 14, 14])




```python
c = c.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
c.shape
```




    torch.Size([1, 32, 1, 1])



- squeeze


```python
d = torch.rand(1, 32, 1, 1)
d.shape
```




    torch.Size([1, 32, 1, 1])




```python
d.squeeze().shape
```




    torch.Size([32])




```python
d.squeeze(0).shape # vs d.squeeze(-4).shape
```




    torch.Size([32, 1, 1])




```python
d.squeeze(-4).shape
```




    torch.Size([32, 1, 1])




```python
d.squeeze(-1).shape
```




    torch.Size([1, 32, 1])




```python
d.squeeze(-2).shape
```




    torch.Size([1, 32, 1])



- expand:broadcasting


```python
a = torch.rand(4, 32, 14, 14)
b = torch.rand(1, 32, 1, 1)
```


```python
b.expand(4, 32, 14, 14).shape
```




    torch.Size([4, 32, 14, 14])




```python
b.expand(-1, 32, -1, -1).shape
```




    torch.Size([1, 32, 1, 1])




```python
b.expand(-1, 32, -1, -4).shape # bug
```




    torch.Size([1, 32, 1, -4])



- repeat: Memory touched


```python
b.shape
```




    torch.Size([1, 32, 1, 1])




```python
b.repeat(4, 32, 1, 1).shape
```




    torch.Size([4, 1024, 1, 1])




```python
b.repeat(4, 1, 1, 1).shape
```




    torch.Size([4, 32, 1, 1])




```python
b.repeat(4, 1, 32, 32).shape
```




    torch.Size([4, 32, 32, 32])



- .t (tensor with <= 2 dimensions)


```python
b.t()
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-44-b510e0f64f40> in <module>
    ----> 1 b.t()


    RuntimeError: t() expects a tensor with <= 2 dimensions, but self is 4D



```python
c = torch.rand(3, 4)
c
```




    tensor([[0.6893, 0.5189, 0.9370, 0.5368],
            [0.8434, 0.8373, 0.8882, 0.6287],
            [0.6004, 0.5860, 0.1956, 0.0710]])




```python
c.t()
```




    tensor([[0.6893, 0.8434, 0.6004],
            [0.5189, 0.8373, 0.5860],
            [0.9370, 0.8882, 0.1956],
            [0.5368, 0.6287, 0.0710]])



- transpose


```python
a = torch.rand(4, 3, 32, 32)
```


```python
b1 = a.transpose(1, 3).view(4, 32*32*3).view(4, 3, 32, 32)
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-48-72a945409913> in <module>
    ----> 1 b1 = a.transpose(1, 3).view(4, 32*32*3).view(4, 3, 32, 32)


    RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.



```python
b1 = a.transpose(1, 3).contiguous().view(4, 32*32*3).view(4, 3, 32, 32)
b1.shape
```




    torch.Size([4, 3, 32, 32])




```python
b2 = a.transpose(1, 3).contiguous().view(4, 32*32*3).view(4, 32, 32, 3).transpose(1, 3)
b2.shape
```




    torch.Size([4, 3, 32, 32])




```python
torch.all(torch.eq(a, b1))
```




    tensor(False)




```python
torch.all(torch.eq(a, b2))
```




    tensor(True)



- permute [b, c, h, w] --> [b, h, w, c]


```python
a = torch.rand(4, 3, 28, 28)
a.transpose(1, 3).shape
```




    torch.Size([4, 28, 28, 3])




```python
b = torch.rand(4, 3, 28, 32)
b.transpose(1, 3).shape
```




    torch.Size([4, 32, 28, 3])




```python
b.transpose(1, 3).transpose(1, 2).shape
```




    torch.Size([4, 28, 32, 3])




```python
b.permute(0, 2, 3, 1).shape
```




    torch.Size([4, 28, 32, 3])



*Summary*

- unsqueeze [-input.dim() - 1, input.dim() + 1)
- repeat --> expand
- transpose --> permute

---

<br />


### 5. Broadcasting

- broadcasting (start with the last dimension)


```python
import torch
import numpy as np
```


```python
x = torch.rand(4, 1, 28, 28)
```


```python
b = 0.2
```


```python
x_b = x + b
x_b.shape
```




    torch.Size([4, 1, 28, 28])




```python
x_b - x
```




    tensor([[[[0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000],
              [0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000],
              [0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000],
              ...,
              [0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000],
              [0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000],
              [0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000]]],


​    

            [[[0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000],
              [0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000],
              [0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000],
              ...,
              [0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000],
              [0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000],
              [0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000]]],


​    

            [[[0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000],
              [0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000],
              [0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000],
              ...,
              [0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000],
              [0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000],
              [0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000]]],


​    

            [[[0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000],
              [0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000],
              [0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000],
              ...,
              [0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000],
              [0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000],
              [0.2000, 0.2000, 0.2000,  ..., 0.2000, 0.2000, 0.2000]]]])

---

<br />


### 6. Merge and split

- cat


```python
import torch
import numpy as np
```


```python
a = torch.rand(4, 32, 8)
b = torch.rand(5, 32, 8)
```


```python
torch.cat([a, b], dim=0).shape
```




    torch.Size([9, 32, 8])




```python
a = torch.rand(4, 3, 16, 32)
b = torch.rand(4, 3, 16, 32)
```


```python
torch.cat([a, b], dim=2).shape
```




    torch.Size([4, 3, 32, 32])



- stack


```python
a = torch.rand(16, 32)
b = torch.rand(16, 32)
```


```python
torch.stack([a, b], dim=0).shape
```




    torch.Size([2, 16, 32])




```python
torch.stack([a, b], dim=1).shape
```




    torch.Size([16, 2, 32])




```python
a = torch.rand(4, 3, 16, 32)
b = torch.rand(4, 3, 16, 32)
```


```python
torch.stack([a, b], dim=2).shape
```




    torch.Size([4, 3, 2, 16, 32])



- stack vs cat


```python
a = torch.rand(32, 8)
b = torch.rand(30, 8)
```


```python
torch.cat([a, b], dim=0).shape
```




    torch.Size([62, 8])




```python
torch.stack([a, b], dim=0).shape
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-13-201fb7282b6d> in <module>
    ----> 1 torch.stack([a, b], dim=0).shape


    RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 0. Got 32 and 30 in dimension 1 at /opt/conda/conda-bld/pytorch_1579022027550/work/aten/src/TH/generic/THTensor.cpp:612


- split: by len


```python
c = torch.rand(4, 32, 8)
```


```python
aa, bb = c.split([1, 3], dim=0)
aa.shape, bb.shape
```




    (torch.Size([1, 32, 8]), torch.Size([3, 32, 8]))




```python
aa, bb = c.split(2, dim=0)
aa.shape, bb.shape
```




    (torch.Size([2, 32, 8]), torch.Size([2, 32, 8]))




```python
aa, bb, cc = c.split([1, 2, 1], dim=0)
aa.shape, bb.shape, cc.shape
```




    (torch.Size([1, 32, 8]), torch.Size([2, 32, 8]), torch.Size([1, 32, 8]))



- chunk: by num (vs split)


```python
c = torch.rand(2, 32, 8)
```


```python
aa, bb = c.split(2, dim=0)
aa.shape, bb.shape
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-19-6536367bfe5f> in <module>
    ----> 1 aa, bb = c.split(2, dim=0)
          2 aa.shape, bb.shape


    ValueError: not enough values to unpack (expected 2, got 1)



```python
aa, bb = c.chunk(2, dim=0)
aa.shape, bb.shape
```




    (torch.Size([1, 32, 8]), torch.Size([1, 32, 8]))

---

<br />


### 7. Basic operation

- basic


```python
import torch
import numpy as np
```


```python
a = torch.rand(3, 4)
b = torch.rand(4)
```


```python
a+b
```




    tensor([[0.5698, 1.0592, 1.5789, 0.4249],
            [1.2410, 1.4248, 1.1539, 0.4428],
            [0.4254, 1.2981, 1.1344, 1.0469]])




```python
torch.add(a, b)
```




    tensor([[0.5698, 1.0592, 1.5789, 0.4249],
            [1.2410, 1.4248, 1.1539, 0.4428],
            [0.4254, 1.2981, 1.1344, 1.0469]])




```python
torch.all(torch.eq(a+b, torch.add(a, b)))
```




    tensor(True)




```python
torch.all(torch.eq(a-b, torch.sub(a, b)))
```




    tensor(True)




```python
torch.all(torch.eq(a*b, torch.mul(a, b)))
```




    tensor(True)




```python
torch.all(torch.eq(a/b, torch.div(a, b)))
```




    tensor(True)



- matmul


```python
a = torch.full([2, 2], 3)
a
```




    tensor([[3., 3.],
            [3., 3.]])




```python
b = torch.ones(2, 2)
b
```




    tensor([[1., 1.],
            [1., 1.]])




```python
torch.mm(a, b)
```




    tensor([[6., 6.],
            [6., 6.]])




```python
torch.matmul(a, b)
```




    tensor([[6., 6.],
            [6., 6.]])




```python
a @ b
```




    tensor([[6., 6.],
            [6., 6.]])



- matmul for linear layer


```python
x = torch.rand(4, 784)
```


```python
w = torch.rand(512, 784)
```


```python
(x @ w.t()).shape
```




    torch.Size([4, 512])



- \>2d tensor matmul


```python
a = torch.rand(4, 3, 28, 64)
b = torch.rand(4, 3, 64, 32)
```


```python
torch.mm(a, b).shape
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-21-bd98cd1646a7> in <module>
    ----> 1 torch.mm(a, b).shape


    RuntimeError: matrices expected, got 4D, 4D tensors at /opt/conda/conda-bld/pytorch_1579022027550/work/aten/src/TH/generic/THTensorMath.cpp:131



```python
torch.matmul(a, b).shape
```




    torch.Size([4, 3, 28, 32])




```python
c = torch.rand(64, 32)
torch.matmul(a, c).shape
```




    torch.Size([4, 3, 28, 32])




```python
d = torch.rand(4, 64, 32)
torch.matmul(a, d).shape
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-25-55e69d3a8519> in <module>
          1 d = torch.rand(4, 64, 32)
    ----> 2 torch.matmul(a, d).shape


    RuntimeError: The size of tensor a (3) must match the size of tensor b (4) at non-singleton dimension 1


- power


```python
a = torch.full([2, 2], 3)
```


```python
a.pow(2)
```




    tensor([[9., 9.],
            [9., 9.]])




```python
a**2
```




    tensor([[9., 9.],
            [9., 9.]])




```python
aa = a**2
aa.sqrt()
```




    tensor([[3., 3.],
            [3., 3.]])




```python
aa.rsqrt()  # reciprocal
```




    tensor([[0.3333, 0.3333],
            [0.3333, 0.3333]])




```python
1 / aa.sqrt()
```




    tensor([[0.3333, 0.3333],
            [0.3333, 0.3333]])




```python
aa**.5
```




    tensor([[3., 3.],
            [3., 3.]])



- Exp log


```python
a = torch.exp(torch.ones(2, 2))
a
```




    tensor([[2.7183, 2.7183],
            [2.7183, 2.7183]])




```python
torch.log(a)
```




    tensor([[1., 1.],
            [1., 1.]])



- Approximation


```python
a = torch.tensor(3.14)
```


```python
a.floor(), a.ceil(), a.trunc(), a.frac()
```




    (tensor(3.), tensor(4.), tensor(3.), tensor(0.1400))




```python
b = torch.tensor(3.499)
b.round()
```




    tensor(3.)




```python
c = torch.tensor(3.5)
c.round()
```




    tensor(4.)



- clamp(min), (min, max)


```python
grad = torch.rand(2, 3) * 15
grad
```




    tensor([[ 5.2049, 10.0966,  6.0705],
            [11.3415,  3.0498, 14.2503]])




```python
grad.max(), grad.min(), grad.median()
```




    (tensor(14.2503), tensor(3.0498), tensor(6.0705))




```python
grad.clamp(10)
```




    tensor([[10.0000, 10.0966, 10.0000],
            [11.3415, 10.0000, 14.2503]])




```python
grad.clamp(0, 10)
```




    tensor([[ 5.2049, 10.0000,  6.0705],
            [10.0000,  3.0498, 10.0000]])

---

<br />


### 8. Statistics properties

- norm-p


```python
import torch
import numpy as np
```


```python
a = torch.full([8], 1)
b = a.view(2, 4)
c = a.view(2, 2, 2)
```


```python
a.norm(1), b.norm(1), c.norm(1)
```




    (tensor(8.), tensor(8.), tensor(8.))




```python
a.norm(2), b.norm(2), c.norm(2)
```




    (tensor(2.8284), tensor(2.8284), tensor(2.8284))




```python
b.norm(1, dim=1)
```




    tensor([4., 4.])




```python
b.norm(2, dim=1)
```




    tensor([2., 2.])




```python
c.norm(1, dim=0)
```




    tensor([[2., 2.],
            [2., 2.]])




```python
c.norm(2, dim=0)
```




    tensor([[1.4142, 1.4142],
            [1.4142, 1.4142]])



- mean, sum, min, max, prod


```python
a = torch.arange(8).view(2, 4).float()
a
```




    tensor([[0., 1., 2., 3.],
            [4., 5., 6., 7.]])




```python
a.min(), a.max(), a.mean(), a.prod()
```




    (tensor(0.), tensor(7.), tensor(3.5000), tensor(0.))




```python
a.sum()
```




    tensor(28.)




```python
a.argmax(), a.argmin()
```




    (tensor(7), tensor(0))



- argmin, argmax


```python
a = torch.rand(4, 10)
a[0]
```




    tensor([0.3720, 0.6761, 0.3741, 0.9730, 0.2760, 0.0788, 0.4316, 0.0789, 0.8999,
            0.8642])




```python
a.argmax()
```




    tensor(3)




```python
a.argmax(dim=1)
```




    tensor([3, 4, 8, 8])



- dim, keepdim


```python
a = torch.rand(4, 10)
```


```python
a.max(dim=1)
```




    torch.return_types.max(
    values=tensor([0.7634, 0.9783, 0.8863, 0.9994]),
    indices=tensor([4, 1, 3, 0]))




```python
a.max(dim=1, keepdim=True)
```




    torch.return_types.max(
    values=tensor([[0.7634],
            [0.9783],
            [0.8863],
            [0.9994]]),
    indices=tensor([[4],
            [1],
            [3],
            [0]]))




```python
a.argmax(dim=1, keepdim=True)
```




    tensor([[4],
            [1],
            [3],
            [0]])



- Top-k or k-th


```python
a = torch.rand(4, 10)
a
```




    tensor([[0.9320, 0.1271, 0.7293, 0.3934, 0.9798, 0.8641, 0.6343, 0.3942, 0.0804,
             0.7824],
            [0.0198, 0.5221, 0.1622, 0.4375, 0.4997, 0.4394, 0.6338, 0.6554, 0.7376,
             0.3378],
            [0.1089, 0.0269, 0.5377, 0.5592, 0.9298, 0.5855, 0.9504, 0.6926, 0.9498,
             0.4973],
            [0.5445, 0.1226, 0.9817, 0.0698, 0.9938, 0.6199, 0.8160, 0.0527, 0.5491,
             0.8526]])




```python
a.topk(3, dim=1)
```




    torch.return_types.topk(
    values=tensor([[0.9798, 0.9320, 0.8641],
            [0.7376, 0.6554, 0.6338],
            [0.9504, 0.9498, 0.9298],
            [0.9938, 0.9817, 0.8526]]),
    indices=tensor([[4, 0, 5],
            [8, 7, 6],
            [6, 8, 4],
            [4, 2, 9]]))




```python
a.topk(3, dim=1, largest=False)
```




    torch.return_types.topk(
    values=tensor([[0.0804, 0.1271, 0.3934],
            [0.0198, 0.1622, 0.3378],
            [0.0269, 0.1089, 0.4973],
            [0.0527, 0.0698, 0.1226]]),
    indices=tensor([[8, 1, 3],
            [0, 2, 9],
            [1, 0, 9],
            [7, 3, 1]]))




```python
a.kthvalue(1, dim=1)
```




    torch.return_types.kthvalue(
    values=tensor([0.0804, 0.0198, 0.0269, 0.0527]),
    indices=tensor([8, 0, 1, 7]))




```python
a.kthvalue(10, dim=1)
```




    torch.return_types.kthvalue(
    values=tensor([0.9798, 0.7376, 0.9504, 0.9938]),
    indices=tensor([4, 8, 6, 4]))




```python
a.kthvalue(3)
```




    torch.return_types.kthvalue(
    values=tensor([0.3934, 0.3378, 0.4973, 0.1226]),
    indices=tensor([3, 9, 9, 1]))




```python
a.kthvalue(1, keepdim=True)
```




    torch.return_types.kthvalue(
    values=tensor([[0.0804],
            [0.0198],
            [0.0269],
            [0.0527]]),
    indices=tensor([[8],
            [0],
            [1],
            [7]]))



- compare


```python
a = torch.randn(4, 10)
```


```python
a > 0
```




    tensor([[False, False, False, False,  True,  True,  True, False,  True, False],
            [ True, False,  True, False, False,  True,  True,  True,  True, False],
            [ True,  True,  True, False, False,  True,  True, False, False, False],
            [False, False,  True,  True, False,  True, False, False,  True,  True]])




```python
torch.gt(a, 0)
```




    tensor([[False, False, False, False,  True,  True,  True, False,  True, False],
            [ True, False,  True, False, False,  True,  True,  True,  True, False],
            [ True,  True,  True, False, False,  True,  True, False, False, False],
            [False, False,  True,  True, False,  True, False, False,  True,  True]])




```python
a != 0
```




    tensor([[True, True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True, True]])



- eq vs equal


```python
a = torch.ones(2, 3)
b = torch.ones(2, 3)
b[:,2] = 0
```


```python
a
```




    tensor([[1., 1., 1.],
            [1., 1., 1.]])




```python
b
```




    tensor([[1., 1., 0.],
            [1., 1., 0.]])




```python
torch.eq(a, b)
```




    tensor([[ True,  True, False],
            [ True,  True, False]])




```python
torch.equal(a, b)
```




    False




```python
torch.eq(a, a)
```




    tensor([[True, True, True],
            [True, True, True]])




```python
torch.equal(a, a)
```




    True



*Summary*

- L1-Norm: Manhattan Distance or Taxicab norm
- L2-Norm: Is the most popular norm, also known as the Euclidean norm
- topk(max_index --> min_index) vs k-th(k --> max_index): topk(10 --> 0) vs k-th(k --> 10)
- eq vs equal

---

<br />


### 9. Advanced operation

- where


```python
import torch
import numpy as np
```


```python
x = torch.randn(2, 2)
x
```




    tensor([[ 0.4610,  0.0117],
            [-1.4521, -3.1001]])




```python
a = torch.ones(2, 2)
a
```




    tensor([[1., 1.],
            [1., 1.]])




```python
b = torch.zeros(2, 2)
b
```




    tensor([[0., 0.],
            [0., 0.]])




```python
torch.where(x > 0.01, a, b)
```




    tensor([[1., 1.],
            [0., 0.]])




```python
torch.gt(x, 0.01)
```




    tensor([[ True,  True],
            [False, False]])



- gather


```python
prob = torch.randn(4, 10)
```


```python
idx = prob.topk(3, dim=1)
idx
```




    torch.return_types.topk(
    values=tensor([[1.2369, 0.9901, 0.1664],
            [1.6307, 0.7344, 0.2915],
            [2.8899, 1.1640, 1.0699],
            [0.8193, 0.4604, 0.1266]]),
    indices=tensor([[4, 6, 0],
            [2, 0, 5],
            [5, 0, 3],
            [2, 7, 8]]))




```python
idx = idx[1]
idx
```




    tensor([[4, 6, 0],
            [2, 0, 5],
            [5, 0, 3],
            [2, 7, 8]])




```python
label = torch.arange(10) + 100
label.expand(4, 10)
```




    tensor([[100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]])




```python
idx.long()
```




    tensor([[4, 6, 0],
            [2, 0, 5],
            [5, 0, 3],
            [2, 7, 8]])




```python
torch.gather(label.expand(4, 10), dim=1, index=idx.long()) #retrieve label
```




    tensor([[104, 106, 100],
            [102, 100, 105],
            [105, 100, 103],
            [102, 107, 108]])

---

<br />


<h2 id="4">STOCHASTIC GRADIENT DESCENT</h2>
