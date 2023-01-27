---

layout: post
title: "Deep learning with TensorFlow 2.0"
author: "Xhy"
categories: DeepLearning
tags: [Machine Learning]
image: ./TF2.0/TF.jpg
---



> At TensorFlow Dev Summit 2019, the TensorFlow team introduced the Alpha version of TensorFlow 2.0

<br />



## `Table of Contents`

* [INSTALLATION][1]
* [GET STARTED][2]
* [TF BASIC OPERATION 1][3]
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

### 3. TensorFlow 2.0

- a. Create a tensorflow virtual environment
```
conda create -n tf2
conda activate tf2
```

- b. Using **conda** command to install TF 2.0
![png](/assets/img/TF2.0/install-tf-gpu.png)
```
conda install tensorflow-gpu==2.1.0
```

- c. Test if the TensorFlow installation is normal
![png](/assets/img/TF2.0/tf-test.png)

---

<br />


<h2 id="2">GET STARTED</h2>

### 1. Linear regression

- Core of [gradient descent](https://www.bilibili.com/video/av15997678?p=3)
![png](/assets/img/PyTorch/gradient-descent.jpg)

- Code implementation

```python
import numpy as np

def compute_loss(points, w, b):
    x, y = points[:, 0], points[:, 1]
    N = float(len(x))

    loss = np.sum((w * x + b - y) ** 2) / N
    return loss

def step_gradient(points, init_w, init_b, learning_rate):
    x, y = points[:, 0], points[:, 1]
    N = float(len(x))

    w, b = init_w, init_b

    # Err 1: x is a array, not scale
    # grad_w = 2 * np.sum(w * x + b - y) * x / N
    # grad_b = 2 * np.sum(w * x + b - y) / N

    grad_w = np.sum(2 * (w * x + b - y) * x / N)
    grad_b = np.sum(2 * (w * x + b - y) / N)

    w_new = w - learning_rate * grad_w
    b_new = b - learning_rate * grad_b

    return w_new, b_new

def gradient_decent(points, init_w, init_b, lr, iterations):
    w, b = init_w, init_b
    for i in range(iterations):
        # Err 2: init_w, init_b just only once
        # w, b = gradient_decent(points, init_w, init_b, lr)
        # print('w =', w, 'b =', b)

        w, b = step_gradient(points, w, b, lr)

    return w, b

def run():
    # 1' read source data and initialize w, b and lr
    points = np.genfromtxt("data.csv", delimiter=",")

    init_w, init_b, lr = 0, 0, 0.0001

    iterations = 100

    # 2' compute loss_before
    loss_before = compute_loss(points, init_w, init_b)

    # 3' update w, b (gradient decent)
    w, b = gradient_decent(points, init_w, init_b, lr, iterations)

    # 4' compute loss_after
    loss_after = compute_loss(points, w, b)

    # 5' output the results
    print('loss_before =', loss_before, '\nloss_after  =', loss_after)
    print('w = ', w, 'b =', b)

if __name__ == '__main__':
    run()
```

```
loss_before = 5565.107834483214
loss_after  = 112.64705664288809
w =  1.4788027175308358 b = 0.03507497059234177
```

Summary: [loss -->  gradient -->  decent]

- y_pre = wx + b
- loss = mse(y_pre - y_true) = sum((wx_i + b - y_i)^2) / N
- grad_w = sum(2 * (wx_i + b - y_i) * x_i / N)
- grad_b = sum(2 * (wx_i + b - y_i) / N)
- w_new = w - lr * grad_w     b_new = b - lr * grad_b

### 2. Handwritten digit recognition problem

*Imagine being given a blank sheet of paper to write down a solution to this problem. What would you do ?*

- Q: Use TensorFlow to recognize the number in the picture of MNIST dataset
   - background
     - why : In order to master the skill of deep learning
     - diff: I have the class, but I want to do it by myself
   - nature : deep learning model, gradient decent method, classification
   - medicine : TensorFlow now
   -  target : Starting from scratch
-  Multidimensional and critical thinking:
   - divergent
- Framework: You must have a global view of the problem
   - input --> hidden --> output
- Execute:
- Check / update:
   - Before: input --> hidden --> output
   - After:  loss --> gradient --> decent
- Summary:  `[loss --> gradient --> decent]`
   - Normalize the train data
   - label to one-hot
   - x = tf.reshape(x, (-1, 28 * 28)）
   - step vs epoch

- Code implementation

```python
import tensorflow as tf
from tensorflow.keras import Sequential, layers, optimizers, datasets

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255.
y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)

y_train = tf.one_hot(y_train, depth=10)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(600)

model = Sequential([
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(10)])

opt = optimizers.SGD(learning_rate=0.001)


def train_epoch(epoch):
    # 4' loop
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # [b, 28, 28] --> [b, 784]
            x = tf.reshape(x, (-1, 784))
            out = model(x)
            # 1' compute loss
            loss = tf.reduce_sum(tf.square(out - y)) / x.shape[0]
        # 2' compute gradient
        grads = tape.gradient(loss, model.trainable_variables)
        # 3' gradient decent(w' = w - lr * grad)
        opt.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print('epoch:', epoch, 'step:', step, 'loss:', loss.numpy())

def train():
    for epoch in range(30):
        train_epoch(epoch)

if __name__ == '__main__':
    train()
```

```
epoch: 0 step: 0 loss: 1.8372002
epoch: 0 step: 100 loss: 0.8855913
...
epoch: 29 step: 0 loss: 0.2764416
epoch: 29 step: 100 loss: 0.32260606
```

---

<br />


<h2 id="3">TF BASIC OPERATION 1</h2>

### 1. Basic operation

#### Create


```python
import tensorflow as tf
```


```python
tf.constant(1)
```




    <tf.Tensor: id=0, shape=(), dtype=int32, numpy=1>




```python
tf.constant(1.)
```




    <tf.Tensor: id=1, shape=(), dtype=float32, numpy=1.0>




```python
tf.constant(2.2, dtype=tf.int32)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-4-cf6ddbb50ed9> in <module>
    ----> 1 tf.constant(2.2, dtype=tf.int32)


    TypeError: Cannot convert 2.2 to EagerTensor of dtype int32



```python
tf.constant(2.2, dtype=tf.double)
```




    <tf.Tensor: id=3, shape=(), dtype=float64, numpy=2.2>




```python
tf.constant([True, False])
```




    <tf.Tensor: id=4, shape=(2,), dtype=bool, numpy=array([ True, False])>




```python
tf.constant('Hello TensorFlow 2.0 !')
```




    <tf.Tensor: id=5, shape=(), dtype=string, numpy=b'Hello TensorFlow 2.0 !'>



#### Tensor property


```python
with tf.device('cpu'):
    a = tf.constant([1])
    b = tf.range(4)

with tf.device('gpu'):
    c = tf.constant([1])
    d = tf.range(4)
    e = tf.constant([1.0])
```


```python
a.device
```




    '/job:localhost/replica:0/task:0/device:CPU:0'




```python
b.device
```




    '/job:localhost/replica:0/task:0/device:CPU:0'



#### int ---> CPU ??


```python
c.device
```




    '/job:localhost/replica:0/task:0/device:CPU:0'




```python
cc = c.gpu()
cc.device
```




    '/job:localhost/replica:0/task:0/device:GPU:0'




```python
d.device
```




    '/job:localhost/replica:0/task:0/device:GPU:0'




```python
e.device
```




    '/job:localhost/replica:0/task:0/device:GPU:0'




```python
aa = a.gpu()
```


```python
aa.device
```




    '/job:localhost/replica:0/task:0/device:GPU:0'




```python
dd = d.cpu()
```


```python
dd.device
```




    '/job:localhost/replica:0/task:0/device:CPU:0'




```python
d.numpy()
```




    array([0, 1, 2, 3])




```python
d.ndim
```




    1




```python
tf.rank(d)
```




    <tf.Tensor: id=47, shape=(), dtype=int32, numpy=1>




```python
tf.rank(tf.ones([3, 4, 2]))
```




    <tf.Tensor: id=51, shape=(), dtype=int32, numpy=3>




```python
d.name
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-47-4672a43555ef> in <module>
    ----> 1 d.name


    AttributeError: Tensor.name is meaningless when eager execution is enabled.


#### Check Tensor Type


```python
import tensorflow as tf
import numpy as np

a = tf.constant([1.])
b = tf.constant([True, False])
c = tf.constant('hello tf')
d = np.arange(4)
```


```python
isinstance(a, tf.Tensor)
```




    True




```python
tf.is_tensor(a) & tf.is_tensor(b) & tf.is_tensor(c)
```




    True




```python
tf.is_tensor(d)
```




    False




```python
a.dtype, b.dtype, c.dtype, d.dtype
```




    (tf.float32, tf.bool, tf.string, dtype('int32'))




```python
a.dtype == tf.float32
```




    True




```python
c.dtype == tf.string
```




    True



#### Convert


```python
a = np.arange(5)
```


```python
a.dtype
```




    dtype('int32')




```python
aa = tf.convert_to_tensor(a)
```


```python
aa
```




    <tf.Tensor: id=59, shape=(5,), dtype=int32, numpy=array([0, 1, 2, 3, 4])>




```python
aa = tf.convert_to_tensor(a, dtype=tf.int64)
```


```python
aa
```




    <tf.Tensor: id=61, shape=(5,), dtype=int64, numpy=array([0, 1, 2, 3, 4], dtype=int64)>




```python
tf.cast(aa, dtype=tf.float32)
```




    <tf.Tensor: id=62, shape=(5,), dtype=float32, numpy=array([0., 1., 2., 3., 4.], dtype=float32)>




```python
aaa = tf.cast(aa, dtype=tf.double)
aaa
```




    <tf.Tensor: id=64, shape=(5,), dtype=float64, numpy=array([0., 1., 2., 3., 4.])>




```python
tf.cast(aaa, dtype=tf.int32)
```




    <tf.Tensor: id=65, shape=(5,), dtype=int32, numpy=array([0, 1, 2, 3, 4])>



#### bool & int


```python
b = tf.constant([0, 1, 2, 3, 4])
b
```




    <tf.Tensor: id=69, shape=(5,), dtype=int32, numpy=array([0, 1, 2, 3, 4])>




```python
bb = tf.cast(b, dtype=tf.bool)
bb
```




    <tf.Tensor: id=72, shape=(5,), dtype=bool, numpy=array([False,  True,  True,  True,  True])>




```python
bbb = tf.cast(bb, dtype=tf.int32)
bbb
```




    <tf.Tensor: id=73, shape=(5,), dtype=int32, numpy=array([0, 1, 1, 1, 1])>



#### tf.Variable


```python
a = tf.range(5)
a
```




    <tf.Tensor: id=77, shape=(5,), dtype=int32, numpy=array([0, 1, 2, 3, 4])>




```python
b = tf.Variable(a)
```


```python
b.dtype
```




    tf.int32




```python
b.name
```




    'Variable:0'




```python
b = tf.Variable(a, name='input_data')
```


```python
b.name
```




    'input_data:0'




```python
b.trainable
```




    True




```python
isinstance(b, tf.Tensor)
```




    False




```python
isinstance(b, tf.Variable)
```




    True




```python
tf.is_tensor(b)
```




    True



#### To numpy


```python
b
```




    <tf.Variable 'input_data:0' shape=(5,) dtype=int32, numpy=array([0, 1, 2, 3, 4])>




```python
b.numpy()
```




    array([0, 1, 2, 3, 4])




```python
a = tf.ones([])
```


```python
a.numpy()
```




    1.0




```python
int(a)
```




    1




```python
float(a)
```




    1.0



#### From Numpy, List


```python
tf.convert_to_tensor(np.ones([2, 3]))
```




    <tf.Tensor: id=112, shape=(2, 3), dtype=float64, numpy=
    array([[1., 1., 1.],
           [1., 1., 1.]])>




```python
tf.convert_to_tensor(np.zeros([2, 3]))
```




    <tf.Tensor: id=113, shape=(2, 3), dtype=float64, numpy=
    array([[0., 0., 0.],
           [0., 0., 0.]])>




```python
tf.convert_to_tensor([1, 2])
```




    <tf.Tensor: id=114, shape=(2,), dtype=int32, numpy=array([1, 2])>




```python
tf.convert_to_tensor([1, 2.])
```




    <tf.Tensor: id=115, shape=(2,), dtype=float32, numpy=array([1., 2.], dtype=float32)>




```python
tf.convert_to_tensor([[1], [2.]])
```




    <tf.Tensor: id=116, shape=(2, 1), dtype=float32, numpy=
    array([[1.],
           [2.]], dtype=float32)>



#### tf.zero


```python
tf.zeros([])
```




    <tf.Tensor: id=117, shape=(), dtype=float32, numpy=0.0>




```python
tf.zeros([1])
```




    <tf.Tensor: id=120, shape=(1,), dtype=float32, numpy=array([0.], dtype=float32)>




```python
tf.zeros([2, 2])
```




    <tf.Tensor: id=123, shape=(2, 2), dtype=float32, numpy=
    array([[0., 0.],
           [0., 0.]], dtype=float32)>




```python
tf.zeros([2, 3, 3])
```




    <tf.Tensor: id=126, shape=(2, 3, 3), dtype=float32, numpy=
    array([[[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]],

           [[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]]], dtype=float32)>



#### tf.zero_like


```python
a = tf.zeros([2, 3, 3])
```


```python
tf.zeros_like(a)
```




    <tf.Tensor: id=130, shape=(2, 3, 3), dtype=float32, numpy=
    array([[[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]],

           [[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]]], dtype=float32)>




```python
a.shape
```




    TensorShape([2, 3, 3])




```python
tf.zeros(a.shape)
```




    <tf.Tensor: id=133, shape=(2, 3, 3), dtype=float32, numpy=
    array([[[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]],

           [[0., 0., 0.],
            [0., 0., 0.],
            [0., 0., 0.]]], dtype=float32)>



#### tf.ones


```python
tf.ones(1)
```




    <tf.Tensor: id=136, shape=(1,), dtype=float32, numpy=array([1.], dtype=float32)>




```python
tf.ones([])
```




    <tf.Tensor: id=137, shape=(), dtype=float32, numpy=1.0>




```python
tf.ones([2])
```




    <tf.Tensor: id=140, shape=(2,), dtype=float32, numpy=array([1., 1.], dtype=float32)>




```python
tf.ones([2, 3])
```




    <tf.Tensor: id=143, shape=(2, 3), dtype=float32, numpy=
    array([[1., 1., 1.],
           [1., 1., 1.]], dtype=float32)>




```python
tf.ones_like(a)
```




    <tf.Tensor: id=146, shape=(2, 3, 3), dtype=float32, numpy=
    array([[[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]],

           [[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]]], dtype=float32)>



#### Fill


```python
tf.fill([2, 2], 0)
```




    <tf.Tensor: id=149, shape=(2, 2), dtype=int32, numpy=
    array([[0, 0],
           [0, 0]])>




```python
tf.fill([2, 2], 1.)
```




    <tf.Tensor: id=152, shape=(2, 2), dtype=float32, numpy=
    array([[1., 1.],
           [1., 1.]], dtype=float32)>



#### Normal


```python
tf.random.normal([2, 2], mean=1, stddev=1)
```




    <tf.Tensor: id=158, shape=(2, 2), dtype=float32, numpy=
    array([[0.85248226, 1.9835291 ],
           [0.7771325 , 1.0879767 ]], dtype=float32)>




```python
tf.random.normal([2, 2])
```




    <tf.Tensor: id=164, shape=(2, 2), dtype=float32, numpy=
    array([[ 0.9717742 , -0.06404414],
           [ 2.4790704 ,  0.444778  ]], dtype=float32)>




```python
tf.random.truncated_normal([2, 2], mean=0, stddev=1)
```




    <tf.Tensor: id=170, shape=(2, 2), dtype=float32, numpy=
    array([[ 0.03271687,  0.17312004],
           [-0.29089022,  0.84833103]], dtype=float32)>



#### Uniform


```python
tf.random.uniform([2, 2], minval=0, maxval=1)
```




    <tf.Tensor: id=19, shape=(2, 2), dtype=float32, numpy=
    array([[0.5943202 , 0.6143656 ],
           [0.5500653 , 0.32760215]], dtype=float32)>




```python
tf.random.uniform([2, 2], minval=0, maxval=100)
```




    <tf.Tensor: id=26, shape=(2, 2), dtype=float32, numpy=
    array([[97.74138 , 99.0658  ],
           [32.210064, 91.799614]], dtype=float32)>



#### Random Permuation


```python
idx = tf.range(10)
idx
```




    <tf.Tensor: id=34, shape=(10,), dtype=int32, numpy=array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])>




```python
idx = tf.random.shuffle(idx)
idx
```




    <tf.Tensor: id=36, shape=(10,), dtype=int32, numpy=array([2, 8, 1, 0, 5, 3, 6, 7, 4, 9])>




```python
a = tf.random.normal([10, 784])
a
```




    <tf.Tensor: id=66, shape=(10, 784), dtype=float32, numpy=
    array([[-0.0759943 ,  0.5909643 , -0.7259282 , ...,  0.34911963,
            -2.3549984 , -0.06266417],
           [ 0.38200593, -0.30257347,  0.05190112, ..., -0.55181193,
            -1.3061525 , -1.042753  ],
           [-2.795243  , -1.0049845 , -1.945913  , ...,  1.8921089 ,
             0.3384327 ,  0.07505874],
           ...,
           [ 1.3826874 ,  0.3215877 ,  1.5792335 , ..., -0.8602143 ,
            -2.22068   , -2.9438102 ],
           [ 1.4700025 ,  1.1525575 , -0.8278866 , ...,  1.0950248 ,
             0.496508  ,  0.93817395],
           [-0.6610565 ,  1.1690682 ,  0.01847431, ...,  0.5124489 ,
             1.8899206 ,  0.55972934]], dtype=float32)>




```python
b = tf.random.uniform([10], maxval=10, dtype=tf.int32)
b
```




    <tf.Tensor: id=70, shape=(10,), dtype=int32, numpy=array([5, 5, 8, 2, 7, 6, 8, 9, 7, 9])>




```python
a = tf.gather(a, idx)
a
```




    <tf.Tensor: id=72, shape=(10, 784), dtype=float32, numpy=
    array([[-2.795243  , -1.0049845 , -1.945913  , ...,  1.8921089 ,
             0.3384327 ,  0.07505874],
           [ 1.4700025 ,  1.1525575 , -0.8278866 , ...,  1.0950248 ,
             0.496508  ,  0.93817395],
           [ 0.38200593, -0.30257347,  0.05190112, ..., -0.55181193,
            -1.3061525 , -1.042753  ],
           ...,
           [ 1.3826874 ,  0.3215877 ,  1.5792335 , ..., -0.8602143 ,
            -2.22068   , -2.9438102 ],
           [-0.23002301,  1.2873333 , -0.43439117, ...,  0.79054   ,
            -0.33982682, -0.38860643],
           [-0.6610565 ,  1.1690682 ,  0.01847431, ...,  0.5124489 ,
             1.8899206 ,  0.55972934]], dtype=float32)>




```python
b = tf.gather(b, idx)
b
```




    <tf.Tensor: id=74, shape=(10,), dtype=int32, numpy=array([8, 7, 5, 5, 6, 2, 8, 9, 7, 9])>



#### tf.constant


```python
tf.constant(1)
```




    <tf.Tensor: id=75, shape=(), dtype=int32, numpy=1>




```python
tf.constant([1])
```




    <tf.Tensor: id=76, shape=(1,), dtype=int32, numpy=array([1])>




```python
tf.constant([1, 2.])
```




    <tf.Tensor: id=77, shape=(2,), dtype=float32, numpy=array([1., 2.], dtype=float32)>




```python
tf.constant([[1, 2.], [3.]])
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-22-2db47d890053> in <module>
    ----> 1 tf.constant([[1, 2.], [3.]])


    ValueError: Can't convert non-rectangular Python sequence to Tensor.


#### Loss


```python
out = tf.random.uniform([4, 10])
out
```




    <tf.Tensor: id=114, shape=(4, 10), dtype=float32, numpy=
    array([[0.39865232, 0.8146142 , 0.7466066 , 0.48542595, 0.09588099,
            0.1204927 , 0.53856754, 0.6199691 , 0.72104144, 0.31082737],
           [0.87862265, 0.32210338, 0.63155115, 0.23569906, 0.82659125,
            0.8151256 , 0.75029194, 0.8083484 , 0.3657925 , 0.33250725],
           [0.52463675, 0.9665301 , 0.7456732 , 0.61061203, 0.66697025,
            0.77250504, 0.05276549, 0.19811642, 0.74909556, 0.04108632],
           [0.46598256, 0.60937536, 0.23974884, 0.49564242, 0.09414244,
            0.8474517 , 0.62777615, 0.4825369 , 0.2641827 , 0.7198379 ]],
          dtype=float32)>




```python
y = tf.range(4)
y
```




    <tf.Tensor: id=118, shape=(4,), dtype=int32, numpy=array([0, 1, 2, 3])>




```python
y = tf.one_hot(y, depth=10)
y
```




    <tf.Tensor: id=122, shape=(4, 10), dtype=float32, numpy=
    array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]], dtype=float32)>




```python
loss = tf.keras.losses.mse(y, out)
loss
```




    <tf.Tensor: id=128, shape=(4,), dtype=float32, numpy=array([0.3132918 , 0.44943458, 0.32934332, 0.28422752], dtype=float32)>




```python
loss = tf.reduce_mean(loss)
loss
```




    <tf.Tensor: id=130, shape=(), dtype=float32, numpy=0.3440743>



#### Vector


```python
net = tf.keras.layers.Dense(10)
net
```




    <tensorflow.python.keras.layers.core.Dense at 0x3468dc08>




```python
net.build((4, 8))
```


```python
net.kernel
```


```python
net.bias
```




    <tf.Variable 'bias:0' shape=(10,) dtype=float32, numpy=array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)>



#### Matrix


```python
x = tf.random.normal([4, 784])
x
```




    <tf.Tensor: id=169, shape=(4, 784), dtype=float32, numpy=
    array([[-0.7347161 ,  0.4363023 , -0.75280005, ..., -0.04806641,
             0.1785243 , -0.24836487],
           [ 0.9954053 ,  0.14712416,  0.46067944, ...,  0.22743599,
             1.2375747 , -0.80820173],
           [-0.09152631,  2.3474286 , -0.33711952, ...,  0.5221496 ,
             0.14307927,  1.1803246 ],
           [-0.31603277,  0.15150416,  0.3816462 , ...,  1.4834361 ,
             0.7699948 , -1.0333261 ]], dtype=float32)>




```python
net = tf.keras.layers.Dense(10)
net.build((4, 784))
```


```python
net(x).shape
```




    TensorShape([4, 10])




```python
net.kernel.shape
```




    TensorShape([784, 10])




```python
net.bias.shape
```




    TensorShape([10])



#### Dim = 3 Tensor


```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
    17465344/17464789 [==============================] - 11s 1us/step



```python
x_train.shape
```




    (25000,)




```python
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=80)
```


```python
x_train.shape
```




    (25000, 80)



#### embedding?
emb = tf.keras.layers.embedding(x_train)

#### Dim = 4 Tensor


```python
x = tf.random.normal((4, 32, 32, 3))
```


```python
net = tf.keras.layers.Conv2D(16, kernel_size=3)
```


```python
net(x)
```




    <tf.Tensor: id=302, shape=(4, 30, 30, 16), dtype=float32, numpy=
    array([[[[-7.80818403e-01,  8.15214992e-01,  5.88986754e-01, ...,
            ],

     5.24938583e-01, -6.62822366e-01,  1.79501530e-02, ...,
               6.43505633e-01,  3.57730448e-01, -2.69943237e-01]]]],
          dtype=float32)>



#### Dim = 5 Tensor


```python
x = tf.random.normal((100, 4, 32, 32, 3))
```


```python
print(x.shape)
```

    (100, 4, 32, 32, 3)


#### Basic indexing


```python
a = tf.ones([1, 5, 5, 3])
```


```python
a[0][0]
```




    <tf.Tensor: id=326, shape=(5, 3), dtype=float32, numpy=
    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]], dtype=float32)>




```python
a[0][0][0]
```




    <tf.Tensor: id=338, shape=(3,), dtype=float32, numpy=array([1., 1., 1.], dtype=float32)>




```python
a[0][0][0][2]
```




    <tf.Tensor: id=354, shape=(), dtype=float32, numpy=1.0>



#### Numpy-style indexing


```python
a = tf.random.normal([4, 28, 28, 3])
```


```python
a[1].shape
```




    TensorShape([28, 28, 3])




```python
a[1, 2].shape
```




    TensorShape([28, 3])




```python
a[1, 2, 3].shape
```




    TensorShape([3])




```python
a[1, 2, 3, 2].shape
```




    TensorShape([])



#### Start:end


```python
a =tf.range(10)
a
```




    <tf.Tensor: id=392, shape=(10,), dtype=int32, numpy=array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])>




```python
a[-1:]
```




    <tf.Tensor: id=408, shape=(1,), dtype=int32, numpy=array([9])>




```python
a[-2:]
```




    <tf.Tensor: id=400, shape=(2,), dtype=int32, numpy=array([8, 9])>




```python
a[:2]
```




    <tf.Tensor: id=412, shape=(2,), dtype=int32, numpy=array([0, 1])>




```python
a[:-1]
```




    <tf.Tensor: id=416, shape=(9,), dtype=int32, numpy=array([0, 1, 2, 3, 4, 5, 6, 7, 8])>



#### Indexing by:


```python
a= tf.random.normal([4, 28, 28, 3])
a.shape
```




    TensorShape([4, 28, 28, 3])




```python
a[0].shape
```




    TensorShape([28, 28, 3])




```python
a[0,:,:,:].shape
```




    TensorShape([28, 28, 3])




```python
a[0,1,:,:].shape
```




    TensorShape([28, 3])




```python
a[:,:,:,0].shape
```




    TensorShape([4, 28, 28])




```python
a[:,:,:,2].shape
```




    TensorShape([4, 28, 28])




```python
a[:,0,:,:].shape
```




    TensorShape([4, 28, 3])



#### start​ : end : step


```python
a = tf.random.normal([4, 28, 28, 3])
```


```python
a[0:2,:,:,:].shape
```




    TensorShape([2, 28, 28, 3])




```python
a[:,0:28:2,0:28:2,:].shape
```




    TensorShape([4, 14, 14, 3])




```python
a[0:,14:,14:,:].shape
```




    TensorShape([4, 14, 14, 3])




```python
a[:,::2,::2,:].shape
```




    TensorShape([4, 14, 14, 3])



#### ::-1


```python
a = tf.range(4)
```


```python
a[::-1]
```




    <tf.Tensor: id=480, shape=(4,), dtype=int32, numpy=array([3, 2, 1, 0])>




```python
a[::-2]
```




    <tf.Tensor: id=484, shape=(2,), dtype=int32, numpy=array([3, 1])>




```python
a[2::-2]
```




    <tf.Tensor: id=488, shape=(2,), dtype=int32, numpy=array([2, 0])>




```python
a = tf.random.normal([2, 4, 28, 28, 3])
a[0].shape
```




    TensorShape([4, 28, 28, 3])



#### ...


```python
a[0,:,:,:,:].shape
```




    TensorShape([4, 28, 28, 3])




```python
a[0,...].shape
```




    TensorShape([4, 28, 28, 3])




```python
a[:,:,:,:,0].shape
```




    TensorShape([2, 4, 28, 28])




```python
a[0,...,2].shape
```




    TensorShape([4, 28, 28])




```python
a[1,0,...,0].shape
```




    TensorShape([28, 28])



#### tf.gather


```python
a = tf.random.normal([4, 35, 8])
```


```python
tf.gather(a, axis=0, indices=[2, 3]).shape
```




    TensorShape([2, 35, 8])




```python
a[2:4].shape
```




    TensorShape([2, 35, 8])




```python
tf.gather(a, axis=0, indices=[2, 1, 4, 0]).shape
```




    TensorShape([4, 35, 8])




```python
tf.gather(a, axis=1, indices=[2, 3, 7, 9, 16]).shape
```




    TensorShape([4, 5, 8])




```python
tf.gather(a, axis=2, indices=[2, 3, 7]).shape
```




    TensorShape([4, 35, 3])



#### tf.gather.nd


```python
a = tf.random.normal([4, 35, 8])
a.shape
```




    TensorShape([4, 35, 8])




```python
tf.gather_nd(a, [0]).shape # a[0]
```




    TensorShape([3, 4])




```python
tf.gather_nd(a, [0, 1]).shape # a[0, 1]
```




    TensorShape([4])




```python
tf.gather_nd(a, [0, 1, 2]).shape # a[0, 1, 2]
```




    TensorShape([])




```python
tf.gather_nd(a, [[0, 1, 2]]).shape # [ a[0, 1, 2] ]
```




    TensorShape([1])




```python
tf.gather_nd(a, [[0, 0],[1, 2]]).shape
```




    TensorShape([2, 8])




```python
tf.gather_nd(a, [[0,0],[1,1],[2,2]]).shape
```




    TensorShape([3, 8])




```python
tf.gather_nd(a, [[0,0,0],[1,1,1],[2,2,2]]).shape
```




    TensorShape([3])




```python
tf.gather_nd(a, [[[0,0,0],[1,1,1],[2,2,2]]]).shape
```




    TensorShape([1, 3])



Recommended indices format:

▪ [[0], [1],…]

▪ [[0,0], [1,1],…]

▪ [[0,0,0], [1,1,1],…]

#### tf.boolean_mask


```python
a = tf.random.normal([4, 28, 28, 3])
a.shape
```




    TensorShape([4, 28, 28, 3])




```python
tf.boolean_mask(a, mask=[True, True, False, False]).shape
```




    TensorShape([2, 28, 28, 3])




```python
tf.boolean_mask(a, mask=[True, True, False, False]).shape
```




    TensorShape([2, 28, 28, 3])




```python
tf.boolean_mask(a, mask=[True, True, False], axis=3).shape
```




    TensorShape([4, 28, 28, 2])




```python
b = tf.range(24)
b = tf.reshape(b,(2, 3, 4))
b
```




    <tf.Tensor: id=911, shape=(2, 3, 4), dtype=int32, numpy=
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],

           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])>




```python
tf.boolean_mask(b, mask=[True])
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-188-0aecea434f71> in <module>
    ----> 1 tf.boolean_mask(b, mask=[True])


    ValueError: Shapes (2,) and (1,) are incompatible



```python
tf.boolean_mask(b, mask=[True, False])
```




    <tf.Tensor: id=941, shape=(1, 3, 4), dtype=int32, numpy=
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]]])>




```python
tf.boolean_mask(b, mask=[[True, False]])
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-191-1a7109819c9c> in <module>
    ----> 1 tf.boolean_mask(b, mask=[[True, False]])


    ValueError: Shapes (2, 3) and (1, 2) are incompatible



```python
tf.boolean_mask(b, mask=[[True, False, False], [False, True, True]])
```




    <tf.Tensor: id=971, shape=(3, 4), dtype=int32, numpy=
    array([[ 0,  1,  2,  3],
           [16, 17, 18, 19],
           [20, 21, 22, 23]])>




```python
tf.boolean_mask(b, mask=[[[True, False, False, True]]])
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-196-8b903455c2c1> in <module>
    ----> 1 tf.boolean_mask(b, mask=[[[True, False, False, True]]])


    ValueError: Shapes (2, 3, 4) and (1, 1, 4) are incompatible



```python
tf.boolean_mask(b, mask=[[[True, False, False, True], [True, False, False, True], [True, False, False, True]], [[True, False, False, True], [True, False, False, True], [True, False, False, True]]])
```




    <tf.Tensor: id=1003, shape=(12,), dtype=int32, numpy=array([ 0,  3,  4,  7,  8, 11, 12, 15, 16, 19, 20, 23])>



Summary:

▪ The number of square bracks([]) represents the number of the dimensions

▪ The two shapes must be compatible

▪ if not, you will encounter the error: Shapes (2, 3, 4) and (1, 1, 4) are incompatible

#### reshape


```python
a = tf.random.normal([4, 28, 28, 3])
```


```python
a.shape, a.ndim
```




    (TensorShape([4, 28, 28, 3]), 4)




```python
b = tf.reshape(a, [4, 784, 3])
b.shape, b.ndim
```




    (TensorShape([4, 784, 3]), 3)




```python
b = tf.reshape(a, [4, -1, 3])
b.shape, b.ndim
```




    (TensorShape([4, 784, 3]), 3)




```python
b = tf.reshape(a, [4, -1])
b.shape, b.ndim
```




    (TensorShape([4, 2352]), 2)




```python
b = tf.reshape(a, [4, 784*3])
b.shape, b.ndim
```




    (TensorShape([4, 2352]), 2)




```python
b = tf.reshape(a, [-1, 3])
b.shape, b.ndim
```

#### reshape is flexible


```python
a = tf.random.normal([4, 28, 28, 3])
```


```python
b = tf.reshape(tf.reshape(a, [4, -1]), [4, 28, 28, 3]).shape
b
```




    TensorShape([4, 28, 28, 3])




```python
b = tf.reshape(tf.reshape(a, [4, -1]), [4, 14, 56, 3]).shape
b
```




    TensorShape([4, 14, 56, 3])




```python
b = tf.reshape(tf.reshape(a, [4, -1]), [4, 1, 784, 3]).shape
b
```




    TensorShape([4, 1, 784, 3])



#### reshape could lead to potential bugs

▪ A: images:     [4, 28, 28, 3]   [b, h, w, 3]

▪ B: reshape to: [4, 784, 3]      [b, pixel, 3]

▪ If you want to reshape to A:

▪ [4, 784, 3] --> [4, 28, 28, 3] & [b, h, w, 3]  [√]

▪ [4, 784, 3] --> [4, 14, 56, 3] & [b, h, w, 3]  [×]

▪ [4, 784, 3] --> [4, 28, 28, 3] & [b, w, h, 3]  [×]

#### tf.transpose


```python
a = tf.random.normal((4, 3, 2, 1))
a.shape
```




    TensorShape([4, 3, 2, 1])




```python
tf.transpose(a).shape
```




    TensorShape([1, 2, 3, 4])




```python
tf.transpose(a, perm=[0, 1, 3, 2]).shape
```




    TensorShape([4, 3, 1, 2])



[b, h, w, 3] --> [b, 3, h, w](TensorFlow --> PyTorchm)


```python
a = tf.random.normal([4, 28, 28, 3])
```


```python
b = tf.transpose(a, [0, 2, 1, 3])
b.shape
```




    TensorShape([4, 28, 28, 3])




```python
c = tf.transpose(a, [0, 3, 2, 1]) # [√]
c.shape
```




    TensorShape([4, 3, 28, 28])




```python
d = tf.transpose(a, [0, 3, 1, 2]) # [×]
d.shape
```




    TensorShape([4, 3, 28, 28])



#### Expand_dims


```python
a = tf.random.normal([4, 35, 8])
```


```python
tf.expand_dims(a, axis=0).shape
```




    TensorShape([1, 4, 35, 8])




```python
tf.expand_dims(a, axis=3).shape
```




    TensorShape([4, 35, 8, 1])




```python
tf.expand_dims(a, axis=-1).shape
```




    TensorShape([4, 35, 8, 1])




```python
tf.expand_dims(a, axis=-3).shape
```




    TensorShape([4, 1, 35, 8])



▪ axis (Left --> Right) [0, 1, 2, 3], then expand the dimension in the left

▪ axis (Rigth --> Left)  [-4, -3, -2, -1], then expand the dimension in the reight

#### Squeeze dim

Only squeeze for shape = 1 idm


```python
tf.squeeze(tf.zeros([1, 2, 1, 1, 3])).shape
```




    TensorShape([2, 3])




```python
a = tf.zeros([1, 2, 1, 3])
```


```python
tf.squeeze(a, axis=0).shape
```




    TensorShape([2, 1, 3])




```python
tf.squeeze(a, axis=-4).shape
```




    TensorShape([2, 1, 3])




```python
tf.squeeze(a, axis=2).shape
```




    TensorShape([1, 2, 3])




```python
tf.squeeze(a, axis=-2).shape
```




    TensorShape([1, 2, 3])



#### Broadcasting (from right)


```python
x = tf.random.normal([4, 32, 32, 3])
```


```python
(x + tf.random.normal([3])).shape
```




    TensorShape([4, 32, 32, 3])




```python
(x + tf.random.normal([32, 32, 1])).shape
```




    TensorShape([4, 32, 32, 3])




```python
(x + tf.random.normal([1,32, 32])).shape
```


    ---------------------------------------------------------------------------

    InvalidArgumentError                      Traceback (most recent call last)

    <ipython-input-6-47933ca14467> in <module>
    ----> 1 (x + tf.random.normal([1,32, 32])).shape


    InvalidArgumentError: Incompatible shapes: [4,32,32,3] vs. [1,32,32] [Op:AddV2] name: add/



```python
(x + tf.random.normal([32, 32, 1])).shape
```




    TensorShape([4, 32, 32, 3])



#### tf.broadcast_to


```python
x.shape
```




    TensorShape([4, 32, 32, 3])




```python
(x + tf.random.normal([4, 1, 1, 1])).shape
```




    TensorShape([4, 32, 32, 3])




```python
b = tf.broadcast_to(tf.random.normal([4, 1, 1, 1]), [4, 32, 32, 3])
```


```python
b.shape
```




    TensorShape([4, 32, 32, 3])



#### Broadcast vs Tile


```python
a = tf.ones([3, 4])
```


```python
a1 = tf.broadcast_to(a, [2, 3, 4])
a1
```




    <tf.Tensor: id=77, shape=(2, 3, 4), dtype=float32, numpy=
    array([[[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]],

           [[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]]], dtype=float32)>




```python
a2 = tf.expand_dims(a, axis=0)
a2
```




    <tf.Tensor: id=83, shape=(1, 3, 4), dtype=float32, numpy=
    array([[[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]]], dtype=float32)>




```python
a3 = tf.tile(a2, [2, 1, 1])
a3
```




    <tf.Tensor: id=87, shape=(2, 3, 4), dtype=float32, numpy=
    array([[[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]],

           [[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]]], dtype=float32)>




```python
a4 = tf.tile(a2, [2, 2, 1])
a4
```




    <tf.Tensor: id=89, shape=(2, 6, 4), dtype=float32, numpy=
    array([[[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]],

           [[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]]], dtype=float32)>




```python
a5 = tf.tile(a2, [2, 2, 2])
a5
```




    <tf.Tensor: id=91, shape=(2, 6, 8), dtype=float32, numpy=
    array([[[1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1.]],

           [[1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1.]]], dtype=float32)>



#### +-*/%//


```python
a = tf.fill([2, 2], 3.)
b = tf.fill([2, 2], 2.)
```


```python
a + b, a - b, a * b, a / b
```




    (<tf.Tensor: id=131, shape=(2, 2), dtype=float32, numpy=
     array([[5., 5.],
            [5., 5.]], dtype=float32)>,
     <tf.Tensor: id=132, shape=(2, 2), dtype=float32, numpy=
     array([[1., 1.],
            [1., 1.]], dtype=float32)>,
     <tf.Tensor: id=133, shape=(2, 2), dtype=float32, numpy=
     array([[6., 6.],
            [6., 6.]], dtype=float32)>,
     <tf.Tensor: id=134, shape=(2, 2), dtype=float32, numpy=
     array([[1.5, 1.5],
            [1.5, 1.5]], dtype=float32)>)




```python
a // b
```




    <tf.Tensor: id=135, shape=(2, 2), dtype=float32, numpy=
    array([[1., 1.],
           [1., 1.]], dtype=float32)>




```python
a % b
```




    <tf.Tensor: id=136, shape=(2, 2), dtype=float32, numpy=
    array([[1., 1.],
           [1., 1.]], dtype=float32)>



#### tf.math.log  tf.exp


```python
a = tf.ones([2, 2])
a
```




    <tf.Tensor: id=139, shape=(2, 2), dtype=float32, numpy=
    array([[1., 1.],
           [1., 1.]], dtype=float32)>




```python
tf.math.log(a)  # log e(a)
```




    <tf.Tensor: id=142, shape=(2, 2), dtype=float32, numpy=
    array([[0., 0.],
           [0., 0.]], dtype=float32)>




```python
tf.exp(a)
```




    <tf.Tensor: id=141, shape=(2, 2), dtype=float32, numpy=
    array([[2.7182817, 2.7182817],
           [2.7182817, 2.7182817]], dtype=float32)>



#### log2, log10 ?


```python
tf.math.log(8.) / tf.math.log(2.)
```




    <tf.Tensor: id=147, shape=(), dtype=float32, numpy=3.0>




```python
tf.math.log(100.) / tf.math.log(10.)
```




    <tf.Tensor: id=152, shape=(), dtype=float32, numpy=2.0>



#### pow, sqrt


```python
b = tf.fill([2, 2], 2.)
b
```




    <tf.Tensor: id=158, shape=(2, 2), dtype=float32, numpy=
    array([[2., 2.],
           [2., 2.]], dtype=float32)>




```python
tf.pow(b, 3)
```




    <tf.Tensor: id=160, shape=(2, 2), dtype=float32, numpy=
    array([[8., 8.],
           [8., 8.]], dtype=float32)>




```python
b ** 3
```




    <tf.Tensor: id=162, shape=(2, 2), dtype=float32, numpy=
    array([[8., 8.],
           [8., 8.]], dtype=float32)>




```python
tf.sqrt(b)
```




    <tf.Tensor: id=164, shape=(2, 2), dtype=float32, numpy=
    array([[1.4142135, 1.4142135],
           [1.4142135, 1.4142135]], dtype=float32)>



#### @ matmul


```python
a = tf.fill([2, 2], 1.)
a
```




    <tf.Tensor: id=170, shape=(2, 2), dtype=float32, numpy=
    array([[1., 1.],
           [1., 1.]], dtype=float32)>




```python
b = tf.fill([2, 2], 2.)
b
```




    <tf.Tensor: id=173, shape=(2, 2), dtype=float32, numpy=
    array([[2., 2.],
           [2., 2.]], dtype=float32)>




```python
a @ b
```




    <tf.Tensor: id=174, shape=(2, 2), dtype=float32, numpy=
    array([[4., 4.],
           [4., 4.]], dtype=float32)>




```python
tf.matmul(a, b)
```




    <tf.Tensor: id=175, shape=(2, 2), dtype=float32, numpy=
    array([[4., 4.],
           [4., 4.]], dtype=float32)>




```python
aa = tf.ones([4, 2, 3])
bb = tf.fill([4, 3, 5], 2.)
```


```python
aa @ bb
```




    <tf.Tensor: id=183, shape=(4, 2, 5), dtype=float32, numpy=
    array([[[6., 6., 6., 6., 6.],
            [6., 6., 6., 6., 6.]],

           [[6., 6., 6., 6., 6.],
            [6., 6., 6., 6., 6.]]], dtype=float32)>




```python
tf.matmul(aa, bb)
```




    <tf.Tensor: id=184, shape=(4, 2, 5), dtype=float32, numpy=
    array([[[6., 6., 6., 6., 6.],
            [6., 6., 6., 6., 6.]],

           [[6., 6., 6., 6., 6.],
            [6., 6., 6., 6., 6.]]], dtype=float32)>



#### With broadcasting


```python
a = tf.ones([4, 2, 3])
a
```




    <tf.Tensor: id=187, shape=(4, 2, 3), dtype=float32, numpy=
    array([[[1., 1., 1.],
            [1., 1., 1.]],

           [[1., 1., 1.],
            [1., 1., 1.]],

           [[1., 1., 1.],
            [1., 1., 1.]],

           [[1., 1., 1.],
            [1., 1., 1.]]], dtype=float32)>




```python
b = tf.fill([3, 5], 2.)
b
```




    <tf.Tensor: id=199, shape=(3, 5), dtype=float32, numpy=
    array([[2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.],
           [2., 2., 2., 2., 2.]], dtype=float32)>




```python
bb = tf.broadcast_to(b, [4, 3, 5])
```


```python
a @ bb
```




    <tf.Tensor: id=202, shape=(4, 2, 5), dtype=float32, numpy=
    array([[[6., 6., 6., 6., 6.],
            [6., 6., 6., 6., 6.]],

           [[6., 6., 6., 6., 6.],
            [6., 6., 6., 6., 6.]],

           [[6., 6., 6., 6., 6.],
            [6., 6., 6., 6., 6.]],

           [[6., 6., 6., 6., 6.],
            [6., 6., 6., 6., 6.]]], dtype=float32)>



#### Recap

Y = X @ W + b


```python
x = tf.ones([4, 2])
w = tf.ones([2, 1])
b = tf.constant(0.1)
```


```python
y = x @ w + b
y
```




    <tf.Tensor: id=215, shape=(4, 1), dtype=float32, numpy=
    array([[2.1],
           [2.1],
           [2.1],
           [2.1]], dtype=float32)>



out = relu(X @ W + b)


```python
out = tf.nn.relu(y)
out
```




    <tf.Tensor: id=217, shape=(4, 1), dtype=float32, numpy=
    array([[2.1],
           [2.1],
           [2.1],
           [2.1]], dtype=float32)>

### 2. Forward propagation

`Q: Build a forward network from scratch`

- Model: [b, 28, 28] --> [b, 784] --> [b, 512] --> [b, 256] --> [b, 10]
- x: [b, 28, 28] --> [b, 784]
- h1 =  x @ w1 + b1      ---> relu    [b, 512]
- h2 = h1 @ w2 + b2     ---> relu    [b, 256]
- h3 = h2 @ w3 + b3                      [b, 10]

`Framework:  Model --> Loss --> Gradient decent`

1. dataset
2. create model
3. out = model(x)
4. loss = mse(out - y)
5. gradient decent



```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras import datasets

(x, y), _ = datasets.mnist.load_data()

# Err 1: x = tf.convert_to_tensor(x, dtype=tf.float32)
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255
y = tf.convert_to_tensor(y, dtype=tf.int32)

y_one_hot = tf.one_hot(y, depth=10)
db = tf.data.Dataset.from_tensor_slices((x, y_one_hot)).batch(100)

# init w, b
# w1 = tf.Variable(tf.random.truncated_normal([784, 512]))
# b1 = tf.Variable(tf.zeros([512]))
# w2 = tf.Variable(tf.random.truncated_normal([512, 256]))
# b2 = tf.Variable(tf.zeros([256]))
# w3 = tf.Variable(tf.random.truncated_normal([256, 10]))
# b3 = tf.Variable(tf.zeros([10]))

w1 = tf.Variable(tf.random.truncated_normal([784, 512], stddev=0.1))
b1 = tf.Variable(tf.zeros([512]))
w2 = tf.Variable(tf.random.truncated_normal([512, 256], stddev=0.1))
b2 = tf.Variable(tf.zeros([256]))
w3 = tf.Variable(tf.random.truncated_normal([256, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3

for epoch in range(10):
    for step, (x, y) in enumerate(db):
        # x: [b, 28, 28] --> [b, 784]
        # h1 =  x @ w1 + b1     ---> relu    [b, 512]
        # h2 = h1 @ w2 + b2     ---> relu    [b, 256]
        # h3 = h2 @ w3 + b3                  [b, 10]

        x = tf.reshape(x, [-1, 28 * 28])

        with tf.GradientTape() as tape:  # tf.Variable
            h1 = x @ w1 + b1
            h1 = tf.nn.relu(h1)
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)
            h3 = h2 @ w3 + b3

            # Err 2: tf.reduce_sum() returns a scalar
            # loss = tf.reduce_mean(tf.reduce_sum((h3 - y) ** 2))
            loss = tf.reduce_mean((h3 - y) ** 2)

        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])

        # w1 = w1 - lr * grads[0]  # w1: tf.Variable --> tensor
        # b1 = b1 - lr * grads[1]
        # w2 = w2 - lr * grads[2]
        # b2 = b2 - lr * grads[3]
        # w3 = w3 - lr * grads[4]
        # b3 = b3 - lr * grads[5]

        w1.assign_sub(lr * grads[0])  # w1: tf.Variable will not be changed
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print('epoch:', epoch, ' step:', step, 'loss =', float(loss))
```

```
epoch: 0  step: 0 loss = 1.0857837200164795
...
epoch: 9  step: 500 loss = 0.10089235007762909
```

---
