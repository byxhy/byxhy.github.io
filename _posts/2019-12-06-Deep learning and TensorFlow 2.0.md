---
layout: post
title: "Deep learning and TensorFlow 2.0"
author: "Xhy"
tags: [ML]
categories: Machine Learning
image: TF.jpg
---


> At TensorFlow Dev Summit 2019, the TensorFlow team introduced the Alpha version of TensorFlow 2.0

<br />



## `Table of Contents`

* [First impressions of deep learning][1]
* [Standard installation method][2]
* [Regression problems][3]
* [Build the model][4]
* [Compilation and training][5]
* [Evaluate the model][6]

[1]: #1
[2]: #2
[3]: #3
[4]: #4
[5]: #5
[6]: #6



<br />



<h2 id="1">[ 1. First impressions of deep learning ]</h2>
### Lesson 1 - Tutorial

- After the course, try to come up with some new ideas.
- Try to use deep learning to solve some problems in your life



### Lesson 2 - Framework of deep learning - 1

- Tensorflow
  - V_0.1 (2015.9)
  - V_1.0 (2017.2)
  - V_2.0 (2019.9)
- Scikit-learn
  - Machine learning, No GPU
- Torch
  - Lua
- Caffe
  - 2013, The first framework for deep learning
  - No auto-grad, C++
  - Facebook, Caffe2 -> PyTorch
  - Torch -> PyTorch
- Keras
  - wrapper
- Teano
  - difficult to develop and debug
  - Google, TensorFlow
  - -> TensorFlow2

- Chainer(Japan)
- MXNet



### Lesson 3 - Framework of deep learning - 2

Forget TensorFlow 1.0 and start with TensorFlow 2.0.



### Lesson 4 - Install Anaconda

![png](/assets/img/TF2.0/Anaconda.png)

[Anaconda Download](https://www.anaconda.com/distribution/)



### Lesson 5 - Install TensorFlow 2.0

```
▪ conda create -n tf2 tensorflow-gpu
▪ conda activate tf2
```



- *If you cannot connect the official source, you can switch to the domestic source*

  Configure the domestic source

  ```
  ▪ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  ▪ conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
  ```

  Display the source address

  ```
  ▪ conda config --set show_channel_urls yes
  ```

  Install the TensorFlow

  ```
  ▪ conda create -n tf2 tensorflow-gpu
  ▪ conda activate tf2
  ```

- *Check your python  and ipython environment*(**Where**)

```
λ activate tf2                                                     

(tf2) λ where python
D:\ProgramData\Anaconda3\envs\tf2\python.exe
D:\ProgramData\Anaconda3\python.exe
C:\Python27\python.exe


(tf2) λ where ipython                                             
D:\ProgramData\Anaconda3\Scripts\ipython.exe                       
```



- *We found that there is no ipython in tf2 virtual environment, so we install it*

```
(tf2) λ pip install ipython
```



```
(tf2) λ where ipython
D:\ProgramData\Anaconda3\envs\tf2\Scripts\ipython.exe
D:\ProgramData\Anaconda3\Scripts\ipython.exe
```



- *Check TensorFlow version is gpu or not*

```
(tf2) λ ipython           
In [1]: import tensorflow as tf                              
In [2]: tf.test.is_gpu_available()     
Out[2]: True      
```



### Lesson6 - Install PyCharm

![png](/assets/img/TF2.0/PyCharm.png)

[PyCharm Download](https://www.jetbrains.com/pycharm/)



Add Python Interpreter

![png](/assets/img/TF2.0/Interpreter.png)

Test

![png](/assets/img/TF2.0/PyCharm_Test.png)



<br />



<h2 id="2">[ 2. Standard installation method ]</h2>
But I still recommend you use conda to install TensorFlow.(Like the tutorial above)



( Lesson7 - Lesson14 )



<br />



<h2 id="3">[ 3. Regression problems ]</h2>
### Lesson15 - Linear Regression - 1

- Linear Regression
- Logistic Regression
- Classification

### Lesson16 - Linear Regression - 2

### Lesson17 - Regression problem practice - 1

### Lesson18 - Regression problem practice - 1



```python
import numpy as np

#1. Calculate the mean square error function
def loss_caculate_func(w, b, points):
    x = points[:, 0]
    y_true = points[:, 1]

    y_pred = w * x + b
    #  MSE = sum(y_pre - y_real) ** 2 / N
    loss = np.sum((y_pred - y_true) ** 2) / float(len(points))

    return loss

#2. Caculate gradient
def step_gradient(points, w_init, b_init, learning_rate):
    # y = wx+b
    # grad_w = 2 * (wx + b - y) * x
    # grad_b = 2 * (wx + b - y) * 1

    N = float(len(points))
    grad_w = np.sum((2 / N) * (w_init * points[:, 0] + b_init - points[:, 1]) * points[:, 0])
    grad_b = np.sum((2 / N) * (w_init * points[:, 0] + b_init - points[:, 1]))

    w_new = w_init - learning_rate * grad_w
    b_new = b_init - learning_rate * grad_b

    return w_new, b_new

#3. Gradient decent
def gradient_decent(points, w_init, b_init, learning_rate, num_iterations):
    w = w_init
    b = b_init

    for i in range(num_iterations):
        w, b = step_gradient(np.array(points), w, b, learning_rate)

    return w, b

#4. Processing flow
    #4.1 read data
    #4.2 set the initial w, b value and caculate the first loss value
    #4.3 gradient decent
    #4.4 caculate the last loss value

def run():
    points = np.genfromtxt("data.csv", delimiter=",")

    w_init = 0
    b_init = 0
    lr = 0.0001
    # lr = 0.001 [Err: invalid value encountered in double_scalars]
    num_iterations = 100

    loss_before = loss_caculate_func(w_init, b_init, points)
    print('loss_before =', loss_before)

    w, b = gradient_decent(points, w_init, b_init, lr, num_iterations)

    loss_after = loss_caculate_func(w, b, points)
    print('loss_after  =', loss_after)

    print('w =', w, ' b =', b)

if __name__ == "__main__":
    run()

```

```
D:\ProgramData\Anaconda3\envs\tf2\python.exe "E:/M/PycharmProjects/Lesson4 - Linear Regression/Linear regression.py"
loss_before = 5565.107834483214
loss_after  = 112.64705664288809
w = 1.4788027175308358  b = 0.03507497059234176
```



### Lesson23 - Handwritten digit recognition problem - 2

*Imagine being given a blank sheet of paper to write down a solution to this problem. What would you do ?*



1. Q: Use TensorFlow to recognize the number in the picture of MNIST dataset
   - background
     - why : In order to master the skill of deep learning
     - diff: I have the class, but I want to do it by myself
   - nature : deep learning model, gradient decent method, classification
   - medicine : TensorFlow now
   -  target : Starting from scratch
2. Multidimensional and critical thinking:
   - divergent
3. Framework: You must have a global view of the problem
   - input  --->  hidden  -->  output
4. Execute:
5. Check / update:
   - Before: input  --->  hidden  -->  output
   - After:  loss   --->  gradient ->  decent
6. Summary:  `[loss   --->  gradient ->  decent]`
   - Normalize the train data
   - label to one-hot
   - x = tf.reshape(x, (-1, 28 * 28)
   - step vs epoch

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



<br />



<h2 id="4">[ 4. TensorFlow 2.0 basic operations ]</h2>
Try to use Jupyter Notebook  to practice these exercises. 



*If you don't set the Jupyter Notebook environment, you may meet this problem*


```python
import tensorflow as tf
```

    -------------------------------------------------------------------------
    ModuleNotFoundError                     Traceback (most recent call last)
    
    <ipython-input-1-64156d691fe5> in <module>
    ----> 1 import tensorflow as tf


    ModuleNotFoundError: No module named 'tensorflow'



*So before you start your notebook, you should add your TensorFlow2.0 environment first.*

```
source activate myenv

conda install ipykernel

python -m ipykernel install --user --name myenv --display-name "Python (myenv)"

juputer notebook
```

