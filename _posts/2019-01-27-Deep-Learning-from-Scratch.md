---
layout: post
title: "Deep Learning from Scratch"
author: "Xhy"
categories: Deep-Learning
tags: [Machine Learning]
image: james-pond.jpg
---

Photo by james-pond


>那么,怎么才能更深入地理解深度学习呢?在笔者看来,最好的办法就
是亲自实现。从零开始编写可实际运行的程序,一边看源代码,一边思考。
笔者坚信,这种做法对正确理解深度学习(以及那些看上去很高级的技术)
是很重要的。 --斋藤康毅

<br />



### Deep Learning from Scratch

#### Chapter 1 – Introduction to Python

#### 1.1 What is python

#### 1.2 Install python

#### 1.3 The python interpreter

#### Arithmetic operations


```python
1 - 2
```




    -1




```python
4 * 5
```




    20




```python
7 /5
```




    1.4




```python
3 ** 2
```




    9



#### Variable


```python
x = 100
print(x)
```

    100



```python
y = 3.14
x * y
```




    314.0




```python
type(x * y)
```




    float



#### list


```python
a = [1, 2, 3, 4, 5]
print(a)
```

    [1, 2, 3, 4, 5]



```python
len(a)
```




    5




```python
type(a)
```




    list




```python
a[0]
```




    1




```python
a[4]
```




    5




```python
a[0:2]  # Star
```




    [1, 2]




```python
a[1:]
```




    [2, 3, 4, 5]




```python
a[:3]
```




    [1, 2, 3]




```python
a[:-1] #The last element
```




    [1, 2, 3, 4]




```python
a[:-2]
```




    [1, 2, 3]



#### dict


```python
me = {'height':190}
me['height']
```




    190




```python
type(me)
```




    dict




```python
me['weight'] = 70
print(me)
```

    {'height': 190, 'weight': 70}


#### bool


```python
hungry = True
sleepy = False
type(hungry)
```




    bool




```python
not hungry
```




    False




```python
hungry and sleepy
```




    False




```python
hungry or sleepy
```




    True



#### if


```python
hungry = True
if hungry:
    print("I\'m hungry")
```

    I'm hungry



```python
hungry = False
if hungry:
    print("I\'m hungry")
else:
    print("I\'m not hungry")        
```

    I'm not hungry


#### for


```python
for i in [1, 2, 3]:
    print(i)
```

    1
    2
    3


#### function


```python
def hello():
    print("Say hi ~")
```


```python
hello()
```

    Say hi ~


#### 1.4 Script


```python
print("I\'m hungry!")
```

    I'm hungry!



```python
run -i './deep-learning-from-scratch/ch01/hungry.py'
```

    I'm hungry!


#### class


```python
class Man:
    def __init__(self, name1, name2):
        self.name1 = name1
        self.name2 = name2
        print("Initialized!")

    def hello(self):
        print("Hello " + self.name1 + "!")

    def goodby(self):
        print("Goodby " + self.name1 + "!")

    def Goodby2(self):        
        print("Goodby " + self.name2 + "!")

m = Man("GouDan", "ErYa")
m.hello()
m.goodby()
m.Goodby2()
```

    Initialized!
    Hello GouDan!
    Goodby GouDan!
    Goodby ErYa!



```python
run -i './deep-learning-from-scratch/ch01/man.py'
```

    Initialized!
    Hello GouDan!
    Goodby GouDan!
    Goodby ErYa!


#### 1.5 Numpy


```python
import numpy as np
x = np.array([1.0, 2.0, 3.0])
print(x)
```

    [1. 2. 3.]



```python
type(x)
```




    numpy.ndarray




```python
y = np.array([3.0, 2.0, 1.0])
x + y
```




    array([4., 4., 4.])




```python
x - y
```




    array([-2.,  0.,  2.])




```python
x * y
```




    array([3., 4., 3.])




```python
x / y
```




    array([0.33333333, 1.        , 3.        ])




```python
x / 2
```




    array([0.5, 1. , 1.5])



#### N-Dimention array


```python
A = np.array([[1, 2], [3, 4]])
print(A)
```

    [[1 2]
     [3 4]]



```python
A.shape
```




    (2, 2)




```python
A.dtype
```




    dtype('int64')




```python
B = np.array([[3, 0], [0, 6]])
A + B
```




    array([[ 4,  2],
           [ 3, 10]])




```python
A * B
```




    array([[ 3,  0],
           [ 0, 24]])




```python
print(A)
```

    [[1 2]
     [3 4]]



```python
A * 10
```




    array([[10, 20],
           [30, 40]])




```python
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
A * B
```




    array([[10, 40],
           [30, 80]])




```python
x = np.array([[51, 55], [14, 19], [0, 4]])
print(x)
```

    [[51 55]
     [14 19]
     [ 0  4]]



```python
x[0]
```




    array([51, 55])




```python
x[0][1]
```




    55




```python
for row in x:
    print(row)
```

    [51 55]
    [14 19]
    [0 4]



```python
x = x.flatten()
print(x)
```

    [51 55 14 19  0  4]



```python
x[np.array([0, 2, 4])]
```




    array([51, 14,  0])




```python
x > 20
```




    array([ True,  True, False, False, False, False])




```python
x[x > 20]
```




    array([51, 55])



#### 1.6 Matplotblib


```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show()
```


![png](/assets/img/Deep Learning from Scratch/output_74_0.png)



```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)
y1 = np.sin(x)

y2 = np.cos(x)

plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle = "--", label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin & cos')
plt.legend()
plt.show()
```


![png](/assets/img/Deep Learning from Scratch/output_75_0.png)


#### imshow


```python
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('./deep-learning-from-scratch/dataset/lena.png')
plt.imshow(img)
plt.show()
```


![png](/assets/img/Deep Learning from Scratch/output_77_0.png)


#### Get More - https://www.scipy-lectures.org/

#### Chapter 2 – Perceptron

#### 2.1 What is perceptron

#### 2.2 Simple logic circuit

#### 2.3 Perceptron implementation

#### AND


```python
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp =  x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
```


```python
AND(0, 0)
```




    0




```python
AND(1, 0)
```




    0




```python
AND(0, 1)
```




    0




```python
AND(1, 1)
```




    1



#### Import weights and offsets


```python
import numpy as np
x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7

w * x
```




    array([0. , 0.5])




```python
np.sum(w * x)
```




    0.5




```python
np.sum(w * x) + b
```




    -0.19999999999999996



#### Perceptron implementation2

#### AND


```python
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
```


```python
AND(0, 0)
```




    0




```python
AND(1, 0)
```




    0




```python
AND(0, 1)
```




    0




```python
AND(1, 1)
```




    1



#### NAND


```python
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
```


```python
NAND(0, 0)
```




    1




```python
NAND(1, 0)
```




    1




```python
NAND(0, 1)
```




    1




```python
NAND(1, 1)
```




    0



#### OR


```python
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
```


```python
OR(0, 0)
```




    0




```python
OR(1, 0)
```




    1




```python
OR(0, 1)
```




    1




```python
OR(1, 1)
```




    1



#### 2.5 multi-layered perceptron

#### XOR


```python
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y  = AND(s1, s2)
    return y
```


```python
XOR(0, 0)
```




    0




```python
XOR(0, 1)
```




    1




```python
XOR(1, 0)
```




    1




```python
XOR(1, 1)
```




    0



#### Chapter 3 – Neural network

#### 3.1 From perceptron to neural network

#### 3.2 Activation function

#### 3.2.3 Activation function - step function


```python
import numpy as np
import matplotlib.pylab as plt

def step_funcion(x):
    return np.array(x>0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_funcion(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.title('step function')
plt.show()
```


![png](/assets/img/Deep Learning from Scratch/output_123_0.png)


#### 3.2.4 Activation function - sigmoid function


```python
import numpy as np
import matplotlib.pylab as plt

def sigmoid_function(x):
    return 1 /( 1 + np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.title('sigmoid function')
plt.show()
```


![png](/assets/img/Deep Learning from Scratch/output_125_0.png)


#### step function VS sigmoid function


```python
import numpy as np
import matplotlib.pyplot as plt

def step_funcion(x):
    return np.array(x>0, dtype=np.int)

def sigmoid_function(x):
    return 1 /( 1 + np.exp(-x))

x  = np.arange(-5.0, 5.0, 0.1)

y1 = step_funcion(x)

y2 = sigmoid_function(x)

plt.plot(x, y1, label="step")
plt.plot(x, y2, linestyle = "--", label="sigmoid")
plt.xlabel("x")
plt.ylabel("y")
plt.title('step & sigmoid')
plt.legend()
plt.show()
```


![png](/assets/img/Deep Learning from Scratch/output_127_0.png)


#### 3.2.7 Activation function - ReLU function


```python
import numpy as np
import matplotlib.pylab as plt

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)

plt.plot(x, y)
plt.ylim(-1, 5.5)
plt.title('ReLU function')
plt.show()
```


![png](/assets/img/Deep Learning from Scratch/output_129_0.png)


#### 3.3 N dimension array operation


```python
import numpy as np

A = np.array([1, 2, 3, 4])
print(A)
```

    [1 2 3 4]



```python
np.ndim(A)
```




    1




```python
A.shape
```




    (4,)




```python
A.shape[0]
```




    4




```python
A.shape[1]
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-31-b42afa0c404d> in <module>
    ----> 1 A.shape[1]


    IndexError: tuple index out of range



```python
B = np.array([[1,2], [3,4], [5,6]])
print(B)
```

    [[1 2]
     [3 4]
     [5 6]]



```python
np.ndim(B)
```




    2




```python
B.shape
```




    (3, 2)



#### Matrix multiplication


```python
A = np.array([[1,2], [3,4]])
A.shape
```




    (2, 2)




```python
B = np.array([[5,6], [7,8]])
B.shape
```




    (2, 2)




```python
np.dot(A, B)
```




    array([[19, 22],
           [43, 50]])




```python
A = np.array([[1,2,3], [4,5,6]])
A.shape
```




    (2, 3)




```python
B = np.array([[1,2], [3,4], [5,6]])
B.shape
```




    (3, 2)




```python
np.dot(A, B)
```




    array([[22, 28],
           [49, 64]])




```python
C = np.array([[1,2], [3,4]])
C.shape
```




    (2, 2)




```python
np.dot(A, C)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-43-bb5afb89b162> in <module>
    ----> 1 np.dot(A, C)


    ValueError: shapes (2,3) and (2,2) not aligned: 3 (dim 1) != 2 (dim 0)


#### A is two dimension and B is one dimentsion


```python
A = np.array([[1,2], [3, 4], [5,6]])
A.shape
```




    (3, 2)




```python
B = np.array([7,8])
B.shape
```




    (2,)




```python
np.dot(A, B)
```




    array([23, 53, 83])




```python

```


```python
import numpy as np

X = np.array([1, 2])
X.shape
```




    (2,)




```python
W = np.array([[1, 3, 5], [2, 4, 6]])
print(W)
```

    [[1 3 5]
     [2 4 6]]



```python
W.shape
```




    (2, 3)




```python
Y = np.dot(X, W)
print(Y)
```

    [ 5 11 17]



```python

```


```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```


```python
X  = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)
```

    (2, 3)
    (2,)
    (3,)



```python
A1 = np.dot(X, W1) + B1
print(A1)
```

    [0.3 0.7 1.1]



```python
Z1 = sigmoid(A1)
print(Z1)
```

    [0.57444252 0.66818777 0.75026011]



```python
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)
print(W2.shape)
print(B2.shape)
```

    (3,)
    (3, 2)
    (2,)



```python
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print(Z2)
```

    [0.62624937 0.7710107 ]



```python
def identity_function(x):
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)
print(Y)
```

    [0.31682708 0.69627909]


##### Code Summary


```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(x):
    return x

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(Z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(Z2, W3) + b3
    y  = identity_function(a3)

    return y
```


```python
network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
```

    [0.31682708 0.69627909]


##### Softmax function


```python
a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
print(exp_a)
```

    [ 1.34985881 18.17414537 54.59815003]



```python
sum_exp_a = sum(exp_a)
print(sum_exp_a)
```

    74.1221542101633



```python
y = exp_a / sum_exp_a
print(y)
```

    [0.01821127 0.24519181 0.73659691]



```python
def softmax(x):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
```


```python
print(softmax(a))
```

    [0.01821127 0.24519181 0.73659691]


##### softmax - matters need attention


```python
a = np.array([1010, 1000, 990])
np.exp(a) / np.sum(np.exp(a))
```

    /home/xhy/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:2: RuntimeWarning: overflow encountered in exp

    /home/xhy/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:2: RuntimeWarning: invalid value encountered in true_divide






    array([nan, nan, nan])




```python

```


```python
c = np.max(a)
a - c
```




    array([  0, -10, -20])




```python
np.exp(a - c) / np.sum(np.exp(a - c))
```




    array([9.99954600e-01, 4.53978686e-05, 2.06106005e-09])



#### So the best softmax function is:


```python
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
```


```python
a = np.array([1010, 1000, 990])
print(softmax(a))
```

    [9.99954600e-01 4.53978686e-05 2.06106005e-09]



```python
a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
```

    [0.01821127 0.24519181 0.73659691]



```python
np.sum(y)
```




    1.0



#### 3.6.1 MNIST


```python
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist   # Must be the same level as the parent directory of load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False) # Change the vps to Global Model
```

    Downloading train-images-idx3-ubyte.gz ...
    Done
    Converting train-images-idx3-ubyte.gz to NumPy Array ...
    Done
    Converting train-labels-idx1-ubyte.gz to NumPy Array ...
    Done
    Converting t10k-images-idx3-ubyte.gz to NumPy Array ...
    Done
    Converting t10k-labels-idx1-ubyte.gz to NumPy Array ...
    Done
    Creating pickle file ...
    Done!



```python
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(x_test.shape)
```

    (60000, 784)
    (60000,)
    (10000, 784)
    (10000, 784)





#### Show the picture


```python
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
```

    5
    (784,)
    (28, 28)



![png](/assets/img/Deep Learning from Scratch/ShowThePicture_ScreenShot.png)



#### 3.6.2 Neural network reasoning


```python
import pickle
from dataset.mnist import load_mnist
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a    
    return y

def get_data():
    (x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("./deep-learning-from-scratch/ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
        return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y
```


```python
x, t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) # Get the index of the element with the highest probability
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
```

    Accuracy:0.9352



#### 3.6.3 Processing Batch


```python
x, _ = get_data()
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']
```


```python
x.shape
```




    (10000, 784)




```python
x[0].shape
```




    (784,)




```python
W1.shape
```




    (784, 50)




```python
W2.shape
```




    (50, 100)




```python
W3.shape
```




    (100, 10)



#### Batch implementation


```python
x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
```

    Accuracy:0.9352



```python
list( range(0, 10) )
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]




```python
list( range(0, 10, 3) )
```




    [0, 3, 6, 9]




```python
x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6],[0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
y = np.argmax(x, axis=1)
print(y)
```

    [1 2 1 0]



```python
y = np.array([1, 2, 1, 0])
t = np.array([1, 2, 0, 0])
print(y==t)
np.sum(y==t)
```

    [ True  True False  True]

    3


---


![png](/assets/img/Deep Learning from Scratch/output_129_0.png)
