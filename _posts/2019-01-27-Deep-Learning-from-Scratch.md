---
layout: post
title: "Deep Learning from Scratch"
author: "Xhy"
categories: Deep-Learning
tags: [improve]
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
run -i './deep-learning-from-scratch/ch01/hungry.py'
```

    I'm hungry!


#### class


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


![png](output_72_0.png)



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


![png](output_73_0.png)


#### imshow


```python
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('./deep-learning-from-scratch/dataset/lena.png')
plt.imshow(img)
plt.show()
```


![png](output_75_0.png)


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




```python

```
