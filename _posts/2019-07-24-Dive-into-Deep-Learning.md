---
layout: post
title: "Dive into Deep Learning"
author: "Xhy"
categories: Machine Learning
tags: [ML]
image: DiDeepLearning.jpg
---

Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola


>In just the past five years, deep learning has taken the world by surprise, driving rapid progress in fields as diverse as computer vision, natural language processing, automatic speech recognition, reinforcement learning, and statistical modeling. With these advances in hand, we can now build cars that drive themselves (with increasing autonomy), smart reply systems that anticipate mundane replies, helping people dig out from mountains of email, and software agents that dominate the world’s best humans at board games like Go, a feat once deemed to be decades away. Already, these tools are exerting a widening impact, changing the way movies are made, diseases are diagnosed, and playing a growing role in basic sciences – from astrophysics to biology. This book represents our attempt to make deep learning approachable, teaching you both the concepts, the context, and the code.

<br />

# Dive into Deep Learning

## Charpter 1 - Deep Learning introduction

## Charpter 2 - Propaedeutics

### 2.1 - How to get source data and install the environment

### 2.2 - Data manipunation

#### 2.2.1 - Create NDArray

```python
from mxnet import nd
```

```python
x = nd.arange(12)
x
```



​    

```
[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11.]
<NDArray 12 @cpu(0)>
```



```python
x.shape
```



```
(12,)
```



```python
x.size
```



```
12
```



```python
x_reshape = x.reshape((3, 4))
x_reshape
```



​    

```
[[ 0.  1.  2.  3.]
 [ 4.  5.  6.  7.]
 [ 8.  9. 10. 11.]]
<NDArray 3x4 @cpu(0)>
```



#### if you don't know the number of rows and columns, just fill in  -1

```python
x1_reshape = x.reshape((-1, 4))
x1_reshape
```



​    

```
[[ 0.  1.  2.  3.]
 [ 4.  5.  6.  7.]
 [ 8.  9. 10. 11.]]
<NDArray 3x4 @cpu(0)>
```



```python
x2_reshape = x.reshape((2, -1))
x2_reshape
```



​    

```
[[ 0.  1.  2.  3.  4.  5.]
 [ 6.  7.  8.  9. 10. 11.]]
<NDArray 2x6 @cpu(0)>
```



#### Tensor

```python
nd.zeros((2, 3, 4))
```



​    

```
[[[0. 0. 0. 0.]
  [0. 0. 0. 0.]
  [0. 0. 0. 0.]]

 [[0. 0. 0. 0.]
  [0. 0. 0. 0.]
  [0. 0. 0. 0.]]]
<NDArray 2x3x4 @cpu(0)>
```



```python
nd.ones((2, 2, 3, 4))
```



​    

```
[[[[1. 1. 1. 1.]
   [1. 1. 1. 1.]
   [1. 1. 1. 1.]]

  [[1. 1. 1. 1.]
   [1. 1. 1. 1.]
   [1. 1. 1. 1.]]]
```

​    

```
 [[[1. 1. 1. 1.]
   [1. 1. 1. 1.]
   [1. 1. 1. 1.]]

  [[1. 1. 1. 1.]
   [1. 1. 1. 1.]
   [1. 1. 1. 1.]]]]
<NDArray 2x2x3x4 @cpu(0)>
```



```python
Y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
Y
```



​    

```
[[2. 1. 4. 3.]
 [1. 2. 3. 4.]
 [4. 3. 2. 1.]]
<NDArray 3x4 @cpu(0)>
```



```python
nd.random.normal(0, 1, shape=(3, 4))
```



​    

```
[[ 2.2122064   0.7740038   1.0434403   1.1839255 ]
 [ 1.8917114  -1.2347414  -1.771029   -0.45138445]
 [ 0.57938355 -1.856082   -1.9768796  -0.20801921]]
<NDArray 3x4 @cpu(0)>
```



#### 2.2.2 - Operation

```python
X = nd.random.normal(0, 1, shape=(3, 4))
X = nd.random.normal(3, 4, shape=(3, 4))
Z = X + Y
Z
```



​    

```
[[ 5.1234813  -0.46640396  8.218193    4.5186186 ]
 [-0.3767681   3.2566516   2.2410665  -1.9860268 ]
 [ 5.627817    4.371192    7.0347834   7.66047   ]]
<NDArray 3x4 @cpu(0)>
```



```python
Z = X * Y
Z
```



​    

```
[[  6.2469625  -1.466404   16.872774    4.555855 ]
 [ -1.3767681   2.5133033  -2.2768006 -23.944107 ]
 [  6.5112696   4.1135755  10.069567    6.66047  ]]
<NDArray 3x4 @cpu(0)>
```



```python
Z = X / Y
Z
```



​    

```
[[ 1.5617406  -1.466404    1.0545484   0.5062061 ]
 [-1.3767681   0.6283258  -0.25297785 -1.4965067 ]
 [ 0.40695435  0.45706394  2.5173917   6.66047   ]]
<NDArray 3x4 @cpu(0)>
```



```python
Z = X.exp()
Z
```



​    

```
[[2.2725355e+01 2.3075379e-01 6.7910698e+01 4.5659122e+00]
 [2.5239295e-01 3.5136368e+00 4.6816543e-01 2.5136315e-03]
 [5.0927472e+00 3.9400439e+00 1.5366631e+02 7.8091791e+02]]
<NDArray 3x4 @cpu(0)>
```



```python
Z = nd.dot(X, Y.T)
Z
```



​    

```
[[ 26.209187  18.919727  18.049719]
 [-22.4907   -25.084373  -9.241012]
 [ 44.74737   46.116432  27.354881]]
<NDArray 3x3 @cpu(0)>
```



```python
print(X)
print(Y)
```

```
[[ 3.1234813  -1.466404    4.2181935   1.5186183 ]
 [-1.3767681   1.2566516  -0.75893354 -5.986027  ]
 [ 1.6278174   1.3711919   5.0347834   6.66047   ]]
<NDArray 3x4 @cpu(0)>

[[2. 1. 4. 3.]
 [1. 2. 3. 4.]
 [4. 3. 2. 1.]]
<NDArray 3x4 @cpu(0)>
```



```python
nd.concat(X, Y, dim=0), nd.concat(X, Y, dim=1)
```



```
(
 [[ 3.1234813  -1.466404    4.2181935   1.5186183 ]
  [-1.3767681   1.2566516  -0.75893354 -5.986027  ]
  [ 1.6278174   1.3711919   5.0347834   6.66047   ]
  [ 2.          1.          4.          3.        ]
  [ 1.          2.          3.          4.        ]
  [ 4.          3.          2.          1.        ]]
 <NDArray 6x4 @cpu(0)>,
 [[ 3.1234813  -1.466404    4.2181935   1.5186183   2.          1.
    4.          3.        ]
  [-1.3767681   1.2566516  -0.75893354 -5.986027    1.          2.
    3.          4.        ]
  [ 1.6278174   1.3711919   5.0347834   6.66047     4.          3.
    2.          1.        ]]
 <NDArray 3x8 @cpu(0)>)
```



```python
X == Y
```



​    

```
[[0. 0. 0. 0.]
 [0. 0. 0. 0.]
 [0. 0. 0. 0.]]
<NDArray 3x4 @cpu(0)>
```



```python
X.sum()
```



​    

```
[15.223075]
<NDArray 1 @cpu(0)>
```



```python
X.norm().asscalar()
```



```
12.088418
```



```python

```
