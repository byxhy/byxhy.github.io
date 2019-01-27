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
len(a)
```

    [1, 2, 3, 4, 5]





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



```python

```
