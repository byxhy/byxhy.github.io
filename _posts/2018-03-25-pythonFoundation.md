---
layout: post
title: "Python Basic Tutorial"
author: "Xhy"
categories: Machine Learning
tags: [documentation,sample]
image: MLFoudation.jpg
---


Photo by JESHOOTS.COM

>声明：本系列根據[莫煩python](https://morvanzhou.github.io/)老师的python基礎教學網站的课程整理而來。學下來感覺不錯，那就秉承着先上路再迭代的思想，出发啦...　另外本系列所有的源碼可以點這裏[下載](http://www.imooc.com/u/1349694/courses?sort=publish)

<br />

## Table of Contents

1. [安装](#introduction)
   1. [What is Jekyll](#what-is-jekyll)
   2. [Never Used Jeykll Before?](#never-used-jekyll-before)
2. [基本使用](#installation)
   1. [GitHub Pages Installation](#github-pages-installation)
   2. [Local Installation](#local-installation)
3. [while 和 for 循环](#configuration)
   1. [Sample Posts](#sample-posts)
   2. [Site Variables](#site-variables)
4. [if 判断](#features)
   1. [Design Considerations](#design-considerations)
   2. [Disqus](#disqus)
   3. [Google Analytics](#google-analytics)
5. [定义功能](#everything-else)
6. [变量形式](#Contributing)
7. [模块安装](#questions)
8. [文件读取](#credits)
9. [class 类](#license)
10. [input 输入](#everything-else)
11. [元组, 列表, 字典](#Contributing)
12. [模块](#questions)
13. [其他](#credits)

<br />

---

<br />


## 1 - Install Python3

## 2 - Basic use

### 2.1 - print function

#### 2.1.1 - print string


```python
print("This is Python basic tutorial")
```

    This is Python basic tutorial



```python
print('This is Python basic tutorial')
```

    This is Python basic tutorial


#### 2.1.2 - string addition


```python
print('This is Python' + ' basic tutorial')
```

    This is Python basic tutorial


#### 2.1.3 - simple computation


```python
print(3 + 4)
```

    7



```python
print(int('3') + 4)
```

    7



```python
print(int('3.4'))
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-12-fa76ed979087> in <module>()
    ----> 1 print(int('3.4'))


    ValueError: invalid literal for int() with base 10: '3.4'



```python
print(float('3.4') + 3)
```

    6.4



```python
print(int(3.4) + 4)
```

    7


### 2.2  - basic computation

#### 2.2.1 - addition and subtraction multiplication and division


```python
2 + 2
```




    4




```python
4 - 2
```




    2




```python
4 * 2
```




    8




```python
4 / 3
```




    1.3333333333333333



#### 2.2.2 - ^ and **


```python
2 ** 2
```




    4




```python
2 ** 3
```




    8




```python
2 ** 4
```




    16



#### 2.2.3 - %


```python
9 % 2
```




    1




```python
7 % 3
```




    1




```python
5 % 3
```




    2



### 2.3  - variable

#### 2.3.1 - naming rules


```python
gun = 94
```


```python
print(gun)
```

    94



```python
apple = 'iPhone X'
```


```python
print(apple)
```

    iPhone X



```python
a, b, c = 9, 8, 7
```


```python
print(a, b, c)
```

    9 8 7


## 3 - While and for loop

### 3.1 - while


```python
condition = 0
while condition < 5:
    print(condition)
    condition += 1       # Err: condition++
```

    0
    1
    2
    3
    4



```python
conditoin = 5
while condition:
    print(condition)
    condition -= 1
```

    5
    4
    3
    2
    1


### 3.2 - for


```python
eg_list = [1, 2, 3, 4, 5, 6.2, 7, 8, 9, 10]
for i in eg_list:
    print(i)
    print('This is inside the for loop')

print('This is outside the for loop')
```

    1
    This is inside the for loop
    2
    This is inside the for loop
    3
    This is inside the for loop
    4
    This is inside the for loop
    5
    This is inside the for loop
    6.2
    This is inside the for loop
    7
    This is inside the for loop
    8
    This is inside the for loop
    9
    This is inside the for loop
    10
    This is inside the for loop
    This is outside the for loop


### 3.3 - range


```python
a = range(1, 10)
for i in a:
    print(i)
```

    1
    2
    3
    4
    5
    6
    7
    8
    9



```python
b = range(10)
for i in b:
    print(i)
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9



```python
c = range(1, 10, 2)
for i in c:
    print(i)
```

    1
    3
    5
    7
    9


### 3.4 - set type

#### 3.4.1 - list


```python
eg_list = [1, 2, 3, 4, 5, 6]
for i in eg_list:
    print(i)
```

    1
    2
    3
    4
    5
    6


#### 3.4.2 - tuple


```python
tup = ('WangEr', 38, 168.5)
for i in tup:
    print(i)
```

    WangEr
    38
    168.5


#### 3.4.3 - dictionary (But the dictionary is out of order. refer OrderedDict)


```python
dic = {}
dic['lan'] = 'python'
dic['version'] = 2.7
dic['platform'] = 64
for key in dic:
    print(key, '=', dic[key])
```

    lan = python
    version = 2.7
    platform = 64


#### 3.4.4 - set (The set collection will remove duplicates and it is out of order too)


```python
s = set(['Python3', 'Python2', 'NXP', 'Python3'])
for i in s:
    print(i)
```

    Python3
    Python2
    NXP


#### 3.4.5 - iterator


```python
# define a Fib class
class Fib(object):
    def __init__(self, max):
        self.max = max
        self.n, self.a, self.b = 0, 0, 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.n < self.max:
            r = self.b
            self.a, self.b = self.b, self.a + self.b
            self.n = self.n + 1
            return r
        raise StopIteration()

# using Fib object
for i in Fib(5):
    print(i)
```

    1
    1
    2
    3
    5


#### 3.4.6 - yield


```python
def fib(max):
    a, b = 0, 1
    while max:
        r = b
        a, b = b, a+b
        max -= 1
        yield r

# using generator
for i in fib(5):
    print(i)
```

    1
    1
    2
    3
    5


## 4 - If

### 4.1 - basic use


```python
x = 9
y = 6
if x < y:
    print('x is less than y')
else:
    print('x is more than y')
```

    x is more than y


#### * Although the following syntax is correct in Python, we still discourage it.


```python
x = 3
y = 5
z = 9
if x < y < z:
    print('x is less than y, and y is less than z')
else:
    print('y is not sure')  
```

    x is less than y, and y is less than z


#### * The following syntax is recommenaded.


```python
x = 3
y = 5
z = 3
if x < y and y < z:
    print('x is less than y, and y is less than z')
else:
    print('y is not sure')
```

    y is not sure



```python
x = 4
y = 4
z = 5
if x == y:
    print('x is equal to y')
else:
    print('x is not equal to y')

if x == z:
    print('x is equal to z')
else:
    print('x is not equal to z')
```

    x is equal to y
    x is not equal to z



```python
x = 3
y = 9
if x > y:
    print('x > y')
elif x < y:
    print('x < y')
else:
    print('x == y')
```

    x < y


#### * Trinocular operator ( var = var1 if condition else var2 )


```python
worked = True
result = 'done' if worked else 'not yet'
print(result)
```

    done


## 5 - Define function

### 5.1 - def use


```python
def add(a, b):
    print('This is a add() function.')
    print('a + b =', a + b)

add(3, 7)
```

    This is a add() function.
    a + b = 10


### 5.2 - function parameters


```python
def add(a, b):
    print('This is a add() function.')
    print('a + b =', a + b)

add(3, 7)
```

    This is a add() function.
    a + b = 10


### 5.3 - default parameters


```python
def sale_car(price, color='red', brand='carmy', is_second_hand=True):
    print('price =', price, ','
          'color =', color, ','
          'brand =', brand, ','
          'is_second_hand =', is_second_hand)

sale_car(10000)
```

    price = 10000 ,color = red ,brand = carmy ,is_second_hand = True


### 5.4 - call yourself, only called inside the script


```python
if __name__ == '__main__':
    #code_here
    print('main')
```

    main


### 5.5 - variable parameters


```python
def sumGrades(name, *score):
    totalScore = 0
    for i in score:
        totalScore += i
    print(name, 'total score is', totalScore)

sumGrades('Lisa', 8)

sumGrades('Mery', 8, 7, 9)
```

    Lisa total score is 8
    Mery total score is 24


### 5.6 - keyword parameters


```python
def keyPara(name, **kw):
    print('name is ', name)
    for k, v in kw.items():
        print(k, v)

keyPara('Mary', Gender = 'woman', country = 'US', age = '23')
```

    name is  Mary
    Gender woman
    country US
    age 23


#### 1) Default , variable  and keyword parameters should be behind all the function parameters

#### 2) Through the variable parameters and keyword parameters, all function can be replaced by universal_func(*args, **kw) .

## 6 - Local variables and global variables

### 6.1 - local variable


```python
globalVar = 100

def localVariable():
    localVar = 90

print(globalVar)

print(localVar)
```

    100



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-74-0161f03053e0> in <module>()
          6 print(globalVar)
          7
    ----> 8 print(localVar)


    NameError: name 'localVar' is not defined


### 6.2 - global variable


```python
a = None

def fun():
    global a
    a = 300

print(a)

fun()

print(a)
```

    None
    300


## 7 - Install the modules

### 7.1 - install Numpy and matplotlib


```python
import numpy as np
import matplotlib.pyplot as plt
```

## 8 - Read and write file

### 8.1 - write

#### 8.1.1 - '\n'


```python
text = 'This is my first test. This is the second line. This the third.'
print(text)
```

    This is my first test. This is the second line. This the third.



```python
print('\n')
text = 'This is my first test.\nThis is the second line.\nThis the third line.'
print(text)
```



    This is my first test.
    This is the second line.
    This the third line.


#### 8.1.2 - read and write file


```python
text = 'This is my first test.\nThis is the second line.\nThis the third line.'

myNewFile = open('Documents/nb.txt','w')
myNewFile.write(text)
myNewFile.close()
```

#### 8.1.3 - '\t'


```python
text = '\tThis is my first test. \n\tThis is the second line. \n\tThis is the third line.'
print(text)

myNewFile = open('Documents/nb.txt','w')
myNewFile.write(text)
myNewFile.close()
```

    	This is my first test.
    	This is the second line.
    	This is the third line.


### 8.2 - write

#### 8.2.1 - append


```python
appendText = '\nThis is appended file.'

myNewFile = open('Documents/nb.txt','a')
myNewFile.write(appendText)
myNewFile.close()
```

#### 8.2.2 - write


```python
writeText = '\nThis is a \'write\' command.'

myNewFile = open('Documents/nb.txt','w')
myNewFile.write(writeText)
myNewFile.close()
```

### 8.3 - read

#### 8.3.1 - read


```python
file = open('Documents/nb.txt','r')
context = file.read()
print(context)
file.close()
```


    This is a 'write' command.


#### 8.3.2 - readline


```python
file = open('Documents/nb.txt','r')
context = file.readline()
print(context)
context = file.readline()
print(context)
context = file.readline()
print(context)
context = file.readline()
print(context)
file.close()
```

    	This is my first test.

    	This is the second line.

    	This is the third line.



#### 8.3.3 - readlines(text → list)


```python
file = open('Documents/nb.txt','r')
context = file.readlines()
print(context)
file.close()

for item in context:
    print(item)
```

    ['\tThis is my first test. \n', '\tThis is the second line. \n', '\tThis is the third line.']
    	This is my first test.

    	This is the second line.

    	This is the third line.


## 9 - Class

### 9.1 - class


```python
class Caculator:
    name = 'caculator'
    price = 20
    def add(self, x, y):
        print('x + y =', x + y)
    def sub(self, x, y):
        print('x - y =', x - y)

cal = Caculator()

cal.add(3, 7)
cal.sub(3, 7)

cal.name
```

    x + y = 10
    x - y = -4





    'caculator'



### 9.2 - class and init


```python
class Caculator:
    name = 'caculator'
    price = 20
    def __init__(self, name, price, width, height):
        self.name = name
        self.price = price
        self.width = width
        self.height = height

cal = Caculator('cw', 19, 20, 18)
print('name =', cal.name)
print('price =', cal.price)
print('width =', cal.width)
print('height =', cal.height)
```

    name = cw
    price = 19
    width = 20
    height = 18


### 9.3 - default parameters in init


```python
class Caculator:
    name = 'caculator'
    price = 20
    def __init__(self, name, price, width = 20, height = 18):
        self.name = name
        self.price = price
        self.width = width
        self.height = height

cal = Caculator('cw', 19)
print('name =', cal.name)
print('price =', cal.price)
print('width =', cal.width)
print('height =', cal.height)
```

    name = cw
    price = 19
    width = 20
    height = 18


## 10 - Input

### 10.1 - input


```python
num = input('please input a number: ')
print(num)
```

    please input a number: 12345
    12345



```python
num = int(input('please input a number: '))
if num > 10:
    print('num > 10')
elif num > 2 and num <= 10:
    print('2 < num <= 10')
else:
    print('num < 2')
```

    please input a number: 4
    2 < num <= 10


### 10.2 - input extension


```python
score=int(input('Please input your score: '))
if score>= 90:
   print('Congradulation, you get an A')
elif score >= 80:
    print('You get a B')
elif score >= 70:
    print('You get a C')
elif score >= 60:
    print('You get a D')
else:
    print('Sorry, You are failed ')
```

    Please input your score: 47
    Sorry, You are failed


## 11 - Tuple \ list \ dictionary

### 11.1 - tuple


```python
a_tuple = (1, 2, 3, 4, 5)
b_tuple =  1, 2, 3, 4, 5
print(a_tuple)
print(b_tuple)

print(a_tuple[0 : 2])
print(b_tuple[2 : 4])
print(b_tuple[0 : -1])            ####### -1 represents the last element in Python, but [0 : -1] is the left closed and right open.
print(b_tuple[-1])
print(b_tuple[0 : len(b_tuple)])
```

    (1, 2, 3, 4, 5)
    (1, 2, 3, 4, 5)
    (1, 2)
    (3, 4)
    (1, 2, 3, 4)
    5
    (1, 2, 3, 4, 5)


### 11.2 - list


```python
a_list = [1, 2, 3, 4, 5]
print(a_list)

print(a_list[-1])
print(a_list[0 : -1])
print(a_list[-1])
print(a_list[0 : len(a_list)])
```

    [1, 2, 3, 4, 5]
    5
    [1, 2, 3, 4]
    5
    [1, 2, 3, 4, 5]


### 11.3 - print by for loop


```python
a_tuple = (1, 2, 3, 4, 5)
b_list  = [6, 7, 8, 9, 10]

for index in range(len(a_tuple)):
    print('index =', index, 'number in tuple', a_tuple[index])

print('\n')

for index in range(len(b_list)):
    print('index =', index, 'number in list', b_list[index])
```

    index = 0 number in tuple 1
    index = 1 number in tuple 2
    index = 2 number in tuple 3
    index = 3 number in tuple 4
    index = 4 number in tuple 5


    index = 0 number in list 6
    index = 1 number in list 7
    index = 2 number in list 8
    index = 3 number in list 9
    index = 4 number in list 10


### 11.4 - list

#### 11.4.1 - append


```python
a = [1, 2, 3, 4, 1, 1, -1]
a.append(0)  # append 0 to the end of the list
print(a)
```

    [1, 2, 3, 4, 1, 1, -1, 0]


#### 11.4.2 - remove


```python
a = [1, 2, 3, 4, 1, 1, -1]
a.remove(2)   # remove the first 2 in list
print(a)
```

    [1, 3, 4, 1, 1, -1]


#### 11.4.3 - find the corresponding index


```python
a = [1, 2, 3, 4, 1, 1, -1]
print(a[0 : 3])
print(a[0 : ])
print(a[ : -1])
print(a[-1])
print(a[-3 : ])
```

    [1, 2, 3]
    [1, 2, 3, 4, 1, 1, -1]
    [1, 2, 3, 4, 1, 1]
    -1
    [1, 1, -1]



```python
a = [1, 2, 3, 4, 1, 1, -1]
print(a.index(1))
print(a.index(3))
```

    0
    2



```python
a = [1, 2, 3, 4, 1, 1, -1]
print(a.count(1))
```

    3


#### 11.4.4 - list sort


```python
a = [1, 2, 3, 4, 1, 1, -1]
a.sort()
print(a)
```

    [-1, 1, 1, 1, 2, 3, 4]



```python
a = [1, 2, 3, 4, 1, 1, -1]
a.sort(reverse = True)
print(a)
```

    [4, 3, 2, 1, 1, 1, -1]


### 11.5 - multidimensional list

#### 11.5.1 - create a two dimensional list


```python
a = [1, 2, 3, 4, 5]

multi_dim_a = [[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9]]

print(a[1])

print(multi_dim_a[0][2])
```

    2
    3


### 11.6 - dictionary

#### 11.6.1 - create dictionary --- dic[key:value]


```python
a_list = [1, 2, 3, 4, 5, 6, 7, 8]

d1 = {'apple':1, 'peer':2, 'orange':3}
d2 = {1:'a', 2:'b', 3:'c'}
d3 = {1:'a', 'b':2, 'c':3}

print(a_list[0])
print(d1['apple'])

print(d2[1])
print(d2['a'])
```

    1
    1
    a



    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-20-a4d791bc3605> in <module>()
          9
         10 print(d2[1])
    ---> 11 print(d2['a'])


    KeyError: 'a'



```python
# dictionary is no order
del d1['peer']


print(d1)
```

    {'apple': 1, 'orange': 3}



```python
print(d3['b'])
print(d3[2])
```

    2



    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-22-3fd50c7cf06e> in <module>()
          1 print(d3['b'])
    ----> 2 print(d3[2])


    KeyError: 2


#### 11.6.2 - dictionary save type


```python
def func():
    return 0

d4 = {'apple':[1, 2, 3], 'pear':{1:3, 3:'a'}, 'orange':func}

print(d4)

print(d4['pear'][3])
```

    {'apple': [1, 2, 3], 'pear': {1: 3, 3: 'a'}, 'orange': <function func at 0x7f8f105779d8>}
    a


## 12 - module

### 12.1 - import method

#### 12.1.1 - import time


```python
import time
print(time.localtime())
```

    time.struct_time(tm_year=2018, tm_mon=4, tm_mday=3, tm_hour=23, tm_min=17, tm_sec=6, tm_wday=1, tm_yday=93, tm_isdst=0)


#### 12.1.2 - import time as  - , -


```python
import time as t
print(t.localtime())
```

    time.struct_time(tm_year=2018, tm_mon=4, tm_mday=3, tm_hour=23, tm_min=20, tm_sec=26, tm_wday=1, tm_yday=93, tm_isdst=0)


#### 12.1.3 - from time import time,localtime (just import function you want)


```python
from time import time, localtime

print(localtime())
print(time())
```

    time.struct_time(tm_year=2018, tm_mon=4, tm_mday=3, tm_hour=23, tm_min=22, tm_sec=11, tm_wday=1, tm_yday=93, tm_isdst=0)
    1522768931.7420359


#### 12.1.4 - import time  import *


```python
from time import *
print(localtime())
```

    time.struct_time(tm_year=2018, tm_mon=4, tm_mday=3, tm_hour=23, tm_min=24, tm_sec=1, tm_wday=1, tm_yday=93, tm_isdst=0)


### 12.2 - import my own  module

#### balance.py


```python
d=float(input('Please enter what is your initial balance: \n'))
p=float(input('Please input what is the interest rate (as a number): \n'))
d=float(d+d*(p/100))
year=1
while year<=5:
    d=float(d+d*p/100)
    print('Your new balance after year:',year,'is',d)
    year=year+1
print('your final year is',d)
```

    Please enter what is your initial balance:
    5000
    Please input what is the interest rate (as a number):
    2.3
    Your new balance after year: 1 is 5232.645
    Your new balance after year: 2 is 5352.995835000001
    Your new balance after year: 3 is 5476.114739205001
    Your new balance after year: 4 is 5602.065378206716
    Your new balance after year: 5 is 5730.91288190547
    your final year is 5730.91288190547



```python
import balance
```

    Please enter what is your initial balance:
    5000
    Please input what is the interest rate (as a number):
    2.3
    Your new balance after year: 1 is 5232.645
    Your new balance after year: 2 is 5352.995835000001
    Your new balance after year: 3 is 5476.114739205001
    Your new balance after year: 4 is 5602.065378206716
    Your new balance after year: 5 is 5730.91288190547
    your final year is 5730.91288190547


## 13 - Others

### 13.1 - continue & break

#### 13.1.1 - true or false


```python
a = True
while a:
    b = input('type something')
    if b == '1':
        a = False
    else:
        pass
print('finish run')
```

    type something3
    type something1
    finish run


#### 13.1.2 - break


```python
while True:
    b = input('type somesthing:')
    if b == '1':
        break
    else:
        pass
print('finish run')
```

    type somesthing:3
    type somesthing:2
    type somesthing:1
    finish run


#### 13.1.3 - continue


```python
while True:
    b = input('type somesthing:')
    if b == '1':
        continue
    elif b == '2':
        break
    else:
        pass
    print('still in while')

print('finish run')
```

    type somesthing:3
    still in while
    type somesthing:1
    type somesthing:2
    finish run


### 13.2 - try

### 13.3 - zip lambda map

### 13.4 - copy & deepcoyp

### 13.5 - threading

### 13.6 - multiprocessing

### 13.7 - tkinter

### 13.8 - pickle

### 13.9 - set

### 13.10 - Regular expression
