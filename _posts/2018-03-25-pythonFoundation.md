---
layout: post
title: "Python foundation for Machine Learning"
author: "Xhy"
categories: Machine Learning
tags: [documentation,sample]
image: MLFoudation.jpg
---


Photo by Farzad Nazifi

>声明：系列筆記參照[莫煩python](https://morvanzhou.github.io/)教學網站的课程整理而來，感謝老師的分享

<br />

## Table of Contents

1. [Install Python](#1---install-python)
   1. [by yourself](#1-1---by-yourself)

2. [Basic use](#2---basic-use)
   1. [print function](#2-1---print-function)
   2. [basic computation](#2-2---basic-computation)
   3. [variable](#2-3---variable)

3. [While and for loop](#3---while-and-for-loop)
   1. [while](#3-1---while)
   2. [for](#3-2---for)
   3. [range](#3-3---range)
   4. [set type](#3-4---set-type)

4. [If](#4---if)
   1. [basic use](#4-1---basic-use)


5. [Define function](#5---define-function)
   1. [def use](#5-1---def-use)
   2. [function parameters](#5-2---function-parameters)
   3. [default parameters](#5-3---default-parameters)
   4. [call yourself, only called inside the script](#5-4---call-yourself-only-called-inside-the-script)
   5. [variable parameters](#5-5---variable-parameters)
   6. [keyword parameters](#5-6---keyword-parameters)

6. [Local variables and global variables](#6---local-variables-and-global-variables)
   1. [local variable](#6-1---local-variable)
   2. [global variable](#6-2---global-variable)

7. [Install the modules](#7---install-the-modules)
   1. [install Numpy and matplotlib](#7-1---install-numpy-and-matplotlib)

8. [Read and write file](#8---read-and-write-file)
   1. [write](#8-1---write)
   2. [write](#8-2---write)
   3. [read](#8-3---read)

9. [Class](#9---class)
   1. [class](#9-1---class)
   2. [class and init](#9-2---class-and-init)
   3. [default parameters in init](#9-3---default-parameters-in-init)

10. [Input](#10---input)
    1. [input](#10-1---input)
    2. [input extension](#10-2---input-extension)

11. [Tuple \ list \ dictionary](#11---tuple-list-dictionary)
    1. [tuple](#11-1---tuple)
    2. [input extension](#11-2---input-extension)

11 - Tuple-list-dictionary


<br />

---

<br />

1104
## 1 - Install Python3

### 1-1 - by yourself

## 2 - Basic use

### 2-1 - print function

#### 2-1-1 - print string


```python
print("This is Python basic tutorial")
```

    This is Python basic tutorial



```python
print('This is Python basic tutorial')
```

    This is Python basic tutorial


#### 2-1-2 - string addition


```python
print('This is Python' + ' basic tutorial')
```

    This is Python basic tutorial


#### 2-1-3 - simple computation


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


### 2-2 - basic computation

#### 2-2-1 - addition and subtraction multiplication and division


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



#### 2-2-2 - ^ and **


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



#### 2-2-3 - %


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



### 2-3 - variable

#### 2-3-1 - naming rules


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

### 3-1 - while


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


### 3-2 - for


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


### 3-3 - range


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


### 3-4 - set type

#### 3-4-1 - list


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


#### 3-4-2 - tuple


```python
tup = ('WangEr', 38, 168.5)
for i in tup:
    print(i)
```

    WangEr
    38
    168.5


#### 3-4-3 - dictionary (But the dictionary is out of order. refer OrderedDict)


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


#### 3-4-4 - set (The set collection will remove duplicates and it is out of order too)


```python
s = set(['Python3', 'Python2', 'NXP', 'Python3'])
for i in s:
    print(i)
```

    Python3
    Python2
    NXP


#### 3-4-5 - iterator


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


#### 3-4-6 - yield


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

### 4-1 - basic use


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


#### * The following syntax is recommended.


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

### 5-1 - def use


```python
def add(a, b):
    print('This is a add() function.')
    print('a + b =', a + b)

add(3, 7)
```

    This is a add() function.
    a + b = 10


### 5-2 - function parameters


```python
def add(a, b):
    print('This is a add() function.')
    print('a + b =', a + b)

add(3, 7)
```

    This is a add() function.
    a + b = 10


### 5-3 - default parameters


```python
def sale_car(price, color='red', brand='carmy', is_second_hand=True):
    print('price =', price, ','
          'color =', color, ','
          'brand =', brand, ','
          'is_second_hand =', is_second_hand)

sale_car(10000)
```

    price = 10000 ,color = red ,brand = carmy ,is_second_hand = True


### 5-4 - call yourself, only called inside the script


```python
if __name__ == '__main__':
    #code_here
    print('main')
```

    main


### 5-5 - variable parameters


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


### 5-6 - keyword parameters


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

#### 2) Through the variable parameters and keyword parameters, all function can be replaced by universal_func(*args,**kw)


## 6 - Local variables and global variables

### 6-1 - local variable


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


### 6-2 - global variable


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

### 7-1 - install Numpy and matplotlib


```python
import numpy as np
import matplotlib.pyplot as plt
```

## 8 - Read and write file

### 8-1 - write

#### 8-1-1 - '\n'


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


#### 8-1-2 - read and write file


```python
text = 'This is my first test.\nThis is the second line.\nThis the third line.'

myNewFile = open('Documents/nb.txt','w')
myNewFile.write(text)
myNewFile.close()
```

#### 8-1-3 - '\t'


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


### 8-2 - write

#### 8-2-1 - append


```python
appendText = '\nThis is appended file.'

myNewFile = open('Documents/nb.txt','a')
myNewFile.write(appendText)
myNewFile.close()
```

#### 8-2-2 - write


```python
writeText = '\nThis is a \'write\' command.'

myNewFile = open('Documents/nb.txt','w')
myNewFile.write(writeText)
myNewFile.close()
```

### 8-3 - read

#### 8-３-1 - read


```python
file = open('Documents/nb.txt','r')
context = file.read()
print(context)
file.close()
```


    This is a 'write' command.


#### 8-３-2 - readline


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



#### 8-３-3 - readlines(text → list)


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

### 9-1 - class


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



### 9-2 - class and init


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


### 9-3 - default parameters in init


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

### 10-1 - input


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


### 10-2 - input extension


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


## 11 - Tuple-list-dictionary

### 11-1 - tuple


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


### 11-2 - list


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


### 11-3 - print by for loop


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


### 11-4 - list

#### 11-4.1 - append


```python
a = [1, 2, 3, 4, 1, 1, -1]
a.append(0)  # append 0 to the end of the list
print(a)
```

    [1, 2, 3, 4, 1, 1, -1, 0]


#### 11-4.2 - remove


```python
a = [1, 2, 3, 4, 1, 1, -1]
a.remove(2)   # remove the first 2 in list
print(a)
```

    [1, 3, 4, 1, 1, -1]


#### 11-4.3 - find the corresponding index


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


#### 11-4.4 - list sort


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


### 11-5 - multidimensional list

#### 11-5.1 - create a two dimensional list


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


### 11-6 - dictionary

#### 11-6.1 - create dictionary --- dic[key:value]


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


#### 11-6.2 - dictionary save type


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

#### 13.2.1 - try, except ... as ...


```python
try:
    file = open('eee.txt', 'r')
except Exception as e:
    print(e)
```

    [Errno 2] No such file or directory: 'eee.txt'


#### 13.2.2 - application

#### The first time: No such file or directory


```python
try:
    file=open('eeee.txt','r+')
except Exception as e:
    print(e)
    response = input('do you want to create a new file:')
    if response=='y':
        file=open('eeee.txt','w')
    else:
        pass
else:
    file.write('ssss')
    file.close()
```

    [Errno 2] No such file or directory: 'eeee.txt'
    do you want to create a new file:y


#### The seconed time: write 'ssss' to eeee.txt


```python
try:
    file=open('eeee.txt','r+')
except Exception as e:
    print(e)
    response = input('do you want to create a new file:')
    if response=='y':
        file=open('eeee.txt','w')
    else:
        pass
else:
    file.write('ssss')
    file.close()
```

### 13.3 - zip lambda map

#### 13.3.1 - zip


```python
a = [1, 2, 3]
b = [4, 5, 6]

ab = zip(a, b)
```


```python
print(ab)
```

    <zip object at 0x7fb53024a948>



```python
print(list(ab)) # view by list
```

    [(1, 4), (2, 5), (3, 6)]



```python
for i,j in zip(a, b):
    print(i/2, j*2)
```

    0.5 8
    1.0 10
    1.5 12


#### 13.3.2 - lambda


```python
fun = lambda x,y : x+y

x = int(input('x = '))
y = int(input('y = '))

print('x + y =',fun(x, y))
```

    x = 3
    y = 7
    x + y = 10


#### 13.3.3 - map


```python
def func(x, y):
    return (x + y)
```


```python
list(map(func, [1],[2]))
```




    [3]




```python
list(map(func, [1,2], [3, 4]))
```




    [4, 6]



### 13.4 - copy & deepcopy

#### 13.4.1 - id


```python
import copy

a = [1, 2, 3]
b = a

id(a)
```




    140416172680264




```python
id(b)
```




    140416172680264




```python
id(a) == id(b)
```




    True




```python
b[0] = 9999
print(a, b)
```

    [9999, 2, 3] [9999, 2, 3]


#### 13.4.2 - copy


```python
import copy

c = [1, 2, 3]
d = copy.copy(c)

print(id(c) == id(d))
```

    False



```python
c[0] = 999
print(c, d)
```

    [999, 2, 3] [1, 2, 3]


#### 13.4.3 - deepcopy


```python
# copy.copy
e = [1, 2, [3, 4]]
f = copy.copy(e)

print(id(e) == id(f))
```

    False



```python
print(id(e[2]) == id(f[2]))
```

    True



```python
e[2][0] = 666
print(e, f)
```

    [1, 2, [666, 4]] [1, 2, [666, 4]]



```python
# copy.deepcopy
g = copy.deepcopy(e)

print(id(e) == id(g))
```

    False



```python
print(id(e[2]) == id(g[2]))
```

    False



```python
e[2][0] = 999
print(e, g)
```

    [1, 2, [999, 4]] [1, 2, [666, 4]]


### 13.5 - threading

#### 13.5.1 - add thread


```python
import threading
threading.active_count()
```




    5




```python
threading.enumerate()
```




    [<_MainThread(MainThread, started 139974580156160)>,
     <Thread(Thread-2, started daemon 139974371399424)>,
     <Heartbeat(Thread-3, started daemon 139974363006720)>,
     <HistorySavingThread(IPythonHistorySavingThread, started 139974337828608)>,
     <ParentPollerUnix(Thread-1, started daemon 139973990807296)>]




```python
threading.current_thread()
```




    <_MainThread(MainThread, started 139974580156160)>




```python
def thread_job():
    print('This is a thread of %s', threading.current_thread())

def main():
    thread = threading.Thread(target=thread_job,)
    thread.start()

if __name__ == '__main__':
    main()
```

    This is a thread of %s <Thread(Thread-9, started 139973982414592)>


#### 13.5.2 - join

#### do not add join


```python
import threading
import time

def thread_job():
    print("T1 start\n")
    for i in range(10):
        time.sleep(0.1)
    print("T1 finish\n")

added_thread = threading.Thread(target=thread_job, name='T1')
added_thread.start()
print("all done\n")
```

    T1 start

    all done

    T1 finish



#### do add join


```python
import threading
import time

def thread_job():
    print("T1 start\n")
    for i in range(10):
        time.sleep(0.1)
    print("T1 finish\n")

added_thread = threading.Thread(target=thread_job, name='T1')
added_thread.start()
added_thread.join()
print("all done\n")
```

    T1 start

    T1 finish

    all done




```python
def T1_job():
    print("T1 start\n")
    for i in range(10):
        time.sleep(0.1)
    print("T1 finish\n")

def T2_job():
    print("T2 start\n")
    print("T2 finish\n")

thread_1 = threading.Thread(target=T1_job, name='T1')
thread_2 = threading.Thread(target=T2_job, name='T2')
thread_1.start()
thread_2.start()
print("all done\n")
```

    T2 start
    T1 start
    all done


    T2 finish


    T1 finish




```python
def T1_job():
    print("T1 start\n")
    for i in range(10):
        time.sleep(0.1)
    print("T1 finish\n")

def T2_job():
    print("T2 start\n")
    print("T2 finish\n")

thread_1 = threading.Thread(target=T1_job, name='T1')
thread_2 = threading.Thread(target=T2_job, name='T2')
thread_1.start()
thread_2.start()
thread_2.join()
thread_1.join()
print("all done\n")
```

    T1 start

    T2 start

    T2 finish

    T1 finish

    all done



#### 13.5.3 - Queue


```python
import threading
import time
from queue import Queue
```


```python
def job(l,q):
    for i in range (len(l)):
        l[i] = l[i]**2
    q.put(l)   # multithreading function can not use return
```


```python
import threading
import time

from queue import Queue

def job(l,q):
    for i in range (len(l)):
        l[i] = l[i]**2
    q.put(l)

def multithreading():
    q =Queue()
    threads = []
    data = [[1,2,3],[3,4,5],[4,4,4],[5,5,5]]
    for i in range(4):
        t = threading.Thread(target=job,args=(data[i],q))
        t.start()
        threads.append(t)
    for thread in threads:
        thread.join()
    results = []
    for _ in range(4):
        results.append(q.get())
    print(results)

if __name__== '__main__':
    multithreading()
```

    [[1, 4, 9], [9, 16, 25], [16, 16, 16], [25, 25, 25]]


#### 13.5.4 - GIL(Global Interpreter Lock)


```python
import threading
from queue import Queue
import copy
import time

def job(l, q):
    res = sum(l)
    q.put(res)

def multithreading(l):
    q = Queue()
    threads = []
    for i in range(4):
        t = threading.Thread(target=job, args=(copy.copy(l), q), name='T%i' % i)
        t.start()
        threads.append(t)
    [t.join() for t in threads]
    total = 0
    for _ in range(4):
        total += q.get()
    print(total)

def normal(l):
    total = sum(l)
    print(total)

if __name__ == '__main__':
    l = list(range(1000000))
    s_t = time.time()
    normal(l*4)
    print('normal: ',time.time()-s_t)
    s_t = time.time()
    multithreading(l)
    print('multithreading: ', time.time()-s_t)
```

    1999998000000
    normal:  0.16218066215515137
    1999998000000
    multithreading:  0.0786292552947998


#### 13.5.5 - Lock

#### do not use lock


```python
import threading

def job1():
    global A
    for i in range(10):
        A+=1
        print('job1',A)

def job2():
    global A
    for i in range(10):
        A+=10
        print('job2',A)

if __name__== '__main__':
    lock=threading.Lock()
    A=0
    t1=threading.Thread(target=job1)
    t2=threading.Thread(target=job2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
```

    job1job2  19
    1job2
     job130
    2job2
     job140
    3job2
     job150
    4job2
     job160
    5job2
     job170
    6job2
     job180
    7job2
     job190
    8job2
     job1100
    9job2
     job1110
    20


#### do use lock


```python
import threading

def job1():
    global A,lock
    lock.acquire()
    for i in range(10):
        A+=1
        print('job1',A)
    lock.release()

def job2():
    global A,lock
    lock.acquire()
    for i in range(10):
        A+=10
        print('job2',A)
    lock.release()

if __name__== '__main__':
    lock=threading.Lock()
    A=0
    t1=threading.Thread(target=job1)
    t2=threading.Thread(target=job2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
```

    job1 1
    job1 2
    job1 3
    job1 4
    job1 5
    job1 6
    job1 7
    job1 8
    job1 9
    job1 10
    job2 20
    job2 30
    job2 40
    job2 50
    job2 60
    job2 70
    job2 80
    job2 90
    job2 100
    job2 110



```python
### 13.6 - multiprocessing
```


```python
### 13.7 - tkinter
```

### 13.8 - pickle

#### 13.8.1 - save


```python
import pickle

a_dict = {'da':111, 2:[23, 1, 4], '23': {1:2, 'd':'sad'}}

file = open('p.pickle', 'wb')
pickle.dump(a_dict, file)
file.close()
```

#### 13.8.2 - reload


```python
with open('p.pickle', 'rb') as file:
    a_dict_reload = pickle.load(file)

print(a_dict_reload)
```

    {'da': 111, 2: [23, 1, 4], '23': {1: 2, 'd': 'sad'}}


### 13.9 - set

#### 13.9.1 - foundation


```python
char_list = ['a', 'b', 'c', 'c', 'd', 'd', 'd']

sentence = 'Welcome Back to This Tutorial'

print(set(char_list))
```

    {'d', 'c', 'a', 'b'}



```python
print(set(sentence))
```

    {'W', 'T', 'u', 'B', 'k', ' ', 'm', 'i', 'o', 's', 'r', 'a', 'l', 't', 'h', 'c', 'e'}



```python
print(set(char_list+ list(sentence)))
```

    {'W', 'T', 'u', 'b', 'B', 'd', 'k', ' ', 'm', 'i', 'o', 's', 'r', 'a', 'l', 't', 'h', 'c', 'e'}


#### 13.9.2 - add


```python
unique_char = set(char_list)
unique_char.add('x')

print(unique_char)
```

    {'b', 'd', 'x', 'a', 'c'}



```python
unique_char.add(['y', 'z']) # unique_char.add(['y', 'z']) this is wrong
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-58-e470cd27c66a> in <module>()
    ----> 1 unique_char.add(['y', 'z'])


    TypeError: unhashable type: 'list'


#### 13.9.3 - remove / discard / clear


```python
unique_char.remove('x')
print(unique_char)
```

    {'b', 'd', 'a', 'c'}



```python
unique_char.discard('d')
print(unique_char)
```

    {'b', 'a', 'c'}



```python
unique_char.clear()
print(unique_char)
```

    set()


#### 13.9.4 - difference and intersection


```python
unique_char = set(char_list)
print(unique_char)
```

    {'d', 'c', 'a', 'b'}



```python
print(unique_char.difference({'a', 'e', 'i'}))
```

    {'d', 'c', 'b'}



```python
print(unique_char.intersection({'a', 'e', 'i'}))
```

    {'a'}


### 13.10 - Regular expression

#### 13.10.1 - matching string


```python
pattern1 = "cat"
pattern2 = "bird"
string = "dog runs to cat"

print(pattern1 in string)
print(pattern2 in string)  
```

    True
    False


#### 13.10.2 - re


```python
import re

pattern1 = "cat"
pattern2 = "bird"
string = "dog runs to cat"

print(re.search(pattern1, string))
print(re.search(pattern2, string))
```

    <_sre.SRE_Match object; span=(12, 15), match='cat'>
    None


#### 13.10.3 - multiple patterns


```python
ptn = r"r[au]n"
print(re.search(ptn, "dog runs to cat"))
```

    <_sre.SRE_Match object; span=(4, 7), match='run'>



```python
print(re.search(r"r[A-Z]n", "dog runs to cat"))
print(re.search(r"r[a-z]n", "dog runs to cat"))
print(re.search(r"r[0-9]n", "dog r2ns to cat"))
print(re.search(r"r[0-9a-z]n", "dog runs to cat"))
```

    None
    <_sre.SRE_Match object; span=(4, 7), match='run'>
    <_sre.SRE_Match object; span=(4, 7), match='r2n'>
    <_sre.SRE_Match object; span=(4, 7), match='run'>


#### 13.10.4 - match by type

####  \d : decimal digit


```python
print(re.search(r"r\dn", "run r4n"))
```

    <_sre.SRE_Match object; span=(4, 7), match='r4n'>


#### \D : any non-decimal digit


```python
print(re.search(r"r\Dn", "run r4n"))
```

    <_sre.SRE_Match object; span=(0, 3), match='run'>


#### \s : any white space [\t\n\r\f\v]


```python
print(re.search(r"r\sn", "r\nn r4n"))
```

    <_sre.SRE_Match object; span=(0, 3), match='r\nn'>


#### \S : opposite to \s, any non-white space


```python
print(re.search(r"r\Sn", "r\nn r4n"))
```

    <_sre.SRE_Match object; span=(4, 7), match='r4n'>


#### \w : [a-zA-Z0-9_]


```python
print(re.search(r"r\wn", "r\nn r4n"))
```

    <_sre.SRE_Match object; span=(4, 7), match='r4n'>


#### \W : opposite to \w


```python
print(re.search(r"r\Wn", "r\nn r4n"))
```

#### \b : empty string (only at the start or end of the word)


```python
print(re.search(r"\bruns\b", "dog runs to cat"))
```

    <_sre.SRE_Match object; span=(4, 8), match='runs'>


#### \B : empty string (but not at the start or end of a word)


```python
print(re.search(r"\B runs \B", "dog   runs  to cat"))
```

    <_sre.SRE_Match object; span=(5, 11), match=' runs '>


#### \\ : match \


```python
print(re.search(r"runs\\", "runs\ to me"))
```

    <_sre.SRE_Match object; span=(0, 5), match='runs\\'>


#### . : match anything (except \n)


```python
print(re.search(r"r.n", "r[ns to me"))
```

    <_sre.SRE_Match object; span=(0, 3), match='r[n'>


#### ^ : match line beginning


```python
print(re.search(r"^dog", "dog runs to cat"))
```

    <_sre.SRE_Match object; span=(0, 3), match='dog'>


#### $ : match line ending


```python
print(re.search(r"cat$", "dog runs to cat"))
```

    <_sre.SRE_Match object; span=(12, 15), match='cat'>


#### ? : may or may not occur


```python
print(re.search(r"Mon(day)?", "Monday"))
print(re.search(r"Mon(day)?", "Mon"))  
```

    <_sre.SRE_Match object; span=(0, 6), match='Monday'>
    <_sre.SRE_Match object; span=(0, 3), match='Mon'>


#### flags


```python
string = """
dog runs to cat.
I run to dog.
"""
print(re.search(r"^I", string))
```

    None



```python
print(re.search(r"^I", string, flags=re.M))  
```

    <_sre.SRE_Match object; span=(18, 19), match='I'>


#### 13.10.5 - duplicate match

#### * : occur 0 or more times


```python
print(re.search(r"ab*", "a"))
print(re.search(r"ab*", "abbbbb"))
```

    <_sre.SRE_Match object; span=(0, 1), match='a'>
    <_sre.SRE_Match object; span=(0, 6), match='abbbbb'>


#### + : occur 1 or more times


```python
print(re.search(r"ab+", "a"))
print(re.search(r"ab+", "abbbbb"))  
```

    None
    <_sre.SRE_Match object; span=(0, 6), match='abbbbb'>


#### {n, m} : occur n to m times


```python
print(re.search(r"ab{2,10}", "a"))
print(re.search(r"ab{2,10}", "abbbbb"))
```

    None
    <_sre.SRE_Match object; span=(0, 6), match='abbbbb'>


#### {n} : occur n  times


```python
print(re.search(r"ab{3}", "abbcbb"))
print(re.search(r"ab{3}", "abbbbb"))
print(re.search(r"ab{4}", "abbbbb"))
print(re.search(r"ab{5}", "abbbbb"))
```

    None
    <_sre.SRE_Match object; span=(0, 4), match='abbb'>
    <_sre.SRE_Match object; span=(0, 5), match='abbbb'>
    <_sre.SRE_Match object; span=(0, 6), match='abbbbb'>


#### 13.10.6 - group


```python
match = re.search(r"ID:(\d+), Date:(.+)", "ID:021523, Date: Feb/12/2017")
print(match.group())
```

    ID:021523, Date: Feb/12/2017



```python
print(match.group(1))
```

    021523



```python
print(match.group(2))
```

     Feb/12/2017


#### group by name


```python
match = re.search(r"ID:(?P<id>\d+), Date:(?P<date>.+)", "ID:021523, Date: Feb/12/2017")
print(match.group('id'))
print(match.group('date'))
```

    021523
     Feb/12/2017


#### 13.10.7 - findall

#### findall


```python
print(re.findall(r"r[ua]n", "run ran ren"))
```

    ['run', 'ran']


#### | : or


```python
print(re.findall(r"run|ran", "run ran ren"))
```

    ['run', 'ran']


#### 13.10.8 - sub vs replace


```python
print(re.sub(r"r[au]ns", "catches", "dog runs to cat"))
```

    dog catches to cat


#### 13.10.9 - split


```python
print(re.split(r"[,;./]", "a;b,c.d;e.y/u/uu"))   
```

    ['a', 'b', 'c', 'd', 'e', 'y', 'u', 'uu']


#### 13.10.9 - compile


```python
compiled_re = re.compile(r"r[ua]n")
print(compiled_re.search("dog ran to cat"))
```

    <_sre.SRE_Match object; span=(4, 7), match='ran'>



## It's Over
