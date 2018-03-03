---
layout: post
title: "ML-Python foundation"
author: "Xhy"
categories: journal
tags: [documentation,sample]
image: Trial.jpg
---

>声明：本系列根據[莫煩Python](https://morvanzhou.github.io/)老师的Python基礎教學網站的课程整理而來。學下來感覺不錯，那就秉承着先上路再迭代的思想，出发啦...　另外本系列所有的源碼可以點這裏[下載](http://www.imooc.com/u/1349694/courses?sort=publish)

---

### 篇节

<br />

[一、安装](#1)

[二、基本使用](#2)

[三、while 和 for 循环](#3)

[四、if 判断](#4)

[五、定义功能](#5)

[六、变量形式](#6)

[七、模块安装](#7)

[八、文件读取](#8)

[九、class 类](#9)

[十、input 输入](#10)

[十一、元组, 列表, 字典](#11)

[十二、模块](#12)

[十三、其他](#13)

---

<br />

<h3 id="1"> 一、安装 ☂</h3>

1] 安装

自己想辦法，正所謂:授人以魚不如授人以漁，假如你真的想的話，你就一定辦得到。另外在新學一件事情的時候，不要糾結與工具，不要糾結與細節，始終謹記你的出發點是什麼！

<br />

<h3 id="2"> 二、基本使用 ☂</h3>

1] print

```Python
print("This is Python basic tutorial")

print('This is Python basic tutorial')

print('This is Python' + ' basic tutorial')

print(3 + 4)

print(int('3') + 4)

print(float('3.4') + 3)

print(int('3.4'))     # ValueError: invalid literal for int() with base 10: '3.4'
```

2] basic computation
- addition subtraction multiplication and division

```Python
2 + 2

4 - 2

4 * 2

4 / 3
```

- ^ and **

```Python
2 ** 2

2 ** 3

2 ** 4
```

- %

```Python
9 % 2

7 % 3

5 % 3
```
3] variables

```Python
gun = 94
print(gun)

apple = 'iPhone X'
print(apple)

a, b, c = 9, 8, 7
print(a, b, c)
```
<br />

<h3 id="3"> 三、while 和 for 循环 ☂</h3>

1] while

```Python
condition = 0
while condition < 5:
    print(condition)
    condition += 1       # Err: condition++

conditoin = 5
while condition:
    print(condition)
    condition -= 1  
```
２] for

```Python
eg_list = [1, 2, 3, 4, 5, 6.2, 7, 8, 9, 10]
for i in eg_list:
    print(i)
    print('This is inside the for loop')

print('This is outside the for loop')
```
３] range

```Python
a = range(1, 10)
for i in a:
    print(i)

b = range(10)
for i in b:
    print(i)

c = range(1, 10, 2)
for i in c:
    print(i)
```

4] set type
- list

```Python
eg_list = [1, 2, 3, 4, 5, 6]
for i in eg_list:
    print(i)
```
- tuple

```Python
tup = ('WangEr', 38, 168.5)
for i in tup:
    print(i)
```
- dictionary (But the dictionary is out of order. refer OrderedDict)

```Python
dic = {}
dic['lan'] = 'python'
dic['version'] = 2.7
dic['platform'] = 64
for key in dic:
    print(key, '=', dic[key])
```
- set (The set collection will remove duplicates and it is out of order too)

```Python
s = set(['Python3', 'Python2', 'NXP', 'Python3'])
for i in s:
    print(i)
```
- iterator

```Python
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
-  yield

```Python
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
<br />

<h3 id="4"> 四、if 判断 ☂</h3>

1] if

```Python
x = 9
y = 6
if x < y:
    print('x is less than y')
else:
    print('x is more than y')
```

2] if else

- Although the following syntax is correct in Python, we still discourage it.

```Python
x = 3
y = 5
z = 9
if x < y < z:
    print('x is less than y, and y is less than z')
else:
    print('y is not sure')  
```

- The following syntax is recommenaded.

```Python
x = 3
y = 5
z = 3
if x < y and y < z:
    print('x is less than y, and y is less than z')
else:
    print('y is not sure')
```
3] if elif else

```Python
x = 3
y = 9
if x > y:
    print('x > y')
elif x < y:
    print('x < y')
else:
    print('x == y')
```

4] Trinocular operator ( var = var1 if condition else var2 )

```Python
worked = True
result = 'done' if worked else 'not yet'
print(result)
```

<h3 id="5"> 五、定义功能 ☂</h3>

1] def

```Python
def add(a, b):
    print('This is a add() function.')
    print('a + b =', a + b)

add(3, 7)
```

2] function parameter

```Python
def add(a, b):
    print('This is a add() function.')
    print('a + b =', a + b)

add(3, 7)
```

3] default parameter

```Python
def sale_car(price, color='red', brand='carmy', is_second_hand=True):
    print('price =', price, ','
          'color =', color, ','
          'brand =', brand, ','
          'is_second_hand =', is_second_hand)

sale_car(10000)
```

４] call yourself, only called inside the script

```Python
if __name__ == '__main__':
    #code_here
    print('main')
```

５] variable parameters

```Python
def sumGrades(name, *score):
    totalScore = 0
    for i in score:
        totalScore += i
    print(name, 'total score is', totalScore)

sumGrades('Lisa', 8)

sumGrades('Mery', 8, 7, 9)
```

６] keyword parameters

```Python
def keyPara(name, **kw):
    print('name is ', name)
    for k, v in kw.items():
        print(k, v)

keyPara('Mary', Gender = 'woman', country = 'US', age = '23')
```

- PA:

1) Default, variable and keyword parameters should be behind all the function parameters

2) Through the variable parameters and keyword parameters, all function can be replaced by universal_func(*args, **kw)

---

    [六、变量形式](#6)
    - 6.1 全局 & 局部 变量

    [七、模块安装](#7)
    - 7.1 模块安装

    [八、文件读取](#8)
    - 8.1 读写文件 1
    - 8.2 读写文件 2
    - 8.3 读写文件 3

    [九、class 类](#9)
    - 9.1 class 类
    - 9.2 class 类 init 功能

    [十、input 输入](#10)
    - 10.1 input 输入

    [十一、元组, 列表, 字典](#11)
    - 11.1 元组 列表
    - 11.2 list 列表
    - 11.3 多维列表
    - 11.4 dictionary 字典

    [十二、模块](#12)
    - 12.1 import 模块
    - 12.2 自己的模块

    [十三、其他](#13)
    - 13.1 continue & break
    - 13.2 try 错误处理
    - 13.3 zip lambda map
    - 13.4 copy & deepcopy 浅复制 & 深复制
    - 13.5 threading 什么是多线程
    - 13.6 multiprocessing 什么是多进程
    - 13.7 什么是 tkinter 窗口
    - 13.8 pickle 保存数据
    - 13.9 set 找不同
    - 13.10 正则表达式




全剧终！

不知道为什么，一直以来就想着早点尽快地学完，没想等真的学完了，竟有一种失落的感觉，就好像一直追一部剧，成了一种习惯，突然之间全剧终了……

或许该是用的时候了，上路吧~

去慕课致谢！谢谢您
