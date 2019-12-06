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
* [Prepare the dataset and save to pickle file][2]
* [Shuffle the dataset][3]
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



<h2 id="1"> 1. First impressions of deep learning</h2>

### *Lesson 1 - Tutorial*

- After the course, try to come up with some new ideas.
- Try to use deep learning to solve some problems in your life



### *Lesson 2 - Framework of deep learning-1*

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
  - -> TensoFlow2

- Chainer(Japan)
- MXNet



### *Lesson 3 - Framework of deep learning-2*

Forget TensorFlow 1.0 and start with TensorFlow 2.0.



### *Lesson 4 - Install Anaconda*

![1575604685668](/home/xhy/.config/Typora/typora-user-images/1575604685668.png)

[Anaconda Download](https://www.anaconda.com/distribution/)



### *Lesson 5 - Install TensorFlow 2*

```
▪ conda create -n tf2 tensorflow-gpu
▪ conda activate tf2
```



- *If you cannot connect the official source, you can switch to the domestic source*

  1. Config the domestic source

     ```
     conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
     ```

     ```
     conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
     ```

  2. Display the source address

     ```
     conda config --set show_channel_urls yes
     ```

  3. Install the TensorFlow

     ```
     conda create -n tf2 tensorflow-gpu
     ```

     ```
     conda activate tf2
     ```

- *Check your python  and ipython environment*

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



### *Lesson6 - Install Anaconda*



### Lesson6 - Install Anaconda

Lesson6 - Install Anaconda

Lesson6 - Install Anaconda

Lesson6 - Install Anaconda

Lesson6 - Install Anaconda

Lesson6 - Install Anaconda

Lesson6 - Install Anaconda




