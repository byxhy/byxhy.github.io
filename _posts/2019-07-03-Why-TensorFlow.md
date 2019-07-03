---
layout: post
title: "Why TensorFlow"
author: "Xhy"
categories: Machine Learning
tags: [Reading]
image: TF by Google IO.gif
---

Photo from TensorFlow @ Google I/O ’19 Recap


>TensorFlow is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of tools, libraries and community resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML powered applications.

<br />

# Why TensorFlow

[TOC]



## Install TensoFlow

## Delete TF

### Install the Python development environment on your system

- ```
  sudo pip3 install -U virtualenv
  ```



### Create a virtual environment (recommended)

- ```
  cd /home/xhy
  ```

- ```
  virtualenv --system-site-packages -p python3 ./venv
  ```

- if you meet this problem,  alueError: Unable to determine SOCKS version from socks://127.0.0.1:1081/  

  Source: https://blog.csdn.net/xingkongyidian/article/details/85162285   

  Reason: Shadowsocks

- ```
  unset all_proxy && unset ALL_PROX
  ```



### Install the TensorFlow pip package

**Virtualenv install**

- ```
  unset all_proxy && unset ALL_PROX
  ```

  ```bsh
  pip install --upgrade tensorflow
  ```

**Verify the install**

```bsh
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

if you meet this error:  Original error was: cannot import name 'multiarray' from 'numpy.core' (/usr/local/lib/python3.6/dist-packages/numpy/core/__init__.py), you can find something from this website(https://blog.csdn.net/langb2014/article/details/78401713)

Flow:

- cd /usr/local/lib/python3.6/dist-packages
- sudo mkdir numpy_backup
- sudo mv numpy* ./numpy_backup
- pip install numpy
  tip: Requirement already satisfied: numpy in /home/xhy/anaconda3/lib/python3.7/site-packages (1.16.4)
- Then the default python numpy library automatic change to anaconda3/lib



**OK**

- python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
- Tensor("Sum:0", shape=(), dtype=float32)



TensorFlow official website

https://www.tensorflow.org/install/pip



Felix,  Sun Jun 23 , 15:16



### Create a virtual conda environment for TF

- conda create -n conda_venv_tf
- conda activate conda_venv_tf
- conda deactivate



### [TensorFlow with Jupyter Notebooks using Virtualenv](http://rndness.com/blog/2018/2/4/tensorflow-with-jupyter-notebooks-using-virtualenv)

##### if still meet this issue:  [No module named tensorflow in jupyter](https://stackoverflow.com/questions/38221181/no-module-named-tensorflow-in-jupyter)

##### Try this:

Jupyter runs under the conda environment where as your tensorflow install lives outside conda. In order to install tensorflow under the conda virtual environment run the following command in your terminal:

```
 conda install -c conda-forge tensorflow
```
