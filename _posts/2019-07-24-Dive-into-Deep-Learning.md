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

# Why TensorFlow


Update the Atom to 1.39.0



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
