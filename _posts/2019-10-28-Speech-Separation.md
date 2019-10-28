---
layout: post
title: "Speech separation"
author: "Xhy"
categories: Speech
tags: [improve]
image: jason-rosewell-ASKeuOZqhYU.jpg
---

Photo by jason-rosewell-ASKeuOZqhYUl


> 语音增强旨在通过利用信号处理算法提高语音的质量和可懂度。主要包括1.语音解混响，混响是由于空间环境对声音信号的反射产生的；2，语音降噪，干扰主要来源于各种环境和人的噪声；3.语音分离，噪声主要来源于其他说话人的声音信号。通过去除这些噪声或者人声来提高语音的质量。现已经应用于现实生活中，如电话、语音识别、助听器、VoIP以及电话会议系统等。 ……

#### <br />




## Table of Contents

* [背景目标][1]
* [相关论文][2]
* [发展趋势][3]

[1]:	#1
[2]:	#2
[3]: #3



<br />

<h3 id="1"> 1. 背景目标</h3>
语音分离的问题来自“鸡尾酒会问题”，虽然酒会上很嘈杂，但每个人都能选择性分离出自己想听的声音，那有没有办法对采集回来有干扰的音频信号也做同样的处理呢？



根据干扰的不同，语音分离任务可以分为以下三类：

- 当干扰为噪声信号时，可以称为“语音增强”（Speech Enhancement）
- 当干扰为其他说话人时，可以称为“多说话人分离”（Speaker Separation）
- 当干扰为目标说话人自己声音的反射波时，可以称为“解混响”（De-reverberation）



\[1] Shixue Wen. Speech Separation Based on Deep Learning

![png](/assets/img/SpeechEnhancement/speech separation.png)

![speech separation](/home/xhy/github/byxhy.github.io/assets/img/SpeechEnhancement/speech separation.png)





<h3 id="2"> 2. 相关论文</h3>

\[1] 李号.基于深度学习的单通道语音分离[D].内蒙古：内蒙古大学，2017.



疑点：

- 谱减法  音乐噪声
- 维纳滤波 高斯白噪声  区别是啥  （平稳噪声，对于非平稳噪声 容易出现语音失真）
- 卡尔曼滤波（非平稳噪声）





切入点：

- 清音（短时谱上类似白噪声）
- 浊音（声带振动，有明显谐波结构）





片段点：

- 听觉系统

  - 响度
  - 音色
  - 音调

- 高频定位性

- 掩蔽效应，强信号会盖过弱信号，使人无法感觉到弱信号的存在

  - 定义
  - 特性

- 噪声

  - 周期性噪声 （陷波滤波器）
  - 脉冲噪声（时域上进行，根据平均幅度确定脉冲阈值，判定，直接幅值衰减或用相邻采样点幅值插值平滑）
  - 宽带噪声（特别地，整个频率上分布均匀平稳的宽带噪声又称为高斯白噪声，谱减法，自适应对消）

- 语音分离效果评价

  - 主观
    - DRT-Diagnostic Rhyme Test （判断韵字测试）
    - MOS-Mean Opinion Score （平均得分意见）
    - DAM-Diagnostic Acceptability Measure （判断满意度测试）
  - 客观
    - PESQ-Perceptual evaluation of speech quality（主观语音质量评估 -0.5-4.5）
    - POLQA（Perceptual Objective Listening Quality Analysis，感知客观语音质量评估），即ITU-T的P.863
    - SAR
    - SDR
    - SIR
    - POLQA 和 PESQ  http://cn.ap.com/info26
    -
    - ![img](http://cn.ap.com/upfile/image/20190721/20190721171947_81845.png)
    - ![img](http://cn.ap.com/upfile/image/20190721/20190721172427_83811.png)

- 基于无监督学习的语音分离技术

  - 谱减法（静音噪声估计-》voice activity detection-》相减-》重构）
  - ![image-20191028081811083](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20191028081811083.png)
  - 维纳滤波
    - 语音噪声
    - 白噪声
  - 自适应滤波
  - ![image-20191028082238254](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20191028082238254.png)
  - 计算听觉场景分析-CASA
    - ![image-20191028082446616](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20191028082446616.png)
    - 难点在于低信噪比时基音提取困难
    - 清音短时谱上表现类似白噪声，容易被当作噪声去掉
    - 目标源聚类问题



- 基于有监督学习的语音分离技术

  - HMM
  - 浅层人工神经网络
    - 权值随机初始化
    - 规模小
    - 数据量小
  - DNN
    - 非线性学习能力
    - 输入为时频分解的特征，没有考虑语音在统计意义上的特点
    - ![Screenshot from 2019-10-28 15-29-14](/home/xhy/Pictures/Screenshot from 2019-10-28 15-29-14.png)
  - NMF（与人类从局部到整体的思想是一致的）
    - 服务范围
    - 服务无法![Screenshot from 2019-10-28 15-35-04](/home/xhy/Pictures/Screenshot from 2019-10-28 15-35-04.png)
    -
  - 我wf

---

## [Looking to Listen: Audio-Visual Speech Separation](http://ai.googleblog.com/2018/04/looking-to-listen-audio-visual-speech.html)