---
layout: post
title: "Speech separation"
author: "Xhy"
categories: Speech
tags: [improve]
image: jason-rosewell.jpg
---

Photo by jason-rosewell


> Speech separation is the task of separating target speech from background interference. Traditionally, speech separation is studied as a signal processing problem. A more recent approach formulates speech separation as a supervised learning problem, where the discriminative patterns of speech, speakers, and background noise are learned from training data. Over the past decade, many supervised separation algorithms have been put forward. In particular, the recent introduction of deep learning to supervised speech separation has dramatically accelerated progress and boosted separation performance. 

#### <br />




## Table of Contents

* [Background][1]
* [Papers][2]
* [Summary][3]
  * Evaluation
  * Algorithm
* [Trend][4]

[1]:	#1
[2]:	#2
[3]: #3
[4]: #4



<br />

<h3 id="1"> 1. Background</h3>
语音分离的问题来自“鸡尾酒会问题”，虽然酒会上很嘈杂，但每个人都能选择性分离出自己想听的声音，那有没有办法对采集回来有干扰的音频信号也做同样的处理呢？



根据干扰的不同，语音分离任务可以分为以下三类：

- 当干扰为噪声信号时，可以称为“语音增强”（Speech Enhancement）
- 当干扰为其他说话人时，可以称为“多说话人分离”（Speaker Separation）
- 当干扰为目标说话人自己声音的反射波时，可以称为“解混响”（De-reverberation）



\[1] Shixue Wen. Speech Separation Based on Deep Learning

![png](/assets/img/SpeechEnhancement/speech separation.png)



<h3 id="2"> 2. Papers</h3>
**\[1] 李号. 基于深度学习的单通道语音分离[D]. 内蒙古大学，2017.**

语音信号知识基础：

- 人体发声系统
  - 产生（人体多器官相互配合带动声带振动发声）
  - 清音（短时谱上类似白噪声，分离难度大）
  - 浊音（声带振动，有明显谐波结构）
  - 基音（声带开启和闭合一次的时间差为一个基音周期）
    - 80 ～ 500Hz
    - 纯净语音谱中，谐波结构清晰，基音特征明显，加入噪声后模糊
  - 短时平稳性（10-30ms内语音是相对平稳的，可认为统计特征不变）
- 人耳听觉感知系统

片段点：

- 听觉系统

  - 响度
  - 音色
  - 音调

- 高频定位性

- 掩蔽效应，强信号会盖过弱信号，使人无法感觉到弱信号的存在

  - 定义
  - 特性

- 不同噪声降噪方案

  - 周期性噪声 （陷波滤波器）
  - 脉冲噪声（时域上进行，根据平均幅度确定脉冲阈值，判定，直接幅值衰减或用相邻采样点幅值插值平滑）
  - 宽带噪声（特别地，整个频率上分布均匀平稳的宽带噪声又称为高斯白噪声，谱减法，自适应对消）

- 语音分离效果评价

  - 主观
    - DRT-Diagnostic Rhyme Test （判断韵字测试）
    - MOS-Mean Opinion Score （平均得分意见）
    - DAM-Diagnostic Acceptability Measure （判断满意度测试）
  - 客观
    - PESQ-Perceptual evaluation of speech quality（语音质量感知）
    - POLQA-Perceptual Objective Listening Quality Analysis（感知客观语音质量评估，即ITU-T的P.863）
    - SAR-Source to Artifact Ratio（信噪伪影比）
    - SDR-Source to Distortion Ratio（信噪失真比）
    - SIR-Source to Interference Ratio（信噪干扰比）
    - [PESQ和POLQA可以用仪器来测试](http://cn.ap.com/info26)![png](/assets/img/SpeechEnhancement/PESQ-POLQA.png)
  
- 基于无监督学习的语音分离技术

  - 谱减法，可能产生音乐噪声
    - 静音噪声估计
    - VAD
    - 谱相减
    - 重构
    - 适用平稳噪声，对于非平稳噪声 容易出现语音失真![png](/assets/img/SpeechEnhancement/Spectral Subtraction.png)
  
  - 维纳滤波，残留噪声类似白噪声
    - 语音噪声
    - 白噪声
    - 卡尔曼滤波（非平稳噪声）
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



---



**\[2] 李素华. EVS音频流无参考客观质量评估研究[D]. 西安电子科技大学，2017.**

音频主客观质量评估方法研究



---



**\[3] 张建伟，陶亮，周健，王华彬.噪声谱估计算法对语音可懂度的影响[J]. 声学学报 2015(05).**

STOI 算法中，同时输入干净的语音 x(n)和经过增强算法重建的干净语音估计 y(n)，STOI 算法会给出一个(0, 1)范围内的值，STOI 值越大，表示处理后的语音的可懂度越高



[STOI](https://github.com/mpariente/pystoi)

![1572694725389](/home/xhy/.config/Typora/typora-user-images/1572694725389.png)



---



[A toolbox for performance measurement in (blind) source separation](http://bass-db.gforge.inria.fr/bss_eval/)

[mir_eval Documentation](https://craffel.github.io/mir_eval/)



---





**\[4] 卓 嘎，次仁尼玛.基于 Matlab的藏语语音频谱仿真和分析[J]. 电子设计工程2019， 19(10):170-173.**



---

Methods for subjective determination of transmission quality



---

[ZZZ] Shixue Wen. Speech Separation Based on Deep Learning



---

<h3 id="3"> 3. Summary</h3>

#### 3.1 Speech quality evaluation

- 常见音频主观质量评估
  - 绝对分类评级（Absolute Category Rating, ACR）![1572680674580](/home/xhy/.config/Typora/typora-user-images/1572680674580.png)
  - 失真分类评级（Degradation Category Rating, DCR）![1572680727347](/home/xhy/.config/Typora/typora-user-images/1572680727347.png)
  - 对照分类评级（Comparison Category Rating, CCR）![1572680755956](/home/xhy/.config/Typora/typora-user-images/1572680755956.png)
  - DAM（Diagnostic Acceptability Measure，满意度测量）
- 客观质量评估
  - 有参考评估方法
    - 短时客观可懂度(Short-Time Objective Intelligibility, STOI)
    - 语音质量感知评估(Perceptual Evaluation of Speech Quality, PESQ)
    - 感知客观听力质量评估(Perceptual Objective Listening Quality Assessment, POLQA)
    - 信噪比(Signal-to-Noise Ratio, SNR)
    - 信噪干扰比(Source to Interference Ratio, SIR)
    - 信噪伪影比(Source to Artifact Ratio, SAR)
    - 信噪失真比(Source to Distortion Ratio, SDR)
  - 无参考评估方法
    - 规划层评估模型
    - 包层评估模型
    - 比特层评估模型
    - 媒体层评估模型
    - 混合评估模型
- 总结
  - 主观质量评估能很好地反映人的真实感受，可以以ACR、DCR和CCR为主
  - 降噪效果在客观质量评估上可以用STOI、PESQ这些指标来衡量
  - SIR、SAR、SDR可以用来评估模型语音分离的性能

#### 3.2 Speech separation algorithm



<h3 id="4"> 4. Trend</h3>

[Looking to Listen: Audio-Visual Speech Separation](http://ai.googleblog.com/2018/04/looking-to-listen-audio-visual-speech.html)

