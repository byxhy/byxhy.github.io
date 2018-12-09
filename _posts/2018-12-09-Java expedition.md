---
layout: post
title: "The road to the Java expedition"
author: "Xhy"
categories: java
tags: [improve]
image: maximilian-weisbecker.jpg
---

Photo by maximilian-weisbecker


>声明：本系列按照[james_yuan](http://www.imooc.com/u/1349694/courses?sort=publish)老师的C++课程的学习路径整理而来，添加少量学习注释。最近沉迷学习，无法自拔，跟着慕课老师james_yuan学习C++，秉承着先上路再迭代的思想，出发啦 ..

<br />

## Table of Contents

* [Java入門 第一季 ](#1)
	* C++语言新特性
		* C++输入输出流
		* C++新特性以及输入输出
		* namespace-命名空间的学习
	* 綜合
		* 练习：求最大值
* [C++远征之离港篇](#2)
	* C++语言新特性
		* C++特性之引用
		* C++特性之const
	* C++语言新亮點
	* 內存管理


<br />
<br />

<h3 id="1"> 一、Java入門 第一季 ☂</h3>

**第1章 Java初体验**

1] 簡介及環境配置
* Ubuntu系統下JDK下載安裝
	* [JDK下載](https://www.oracle.com/technetwork/java/javase/downloads/jdk11-downloads-5066655.html)
	* sudo apt-get install ./jdk-11.0.1_linux-x64_bin.deb
* 配置環境配置
	* cd /etc/profile.d/
	* sudo gedit mypath.sh
	* export JAVA_HOME=/usr/lib/jvm/jdk-11.0.1
	* export PATH=$PATH:$JAVA_HOME/bin
* 使環境變量生效和測試是否安裝成功
	* source mypath.sh
	* java -version
* Install Eclipse IDE
	* tar xfz eclipse-java-2018-09-linux-gtk-x86_64.tar.gz
	* ./eclipse

2] C++新特性以及输入输出

9] 拷贝构造函数

* 类内定义的函数优先编译为内联函数

```c++
/*
 **************************************************************************    	 
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : Encapsulation.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 01-12-2017
 *      Description:     
 **************************************************************************    	 
 */

```
update java expedition to line-94
