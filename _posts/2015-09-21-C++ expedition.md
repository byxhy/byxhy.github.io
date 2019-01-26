---
layout: post
title: "The road to the C++ expedition"
author: "Xhy"
categories: c++
tags: [improve]
image: joshua-earle.jpg
---

Photo by joshua-earle


>声明：本系列按照[james_yuan](http://www.imooc.com/u/1349694/courses?sort=publish)老师的C++课程的学习路径整理而来，添加少量学习注释。最近沉迷学习，无法自拔，跟着慕课老师james_yuan学习C++，秉承着先上路再迭代的思想，出发啦 ..

<br />

## Table of Contents

* [C++远征之起航篇](#1)
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
* [C++远征之封装篇（上）](#3)
	* C++类与对象初体验
		* 簡介
		* C++类与对象初体验
	* C++對象的封裝
		* C++初始字符串类型
		* C++属性封装代码演示
		* C++精彩的类外定义
	* C++對象的生離死別
	  * 默認构造函数演示
		* 构造函数初始化列表
		* 拷贝构造函数
* [C++远征之封装篇（下）](#4)
	* C++对象成員與对象数组
		* 对象数组实践
		* 对象成員
	* C++语言新亮點
		* C++新亮點之默认参数
		* C++新亮點之重載函數
		* C++新亮點之內聯函數
* [C++远征之继承篇](#5)
	* C++语言新特性
		* C++输入输出流
		* C++新特性以及输入输出
		* namespace-命名空间的学习
	* 綜合
		* 练习：求最大值
* [C++远征之多态篇](#6)
	* C++语言新特性
		* C++特性之引用
		* C++特性之const
	* C++语言新亮點
		* C++新亮點之默认参数
		* C++新亮點之重載函數
		* C++新亮點之內聯函數
* [C++远征之模板篇](#7)
	* C++语言新特性
		* C++特性之引用
		* C++特性之const
	* C++语言新亮點
		* C++新亮點之默认参数
		* C++新亮點之重載函數
		* C++新亮點之內聯函數

<br />
<br />

<h3 id="1"> 一、C++远征之起航篇 ☂</h3>

1] C++输入输出流

```c++
/*
 **************************************************************************    	 
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : Hello.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 11-02-2016
 *      Description:     
 **************************************************************************    	 
 */

 #include <iostream>
 #include <stdlib.h>

 using namespace std;

 int main(int argc, const char * argv[])
 {
     cout << "Hello tomorrow !" << endl;

     system("pause");

     return 0;
 }
```

2] C++新特性以及输入输出

```c++
/*
 **************************************************************************    	 
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : CinCout.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 11-03-2016
 *      Description:     
 **************************************************************************    	 
 */

 #include <iostream>
 #include <stdlib.h>

 using namespace std;

 int main(int argc, const char * argv[])
 {
     int x = 0;

     cout << "Please input a integer number: ";
     cin >> x;  //VS: getline(cin, x);

     cout << oct << x << endl;
     cout << dec << x << endl;
     cout << hex << x << endl;

     bool y = 0;
     cout << "Please input a bool value: ";
     cout << boolalpha << y << endl;

     //PA: C++11的safe-bool标准: 只有在上下文需要判断bool条件的时候才会自动转换为bool类型
     //这里没有需要判断的情况，所以输入除1外的其他值，经过cin处理直接就变成false了
     //所以，不要输入bool值了
     //想想cin如果在一个循环里会怎样(提示: clear)


     //Err: Infinite Loop
     //while (true)
     //{
     //cin >> y;
     //cout << boolalpha << y << endl;
     //}


     //True:
     //while (true)
     //{
     //cin.clear();
     //cin >> y;
     //cout << boolalpha << y << endl;
     //}


     system("pause");

     return 0;
 }
```


3] namespace-命名空间的学习

```c++
/*
 **************************************************************************    	 
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : Namespace.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 11-03-2016
 *      Description:     
 **************************************************************************    	 
 */

 #include <iostream>
 #include <stdlib.h>

 using namespace std;

 namespace CompanyA
 {
     int x = 1;

     void fun()
     {
         cout << "CompanyA" << endl;
     }
 }

 //这是一个好的习惯，对于协作来说
 namespace CompanyB
 {
     int x = 2;

     void fun()
     {
         cout << "CompanyB" << endl;
     }

     void fun2()
     {
         cout << "Company2B" << endl;
     }
 }

 //必须先有定义好的名字空间，才能去使用，不能放在实现之前
 using namespace CompanyB;

 int main(int argc, const char * argv[])
 {
     cout << CompanyA::x << endl;
     CompanyA::fun();

     //推荐这种写法
     cout << CompanyB::x << endl;
     CompanyB::fun();

     //VS: 使用了名字空间 CompanyB
     cout << x << endl;
     fun();
     fun2();

     system("pause");

     return 0;
 }
```

4] 练习：求最大值
```c++
/*
 **************************************************************************    	 
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : GetMaxOrMin.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 11-05-2016
 *      Description:     
 **************************************************************************    	 
 */

 #include <iostream>
 #include <stdlib.h>

 using namespace std;

 namespace CompanyA
 {
     int getMaxOrMin(int * arr, int count, bool isMax = true);
 }


 int main(int argc, const char * argv[])
 {
     int arr[7] = { 3, 4, 1, 8, 9, 1, 2 };

     cout << "Max is " << CompanyA::getMaxOrMin(arr, 7) << endl;
     cout << "Min is " << CompanyA::getMaxOrMin(arr, 7, false) << endl;

     system("pause");

     return 0;
 }

 int CompanyA::getMaxOrMin(int *arr, int count, bool isMax)
 {
     //PA: 一定要赋第一个值给ret
     int ret = * arr;

     if (isMax)
     {
         //从第一个起
		     for (int i = 1; i < count; i++)
         {
             ret = (ret >= arr[i]) ? ret : arr[i];
         }
     }
     else
     {
         for (int i = 1; i < count; i++)
         {
             ret = (ret <= arr[i]) ? ret : arr[i];
         }
     }

     return ret;
 }
```
<h3 id="2"> 二、C++远征之离港篇 ☂</h3>
1] C++特性之引用

```c++
/*
 **************************************************************************    	 
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : Reference.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 11-05-2016
 *      Description:     
 **************************************************************************    	 
 */

 #include <iostream>
 #include <stdlib.h>

 using namespace std;

 //1' 基本数据类型的引用 &
 int testBasicStrRef()
 {
     int x = 0;
     cout << "x = " << x << endl;

     int &y = x;  //引用必须初始化
     y = 3;

     cout << "x = " << x << endl;

     return 0;
 }

 //2' struct的引用 &
 int testStructRef()
 {
     typedef struct
     {
         //Err: 非类范围内的匿名联合的成员不允许类内初始值设定项
         //int x = 1;
         //int y = 2;

         int x;
         int y;
     }Coor;

     Coor c1 = {1, 2};
     cout << "c1.x = " << c1.x << "  c1.y = " << c1.y << endl;

     Coor &c2 = c1;  //引用必须初始化
     c2.x = 3;
     c2.y = 4;

     cout << "c1.x = " << c1.x << "  c1.y = " << c1.y << endl;

     return 0;
 }

 //3' 指针的引用 *&
 int testPointRef()
 {
     int x = 9;
     cout << "x  = " << x << endl;

     int * p = &x;
     cout << "* p = " << * p << endl;

     int * &q = p;  //引用必须初始化
     * q = 3;
     cout << "x  = " << x << endl;
     cout << "* p = " << * p << endl;

     return 0;
 }

 //VS: void swap(int x, int y)
 void swap(int &x, int &y)
 {
     int tmp = x;

     x = y;
     y = tmp;
 }
 //4' 引用作函数参数 Vs 指针
 int testFuncPrameRef()
 {
     int x = 3;
     int y = 9;
     cout << "x = " << x << " y = " << y << endl;

     swap(x, y);
     cout << "x = " << x << " y = " << y << endl;

     return 0;
 }


 int main(int argc, const char * argv[])
 {
     testBasicStrRef();
     testStructRef();
     testPointRef(); //PA: 理解下
     testFuncPrameRef();

     system("pause");

     return 0;
 }
```

2] C++特性之const

```c++
/*
 **************************************************************************    	 
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : Const.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 11-06-2016
 *      Description:     
 **************************************************************************    	 
 */

 #include <iostream>
 #include <stdlib.h>

 using namespace std;

 //1' 修饰基本数据类型
 int testBasicStrCon()
 {
     //Vs: #define X  3
     const int x = 6;
     int const y = 9;

     //x = 5;   //Err: const修饰之后的值不能改变
     //y = 8;

     cout << "x = " << x << endl;
     cout << "y = " << y << endl;

     return 0;
 }

 //2' const 修饰指针常量左边(左数右指法、整体法, 有没有开锁就是了)
 int testPointConLift()
 {
     int x = 6;
     int const * p = &x;  //左数，开锁就是值了,那值不能改变
     cout << "* p = " << * p << endl;

     //* p = 4;		   //Err: 值不能改变

     x = 4;
     cout << "* p = " << * p << endl;

     return 0;
 }

 //3' const 修饰指针常量右边
 int testPointConRight()
 {
     int x = 6;
     int y = 9;

     int * const p = &x;  //右指，没开锁,那就是指针不能变

     cout << "* p = " << * p << endl;

     * p = 4;
     cout << "* p = " << * p << endl;
     cout << " x = " << x << endl;

     //p = &y;		   //Err: 指针不能变

     return 0;
 }


 //4' const 修饰指针，将权限大的变量赋值给权限小的
 int testPointLimit()
 {
     int x = 6;
     int const * p = &x;	//PA: 权限大的 赋值给 权限小的
     cout << "* p = " << * p << endl;

     //PA: 反过来就不行了
     //const int y = 4;
     //int * q = &y;		//通过*q有可能就操作了y的值，报错

     return 0;
 }

 //5' const 修饰引用
 int testReference()
 {
     int x = 6;
     int const &y = x;

     cout << "y = " << y << endl;

     //y = 4;		//error!!!
     x = 4;
     cout << "y = " << y << endl;

     return 0;
 }

 void fun(const int &a, const int &b)
 {
     //a = 1;	 //error !!!
     //b = 2;	 //error !!!
 }

 //6' const 修饰函数形参(避免改变传入的变量的值)
 int testFuncParameter()
 {
     int x = 6;
     int y = 9;
     cout << "x = " << x << " y = " << y << endl;

     fun(x, y);
     cout << "x = " << x << " y = " << y << endl;

     return 0;
 }


 int main(int argc, const char * argv[])
 {
     //testBasicStrCon();

     //testPointConLift();

     //testPointConRight();

     //testPointLimit();

     //testReference();

     testFuncParameter();

     system("pause");

     return 0;
 }
 ```
 3] C++亮點之默认参数

```c++
/*
 **************************************************************************    	 
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : DefaultPara.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 11-06-2016
 *      Description:     
 **************************************************************************    	 
 */

 #include <iostream>
 #include <stdlib.h>

 using namespace std;

 //3' C++函数特性之默认参数	Q:要求，好处
 void fun_default(int a = 10, int b = 20, int c = 30)
 {
    cout << "a = " << a << "," << "b = " << b << "," << "c = " << c << endl;
 }
 int main()
 {
     fun_default();
     fun_default(100);
     fun_default(100, 200);
     fun_default(100, 200, 300);

     return 0;
 }
```
4] C++亮點之重載函數

```c++
/*
 **************************************************************************    	 
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : Overload.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 11-06-2016
 *      Description:     
 **************************************************************************    	 
 */

 #include <iostream>
 #include <stdlib.h>

 using namespace std;

 //4' C++函数特性之重载函数	Q:要求，好处
 void funOverload(double i = 0.1, double j = 0.2)
 {
     cout << "i = " << i << "," << "j = " << j << endl;
 }

 void funOverload(int a = 10, int b = 20, int c = 30)
 {
     cout << "a = " << a << "," << "b = " << b << "," << "c = " << c << endl;
 }

 int main()
 {
     //Err: 找全部参数都有默认值的函数，如果两个函数都有默认参数，则报错
     //Err: funOverload();   

     funOverload(1, 2);
     funOverload(0.1);

     funOverload(100);
     funOverload(100, 200);
     funOverload(100, 200, 300);

     return 0;
 }
```
5] C++亮點之內聯函數

```c++
/*
 **************************************************************************    	 
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : Inline.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 11-06-2016
 *      Description:     
 **************************************************************************    	 
 */

 #include <iostream>
 #include <stdlib.h>

 using namespace std;

 //5'  C++函数特性之内联函数	Q:要求，好处
 inline void funInline(double i = 0.1, double j = 0.2)
 {
     cout << "i = " << i << "," << "j = " << j << endl;
 }

 inline void funInline(int a = 10, int b = 20, int c = 30)
 {
     cout << "a = " << a << "," << "b = " << b << "," << "c = " << c << endl;
 }

 int main()
 {
     funInline(1, 2);   //首先去找全部参数都有默认值的函数，若都有，就报错
     funInline(100);
     funInline(100, 200);
     funInline(100, 200, 300);

     funInline(0.1);

     return 0;
 }
```

6] 內存管理

```c++
/*
 **************************************************************************    	 
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : MemoryManagement.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 11-06-2016
 *      Description:     
 **************************************************************************    	 
 */

 #include <iostream>
 #include <string.h>
 #include <stdlib.h>

 using namespace std;

 //1' C++申请内存，四部曲，对比malloc
 int testMemory()
 {
     int * p = new int;
     //1、判断申请是否成功
     if (NULL == p)
     {
          cout << "new fail" << endl;
         return 0;
     }

     //2、使用
     * p = 10;

     cout << "* p = " << * p << endl;

     //3、释放
     delete p;


     //4、置空
     p = NULL;

     return 0;
 }

 //2' C++申请内存块，四部曲
 int testMemoryBlock()
 {
     char * str1 = new char[50];

     //1、判断申请是否成功
     if (NULL == str1)
     {
         cout << "new fail" << endl;
        return 0;
     }

     //2、使用
     //char * str2 = "Cannot stoping ...";  //Err: ??????(程序静态区)
     const char * str2;
     str2 = "Cannot stoping ...";

     //Err: int len2 = sizeof(str2);  // sizeof得到的是指针地址大小
     int len = strlen(str2) + 1;  // 要加上结束符

     strcpy_s(str1, len, str2);

     cout << str1 << endl;

     //3、释放
     delete[] str1;

     //4、置空
     str1 = NULL;

     return 0;
 }

//3' VS malloc，四部曲
 int testVsMalloc()
 {
     //VS: char *str1 = new char[50];
     int n = 50;
     char * str1 = (char*)malloc(n * sizeof(char));

     //1、判断申请是否成功
     if (NULL == str1)
     {
         cout << "malloc fail" << endl;
         return 0;
     }

     //2、使用
     const char * str2;
     str2 = "Cannot stoping ...";

     //Err: int len2 = sizeof(str2);  // sizeof得到的是指针地址大小
     int len = strlen(str2) + 1;  // 要加上结束符

     strcpy_s(str1, len, str2);

     cout << str1 << endl;

     //3、释放
     //Vs: delete[] str1;
     free(str1);

     //4、置空
     str1 = NULL;

     return 0;
}

 int main(int argc, const char * argv[])
 {
     //testMemory();

     //testMemoryBlock();

     testVsMalloc();

     system("pause");

     return 0;
 }
```

<h3 id="3"> 三、C++远征之封裝篇（上） ☂</h3>

1] 简介

2] C++类与对象初体验

```c++
/*
 **************************************************************************    	 
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : ClassAndObject.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 12-26-2016
 *      Description:     
 **************************************************************************    	 
 */

 #include <iostream>
 #include <stdlib.h>

 using namespace std;

 class Coordinate
 {
 public:
     void prinftX()
     {
         cout << "x = " << x << endl;
     }
     void prinftY()
     {
         cout << "y = " << y << endl;
     }

     void operation()
     {
         x += 10;
         y -= 1;
     }

 public:
     int x;
     int y;
 };

 int main(int argc, const char * argv[])
 {
     //1、从栈中实例化类
     Coordinate coor;

     coor.x = 10;
     coor.y = 20;

     coor.prinftX();
     coor.prinftY();

     cout << endl;


     //2、从堆中实例化类
     Coordinate * p = new Coordinate();
     if (NULL == p)
     {
         printf("new failed !\n");
         return 0;
     }

     p->x = 100;
     p->y = 200;

     p->prinftX();
     p->prinftY();

     delete p;
     p = NULL;

     cout << endl;

     //3、从堆中实例化多个类
     Coordinate * q = new Coordinate[5];
     if (NULL == q)
     {
         printf("new failed !\n");
          return 0;
     }

     for (int i = 0; i < 5; i++)
     {
         q[i].x = i * 100;
          q[i].y = (i + 1) * 100;
     }

     cout << "Before:" << endl;

     for (int i = 0; i < 5; i++)
     {
         q[i].prinftX();
         q[i].prinftY();

         cout << endl;
     }

     cout << "After:" << endl;
     q[1].operation();
     q[4].operation();

     for (int i = 0; i < 5; i++)
     {
         q[i].prinftX();
         q[i].prinftY();

          cout << endl;
     }

     delete[] q;
     q = NULL;

     system("pause");

     return 0;
 }
```
3] C++初始字符串类型

```c++
/*
 **************************************************************************    	 
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : InitString.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 12-28-2016
 *      Description:     
 **************************************************************************    	 
 */

 #include <iostream>
 #include <string>
 #include <stdlib.h>
 #include <afx.h>

 using namespace std;

 void testString();
 void testCharPoint();
 void testStringMore();
 void pCharToString();
 void printCString(CString &csInput);
 void testCString();

 int main(int argc, const char * argv[])
 {
     //testString();
     //testCharPoint();
     //testStringMore();
     //pCharToString();
     //CString csInfo = "printCString";
     //printCString(csInfo);
     testCString();

     system("pause");

     return 0;
 }

//1、热身
 void testString()
 {
     string name;

     cout << "Please input your name: ";
     getline(cin, name);		//VS: cin >> name;

     if (name.empty())		  //Err: if (NULL == name)
     {
         cout << "The input is null ..." << endl;
         return;
     }

     cout << "Hello " << name << " !" << endl;

     if ("imooc" == name)	   //Err: if (imooc == name)
     {
         cout << "You are Administrator !" << endl;
     }

     cout << "Your name's length is " << name.size() << "." << endl;

     cout << "Your name's first letter is " << name[0] << "." << endl;  //PA
 }

//2、char* 的使用：可以指向一个字符，也可以表示字符数组的首地址功能
 void testCharPoint()
 {
     //PA: ""里面包含了'\0'
     char ch1[17] = "study char point";

     cout << "	ch1:" << ch1 << endl;
     cout << " ch1[0]:" << ch1[0] << endl;
     cout << "ch1[14]:" << ch1[14] << endl;
     cout << "ch1[15]:" << ch1[15] << endl;
     cout << "ch1[16]:" << ch1[16] << endl;  //Q?
     cout << "strlen(ch1) = " << strlen(ch1) << endl << endl;

     //Err: char * ch2 = "test char point";
     const char * ch2 = "study char point";
     cout << "	ch2:" << ch2 << endl;
     cout << "   * ch2:" << * ch2 << endl;

     char * ch3 = ch1;
     cout << "	ch3:" << ch3 << endl;
     cout << "   * ch3:" << * ch3 << endl << endl;

     //Err: char * ch4 = ch1[3];
     char * ch4 = &ch1[3];
     cout << "&ch1[3]:" << ch4 << endl << endl;

     char ch5 = 'c';
     cout << "	ch5:" << ch5 << endl << endl;

     //Err: char * ch6 = ch5;
     char * ch6 = &ch5;
     cout << "	ch6:" << ch6 << endl;  // Pointer address
     cout << "   * ch6:" << * ch6 << endl;
 }

//3、string 的使用: <string>
 void testStringMore()
 {
     string s1 = "This";
     //EQ: string s2 = string(" is");
     string s2 = " is";
     string s3 = string(" a").append("program.");
     //Q: 什么时候不能直接相加，连接？s4 = "hello" + "world";
     //Q: 加上string呢？ s4 = string("hello") + string("world");
     string s4 = s1 + s2 + s3;

     cout << "s4:" << s4 << endl;
     cout << "s4.size():" << s4.size() << endl;

     //Err: string s5 = s4.insert(s4.end()-9, ' '); insert返回的不是字符串
     s4.insert(s4.end() - 8, 1, ' ');
     //EQ: s4.insert(s4.end()-8, ' '); 其中 1 代表个数，1个可以不写

     cout << "s4:" << s4 << endl;
     cout << "s4.size():" << s4.size() << endl;
 }

//4、char*  <---->  string 借助c_str();
 void pCharToString()
 {
     //Err: char * ch1 = "pCharToString";
     const char * ch1 = "pCharToString";
     string s1 = string(ch1);

     cout << "ch1: " << ch1 << endl;
     cout << " s1: " << s1 << endl << endl;


     cout << "* ch1+1  : " << * ch1 + 1 << endl;
     cout << "* (ch1+1): " << * (ch1 + 1) << endl << endl;


     cout << "* ch1	: " << * ch1 << endl;
     cout << "* (ch1+1): " << * (ch1 + 1) << endl;
     cout << "* (ch1+2): " << * (ch1 + 2) << endl;
     cout << "* (ch1+3): " << * (ch1 + 3) << endl << endl;


     cout << " s1[0]: " << s1[0] << endl;
     cout << " s1[1]: " << s1[1] << endl;
     cout << " s1[2]: " << s1[2] << endl;
     cout << " s1[3]: " << s1[3] << endl;
     cout << " s1[s1.size() - 1]: " << s1[s1.size() - 1] << endl << endl;


     string s2 = "StringToPChar";
     const char * ch2 = s2.c_str();
     cout << "ch2: " << ch2 << endl << endl;

     //PA: c_str()返回的是一个const char* 以空字符结束指针，并且是临时的，会被改变
     //如果s2改变了，const char*的值也随之改变
     s2 = "unSafe!!!";
     cout << "ch2: " << ch2 << endl;


     //要么随用随转换可以把c_str()的值保存起来
     //int len1 = s2.size() + 1; //Err
     int len1 = s2.size() + 1;
     int len2 = strlen(s2.c_str()) + 1;

     cout << "len1 = " << len1 << " len2 = " << len2 << endl;

     char * ch3 = new char[len1];
     strcpy_s(ch3, len1, s2.c_str());
     cout << "ch3: " << ch3 << endl;

     delete[] ch3;
     ch3 = NULL;
 }

//5、CString 的使用: CString 常用于 MFC 编程中，是属于 MFC 的类需要 <afx.h>
 void printCString(CString &csInput)
 {
     int n = csInput.GetLength(); //类比string
     //cout << "cstr.GetLength() = " << cstr.GetLength() << endl;

     for (int i = 0; i<n; i++)
     {
         printf("%c", csInput[i]); //直接转换成数组了
     }

     printf("\n");

     cout << "csInput: " << csInput << endl;
 }

//6、CString 的基本使用
 void testCString()
 {
     char * ch = "Hello";
     string s = "World";
     //Q: s.c_str()
     CString cstr1(ch), cstr2(s.c_str()), cstr3("Program");

     printCString(cstr1);
     printCString(cstr2);
     printCString(cstr3);

     CString cstr4, cstr5;
     cstr4 = cstr1 + cstr2 + cstr3;
     cstr5 = cstr1 + " " + cstr2 + " " + cstr3;

     printCString(cstr4);
     printCString(cstr5);
 }
```
4]  C++属性封装代码演示

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

 #include <iostream>
 #include <string>
 #include <stdlib.h>

 using namespace std;

 class Student
 {
 public:
     //注意命名格式
     void setName(string name)
     {
         m_strName = name;
     }

     string getName()
     {
         return m_strName;
     }

     void setGender(string gender)
     {
         m_strGender = gender;
     }

     string getGender()
     {
         return m_strGender;
     }

     void initScore()
     {
         m_iScore = 0;
     }

     int getScore()
     {
         return m_iScore;
     }

     void study(int score)
     {
         m_iScore += score;
     }

private:
     string   m_strName;
     string   m_strGender;
     int      m_iScore;
};

int main(int argc, char const *argv[])
{
     //PA: 但凡new，就把内存释放，指针置空这两步先做了
     Student * stu = new Student;

     //If: 如果不初始化呢？ 以后的构造函数就能解决这个问题
     stu->initScore();
     stu->setName("Lisa"); //PA: string类型的双引号一定要带上
     stu->setGender("girl");
     stu->study(2);
     stu->study(8);

     cout << stu->getName() << " " << stu->getGender() << " " << stu->getScore() << endl;

     delete stu;
     stu = NULL;

     system("pause");

     return 0;
}
```
5] C++精彩的类外定义

```c++
/*
 **************************************************************************    	 
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : Encapsulation2.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 01-13-2017
 *      Description:     
 **************************************************************************    	 
 */

 #include <stdlib.h>
 #include "Teacher.h"

 using namespace std;

 int main(int argc, char const *argv[])
 {
     //PA: 但凡new，就把内存释放，指针置空这两步先做了
     Teacher * t = new Teacher;

     t->setName("Confucius"); //PA: string类型的双引号一定要带上
     t->setGender("Man");

     cout << t->getName() << " " << t->getGender() << " ";

     t->teach();

     delete t;
     t = NULL;

     system("pause");

     return 0;
 }
```
Teacher.h
```c++
#ifndef _TEACHER_H_
#define _TEACHER_H_

#include <iostream>
#include <string>

using namespace std;

class Teacher
{
public:
    Teacher()
    {
        cout << "Teacher" << endl;
    }
    ~Teacher()
    {
        cout << "~Teacher" << endl;
    }

    //注意命名格式
    void setName(string name);
    string getName();

    void setGender(string gender);
    string getGender();

    void teach();

private:
    string m_strName;
    string m_strGender;
};

void Teacher::setName(string name)
{
    m_strName = name;
}
string Teacher::getName()
{
    return m_strName;
}

void Teacher::setGender(string gender)
{
    m_strGender = gender;
}
string Teacher::getGender()
{
    return m_strGender;
}

void Teacher::teach()
{
    cout << "is teaching now ~" << endl;
}

#endif
```
6] 默認构造函数演示

```c++
/*
 **************************************************************************    	 
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : Constructor.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 01-14-2017
 *      Description:     
 **************************************************************************    	 
 */

 #include <stdlib.h>
 #include "Teacher.h"

 using namespace std;

 int main(int argc, char const *argv[])
 {
     //1. 堆中实例化
     Teacher * t1 = new Teacher();

     t1->setName("Confucius"); //PA: string类型的双引号一定要带上
     t1->setGender("Man");

     cout << t1->getName() << " " << t1->getGender() << " ";
     t1->teach();

     delete t1;
     t1 = NULL;


     //2. 栈中实例化
     Teacher t2;

     t2.setName("Confucius");
     t2.setGender("Man");

     cout << t2.getName() << " " << t2.getGender() << " ";
     t2.teach();

     system("pause");

     return 0;
 }
```
7] 构造函数初始化列表

* 初始化列表先于构造函数运行
* 初始化列表只能用于构造函数
* 初始化列表可以同时初始化多个数据成员
* 什么情况下才用，跟放在构造函数有什么区别   提示: Const double m_dPi;

```c++
/*
 **************************************************************************    	 
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : ConstructorInitialize.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 01-14-2017
 *      Description:     
 **************************************************************************    	 
 */

 #include <stdlib.h>
 #include "Teacher.h"

 using namespace std;

 int main(int argc, char const *argv[])
 {
    //1. 初始化列表
     Teacher t1;

     //放在初始化列表里去了
     //t1.setName("Confucius");
     //t1.setGender("Man");

     cout << t1.getName() << " " << t1.getGender() << " " << t1.getMax() << " ";
     t1.teach();


     //2. 构造函数默认参数, 初始化列表编码
     Teacher t2("LiYang");
     cout << t2.getName() << " " << t2.getGender() << " " << t2.getMax() << " ";
     t2.teach();

     Teacher t3("LiLei", "Female");
     cout << t3.getName() << " " << t3.getGender() << " " << t3.getMax() << " ";
     t3.teach();

     Teacher t4("LiLei", "Female", 200);
     cout << t4.getName() << " " << t4.getGender() << " " << t4.getMax() << " ";
     t4.teach();


     system("pause");

     return 0;
 }
```
Teacher.h
```c++
#ifndef _TEACHER_H_
#define _TEACHER_H_

#include <iostream>
#include <string>

using namespace std;

class Teacher
{
public: //Star
    Teacher(string name = "Confucius", string gender = "Man", int max = 100) : m_strName(name), m_strGender(gender), m_iMax(max)
    {
        //m_strName = name;
        //m_strGender = gender;
        //Err: m_iMax = max;

        cout << "Teacher" << endl;
    }

    ~Teacher()
    {
        cout << "~Teacher" << endl;
    }

    //注意命名格式
    void setName(string name);
    string getName();

    void setGender(string gender);
    string getGender();

    int getMax();

    void teach();

private:
    string m_strName;
    string m_strGender;
    const int m_iMax;
};

void Teacher::setName(string name)
{
     m_strName = name;
}
string Teacher::getName()
{
     return m_strName;
}

void Teacher::setGender(string gender)
{
     m_strGender = gender;
}
string Teacher::getGender()
{
     return m_strGender;
}

int Teacher::getMax()
{
     return m_iMax;
}

void Teacher::teach()
{
     cout << "is teaching now ~" << endl;
}

#endif
```
8] 拷贝构造函数

* 如果不定义拷贝构造函数, 则系统自动生成, 就像不定义构造函数一样
* 当采用直接初始化或复制初始化实例化对象时, 系统自动调用拷贝构造函数

```c++
/*
 **************************************************************************    	 
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : CopyConstructor.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 01-22-2017
 *      Description:     
 **************************************************************************    	 
 */

 #include <stdlib.h>
 #include "Teacher.h"

 using namespace std;

 int main(int argc, char const *argv[])
 {
     Teacher t1;
     Teacher t2 = t1;
     Teacher t3(t1);

     system("pause");

     return 0;
 }
```
Teacher.h
```c++
#ifndef _TEACHER_H_
#define _TEACHER_H_

#include <iostream>
#include <string>

using namespace std;

class Teacher
{
public:
    Teacher() //如果不定义拷贝构造函数, 则系统自动生成, 就像不定义构造函数一样
    {
        cout << "Teacher" << endl;
    }

    Teacher(const Teacher& teac)  //当采用直接初始化或复制初始化实例化对象时, 系统自动调用拷贝构造函数
    {
        cout << "Teacher" << endl;
    }

    ~Teacher()
    {
        cout << "~Teacher" << endl;
    }
};
#endif

```
9] 析构函数演示

```c++
/*
 **************************************************************************    	 
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : Destroy.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 01-28-2017
 *      Description:     
 **************************************************************************    	 
 */

 #include <stdlib.h>
 #include "Teacher.h"

 using namespace std;

 int main(int argc, char const *argv[])
 {
     Teacher * t = new Teacher[5];

     //Q: 1、如果没有呢？(就没有调用析构函数，栈的自动调用) 2、放到析构函数呢？(最好，就是析构的作用)
     delete[] t;
     t = NULL;

     system("pause");

     return 0;
 }
```

<h3 id="4"> 四、C++远征之封裝篇（下） ☂</h3>

1] 对象数组实践

```c++
/*
 **************************************************************************    	 
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : ObjectArray.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 01-29-2017
 *      Description:     
 **************************************************************************    	 
 */

 #include <iostream>
 #include "Coordinate.h"
 #include <stdlib.h>

 using namespace std;

 int main(int argc, char const *argv[])
 {
     //1. Instantiation in the stack
     Coordinate c1[3];
     c1[0].m_iX = 0;
     c1[0].m_iY = 0;

     c1[1].m_iX = 1;
     c1[1].m_iY = 1;

     c1[2].m_iX = 2;
     c1[2].m_iY = 2;

     for (int i = 0; i<3; i++) {
         cout << "c1[" << i << "].m_iX = " << c1[i].m_iX << endl;
         cout << "c1[" << i << "].m_iY = " << c1[i].m_iY << endl;
         cout << endl;
     }


     //2. Instantiation in the heap
     Coordinate * c2 = new Coordinate[3];

     c2->m_iX = 10;
     c2[0].m_iY = 10;

     //先++，再 c2[1].m_iX 就是上面的 c2[2] 了。
     //c2++;
     //c2[1].m_iX = 20;
     //c2->m_iY = 20;


     c2[1].m_iX = 20;
     c2++;
     c2->m_iY = 20;

     //PA:上面的c2++后，c2[1]即是原来的c2[2]了
     c2[1].m_iX = 30;
     c2++;
     c2->m_iY = 30;

     cout << endl;
     for (int i = 0; i<3; i++) {
         cout << "c2->m_iX = " << c2->m_iX << endl;
         cout << "c2->m_iY = " << c2->m_iY << endl;
         c2--;   //逆序
         cout << endl;
     }

     //PA:经过for后，c2指向第一个的前一个，所以要加一，否则释放内存会失败
     c2++;

     //Q:delete c2;  发生内存泄漏
     delete []c2; //PA: 内存块释放时一定要注意释放第一个地址
     c2 = NULL;


     system("pause");

     return 0;
 }
```

Coordinate.h

```c++
#pragma once

#include <iostream>

using namespace std;

class Coordinate
{
public:
    Coordinate();
    ~Coordinate();

public:
    int m_iX;
    int m_iY;
};
```

Coordinate.cpp

```c++
#include "Coordinate.h"

Coordinate::Coordinate()
{
    cout << "Coordinate()" << endl;
}

Coordinate::~Coordinate()
{
    cout << "~Coordinate()" << endl;
}
```
2] 拷贝构造函数

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
update c++ expedition to line-1724
