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
    * C++對象的生離死別
        * C++新亮點之默认参数
        * C++新亮點之重載函數
        * C++新亮點之內聯函數
* [C++远征之封装篇（下）](#4)
    * C++语言新特性
        * C++特性之引用
        * C++特性之const
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
 ********************************************************************************
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : Hello.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 11-02-2016
 *      Description:     
 ********************************************************************************
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
 ********************************************************************************
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : CinCout.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 11-03-2016
 *      Description:     
 ********************************************************************************
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
    //    这里没有需要判断的情况，所以输入除1外的其他值，经过cin处理直接就变成false了
    //    所以，不要输入bool值了
    //    想想cin如果在一个循环里会怎样(提示: clear)


    //Err: Infinite Loop
    //while (true)
    //{
    //    cin >> y;
    //    cout << boolalpha << y << endl;
    //}


    //True:
    //while (true)
    //{
    //    cin.clear();
    //    cin >> y;
    //    cout << boolalpha << y << endl;
    //}


    system("pause");

    return 0;
}
```


3] namespace-命名空间的学习

```c++
/*
 ********************************************************************************
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : Namespace.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 11-03-2016
 *      Description:     
 ********************************************************************************
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
 ********************************************************************************
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : GetMaxOrMin.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 11-05-2016
 *      Description:     
 ********************************************************************************
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
 ********************************************************************************
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : Reference.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 11-05-2016
 *      Description:     
 ********************************************************************************
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
 ********************************************************************************
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : Const.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 11-06-2016
 *      Description:     
 ********************************************************************************
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

     //* p = 4;           //Err: 值不能改变

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

     //p = &y;           //Err: 指针不能变

     return 0;
 }


 //4' const 修饰指针，将权限大的变量赋值给权限小的
 int testPointLimit()
 {
     int x = 6;
     int const * p = &x;    //PA: 权限大的 赋值给 权限小的
     cout << "* p = " << * p << endl;

     //PA: 反过来就不行了
     //const int y = 4;
     //int * q = &y;        //通过*q有可能就操作了y的值，报错

     return 0;
 }

 //5' const 修饰引用
 int testReference()
 {
     int x = 6;
     int const &y = x;

     cout << "y = " << y << endl;

     //y = 4;        //error!!!
     x = 4;
     cout << "y = " << y << endl;

     return 0;
 }

 void fun(const int &a, const int &b)
 {
     //a = 1;     //error !!!
     //b = 2;     //error !!!
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
 ********************************************************************************
 *      Copyright (C), 2015-2115, Xhy Tech. Stu.
 *      FileName   : DefaultPara.cpp
 *      Author     : X h y
 *      Version    : 2.1   
 *      Date       : 11-06-2016
 *      Description:     
 ********************************************************************************
 */

 #include <iostream>
 #include <stdlib.h>

 using namespace std;

 //3' C++函数特性之默认参数    Q:要求，好处
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
********************************************************************************
*      Copyright (C), 2015-2115, Xhy Tech. Stu.
*      FileName   : Overload.cpp
*      Author     : X h y
*      Version    : 2.1   
*      Date       : 11-06-2016
*      Description:     
********************************************************************************
*/

#include <iostream>
#include <stdlib.h>

using namespace std;

//4' C++函数特性之重载函数    Q:要求，好处
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
********************************************************************************
*      Copyright (C), 2015-2115, Xhy Tech. Stu.
*      FileName   : Inline.cpp
*      Author     : X h y
*      Version    : 2.1   
*      Date       : 11-06-2016
*      Description:     
********************************************************************************
*/

#include <iostream>
#include <stdlib.h>

using namespace std;

//5'  C++函数特性之内联函数    Q:要求，好处
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
********************************************************************************
*      Copyright (C), 2015-2115, Xhy Tech. Stu.
*      FileName   : MemoryManagement.cpp
*      Author     : X h y
*      Version    : 2.1   
*      Date       : 11-06-2016
*      Description:     
********************************************************************************
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
********************************************************************************
*      Copyright (C), 2015-2115, Xhy Tech. Stu.
*      FileName   : ClassAndObject.cpp
*      Author     : X h y
*      Version    : 2.1   
*      Date       : 12-26-2016
*      Description:     
********************************************************************************
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

---
9] 9999999999999

```c++
/*
********************************************************************************
*      Copyright (C), 2015-2115, Xhy Tech. Stu.
*      FileName   : DefaultPara.cpp
*      Author     : X h y
*      Version    : 2.1   
*      Date       : 12-26-2016
*      Description:     
********************************************************************************
*/



```
update c++ expedition to line-923
