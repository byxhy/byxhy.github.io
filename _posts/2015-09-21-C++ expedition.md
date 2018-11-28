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

[一、C++远征之起航篇](#1)

[二、C++远征之离港篇](#2)

[三、C++远征之封装篇（上）](#3)

[四、C++远征之封装篇（下）](#4)

[五、C++远征之继承篇](#5)

[六、C++远征之多态篇](#6)

[七、C++远征之模板篇](#7)

<br />

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
        * C++特性之默认参数
        * List item four
* List item two
* List item three
* List item four
* List item three
* List item four

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
update c++ expedition to line-528
