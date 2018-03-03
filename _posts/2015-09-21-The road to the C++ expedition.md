---
layout: post
title: "The road to the C++ expedition"
author: "Xhy"
categories: journal
tags: [documentation,sample]
image: Trial.jpg
---
Photo by Jeremy Bishop

>声明：本系列按照[james_yuan](http://www.imooc.com/u/1349694/courses?sort=publish)老师的C++课程的学习路径整理而来，添加少量学习注释。最近沉迷学习，无法自拔，跟着慕课老师james_yuan学习C++，秉承着先上路再迭代的思想，出发啦 ..

<br />.

### 篇节

<br />

[一、C++远征之起航篇](#1)

[二、C++远征之离港篇](#2)

[三、C++远征之封装篇（上）](#3)

[四、C++远征之封装篇（下）](#4)

[五、C++远征之继承篇](#5)

[六、C++远征之多态篇](#6)

[七、C++远征之模板篇](#7)

---

<br />

```c++
/*
********************************************************************************
*      Copyright (C), 2015-2115, Xhy Tech. Stu.
*      FileName   : Hello.cpp
*      Author     : X h y
*      Version    : 2.0   
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

<h3 id="1"> 一、C++远征之起航篇 ☂</h3>

1] C++输入输出流


    /*
    ********************************************************************************
    *      Copyright (C), 2015-2115, Xhy Tech. Stu.
    *      FileName   : Hello.cpp
    *      Author     : X h y
    *      Version    : 2.0   
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


2] C++新特性以及输入输出


    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : CinCout.cpp
     *      Author     : X h y
     *      Version    : 2.0   
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

        //PA: C++11的safe-bool标准: 只有在上下文需要判断bool条件的时候才会自动转换为bool类型
        //    这里没有需要判断的情况，所以输入除1外的其他值，经过cin处理直接就变成false了
        //    所以，不要输入bool值了

        //Err: cin >> y;  

        cout << boolalpha << y << endl;

        system("pause");

        return 0;
    }


3] namespace-命名空间的学习

    /*
    ********************************************************************************
    *      Copyright (C), 2015-2115, Xhy Tech. Stu.
    *      FileName   : Namespace.cpp
    *      Author     : X h y
    *      Version    : 2.0   
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

4] 练习：求最大值

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : GetMaxOrMin.cpp
     *      Author     : X h y
     *      Version    : 2.0   
     *      Date       : 11-05-2016
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <stdlib.h>

    using namespace std;

    namespace CompanyA
    {
        int getMaxOrMin(int *arr, int count, bool isMax = true);
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
        int ret = *arr;

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

<h3 id="2"> 二、C++远征之离港篇 ☂</h3>
1] C++特性之引用

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : Reference.cpp
     *      Author     : X h y
     *      Version    : 2.0   
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

    	int *p = &x;
    	cout << "*p = " << *p << endl;

    	int *&q = p;  //引用必须初始化
    	*q = 3;
    	cout << "x  = " << x << endl;
    	cout << "*p = " << *p << endl;

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
    	testPointRef();
    	testFuncPrameRef();

    	system("pause");

    	return 0;
    }
    　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于11-05-2016

2] C++特性之const

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : Const.cpp
     *      Author     : X h y
     *      Version    : 2.0   
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

    	cout << "x = " << x << endl;
    	cout << "y = " << y << endl;

    	return 0;
    }

    //2' const 修饰指针常量左边(左数右指法、整体法, 有没有开锁就是了)
    int testPointConLift()
    {
    	int x = 6;
    	int const *p = &x;  //左数，开锁就是值了,那值不能改变
    	cout << "*p = " << *p << endl;

    	//*p = 4;           //Err: 值不能改变

    	x = 4;
    	cout << "*p = " << *p << endl;

    	return 0;
    }

    //3' const 修饰指针常量右边
    int testPointConRight()
    {
    	int x = 6;
    	int y = 9;

    	int *const p = &x;  //右指，没开锁,那就是指针不能变

    	cout << "*p = " << *p << endl;

    	*p = 4;
    	cout << "*p = " << *p << endl;
    	cout << " x = " << x << endl;

    	//p = &y;           //Err: 指针不能变

    	return 0;
    }

    //4' const 修饰指针，将权限大的变量赋值给权限小的
    int testPointLimit()
    {
    	int x = 6;
    	int const *p = &x;    //PA: 权限大的 赋值给 权限小的
    	cout << "*p = " << *p << endl;

    	//PA: 反过来就不行了
    	const int y = 4;
    	//int *q = &y;        //通过*q有可能就操作了y的值，报错

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


    　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于11-06-2016



3] C++特性之默认参数

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : DefaultPara.cpp
     *      Author     : X h y
     *      Version    : 2.0   
     *      Date       : 11-06-2016
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>

    using namespace std;

    //1' C++函数特性之默认参数    Q:要求，好处
    void fun_default(int a = 10, int b = 20, int c = 30)
    {
    	cout << "a = " << a << "," << "b = " << b << "," << "c = " << c << endl;
    }    

    int main(int argc, const char * argv[])
    {
        fun_default();
    	fun_default(100);
    	fun_default(100, 200);
    	fun_default(100, 200, 300);
    }
    　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于11-06-2016



4] C++特性之重载函数

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : Overload .cpp
     *      Author     : X h y
     *      Version    : 2.0   
     *      Date       : 11-06-2016
     *      Description:     
     ********************************************************************************
     */   

    #include <iostream>

    using namespace std;

    //2' C++函数特性之重载函数    Q:要求，好处
    void funOverload(double i = 0.1, double j = 0.2)
    {
    	cout << "i = " << i << "," << "j = " << j << endl;
    }

    void funOverload(int a = 10, int b = 20, int c = 30)
    {
    	cout << "a = " << a << "," << "b = " << b << "," << "c = " << c << endl;
    }

    int main(int argc, const char * argv[])
    {
        //Err: 找全部参数都有默认值的函数，因为两个函数都有默认参数，所以报错
    	//Err: funOverload();   

    	funOverload(1, 2);
    	funOverload(0.1);

    	funOverload(100);
    	funOverload(100, 200);
    	funOverload(100, 200, 300);

        return 0;
    }
    　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于11-06-2016


5] C++特性之内联函数

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : Inline .cpp
     *      Author     : X h y
     *      Version    : 2.0   
     *      Date       : 11-06-2016
     *      Description:     
     ********************************************************************************
     */   

    #include <iostream>

    using namespace std;

    //3'  C++函数特性之内联函数    Q:要求，好处
    inline void funInline(double i = 0.1, double j = 0.2)
    {
    	cout << "i = " << i << "," << "j = " << j << endl;
    }

    inline void funInline(int a = 10, int b = 20, int c = 30)
    {
    	cout << "a = " << a << "," << "b = " << b << "," << "c = " << c << endl;
    }

    int main(int argc, const char * argv[])
    {
        funInline(1, 2);   //首先去找全部参数都有默认值的函数，若都有，就报错
    	funInline(100);
    	funInline(100, 200);
    	funInline(100, 200, 300);

    	funInline(0.1);

        return 0;
    }      　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于11-06-2016


6] C++特性之申请内存

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : MemoryManagement .cpp
     *      Author     : X h y
     *      Version    : 2.0   
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
    	int *p = new int;
    	//1、判断申请是否成功
    	if (NULL == p)
    	{
    		cout << "new fail" << endl;
    		return 0;
    	}

    	//2、使用
    	*p = 10;

    	cout << "*p = " << *p << endl;

    	//3、释放
    	delete p;


    	//4、置空
    	p = NULL;

    	return 0;
    }


    //2' C++申请内存块，四部曲
    int testMemoryBlock()
    {
    	char *str1 = new char[50];

    	//1、判断申请是否成功
    	if (NULL == str1)
    	{
    		cout << "new fail" << endl;
    		return 0;
    	}

    	//2、使用
    	//char *str2 = "Cannot stoping ...";  //Err: ??????(程序静态区)
    	const char *str2;
    	str2 = "Cannot stoping ...";

    	//Err: int len2 = sizeof(str2);  // sizeof得到的是类型大小
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
    	char *str1 = (char*)malloc(n * sizeof(char));

    	//1、判断申请是否成功
    	if (NULL == str1)
    	{
    		cout << "malloc fail" << endl;
    		return 0;
    	}

    	//2、使用
    	const char *str2;
    	str2 = "Cannot stoping ...";

    	//Err: int len2 = sizeof(str2);  // sizeof得到的是类型大小
    	int len = strlen(str2) + 1;  // 要加上结束符

    	strcpy_s(str1, len, str2);

    	cout << str1 << endl;

    	//3、释放
    	//Vs: delete []str1;
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
    　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于11-06-2016

[Back ↫](#0)

<h3 id="3"> 三、C++远征之封装篇（上） ☂</h3>
1] 简介
2] 类和对象初体验

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : ClassAndObject.cpp
     *      Author     : X h y
     *      Version    : 2.0   
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
    	Coordinate *p = new Coordinate();
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


    	//3、从堆中实例化类
    	Coordinate *q = new Coordinate[5];
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
    　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于12-26-2016      

3] 初始化字符串类型
☭ 目标：
1、提示用户输入姓名
2、接受用户的输入，进行输入参数判断，
3、如果为空，则告诉用户输入为空，直接返回退出
4、如果为imooc，则告诉用户为Administrator，接着进行下面的步骤
5、向用户问好，hello xxx
6、告诉用户姓名的长度
7、告诉用户姓名的第一个字母

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : string.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 12-27-2016
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <string>

    using namespace std;

    int main(int argc, const char * argv[])
    {
        string name;

        cout << "Please input your name: ";
        getline(cin, name);        //VS: cin >> name;

        if (name.empty())          //Err: if (NULL == name)
        {
            cout << "The input is null ..." << endl;
            return 0;
        }

        cout << "Hello " << name << " !" << endl;

        if ("imooc" == name)       //Err: if (imooc == name)
        {
            cout << "You are Administrator !" << endl;
        }

        cout << "Your name's length is " << name.size() << "." << endl;

        cout << "Your name's first letter is " << name[0] << "." << endl;

        return 0;
    }
    　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于12-27-2016      
4] char*, string, CString 比较（补充）

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : string.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 12-28-2016
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <string>
    #include <afx.h>

    using namespace std;

    void testCharPoint();
    void testString();

    void printCString(CString &cstr);
    void testCString();

    int main(int argc, char const *argv[])
    {
        testCharPoint();
        testString();
        testCString();

        return 0;
    }

    //1、char* 的使用：可以指向一个字符，也可以表示字符数组的首地址e功能
    void testCharPoint()
    {
        //PA: ""里面包含了'\0'
        char ch1[17] = "study char point";
        cout << "    ch1:" << ch1 << endl;
        cout << "ch1[17]:" << ch1[17] << endl;

        //Err: char *ch2 = "test char point";
        const char *ch2 = "study char point";
        cout << "    ch2:" << ch2 << endl;
        cout << "   *ch2:" << *ch2 << endl;

        char *ch3 = ch1;
        cout << "    ch3:" << ch3 << endl;
        cout << "   *ch3:" << *ch3 << endl;

        //Err: char *ch4 = ch1[3];
        char *ch4 = &ch1[3];
        cout << "&ch1[3]:" << ch4 << endl;

        char ch5 = 'c';
        cout << "    ch5:" << ch5 << endl;

        //Err: char *ch6 = ch5;
        char *ch6 = &ch5;
        cout << "    ch6:" << ch6 << endl;
        cout << "   *ch6:" << *ch6 << endl;
    }

    //2、string 的使用: <string>
    void testString()
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
        s4.insert(s4.end()-8, 1, ' ');
        //EQ: s4.insert(s4.end()-8, ' '); 其中 1 代表个数，1个可以不写

        cout << "s4:" << s4 << endl;
        cout << "s4.size():" << s4.size() << endl;
    }

    //CString 不是标准 C++库定义的类型，没有对<<运算符进行重载，需要自己实现
    void printCString(CString &cstr)
    {
        int n = cstr.GetLength(); //类比string

        for (int i=0; i<n; i++)
        {
            printf("%c\n", cstr[i]); //直接转换成数组了
        }

        printf("\n");
    }

    //3、CString 的使用: CString 常用于 MFC 编程中，是属于 MFC 的类需要 <afx.h>
    void testCString()
    {
        char *ch = "Hello";
        string s = "World";
        //Q: s.c_str())
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
    　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于12-28-2016
5] C++初始封装代码演示
☭ 类内定义的函数优先编译为内联函数

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : encapsulation.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 01-12-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>

    using namespace std;

    class Student
    {
    public:
        //注意命名格式
        void setName(string _name)
        {
            m_strName = _name;
        }
        string getName()
        {
            return m_strName;
        }

        void setGender(string _gender)
        {
            m_strGender = _gender;
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
        void study(int _score)
        {
            m_iScore += _score;
        }

    private:
        string m_strName;
        string m_strGender;
        int m_iScore; //
    };

    int main(int argc, char const *argv[])
    {
        //PA: 但凡new，就把内存释放，指针置空这两步先做了
        Student *stu = new Student;

        //If: 如果不初始化呢？ 以后的构造函数就能解决这个问题
        stu->initScore();
        stu->setName("Lisa");
        stu->setGender("girl");
        stu->study(2);
        stu->study(8);

        cout << stu->getName() << " " << stu->getGender() << " " << stu->getScore() << endl;

        delete stu;
        stu = NULL;

        return 0;
    }
    　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于01-12-2017
6] C++类外定义（同文件）

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : class.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 01-13-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>

    using namespace std;

    class Teacher
    {
    public:
        //注意命名格式
        void setName(string _name);
        string getName();

        void setGender(string _gender);
        string getGender();

        void teach();

    private:
        string m_strName;
        string m_strGender;
    };

    void Teacher::setName(string _name)
    {
        m_strName = _name;
    }
    string Teacher::getName()
    {
        return m_strName;
    }

    void Teacher::setGender(string _gender)
    {
        m_strGender = _gender;
    }
    string Teacher::getGender()
    {
        return m_strGender;
    }

    void Teacher::teach()
    {
        cout << "is teaching now ~" << endl;
    }

    int main(int argc, char const *argv[])
    {
        //PA: 但凡new，就把内存释放，指针置空这两步先做了
        Teacher *t = new Teacher;

        t->setName("Confucius"); //PA: string类型的双引号一定要带上
        t->setGender("Man");

        cout << t->getName() << " " << t->getGender() << " ";

        t->teach();

        delete t;
        t = NULL;

        return 0;
    }
    　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于01-13-2017    
7] C++类外定义（分文件）

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : class.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 01-13-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include "Teacher.h"
    using namespace std;

    int main(int argc, char const *argv[])
    {
        //PA: 但凡new，就把内存释放，指针置空这两步先做了
        Teacher *t = new Teacher;

        t->setName("Confucius"); //PA: string类型的双引号一定要带上
        t->setGender("Man");

        cout << t->getName() << " " << t->getGender() << " ";

        t->teach();

        delete t;
        t = NULL;

        return 0;
    }

在Teacher.h里声明：

    #ifndef SEEDS_TEAHCER_H
    #define SEEDS_TEAHCER_H

    #endif //SEEDS_TEAHCER_H

    #include <iostream>
    #include <string>

    using namespace std;

    class Teacher
    {
    public:
        //注意命名格式
        void setName(string _name);
        string getName();

        void setGender(string _gender);
        string getGender();

        void teach();

    private:
        string m_strName;
        string m_strGender;
    };

在Teacher.c里定义：  

    #include "Teacher.h"
    #include <iostream>
    using namespace std;

    void Teacher::setName(string _name)
    {
        m_strName = _name;
    }
    string Teacher::getName()
    {
        return m_strName;
    }

    void Teacher::setGender(string _gender)
    {
        m_strGender = _gender;
    }
    string Teacher::getGender()
    {
        return m_strGender;
    }

    void Teacher::teach()
    {
        cout << "is teaching now ~" << endl;
    }
    　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于01-13-2017
8] 构造函数演示

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : structure.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 01-14-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include "Teacher.h"

    using namespace std;

    int main(int argc, char const *argv[])
    {
        //PA: 但凡new，就把内存释放，指针置空这两步先做了
        Teacher *t1 = new Teacher();
        Teacher *t2 = new Teacher("t2");
        Teacher *t3 = new Teacher("t3", "Girl");

        cout << t1->getName() << " " << t1->getGender() << " ";
        t1->teach();

        cout << t2->getName() << " " << t2->getGender() << " ";
        t2->teach();

        cout << t3->getName() << " " << t3->getGender() << " ";
        t3->teach();

        delete t1;
        t1 = NULL;
        delete t2;
        t2 = NULL;
        delete t3;
        t3 = NULL;

        return 0;
    }

在Teacher.h里声明：

    #ifndef SEEDS_TEAHCER_H
    #define SEEDS_TEAHCER_H

    #endif //SEEDS_TEAHCER_H

    #include <iostream>

    using namespace std;

    class Teacher
    {
    public:
        Teacher(string _name = "Confucius", string _gender = "Man");
        ~Teacher();

        //注意命名格式
        void setName(string _name);
        string getName();

        void setGender(string _gender);
        string getGender();

        void teach();

    private:
        string m_strName;
        string m_strGender;
    };

在Teacher.c里定义：  

    #include "Teacher.h"

    #include <iostream>
    using namespace std;

    Teacher::Teacher(string _name, string _gender)
    {
        m_strName = _name;
        m_strGender = _gender;
    }

    Teacher::~Teacher()
    {

    }

    void Teacher::setName(string _name)
    {
        m_strName = _name;
    }
    string Teacher::getName()
    {
        return m_strName;
    }

    void Teacher::setGender(string _gender)
    {
        m_strGender = _gender;
    }
    string Teacher::getGender()
    {
        return m_strGender;
    }

    void Teacher::teach()
    {
        cout << "is teaching now ~" << endl;
    }　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于01-14-2017

9] 初始化列表编码

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : initialize.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 01-15-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <iostream>
    #include "Teacher.h"

    using namespace std;

    int main(int argc, char const *argv[])
    {
        //PA: 但凡new，就把内存释放，指针置空这两步先做了
        Teacher *t1 = new Teacher();
        Teacher *t2 = new Teacher("t2");
        Teacher *t3 = new Teacher("t3", "Girl", 7);

        cout << t1->getName() << " " << t1->getGender() << " " << t1->getMax() << " ";
        t1->teach();

        cout << t2->getName() << " " << t2->getGender() << " " << t2->getMax() << " ";
        t2->teach();

        cout << t3->getName() << " " << t3->getGender() << " " << t3->getMax() << " ";
        t3->teach();

        delete t1;
        t1 = NULL;
        delete t2;
        t2 = NULL;
        delete t3;
        t3 = NULL;

        return 0;
    }    
在Teacher.h里声明：

    #ifndef SEEDS_TEAHCER_H
    #define SEEDS_TEAHCER_H

    #endif //SEEDS_TEAHCER_H

    #include <iostream>
    #include <string>

    using namespace std;

    class Teacher
    {
    public:
        Teacher(string _name = "Confucius", string _gender = "Man", int _max = 5);
        ~Teacher();

        //注意命名格式
        void setName(string _name);
        string getName();

        void setGender(string _gender);
        string getGender();

        void teach();

        int getMax();

    private:
        string m_strName;
        string m_strGender;
        const int m_iMax;
    };

在Teacher.c里定义：  

    #include "Teacher.h"
    #include <iostream>
    using namespace std;

    //1. 逗号隔开各个参数
    //2. 因为声明的参数有默认值，会直接传过来
    //3. 括号初始化
    Teacher::Teacher(string _name, string _gender, int _max):
    m_strName(_name), m_strGender(_gender), m_iMax(_max)
    {
    	//Error !!!   const修饰的常量只能在初始化列表里初始化
    	//m_iMax = 5;
    	cout << "(string _name, string _gender, int _max)" << endl;
    }

    Teacher::~Teacher()
    {

    }

    void Teacher::setName(string _name)
    {
        m_strName = _name;
    }
    string Teacher::getName()
    {
        return m_strName;
    }

    void Teacher::setGender(string _gender)
    {
        m_strGender = _gender;
    }
    string Teacher::getGender()
    {
        return m_strGender;
    }

    void Teacher::teach()
    {
        cout << "is teaching now ~" << endl;
    }

    int Teacher::getMax()
    {
        return m_iMax;
    }　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于01-15-2017

### 004）C++远征之封装篇（下）
#####1.
#####2.
#####3.Const重出江湖

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : ConstComeBack.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 03-05-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <stdlib.h>

    #include "Line.h"

    #define END {cout << endl;}

    using namespace std;

    int main()
    {

    	//5-2. 常对象成员、常成员函数
    	Line *L1 = new Line(2, 3, 4, 5);
    	//调用普通成员函数
    	L1->printInfo();
    	END;

    	const Line *L2 = new Line(5, 6, 7, 8);
    	//调用常成员函数
    	L2->printInfo();
    	END;

    	delete L1;
    	L1 = NULL;
    	delete L2;
    	L2 = NULL;
    	END; END;

    	//5-4 常指针与常引用
    	Coordinate Coor1 = Coordinate(4, 5);

    	//给Coor1取个别名
    	Coordinate &Coor2 = Coor1;
    	//1. 当有普通成员函数的时候调用普通成员函数
    	cout << "Coor2.getX() = " << Coor2.getX() << endl;

    	//2. 当没有普通成员函数的时候调用常成员函数（但常指针和常引用只调用常成员函数）
    	cout << "Coor2.getY() = " << Coor2.getY() << endl;

    	Coordinate *pCoor1 = &Coor1;
    	cout << "pCoor1->getX() = " << pCoor1->getX() << endl;

    	pCoor1->setX(7);
    	cout << "pCoor1->getX() = " << pCoor1->getX() << endl;
    	END;

    	//3. 但常指针和常引用只调用常成员函数
    	//EQ: const Coordinate  &Coor3 = Coor1;
    	const Coordinate  &Coor3 = Coor1;

    	//Err: （对象包含与成员函数不兼容）cout << "Coor3.getX() = " << Coor3.getX() << endl;
    	Coor3.printInfo(); //重载后，自动调用const
    	END;

    	Coordinate Coor4 = Coordinate(8, 9);
    	//常指针，//开锁法,指针值不能变
    	const Coordinate *pCoor3 = &Coor1;
    	cout << "pCoor3->getX(): " << pCoor3->getX() << endl;
    	pCoor3->printInfo();

    	pCoor3 = &Coor4;

    	//开锁法,指针地址不能变
    	Coordinate *const pCoor2 = &Coor1;
    	//调用普通成员函数,因为Coordinate为普通类型，不是看pCoor2
    	//PA:不是看pCoor2，看Coordinate
    	cout << "pCoor2->getX(): " << pCoor2->getX() << endl;
    	pCoor2->printInfo();

    	//Err: pCoor2 = &Coor4;

    	system("pause");

    	return 0;
    }
Coordinate.h

    #include "Coordinate.h"
    class Line
    {
    public:
    	Line(int x1, int y1, int x2, int y2);
    	~Line();

    	//会互为重载
    	void printInfo();
    	void printInfo() const;

    private:
    	//常对象成员
    	const Coordinate m_CoorA;
    	const Coordinate m_CoorB;
    };
Coordinate.cpp

    #include <iostream>
    #include "Line.h"

    using  namespace std;

    Line::Line(int x1, int y1, int x2, int y2) : m_CoorA(x1, y1), m_CoorB(x2, y2)
    {
    	cout << "Line()" << endl;
    }

    Line::~Line()
    {
    	cout << "~Line()" << endl;
    }

    void Line::printInfo()
    {
    	//调用常成员函数
    	//getX()必须为const
    	cout << "Line::printInfo()" << endl;
    	cout << "m_CoorA.getX(), m_CoorA.getY(): " << "(" << m_CoorA.getX() << ", " << m_CoorA.getY() << ")" << endl;
    	cout << "m_CoorB.getX(), m_CoorB.getY(): " << "(" << m_CoorB.getX() << ", " << m_CoorB.getY() << ")" << endl;
    }

    //调用常成员函数
    void Line::printInfo() const
    {
    	cout << "Line::printInfo() const " << endl;
    	cout << "m_CoorA.getX(), m_CoorA.getY(): " << "(" << m_CoorA.getX() << ", " << m_CoorA.getY() << ")" << endl;
    	cout << "m_CoorB.getX(), m_CoorB.getY(): " << "(" << m_CoorB.getX() << ", " << m_CoorB.getY() << ")" << endl;
    }
Coordinate.h

    class Coordinate
    {
    public:
    	Coordinate(int x, int y);
    	~Coordinate();

    	void setX(int x);
    	void setY(int y);

    	int getX();
    	//const
    	int getX() const;
    	int getY() const;

    	void printInfo();
    	void printInfo() const;

    public:
    	int m_iX;
    	int m_iY;
    };
Coordinate.cpp

    #include "Coordinate.h"
    #include <iostream>

    using namespace std;

    Coordinate::Coordinate(int x, int y)
    {
    	m_iX = x;
    	m_iY = y;

    	cout << "Coordinate() " << "(" << m_iX << "," << m_iY << ")" << endl;
    }

    Coordinate::~Coordinate()
    {
    	cout << "~Coordinate()" << endl;
    }

    void Coordinate::setX(int x)
    {
    	m_iX = x;
    }
    void Coordinate::setY(int y)
    {
    	m_iY = y;
    }

    //普通成员函数
    int Coordinate::getX()
    {
    	cout << "getX()" << endl;

    	return m_iX;
    }
    //常成员函数
    int Coordinate::getX() const
    {
    	cout << "getX() const" << endl;
    	return m_iX;
    }
    int Coordinate::getY() const
    {
    	cout << "getY() const" << endl;
    	return m_iY;
    }

    void Coordinate::printInfo()
    {
    	cout << "printInfo()" << endl;
    	cout << "(" << m_iX << ", " << m_iY << ")" << endl;
    }

    void Coordinate::printInfo() const
    {
    	cout << "printInfo() const" << endl;
    	cout << "(" << m_iX << ", " << m_iY << ")" << endl;
    }
    　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于03-05-2017   

### 005）C++远征之继承篇
#####1.继承演示
    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : Inherit.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 03-21-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <stdlib.h>

    #include "Worker.h"

    using namespace std;

    int main()
    {
    	//先生成基类，再生成派生类，析构则相反
    	Worker *p = new Worker();

    	//继承而来
    	//p->m_sName = "However";
    	p->m_iAge = 23;
    	p->eat();

    	p->work();
    	p->m_iSalary = 1000;

    	delete p;
    	p = NULL;

    	system("pause");

    	return 0;
    }
Worker.h

    #include "Person.h"
    class Worker :
    	public Person
    {
    public:
    	Worker();
    	~Worker();

    	void work();

    public:
    	int m_iSalary;
    };
Worker.cpp

    #include <iostream>

    #include "Worker.h"

    using namespace std;

    Worker::Worker()
    {
    	cout << "Worker()" << endl;
    }

    Worker::~Worker()
    {
    	cout << "~Worker()" << endl;
    }

    void Worker::work()
    {
    	cout << "work()" << endl;
    }
Person.h

    #include <iostream>
    #include <string>

    using namespace std;

    class Person
    {
    public:
    	Person();
    	~Person();

    	void eat();

    public:
    	//PA: string 是类，不是关键字
    	string m_sName;
    	int m_iAge;
    };

Person.cpp

    #include <iostream>

    #include "Person.h"

    using namespace std;

    Person::Person()
    {
    	cout << "Person()" << endl;
    }

    Person::~Person()
    {
    	cout << "~Person()" << endl;
    }

    void Person::eat()
    {
    	cout << "eat()" << endl;
    }
        　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于03-21-2017      
#####2.公有继承
    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : PublicInherit.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 03-22-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <stdlib.h>

    #include "Worker.h"

    #include "Person.h"

    using namespace std;

    int main()
    {
    	//先生成基类，再生成派生类，析构则相反
    	Worker *w = new Worker();
    	Person *p = new Worker();

        //在public继承的前提下，有1,2,3
    	//1. public----->public
    	//由public继承而来的public，基类和派生类都可以使用
    	w->m_public_iAge = 30;
    	w->eat_public();
    	w->work_public_test();


    	//2. protected----->protected
    	//由public继承而来的protected，基类和派生类都可以使用,但只能在函数成员内部使用，基类派生类都不能直接操作数据成员
    	w->work_protected_test();   //Err,VS: w->work_protected();

    	//3. private----->none 私有的，除了自己，谁也别想用，自己用，还得在房子里面用
    	p->eat_public();

    	delete w;
    	w = NULL;
    	delete p;
    	p = NULL;

    	system("pause");

    	return 0;
    }
Worker.h

    #include "Person.h"
    class Worker :
    	public Person
    {
    public:
    	Worker();
    	~Worker();

    	void work_public_test();
    	void work_protected_test();
    	void work_private_test();

    	int m_public_iSalary;

    protected:
    	void work_protected();

    private:
    	void work_private();
    };
Worker.cpp

    #include <iostream>

    #include "Worker.h"

    using namespace std;

    Worker::Worker()
    {
    	cout << "Worker()" << endl;
    }

    Worker::~Worker()
    {
    	cout << "~Worker()" << endl;
    }

    void Worker::work_public_test()
    {
    	m_protected_sName = "Jack";
    	m_protected_iAge = 29;

    	cout << "work_public_test()" << endl;
    }
    void Worker::work_private_test()
    {
    	this->work_private();

    	cout << "work_private_test()" << endl;
    }
    void Worker::work_protected_test()
    {
    	this->work_protected();

    	cout << "work_protected_test()" << endl;
    }

    void Worker::work_protected()
    {
    	m_protected_sName = "Jack";
    	m_protected_iAge = 29;

    	cout << "work_protected()" << endl;
    }
    void Worker::work_private()
    {
    	m_protected_sName = "Jack";
    	m_protected_iAge = 29;

    	cout << "work_private()" << endl;
    }
Person.h

    #include <iostream>
    #include <string>

    using namespace std;

    class Person
    {
    public:
    	Person();
    	~Person();

    	void eat_public();

    	//PA: string 是类，不是关键字
    	string m_public_sName;
    	int m_public_iAge;

    protected:
    	void eat_protected();

    	string m_protected_sName;
    	int m_protected_iAge;

    private:
    	void eat_private();

    	string m_private_sName;
    	int m_private_iAge;
    };

Person.cpp

    #include <iostream>

    #include "Person.h"

    using namespace std;

    Person::Person()
    {
    	cout << "Person()" << endl;
    }

    Person::~Person()
    {
    	cout << "~Person()" << endl;
    }

    void Person::eat_public()
    {
    	m_private_iAge = 29;
    	m_protected_iAge = 27;

    	cout << "eat_public()" << endl;
    }   	
        　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于03-22-2017      
#####3.保护继承和私有继承
    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : ProtectedPrivateInherit.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 03-31-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <stdlib.h>

    #include "Person.h"
    #include "Soldier.h"
    #include "Infantry.h"


    using namespace std;

    int main()
    {
    	Soldier *p = new Soldier();

    	Infantry *q = new Infantry();

    	p->work();

    	//Err:继承到Soldier的protected下了，不能直接访问
    	//p->play();

    	//通过Infantry的public继承来证明，Person是被继承到Soldier的protected下面了
    	//Err: q->play();
    	q->testProtected();

    	delete p;
    	p = NULL;

    	delete q;
    	q = NULL;

    	system("pause");

    	return 0;
    }
Person.h

    //VS: #pragma once

    #ifndef _PERSON_H_
    #define _PERSON_H_

    #include <iostream>
    #include <string>

    using namespace std;

    class Person
    {
    public:
    	Person();
    	~Person();

    	void play();

    protected:
    	//PA: string 是类，不是关键字
    	string m_sName;
    };
    #endif
Person.cpp

    #include <iostream>

    #include "Person.h"

    using namespace std;

    Person::Person()
    {
    	cout << "Person()" << endl;
    }

    Person::~Person()
    {
    	cout << "~Person()" << endl;
    }

    void Person::play()
    {
    	cout << "Person::play() " << m_sName << endl;
    }
Soldier.h

    #ifndef _SOLDIER_H_
    #define _SOLDIER_H_

    #include "Person.h"

    class Soldier :	protected Person
    {
    public:
    	Soldier();
    	~Soldier();

    	void work();
    protected:
    	int m_iAge;
    };
    #endif

Soldier.cpp

    #include <iostream>

    #include "Soldier.h"

    using namespace std;

    Soldier::Soldier()
    {
    	cout << "Soldier()" << endl;
    }

    Soldier::~Soldier()
    {
    	cout << "~Soldier()" << endl;
    }

    void Soldier::work()
    {
    	m_sName = "Tom";
    	m_iAge = 26;

    	cout << "(Person)m_sName = " << m_sName << " (Soldier)m_iAge = " << m_iAge << endl;
    } 	

Infantry.h


    #ifndef _INFANTRY_H_
    #define _INFANTRY_H_

    #include "Soldier.h"

    class Infantry : public Soldier
    {
    public:
    	Infantry();
    	~Infantry();

    	void testProtected();
    };
    #endif

Infantry.cpp

    #include <iostream>

    #include "Infantry.h"

    using namespace std;

    Infantry::Infantry()
    {
    	cout << "Infantry::Infantry()" << endl;
    }


    Infantry::~Infantry()
    {
    	cout << "~Infantry::Infantry()" << endl;
    }

    void Infantry::testProtected()
    {
    	Person::m_sName = "Uoga";
    	Person::play();

    	cout << "testProtected()" << endl;
    }
        　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于03-31-2017
#####4.C++隐藏
    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : Hide.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 04-11-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <string>
    #include "Soldier.h"

    using namespace std;

    int main()
    {
    	Soldier *p = new Soldier();

    	p->play();
    	//Err:p->play(7);
    	p->Person::play(7); //PA: 对于同名的函数不重载，只隐藏

    	p->work();

    	delete p;
    	p = NULL;

    	system("pause");

    	return 0;
    }
Person.h

    #ifndef _PERSON_H_
    #define _PERSON_H_

    #include <iostream>

    using namespace std;

    class Person
    {
    public:
    	Person();
    	~Person();

    	void play(int x);

    protected:
    	string m_sName;
    };
    #endif
Person.cpp

    #include <iostream>
    #include <string>

    #include "Person.h"

    using namespace std;

    Person::Person()
    {
    	cout << "Person()" << endl;
    }

    Person::~Person()
    {
    	cout << "~Person()" << endl;
    }

    void Person::play(int x)
    {
    	cout << "Person::play(int x)" << endl;
    }
Soldier.h

     #ifndef _SOLDIER_H_
    #define _SOLDIER_H_

    #include <iostream>
    #include "Person.h"

    class Soldier : public Person
    {
    public:
    	Soldier();
    	~Soldier();

    	void play();
    	void work();

    protected:
    	string m_sName;
    };
    #endif
Soldier.cpp

    #include <iostream>
    #include <string>  //不加头文件，cout << "m_sName = " << m_sName << endl;报错,是个类，必须要头文件
    #include "Soldier.h"

    using namespace std;

    Soldier::Soldier()
    {
    	cout << "Soldier()" << endl;
    }

    Soldier::~Soldier()
    {
    	cout << "~Soldier()" << endl;
    }

    void Soldier::play()
    {
    	cout << "Soldier::play()" << endl;
    	m_sName = "July";
    	Person::m_sName = "Mercy"; //只能通过这种方式区调用
    }

    void Soldier::work()
    {
    	cout << "Soldier::work()" << endl;
    	cout << "m_sName = " << m_sName << endl;
    	cout << "Person::m_sName = " << Person::m_sName << endl;
    }

        　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于04-11-2017

#####5. is A关系
    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : isA.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 04-12-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <stdlib.h>

    #include "Person.h"
    #include "Soldier.h"

    using namespace std;

    void test1(Person p)
    {
    	p.play();
    }

    void test2(Person &p)
    {
    	p.play();
    }

    void test3(Person *p)
    {
    	p->play();
    }

    int main()
    {
    	//1. 士兵是个人，用子类初始化父类对象时，只会把父类有的数据成员传递过去，其他的就会被截断
    	Soldier s1;
    	Person p1 = s1;

    	p1.play(); //打印的是Soldier，证明将s1的值传递过去了


    	Person *p2 = &s1;
    	p2->play();

    	//也只能访问子类里共有的数据成员和函数，其他的访问不了
    	//Err: p2->work();


    	//2. 是个人不一定都是士兵
    	//Err: s1 = p1;
    	//Err: Soldier *s2 = &p1;


    	//3. 但有继承关系时，小心内存泄漏，virtual
    	Person *p3 = new Soldier;

    	p3->play();

    	delete p3; //如果没有virtual修饰person的析构函数，就不会被释放，有可能内存会泄漏
    	p3 = NULL;

    	//4. 函数测试
    	Soldier s4;
    	Person p4;

    	test1(s4); //会实例化一个临时对象Person，之后临时对象会被释放，调用一次析构函数
    	test1(p4);

    	test2(s4); //将传入的参数取一个别名，通过别名来调用p，没有实例化对象
    	test2(p4);

    	test3(&s4); //和test2()是一样的，也没有实例化过程，所以后两者的效率最高
    	test3(&p4);

    	system("pause");

    	return 0;
    }
Person.h

    #ifndef _PERSON_H_
    #define _PERSON_H_

    #include <iostream>
    #include <string>

    using namespace std;

    class Person
    {
    public:
    	Person(string name = "Person");
    	//VS: ~Person();
    	virtual ~Person(); //但凡可能会继承时，都加上virtual关键字

    	void play();

    protected:
    	//PA: string 是类，不是关键字
    	string m_sName;
    };
    #endif
Person.cpp

    #include "Person.h"
    #include <iostream>

    using namespace std;

    Person::Person(string name)
    {
    	m_sName = name;
    	cout << "Person()" << endl;
    }

    Person::~Person()
    {
    	cout << "~Person()" << endl;
    }

    void Person::play()
    {
    	cout << "Person::play() " << m_sName << endl;
    }
Soldier.h

    #ifndef _SOLDIER_H_
    #define _SOLDIER_H_

    #include "Person.h"

    class Soldier :	public Person
    {
    public:
    	Soldier(string  name = "Soldier", int age = 20);
    	//VS: ~Soldier();
    	virtual ~Soldier();

    	void work();

    protected:
    	int m_iAge;
    };

    #endif 	
Soldier.cpp

    #include <iostream>

    #include "Soldier.h"

    using namespace std;

    Soldier::Soldier(string  name, int age)
    {
    	m_sName = name;
    	m_iAge = age;

    	cout << "Soldier()" << endl;
    }

    Soldier::~Soldier()
    {
    	cout << "~Soldier()" << endl;
    }

    void Soldier::work()
    {
    	cout << "(Soldier)m_sName = " << m_sName << " (Soldier)m_iAge = " << m_iAge << endl;
    } 	

        　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于04-12-2017

#####6.多重继承
    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : MultipleInherit.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 04-22-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <stdlib.h>

    #include "Infantry.h"

    using namespace std;

    void test1(Person p)
    {
    	p.play();
    }

    void test2(Person &p)
    {
    	p.play();
    }

    void test3(Person *p)
    {
    	p->play();
    }

    int main()
    {
    	//1. 多重继承，区别多继承
    	Infantry infantry;

    	test1(infantry); //会实例化一个临时对象Person，之后临时对象会被释放，调用一次析构函数

    	test2(infantry); //将传入的参数取一个别名，通过别名来调用p，没有实例化对象

    	test3(&infantry); //和test2()是一样的，也没有实例化过程，所以后两者的效率最高

    	system("pause");

    	return 0;
    }
Person.h

    #ifndef PERSON_H_
    #define PERSON_H_

    #include <iostream>
    #include <string>

    using namespace std;

    class Person
    {
    public:
    	Person(string name = "Person", int age = 18);
    	//VS: ~Person();
    	virtual ~Person(); //但凡可能会继承时，都加上virtual关键字

    	void play();

    protected:
    	//PA: string 是类，不是关键字
    	string m_sName;
    	int m_iAge;
    };
    #endif
Person.cpp

    #include "Person.h"

    #include <iostream>

    using namespace std;

    Person::Person(string name, int age)
    {
    	m_sName = name;
    	m_iAge = age;

    	cout << "Person::Person(string name, int age)" << endl;
    }

    Person::~Person()
    {
    	cout << "Person::~Person()" << endl;
    }

    void Person::play()
    {
    	cout << "Person::play()" << endl;

    	cout << "m_sName = " << m_sName << endl;
    	cout << "m_iAge  = " << m_iAge << endl;
    }
Soldier.h

    #ifndef SOLDIER_H_
    #define SOLDIER_H_

    #include "Person.h"

    class Soldier :	public Person
    {
    public:
    	Soldier(string  name = "Soldier", int age = 19);
    	//VS: ~Soldier();
    	virtual ~Soldier();

    	void work();
    };
    #endif
Soldier.cpp

    #include "Person.h"

    #include <iostream>

    using namespace std;

    Person::Person(string name, int age)
    {
    	m_sName = name;
    	m_iAge = age;

    	cout << "Person::Person(string name, int age)" << endl;
    }

    Person::~Person()
    {
    	cout << "Person::~Person()" << endl;
    }

    void Person::play()
    {
    	cout << "Person::play()" << endl;

    	cout << "m_sName = " << m_sName << endl;
    	cout << "m_iAge  = " << m_iAge << endl;
    } 	
Infantry.h

    #ifndef INFANTRY_H
    #define INFANTRY_H

    #include "Soldier.h"

    class Infantry : public Soldier
    {
    public:
    	Infantry(string name = "Infantry", int age = 20);
    	virtual ~Infantry();

    	void attack();
    };

    #endif
Infantry.cpp

    #include <iostream>

    #include "Infantry.h"

    using namespace std;

    Infantry::Infantry(string name, int age)
    {
    	m_sName = name;
    	m_iAge = age;

    	cout << "Infantry::Infantry()" << endl;
    }

    Infantry::~Infantry()
    {
    	cout << "Infantry::~Infantry()" << endl;
    }

    void Infantry::attack()
    {
    	cout << "Infantry::attack()" << endl;
    }
        　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于04-22-2017
#####7.多继承 VS 多重继承
    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : MoreInherit.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 04-22-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <stdlib.h>

    #include "MigrateWorker.h"

    using namespace std;

    int main()
    {
    	MigrateWorker *p = new MigrateWorker("Masou", 10009);

    	p->sow();
    	p->work();

    	delete p;

    	system("pause");

    	return 0;
    }
Farmer.h

    #ifndef FARMER_H_
    #define FARMER_H_

    #include <iostream>
    #include <string>

    using namespace std;

    class Farmer
    {
    public:
    	Farmer(string name = "Farmer");
    	virtual ~Farmer(); //但凡可能会继承时，都加上virtual关键字

    	void sow();

    protected:
    	//PA: string 是类，不是关键字
    	string m_sName;
    };

    #endif
Farmer.cpp

    #include "Farmer.h"
    #include <iostream>
    using namespace std;
    Farmer::Farmer(string name)
    {
    	m_sName = name;

    	cout << "Farmer::Farmer(string name)" << endl;
    }
    Farmer::~Farmer()
    {
    	cout << "Farmer::~Farmer()" << endl;
    }
    void Farmer::sow()
    {
    	cout << "Farmer::sow()" << endl;

    	cout << "m_sName = " << m_sName << endl;
    }
Worker.h

    #ifndef WORKER_H_
    #define WORKER_H_

    #include "Worker.h"

    class Worker
    {
    public:
    	Worker(int code = 202212);

    	virtual ~Worker();

    	void work();

    protected:
    	int m_iCode;
    };

    #endif 	
Worker.cpp

    #include <iostream>

    #include "Worker.h"

    using namespace std;

    Worker::Worker(int code)
    {
    	m_iCode = code;

    	cout << "Worker::Worker(int code)" << endl;
    }

    Worker::~Worker()
    {
    	cout << "Worker::~Worker()" << endl;
    }

    void Worker::work()
    {
    	cout << "Worker::work()" << endl;

    	cout << "m_iCode  = " << m_iCode << endl;
    }
MigrateWorker.h

    #ifndef MIGRATEWORKER_H
    #define MIGRATEWORKER_H

    #include "Farmer.h"
    #include "Worker.h"

    //多继承
    class MigrateWorker : public Farmer, public Worker
    {
    public:
    	MigrateWorker(string name = "MigrateWorker", int code = 303313);
    	virtual ~MigrateWorker();
    };

    #endif
MigrateWorker.cpp

    #include <iostream>

    #include "MigrateWorker.h"

    using namespace std;

    MigrateWorker::MigrateWorker(string name, int code) : Farmer(name), Worker(code)
    {
    	cout << "MigrateWorker::MigrateWorker(string name, int code)" << endl;
    }

    MigrateWorker::~MigrateWorker()
    {
    	cout << "MigrateWorker::~MigrateWorker()" << endl;
    }

        　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于04-22-2017
####8.虚继承
    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : VirtualInherit.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 04-24-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <stdlib.h>

    #include "MigrateWorker.h"

    using namespace std;

    int main()
    {
    	MigrateWorker *p = new MigrateWorker("Yellow", 10009);

    	p->sow();
    	p->work();

    	delete p;

    	system("pause");

    	return 0;
    }
Person.h

    #ifndef PERSON_H_
    #define PERSON_H_

    #include <iostream>
    #include <string>

    using namespace std;

    class Person
    {
    public:
    	Person(string color);
    	~Person();

    protected:
    	string m_strColor;
    };
    #endif
Person.cpp

    #include <iostream>
    #include <string>

    #include "Person.h"

    using namespace std;

    Person::Person(string color)
    {
    	m_strColor = color;

    	cout << "Person::Person(string color)" << endl;
    }

    Person::~Person()
    {
    	cout << "~Person::Person()" << endl;
    }

Farmer.h

    #ifndef FARMER_H_
    #define FARMER_H_

    #include <iostream>
    #include <string>

    #include "Person.h"

    using namespace std;

    class Farmer : virtual public Person
    {
    public:
    	Farmer(string color, string name = "Farmer");
    	virtual ~Farmer(); //但凡可能会继承时，都加上virtual关键字

    	void sow();

    protected:
    	//PA: string 是类，不是关键字
    	string m_sName;
    };
    #endif
Farmer.cpp

    #include "Farmer.h"
    #include <iostream>

    using namespace std;

    Farmer::Farmer(string color, string name) : Person(color)
    {
    	m_sName = name;

    	cout << "Farmer::Farmer(string name)" << endl;
    }

    Farmer::~Farmer()
    {
    	cout << "Farmer::~Farmer()" << endl;
    }

    void Farmer::sow()
    {
    	cout << "Farmer::sow()" << endl;

    	cout << "m_sName = " << m_sName << endl;
    	cout << "m_strColor  = " << m_strColor << endl;
    }
Worker.h

    #ifndef WORKER_H_
    #define WORKER_H_

    #include <iostream>
    #include <string>

    #include "Worker.h"
    #include "Person.h"

    using namespace std;

    class Worker : virtual public Person
    {
    public:
    	Worker(string color, int code = 202212);

    	virtual ~Worker();

    	void work();

    protected:
    	int m_iCode;
    };

    #endif
Worker.cpp

     #include <iostream>

    #include "Worker.h"

    using namespace std;

    Worker::Worker(string color, int code) : Person(color)
    {
    	m_iCode = code;

    	cout << "Worker::Worker(int code)" << endl;
    }

    Worker::~Worker()
    {
    	cout << "Worker::~Worker()" << endl;
    }

    void Worker::work()
    {
    	cout << "Worker::work()" << endl;

    	cout << "m_iCode  = " << m_iCode << endl;
    	cout << "m_strColor  = " << m_strColor << endl;
    }
MigrateWorker.h

    #ifndef MIGRATEWORKER_H
    #define MIGRATEWORKER_H

    #include "Farmer.h"
    #include "Worker.h"

    //多继承
    class MigrateWorker :  public Farmer, public Worker
    {
    public:
    	MigrateWorker(string color, int code = 303313, string name = "MigrateWorker");
    	virtual ~MigrateWorker();
    };

    #endif
MigrateWorker.cpp

    #include <iostream>

    #include "MigrateWorker.h"

    using namespace std;

    //不允许使用间接非虚拟基类
    MigrateWorker::MigrateWorker(string color, int code, string name) : Worker(color, code), Farmer(color, name), Person(color)
    {
    	cout << "MigrateWorker::MigrateWorker(string name, int code)" << endl;
    }

    MigrateWorker::~MigrateWorker()
    {
    	cout << "MigrateWorker::~MigrateWorker()" << endl;
    }
        　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于04-24-2017
### 006）C++远征之多态篇
####1.虚函数
    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : VirtualFunction.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 04-27-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <stdlib.h>

    #include "Circle.h"
    #include "Rect.h"

    using namespace std;

    int main()
    {
    	Shape *p1 = new Rect(3, 9);
    	Shape *p2 = new Circle(5);

    	//PA：观察对Shape的caclArea() 加与不加Virtual关键字的区别
    	p1->caclArea();
    	p2->caclArea();

    	delete p1;
    	p1 = NULL;

    	delete p2;
    	p2 = NULL;

    	system("pause");

    	return 0;
    }
Shape.h

    #ifndef	SHAPE_H_
    #define SHAPE_H_

    #include <iostream>
    #include <string>

    using namespace std;

    class Shape
    {
    public:
    	Shape();

    	virtual ~Shape();

    	virtual double caclArea();
    };
    #endif
Shape.cpp

    #include <iostream>

    #include "Shape.h"

    using namespace std;

    Shape::Shape()
    {
    	cout << "Shape::Shape()" << endl;
    }

    Shape::~Shape()
    {
    	cout << "~Shape::Shape()" << endl;
    }

    double Shape::caclArea()
    {
    	cout << "Shape::caclArea()" << endl;

    	return 0;
    }
Circle.h

    #ifndef CIRCLE_H_
    #define CIRCLE_H_

    #include <iostream>
    #include <string>

    #include "Shape.h"

    using namespace std;

    class Circle : virtual public Shape
    {
    public:
    	Circle(double r);
    	virtual ~Circle();

    	virtual double caclArea();

    protected:
    	double m_dR;
    };

    #endif
Circle.cpp

    #include <iostream>

    #include "Circle.h"

    using namespace std;

    Circle::Circle(double r)
    {
    	m_dR = r;

    	cout << "Circle::Circle(double r)" << endl;
    }

    Circle::~Circle()
    {
    	cout << "~Circle::Circle()" << endl;
    }

    double Circle::caclArea()
    {
    	cout << "Circle::caclArea()" << endl;

    	return 3.14 * m_dR *m_dR;
    }
Rect.h

    #ifndef RECT_H_
    #define RECT_H_

    #include <iostream>
    #include <string>

    #include "Shape.h"

    using namespace std;

    class Rect : virtual public Shape
    {
    public:
    	Rect(double width, double height);
    	virtual ~Rect();

    	virtual double caclArea();

    protected:
    	double m_dWidth, m_dHeight;
    };

    #endif
Rect.cpp

    #include <iostream>
    #include <string>

    #include "Rect.h"

    using namespace std;

    Rect::Rect(double width, double height)
    {
    	m_dWidth  = width;
    	m_dHeight = height;

    	cout << "Rect::Rect(double width, double height)" << endl;
    }

    Rect::~Rect()
    {
    	cout << "~Rect::Rect()" << endl;
    }

    double Rect::caclArea()
    {
    	cout << "Rect::caclArea()" << endl;
    	return m_dWidth * m_dHeight;
    }
        　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于04-27-2017
####2.虚析构函数
1、虚函数表指针原理
2、隐藏与覆盖

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : VirtualDestructor.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 04-30-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <stdlib.h>

    #include "Circle.h"

    using namespace std;

    int main(void)
    {
    	Shape *p2 = new Circle(5, 3, 4);

    	//PA：观察加与不加Virtual关键字对析构函数的影响
    	p2->caclArea();

    	delete p2;
    	p2 = NULL;

    	system("pause");

    	return 0;
    }
Shape.h

    #ifndef	SHAPE_H_
    #define SHAPE_H_

    #include <iostream>
    #include <string>

    using namespace std;

    class Shape
    {
    public:
    	Shape();

    	//PA：观察加与不加Virtual关键字（避免使用父类指针释放子类对象时造成内存泄露）
    	virtual ~Shape();

    	virtual double caclArea();
    };

    #endif
Shape.cpp

    #include <iostream>

    #include "Shape.h"

    using namespace std;

    Shape::Shape()
    {
    	cout << "Shape::Shape()" << endl;
    }

    Shape::~Shape()
    {
    	cout << "~Shape::Shape()" << endl;
    }

    double Shape::caclArea()
    {
    	cout << "Shape::caclArea()" << endl;

    	return 0;
    }
Circle.h

    #ifndef CIRCLE_H_
    #define CIRCLE_H_

    #include <iostream>
    #include <string>

    #include "Shape.h"
    #include "Coordinate.h"

    using namespace std;

    class Circle : public Shape
    {
    public:
    	Circle(double r, int x, int y);

    	virtual double caclArea();

    protected:
    	double m_dR;
    	Coordinate *m_pCenter;
    };

    #endif
Circle.cpp

    #include <iostream>

    #include "Circle.h"

    using namespace std;

    Circle::Circle(double r, int x, int y)
    {
    	m_dR = r;

    	m_pCenter = new Coordinate(x, y);

    	cout << "Circle::Circle" << endl;
    }

    Circle::~Circle()
    {
    	delete m_pCenter;
    	m_pCenter = NULL;

    	cout << "~Circle::Circle()" << endl;
    }

    double Circle::caclArea()
    {
    	cout << "Circle::caclArea()" << endl;

    	return 3.14 * m_dR *m_dR;
    }
Coordinate.h

    #ifndef COORDINATE_H_
    #define COORDINATE_H_

    class Coordinate
    {
    public:
    	Coordinate(int x, int y);
    	virtual ~Coordinate();

    protected:
    	int m_iX;
    	int m_iY;
    };

    #endif
Coordinate.cpp

    #include <iostream>

    #include "Coordinate.h"

    using namespace std;

    Coordinate::Coordinate(int x, int y)
    {
    	m_iX = x;
    	m_iY = y;

    	cout << "Coordinate() " << "(" << m_iX << "," << m_iY << ")" << endl;
    }

    Coordinate::~Coordinate()
    {
    	cout << "~Coordinate()" << endl;
    }
        　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于04-30-2017
####3.纯虚函数
    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : PureVirtualFunction.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 04-30-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <stdlib.h>

    #include "Dustman.h"
    #include "Worker.h"

    using namespace std;

    int main(void)
    {
    	//不能实例化抽象类
    	//Worker woker();

    	Dustman *p = new Dustman;

    	//PA：观察加与不加Virtual关键字对析构函数的影响
    	p->work();

    	delete p;
    	p = NULL;

    	system("pause");

    	return 0;
    }
Person.h

    #ifndef	PERSON_H_
    #define PERSON_H_

    class Person
    {
    public:
    	virtual void work() = 0; //纯虚函数
    };

    #endif

Worker.h

     #ifndef WORKER_H_
    #define WORKER_H_

    #include <iostream>
    #include <string>

    #include "Person.h"

    using namespace std;

    class Worker : public Person
    {
    public:
    	Worker(int code = 001);
    	virtual ~Worker();

    	void printCode();

    protected:
    	int m_iCode;
    };

    #endif
Worker.cpp

    #include <iostream>
    #include <string>

    #include "Worker.h"

    using namespace std;

    Worker::Worker(int code)
    {
    	m_iCode = code;

    	cout << "Worker::Worker()" << endl;
    }

    Worker::~Worker()
    {
    	cout << "~Worker::Worker()" << endl;
    }

    void Worker::printCode()
    {
    	cout << "m_iCode = " << m_iCode << endl;
    }
Dustman.h

    #ifndef DUSTMAN_H_
    #define DUSTMAN_H_

    #include "Worker.h"

    class Dustman : public Worker
    {
    public:
    	Dustman();
    	virtual ~Dustman();

    	virtual void work();
    };

    #endif
Dustman.cpp

    #include <iostream>

    #include "Dustman.h"

    using namespace std;

    Dustman::Dustman()
    {
    	cout << "Dustman::Dustman()" << endl;
    }

    Dustman::~Dustman()
    {
    	cout << "~Dustman::Dustman()" << endl;
    }

    void Dustman::work()
    {
    	cout << "I`m woring now..." << endl;
    }
        　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于04-30-2017

####4.接口类
    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : InterfaceClass.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 04-30-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <stdlib.h>

    #include "FighterPlane.h"

    using namespace std;

    void flyMatch(Flyable *p)
    {
    	p->takeoff();
    	p->land();
    }

    int main(void)
    {
    	FighterPlane *p = new FighterPlane;

    	flyMatch(p);

    	delete p;
    	p = NULL;

    	system("pause");

    	return 0;
    }
Flyable.h

    #ifndef	FLYABLE_H_
    #define FLYABLE_H_

    //只含纯虚函数的类为接口
    class Flyable
    {
    public:

    	virtual void takeoff() = 0;
    	virtual void land() = 0;
    };

    #endif

Plane.h

    #ifndef PLANE_H_
    #define PLANE_H_

    #include <iostream>
    #include <string>

    #include "Flyable.h"

    class Plane : public Flyable
    {
    public:
    	Plane(int code = 001);
    	virtual ~Plane();

    	void printCode();

    	//在战斗机里实现
    	//virtual void takeoff();
    	//virtual void land();


    protected:
    	int m_iCode;
    };

    #endif
Plane.cpp

    #include <iostream>
    #include <string>

    #include "Plane.h"

    using namespace std;

    Plane::Plane(int code)
    {
    	m_iCode = code;

    	cout << "Plane::Plane()" << endl;
    }

    Plane::~Plane()
    {
    	cout << "~Plane::Plane()" << endl;
    }

    void Plane::printCode()
    {
    	cout << "m_iCode = " << m_iCode << endl;
    }

    //在战斗机里面实现
    //void Plane::takeoff()
    //{
    //	cout << "Plane::takeoff()" << endl;
    //}
    //
    //void Plane::land()
    //{
    //	cout << "Plane::land()" << endl;
    //} 	
FighterPlane.h

    #ifndef FIGHTERPLANE_H_
    #define FIGHTERPLANE_H_

    #include "Plane.h"

    class FighterPlane : public Plane
    {
    public:
    	FighterPlane();
    	virtual ~FighterPlane();

    	virtual void work();

    	//虚函数，在这里和Plane类里都实现了一遍，用那个类就会调用到那个类
    	virtual void takeoff();
    	virtual void land();
    };

    #endif
FighterPlane.cpp

    #include "FighterPlane.h"

    using namespace std;

    FighterPlane::FighterPlane()
    {
    	cout << "FighterPlane::FighterPlane()" << endl;
    }

    FighterPlane::~FighterPlane()
    {
    	cout << "~FighterPlane::FighterPlane()" << endl;
    }

    void FighterPlane::work()
    {
    	cout << "I`m woring now..." << endl;
    }


    void FighterPlane::takeoff()
    {
    	cout << "FighterPlane::takeoff()" << endl;
    }

    void FighterPlane::land()
    {
    	cout << "FighterPlane::land()" << endl;
    }
        　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于04-30-2017
####5.运行时类型识别
    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : RTTI.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 04-30-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <stdlib.h>
    #include <typeinfo.h>  

    #include "Plane.h"
    #include "Bird.h"

    using namespace std;

    void doSomething(Flyable *obj)
    {
    	obj->takeoff();

    	//注意写法
    	if (typeid(*obj) == typeid(Bird))
    	{
    		Bird *bird = dynamic_cast<Bird *>(obj);
    		bird->foraging();
    	}

    	if (typeid(*obj) == typeid(Plane))
    	{
    		Plane *plane = dynamic_cast<Plane *>(obj);
    		plane->carry();
    	}

    	obj->land();
    }

    int main(void)
    {
    	Bird  *b = new Bird;
    	Plane *p = new Plane;

    	doSomething(b);
    	cout << endl;

    	doSomething(p);

    	delete b;
    	b = NULL;
    	delete p;
    	p = NULL;

    	system("pause");

    	return 0;
    }
Flyable.h

    #ifndef	FLYABLE_H_
    #define FLYABLE_H_

    //只含纯虚函数的类为接口
    class Flyable
    {
    public:

    	virtual void takeoff() = 0;
    	virtual void land() = 0;
    };

    #endif

Bird.h

    #ifndef BIRD_H_
    #define BIRD_H_

    #include "Flyable.h"

    class Bird :public Flyable
    {
    public:
    	Bird();
    	virtual ~Bird();

    	void foraging();

    	//虚函数，在这里和Plane类里都实现了一遍，用那个类就会调用到那个类
    	virtual void takeoff();
    	virtual void land();
    };

    #endif
Bird.cpp

    #include <iostream>

    #include "Bird.h"

    using namespace std;

    Bird::Bird()
    {
    	cout << "Bird::Bird()" << endl;
    }

    Bird::~Bird()
    {
    	cout << "~Bird::Bird()" << endl;
    }

    void Bird::foraging()
    {
    	cout << "Bird::foraging()" << endl;
    }


    void Bird::takeoff()
    {
    	cout << "Bird::takeoff()" << endl;
    }

    void Bird::land()
    {
    	cout << "Bird::land()" << endl;
    }
Plane.h

    #ifndef PLANE_H_
    #define PLANE_H_

    #include <iostream>
    #include <string>

    #include "Flyable.h"

    class Plane : public Flyable
    {
    public:
    	Plane();
    	virtual ~Plane();

    	void carry();

    	virtual void takeoff();
    	virtual void land();
    };

    #endif
Plane.cpp

    #include <iostream>
    #include <string>

    #include "Plane.h"

    using namespace std;

    Plane::Plane()
    {
    	cout << "Plane::Plane()" << endl;
    }

    Plane::~Plane()
    {
    	cout << "~Plane::Plane()" << endl;
    }

    void Plane::carry()
    {
    	cout << "Plane::carry()" << endl;
    }

    void Plane::takeoff()
    {
    	cout << "Plane::takeoff()" << endl;
    }

    void Plane::land()
    {
    	cout << "Plane::land()" << endl;
    }
        　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于04-30-2017
####6.异常处理
    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : TryCatch.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 04-30-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <stdlib.h>

    #include "IndexException.h"
    #include "Exception.h"

    using namespace std;

    void testThrow()
    {
    	throw IndexException();
    }

    int main(void)
    {
    	try
    	{
    		testThrow();
    	}

    	//EQ: catch (IndexException &e)
    	catch (Exception &e)
    	{
    		e.printException();
    	}

    	system("pause");

    	return 0;
    }
Exception.h

    #ifndef EXCEPTION_H_
    #define EXCEPTION_H_

    class Exception
    {
    public:
    	virtual ~Exception();

    	virtual void printException();
    };

    #endif
Exception .cpp

    #include <iostream>

    #include "Exception.h"

    using namespace std;

    Exception::~Exception()
    {
    	cout << "~Exception::Exception()" << endl;
    }

    void Exception::printException()
    {
    	cout << "Exception::printException()" << endl;
    }
IndexException.h

    #ifndef INDEXEXCEPTION_H_
    #define INDEXEXCEPTION_H_

    #include <iostream>
    #include <string>

    #include "Exception.h"

    class IndexException : public Exception
    {
    public:
    	virtual void printException();
    };

    #endif
IndexException.cpp

    #include <iostream>
    #include <string>

    #include "IndexException.h"

    using namespace std;

    void IndexException::printException()
    {
    	cout << "Tip: IndexException::printException()" << endl;
    }

        　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于04-30-2017
### 007）C++远征之模板篇
####1.友元函数
    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : FriendFunc.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 05-01-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <stdlib.h>

    #include "Time.h"
    #include "Match.h"

    using namespace std;

    void getTime(Time &t);

    int main(void)
    {
    	Time t(19, 59, 20);
    	getTime(t);


    	Match m;
    	m.printTime(t);

    	system("pause");

    	return 0;
    }

    void getTime(Time &t)
    {
    	cout << "1.通过声明友元全局函数而打印" << endl;

    	cout << t.m_iHour << ":" << t.m_iMin << ":" << t.m_iSec << endl;

    	cout << endl;
    }
Time.h

    #ifndef TIME_H_
    #define TIME_H_

    #include "Match.h"

    class Time
    {
    	//1.声明友元全局函数
    	friend void getTime(Time &t);

    	//2.声明友元成员函数
    	friend void Match::printTime(Time &t);

    public:
    	Time(int hour, int min, int sec);
    	virtual ~Time();

    private:
    	int m_iHour;
    	int m_iMin;
    	int m_iSec;
    };

    #endif

Time.cpp

    #include <iostream>

    #include "Time.h"

    using namespace std;

    Time::Time(int hour, int min, int sec)
    {
    	m_iHour = hour;
    	m_iMin = min;
    	m_iSec = sec;
    }

    Time::~Time()
    {

    }
Match.h

    #ifndef MATCH_H_
    #define MATCH_H_

    //声明Time的时候，都不包含头文件 ???
    //#include "Time.h"

    class Time;
    class Match
    {
    public:
    	void printTime(Time &t);
    };

    #endif
Match.cpp

    #include <iostream>

    #include "Time.h"
    #include "Match.h"


    using namespace std;

    void Match::printTime(Time &t)
    {
    	cout << "2.通过声明友元成员函数而打印" << endl;

    	//不知道为什么编译器提示错误：成员不可访问，实际编译通过
    	cout << t.m_iHour << "时" << t.m_iMin << "分" << t.m_iSec << "秒" << endl;

    	cout << endl;
    }
        　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于05-01-2017
###2.友元类
    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : FriendClass.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 05-01-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <stdlib.h>

    #include "Time.h"
    #include "Match.h"

    using namespace std;

    int main(void)
    {
    	Match m(18, 9, 10);

    	m.testTime();

    	system("pause");

    	return 0;
    }
Time.h

    ifndef TIME_H_
    #define TIME_H_

    //声明Match的时候，都不包含头文件 ???
    //#include "Match.h"

    class Match;

    class Time
    {
    	//3.声明友元类
    	friend Match;

    public:
    	Time(int hour, int min, int sec);
    	virtual ~Time();

    private:
    	void printTime();

    	int m_iHour;
    	int m_iMin;
    	int m_iSec;
    };

    #endif
Time.cpp

    #include <iostream>

    #include "Time.h"

    using namespace std;

    Time::Time(int hour, int min, int sec)
    {
    	m_iHour = hour;
    	m_iMin = min;
    	m_iSec = sec;
    }

    Time::~Time()
    {

    }

    void Time::printTime()
    {
    	cout << m_iHour << "时" << m_iMin << "分" << m_iSec << "秒" << endl;
    }
Match.h

    #ifndef MATCH_H_
    #define MATCH_H_

    //声明Time的时候，都不包含头文件 ???
    //#include "Time.h"

    class Time;
    class Match
    {
    public:
    	void printTime(Time &t);
    };

    #endif
Match.cpp

    #include <iostream>

    #include "Time.h"
    #include "Match.h"


    using namespace std;

    void Match::printTime(Time &t)
    {
    	cout << "2.通过声明友元成员函数而打印" << endl;

    	//不知道为什么编译器提示错误：成员不可访问，实际编译通过
    	cout << t.m_iHour << "时" << t.m_iMin << "分" << t.m_iSec << "秒" << endl;

    	cout << endl;
    }
        　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于05-01-2017
###3.静态函数
    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : StaticFunc.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 05-04-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <stdlib.h>

    #include "Tank.h"

    using namespace std;

    int main(void)
    {
    	//1. 没有实例化对象，静态成员已经存在，可以直接打印使用
    	cout << Tank::getCount() << endl;

    	//2. 实例化一个对象，增加一辆tank
    	Tank *p = new Tank('A');
    	cout << Tank::getCount() << endl;

    	//5. 是否相互调用，静态和非静态函数或成员（见类内例子）--> this指针
    	p->fire();


    	//3. 销毁后，减少一辆tank
    	delete p;
    	p = NULL;
    	cout << Tank::getCount() << endl;

    	system("pause");

    	return 0;
    }
Tank.h

    #ifndef TANK_H_
    #define TANK_H_

    class Tank
    {

    public:
    	Tank(char code);
    	virtual ~Tank();

    	void fire();

    	//4. 是否能用const去修饰静态成员函数
    	//A: 静态成员函数上不允许修饰符
    	//Err: static int getCount() const; //error C2272: “getCount”: 静态成员函数上不允许修饰符
    	static int getCount();

    private:
    	static int s_iCount;
    	char m_cCode;
    };

    #endif
Tank.cpp

    #include <iostream>

    #include "Tank.h"

    using namespace std;

    //静态变量要在类外初始化（类似全局函数，待深入）
    //静态数据成员不能在构造函数初始化，必须单独初始化
    int Tank::s_iCount = 5;

    Tank::Tank(char code)
    {
    	m_cCode = code;

    	//每实例化一次，tank就要增加一辆
    	s_iCount++;

    	cout << "Tank::Tank()" << endl;
    }

    Tank::~Tank()
    {
    	//每次销毁一次，tank就要减少一辆
    	s_iCount--;

    	cout << "~Tank::Tank()" << endl;
    }

    void Tank::fire()
    {
    	//1. 普通成员函数可以调用静态成员函数(this指针)
    	cout << getCount() << endl;

    	cout << "Tank::fire()" << endl;
    }

    int Tank::getCount()
    {
    	//2. 静态成员函数 不可以 调用普通成员函数(没有this指针)
    	//Err: fire(); //非静态成员函数的非法调用

    	return s_iCount;
    }

        　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于05-04-2017
###4.一元运算符重载
    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : OneVariableOperator.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 05-07-2017
     *      Description:     
     ********************************************************************************
     */

    #include <iostream>
    #include <stdlib.h>

    #include "Coordinate.h"

    using namespace std;

    int main(void)
    {
    	Coordinate coor1(4, 6);

    	//coor1.printXY();

    	//1、- 号重载
    	//这就是引用和*this的好处，可以连接起来，另外这里重载了两次
    	( -(-coor1) ).printXY();


    	//3.1（++前置）成员函数 重载
    	Coordinate coor2(1, 2);

    	(++coor2).printXY();

    	//3.2（++后置）成员函数 重载
    	Coordinate coor3(3, 4);

    	(coor3++).printXY(); //还是    (3, 4)

    	coor3.printXY();     //再调用为(4, 5)

    	system("pause");

    	return 0;
    }
Coodinate.h

    #ifndef COORDINATE_H_
    #define COORDINATE_H_

    class Coordinate
    {
    public:
    	Coordinate(int x, int y);
    	virtual ~Coordinate();


    	//PA: 同一个操作赋只能有一种方式重载，不然报错不明确
    	//1.1 成员函数重载
    	Coordinate &operator-();

    	//1.2 友元函数重载（不属于任何一个类）
    	//friend Coordinate &operator-(Coordinate &c);


    	//2.1.1 （++前置）成员函数 重载
    	Coordinate &operator++();

    	//2.1.2 （++前置）友元函数 重载
    	//friend Coordinate &operator++(Coordinate &c);

    	//2.2.1 （++后置）成员函数 重载
    	Coordinate operator++(int);

    	//2.2.2 （++后置）成员函数 重载 ???????
    	//friend Coordinate &operator++(Coordinate &c, int);

    	void printXY();

    private:
    	int m_iX;
    	int m_iY;
    };

    #endif
Coordinate.cpp

   #include "Coordinate.h"

#include <iostream>

using namespace std;

Coordinate::Coordinate(int x, int y)
{
	m_iX = x;
	m_iY = y;
}

Coordinate::~Coordinate()
{

}

void Coordinate::printXY()
{
	cout << "(" << m_iX << " , " << m_iY << ")" << endl;
}

//1.1 成员函数重载
//这里面其实是有this指针的
Coordinate &Coordinate::operator-()
{
	this->m_iX = -this->m_iX;
	m_iY = -m_iY;

	return *this;
}

//1.2 友元函数重载（不属于任何一个类）
//Coordinate &operator-(Coordinate &c)
//{
//	c.m_iX = -c.m_iX;
//	c.m_iY = -c.m_iY;
//
//	return c;
//}


//2.1.1 （++前置）成员函数 重载，类似++i
Coordinate &Coordinate::operator++()
{
	++m_iX; //m_iX++; //因为后面返回*this，m_iX的值都加了一次
	++m_iY; //m_iY++;

	return *this;
}

//2.1.2 （++前置）友元函数 重载
//Coordinate &operator++(Coordinate &c)
//{
//	++c.m_iX;
//	++c.m_iY;
//
//	return c;
//}

//2.2.1 （++后置）成员函数 重载，类似i++
Coordinate Coordinate::operator++(int)
{
	Coordinate old(*this);

	m_iX++;
	m_iY++;

	return old; //返回原来的，下一次调用的时候值才增加
}


//2.2.2 （++后置）成员函数 重载 ???????
//Coordinate operator++(Coordinate &c, int)
//{
//	Coordinate old(c);
//
//	c.m_iX++;
//	c.m_iY++;
//
//	return old;
//}
Soldier.h

    #ifndef _SOLDIER_H_
    } 	
Soldier.cpp

    #include <iostream>

    } 	
Infantry.h

    #ifndef _INFANTRY_H_

    #endif
Infantry.cpp

    #include <iostream>
    }
        　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于05-01-2017

###5.template
    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : TryCatch.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 05-01-2017
     *      Description:     
     ********************************************************************************
     */

     #include <iostream>

    }
Person.h

    #include <iostream>
    }
Person.cpp

    #include <iostream>
    }
Soldier.h

    #ifndef _SOLDIER_H_
    } 	
Soldier.cpp

    #include <iostream>

    } 	
Infantry.h

    #ifndef _INFANTRY_H_

    #endif
Infantry.cpp

    #include <iostream>
    }
        　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于05-01-2017        
###6.template
    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : TryCatch.cpp
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 05-01-2017
     *      Description:     
     ********************************************************************************
     */

     #include <iostream>

    }
Person.h

    #include <iostream>
    }
Person.cpp

    #include <iostream>
    }
Soldier.h

    #ifndef _SOLDIER_H_
    } 	
Soldier.cpp

    #include <iostream>

    } 	
Infantry.h

    #ifndef _INFANTRY_H_

    #endif
Infantry.cpp

    #include <iostream>
    }
        　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　更于05-01-2017    
全剧终！

不知道为什么，一直以来就想着早点尽快地学完，没想等真的学完了，竟有一种失落的感觉，就好像一直追一部剧，成了一种习惯，突然之间全剧终了……

或许该是用的时候了，上路吧~

去慕课致谢！谢谢您
