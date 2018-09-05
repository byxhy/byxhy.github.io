---
layout: post
title: "Algorithm thought"
author: "Xhy"
categories: No-Standard
tags: [algorithm,sample]
image: sabine-schulte.jpg
---


Photo by sabine schulte

>声明：系列筆記參照[莫煩python](https://morvanzhou.github.io/)教學網站的课程整理而來，感謝老師的分享

<br />

## Table of Contents

1. [Selection Sort](#1---selection-sort)
   1. [basic selection sort](#1-1---basic-selection-sort)
	 2. [using template](#1-2---using-template)
	 3. [generate test cases](#1-3---generate-test-cases)

2. [Basic use](#2---basic-use)
   1. [print function](#2-1---print-function)
   2. [basic computation](#2-2---basic-computation)
   3. [variable](#2-3---variable)

3. [While and for loop](#3---while-and-for-loop)
   1. [while](#3-1---while)
   2. [for](#3-2---for)
   3. [range](#3-3---range)
   4. [set type](#3-4---set-type)

4. [If](#4---if)
   1. [basic use](#4-1---basic-use)


5. [Define function](#5---define-function)
   1. [def use](#5-1---def-use)
   2. [function parameters](#5-2---function-parameters)
   3. [default parameters](#5-3---default-parameters)
   4. [call yourself, only called inside the script](#5-4---call-yourself-only-called-inside-the-script)
   5. [variable parameters](#5-5---variable-parameters)
   6. [keyword parameters](#5-6---keyword-parameters)

6. [Local variables and global variables](#6---local-variables-and-global-variables)
   1. [local variable](#6-1---local-variable)
   2. [global variable](#6-2---global-variable)

7. [Install the modules](#7---install-the-modules)
   1. [install Numpy and matplotlib](#7-1---install-numpy-and-matplotlib)

8. [Read and write file](#8---read-and-write-file)
   1. [write](#8-1---write)
   2. [write](#8-2---write)
   3. [read](#8-3---read)

9. [Class](#9---class)
   1. [class](#9-1---class)
   2. [class and init](#9-2---class-and-init)
   3. [default parameters in init](#9-3---default-parameters-in-init)

10. [Input](#10---input)
    1. [input](#10-1---input)
    2. [input extension](#10-2---input-extension)

11. [Tuple \ list \ dictionary](#11---tuple-list-dictionary)
    1. [tuple](#11-1---tuple)
    2. [input extension](#11-2---input-extension)


<br />

---

<br />


## 1 - Selection Sort

### 1-1 - basic selection sort

```c++
##include <iostream>
#include <algorithm>
#include <string>
#include <stdlib.h>

using namespace std;

void selectionSort(int arr[], int n)
{
    for (int i = 0; i < n - 1; i++) {
        int minIndex = i; //放在那很关键

        for (int j = i + 1; j < n; j++) {
        if (arr[j] < arr[minIndex]) {
                minIndex = j;
            }
        }

        swap(arr[i], arr[minIndex]);
    }
}

int main(void)
{
    //1. int
    int a[10] = { 4, 6, 2, 2, 8, 3, 9, 1, 0, 11 };

    selectionSort(a, 10);

    for (int m = 0; m < 10; m++) {
        cout << a[m] << " ";
    }

    cout << endl;

    system("pause");

    return 0;
}
```
### 1-2 - using template

SelectionSort.cpp
```c++
#include "Student.h"

#include <iostream>
#include <algorithm>
#include <string>
#include <stdlib.h>

using namespace std;

template <typename T> //泛型
void selectionSort(T arr[], int n)
{
    for (int i = 0; i < n - 1; i++) {
        int minIndex = i; //放在那很关键

        for (int j = i + 1; j < n; j++) {
            //Err: if (arr[minIndex] > arr[j])
            //要与重载操作符方向一致，在比较结构体的时候
            if (arr[j] < arr[minIndex]) {
                minIndex = j;
            }
        }

        swap(arr[i], arr[minIndex]);
    }
}

int main(void)
{
    //1. int
    int a[10] = { 4, 6, 2, 2, 8, 3, 9, 1, 0, 11 };

    selectionSort(a, 10);

    for (int m = 0; m < 10; m++) {
        cout << a[m] << " ";
    }
    cout << endl;

    //2. double
    double b [4] = { 4.4, 1.1, 2.2, 6.6 };

    selectionSort(b, 4);

    for (int m = 0; m < 4; m++)	{
        cout << b[m] << " ";
    }
    cout << endl;

    //3. string
    string c[4] = { "A", "D", "C", "B" };

    selectionSort(c, 4);

    for (int m = 0; m < 4; m++)	{
        cout << c[m] << " ";
    }
    cout << endl;

    //4. Student
    Student d[4] = { { "A", 90 }, { "D", 92 }, { "C", 92 }, { "B", 97 } };

    selectionSort(d, 4);

    for (int m = 0; m < 4; m++)	{
        //cout << d[m] << " "; //输出和小于号，都重载过了
        cout << d[m];
    }
    cout << endl;

    system("pause");

    return 0;
}
```

Student.h
```c++
#ifndef STUDENT_H_
#define STUDENT_H_

#include <iostream>
#include <string>

using namespace std;

struct Student
{
    string name;
    int    score;

    //自定义比较的威力
    bool operator<(const Student &otherStudent)
    {
        //要与后面排序计算方向一致，两个学生的分数相等按名字顺序排，不等按分数排
        return score != otherStudent.score ? score < otherStudent.score : name < otherStudent.name;
	}

    friend ostream& operator<<(ostream &os, const Student &student)
    {
        os << "Student: " << student.name << " " << student.score << endl;

        return os;
    }
};

#endif
```
### 1-3 - generate test cases

SelectionSort.cpp
```c++
#include "SortTestHelper.h"

#include <iostream>
#include <algorithm>
#include <string>
#include <stdlib.h>


using namespace std;

template <typename T> //泛型
void selectionSort(T arr[], int n)
{
    for (int i = 0; i < n - 1; i++) {
        int minIndex = i; //放在那很关键

        for (int j = i + 1; j < n; j++) {
            //Err: if (arr[minIndex] > arr[j])
            //要与重载操作符方向一致，在比较结构体的时候
            if (arr[j] < arr[minIndex]) {
                minIndex = j;
            }
        }

        swap(arr[i], arr[minIndex]);
    }
}

int main(void)
{
    int n = 100;

    int *arr = SortTestHelper::generateRandomArray(n, 0, n);

    selectionSort(arr, n);

    SortTestHelper::printArray(arr, n);

    delete []arr;
    arr = NULL;

    system("pause");

    return 0;
}
```

SortTestHelper.h

```c++
#ifndef SORTTESTHELPER_H_
#define SORTTESTHELPER_H_

#include <iostream>
#include <string>
#include <ctime>
#include <cassert>

using namespace std;

namespace SortTestHelper
{
    //调用后记得释放
    int *generateRandomArray(int n, int rangeL, int rangeR)
    {
        assert(rangeL <= rangeR);

        int *arr = new int[n];

        srand(time(NULL));

        for (int i = 0; i < n; i++)
            arr[i] = rand() % (rangeR - rangeL + 1) + rangeL;

        return arr;
    }

    void printArray(int *arr, int n)
    {
        if (NULL == arr) {
            cout << "Input array is error." << endl;
            return;
    }

    for (int i = 0; i < n; i++)
        cout << arr[i] << " ";

        cout << endl;
    }

}

#endif
```
## 2 - Basic use

### 2-1 - print function

#### 2-1-1 - print string

<br />
