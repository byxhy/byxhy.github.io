---
layout: post
title: "Algorithm thought"
author: "Xhy"
categories: No-Standard
tags: [algorithm,sample]
image: sabine-schulte.jpg
---


Photo by sabine schulte

>声明：系列筆記參照[liuyubobobo老師](https://coding.imooc.com/learn/list/71.html/)教學课程整理而來，感謝老師的分享

<br />

## Table of Contents

1. [Selection Sort](#1---selection-sort)
   1. [basic selection sort](#1-1---basic-selection-sort)
	 2. [using template](#1-2---using-template)
	 3. [generate test cases](#1-3---generate-test-cases)
   4. [detect performance](#1-4---detect-performance)

<br />

---

<br />


## 1 - Selection Sort

### 1-1 - basic selection and bubble sort

```c++
#include <iostream>

void swap_vehicle(int &a, int &b)
{
    int c = a;
    a = b;
    b = c;
}
void selectionSort(int arr[], int length)
{
    //1. target: output the array in order
    //2. method: selec the smallest number int each round and swap
    //3. Go

    for (int i = 0; i < length - 1; i++) {
        int min = arr[i];
        int min_index = i;

        for (int j = i + 1; j < length; j++) {
            if (min > arr[j]) {
                min = arr[j];
                min_index = j;
            }
        }

        if(min_index != i)
            swap_vehicle(arr[i], arr[min_index]);
    }

}
void selectionSort2(int arr[], int length)
{
    for (int i = 0; i < length - 1; i++) { //PA-1-border
        int min_index = i;

        for (int j = i + 1; j < length; j++) { //PA-2-border
            if (arr[min_index] > arr[j]) {
                min_index = j;
            }
        }

        if (min_index != i)
            swap_vehicle(arr[i], arr[min_index]); 
    }

}

void bubbleSort(int arr[], int length)
{
    //1. target: output the array in order
    //2. method: compare the two numbers from tail and swap them
    //3. Go
    for (int i = 0; i < length - 1; i++) {
        for (int j = length - 1; j > i; j--) {
            if (arr[j - 1] > arr[j]) {
                swap_vehicle(arr[j - 1], arr[j]);
            }
        }   
    }

}

void bubbleSort2(int arr[], int length)
{
    for (int i = 0; i < length - 1; i++) {
        int swap_counts = 0;
        for (int j = length - 1; j > i; j--) {
            if (arr[j - 1] > arr[j]) {
                swap_vehicle(arr[j - 1], arr[j]);
                swap_counts++;
            }
        }

        if (0 == swap_counts) //PA-3-stop condition
            break;
    }

}

int main()
{
    int a[10] = { 4, 6, 2, 2, 8, 3, 9, 1, 0, 11 };

    //selectionSort(a, 10);
    //selectionSort2(a, 10);
    //bubbleSort(a, 10);
    bubbleSort2(a, 10);


    for (int m = 0; m < 10; m++) {
        std::cout << a[m] << " ";
    }
    std::cout << std::endl;

    system("pause");
}

```

**Summary:**

1. Pay attention the border
2. Find the stop conditions
3. 

<br />


### 1-2 - using template

SelectionSort.cpp
```c++
#include "Student.h"

#include <iostream>
#include <algorithm>
#include <string>
#include <stdlib.h>

using namespace std;

template <typename T>
void selectionSort(T arr[], int n)
{
    for (int i = 0; i < n - 1; i++) {
        int minIndex = i; //It matters where you put it.

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
        //cout << d[m] << " "; //The output symbol is overloaded
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

    //custom comparison
    bool operator<(const Student &otherStudent)
    {
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

template <typename T>
void selectionSort(T arr[], int n)
{
    for (int i = 0; i < n - 1; i++) {
        int minIndex = i;

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
    //Remenber to release
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

### 1-4 - detect performance

SelectionSort.cpp
```c++
#include "SortTestHelper.h"

#include <iostream>
#include <algorithm>
#include <string>
#include <stdlib.h>


using namespace std;

template <typename T>
void selectionSort(T arr[], int n)
{
    for (int i = 0; i < n - 1; i++) {
        int minIndex = i;

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
    int n = 1000;

    int *arr = SortTestHelper::generateRandomArray(n, 0, n);

    SortTestHelper::testSort("selectionSort", selectionSort, arr, n);

    SortTestHelper::printArray(arr, n);

    delete [] arr;
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
    //Remenber to release
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

    template <typename T>
    bool isSorted(T arr[], int n)
    {
        for (int i = 0; i < n - 1; i++) {
            if (arr[i] > arr[i+1]) {
                cout << "Sort failed." << endl;
                return false;
            }
        }

        return true;
    }

    template <typename T>
    void testSort(string funcName, void(*sort)(T arr[], int), T arr[], int n)
    {
        clock_t startTime = clock();
        sort(arr, n);
        clock_t stopTime = clock();

        //Check
        assert( isSorted(arr, n) );

        cout << funcName + " cost time: " << double(stopTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    }

}

#endif
```


<br />
