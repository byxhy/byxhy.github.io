---
layout: post
title: "Algorithm thought"
author: "Xhy"
categories: No-Standard
tags: [algorithm,sample]
image: sabine-schulte.jpg
---


Photo by sabine schulte

>This is a multithreading template adapted from Microsoft official [document](https://msdn.microsoft.com/en-us/library/windows/desktop/ms682516(v=vs.85).aspx).

<br />

The following is a simple example that demonstrates how to create a new thread that executes the locally defined function, MyThreadFunction.

```c++
#include <iostream>
#include <algorithm>
#include <string>
#include <stdlib.h>

using namespace std;

void selectionSort(int arr[], int n)
{
	for (int i = 0; i < n - 1; i++) {
		int minIndex = i; //放在那很关键

		for (int j = i + 1; j < n - 1; j++) {
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

<br />
