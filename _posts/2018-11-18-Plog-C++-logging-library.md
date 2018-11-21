---
layout: post
title: "C++ logging library"
author: "Xhy"
categories: c++
tags: [logging]
image: markus-spiske-78531.jpg
---


Photo by markus-spiske-78531

>Plog is a C++ logging library that is designed to be as simple, small and flexible as possible. It is created as an alternative to existing large libraries and provides some unique features as CSV log format and automatic 'this' pointer capture.

<br />


### Statement

The c++ logging library is copy from ([Sergey Podobryf's github](https://github.com/SergiusTheBest/plog)). I did some simple redevelopment on this basis.


### Usage

At first your project needs to know about plog. For that you have to:

* Add plog/include to the project include paths
* Add #include <plog/Log.h> into your cpp/h files (if you have precompiled headers it is a good place to add this include there)
* Add #include "Clog.h" into your cpp/h files

### Here is Clog code

Clog.h
```c++
#ifndef _CLOG_H_
#define _CLOG_H_

#include <plog/Log.h>

#include <iostream>
#include <string>

bool InitLogDir(std::string dir, std::string logname);

const std::string GetCurrentDate();

#endif
```

Clog.cpp
```c++
#include "Clog.h"

#include <Windows.h>

bool InitLogDir(std::string dir, std::string logname)
{
    // 1. Create root directory
    if (!CreateDirectoryA(dir.c_str(), 0))
    {
        if (ERROR_ALREADY_EXISTS == GetLastError())
        {
            // 2. Create date directory
            std::string logdir = dir + "\\" + GetCurrentDate();
            if (!CreateDirectoryA(logdir.c_str(), 0))
            {
                if (ERROR_ALREADY_EXISTS == GetLastError())
                {
                    //ERROR_ALREADY_EXISTS
                }
                else
                {
                    plog::init(plog::debug, "c:\\Clog.txt");
                    LOGE << "Create date directory failed: GetLastError() = " << GetLastError();
                    return false;
                }
            }
        }
        else
        {
            plog::init(plog::debug, "c:\\Clog.txt");
            LOGE << "Create root directory failed: GetLastError() = " << GetLastError();
            return false;
        }
    }

    // 2. Create date directory
    std::string logdir = dir + "\\" + GetCurrentDate();
    if (!CreateDirectoryA(logdir.c_str(), 0))
    {
        if (ERROR_ALREADY_EXISTS == GetLastError())
        {
            //ERROR_ALREADY_EXISTS
        }
        else
        {
            plog::init(plog::debug, "c:\\Clog.txt");
            LOGE << "Create date directory failed: GetLastError() = " << GetLastError();
            return false;
        }

    }

    // 3. Init the plog
    std::string logfile = dir + "\\" + GetCurrentDate() + "\\" + logname;
    plog::init(plog::debug, logfile.c_str());

    return true;
}

const std::string GetCurrentDate()
{
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[20];
    localtime_s(&tstruct, &now);

    strftime(buf, sizeof(buf), "%Y%m%d", &tstruct);

    return buf;
}
```

main.cpp
```c++
#include "Clog.h"

int testclog();

int main()
{
    testclog();

    return 0;
}

int testclog()
{
    //Only one level of directory is supported
    InitLogDir("E:\\Clog", "log.txt");

    LOGD << "Hello Plog";
    LOGE << "Hello Plog";
    LOGW << "Hello Clog";
    LOGI << "Hello Clog";

    return 0;
}
```

log.txt
```c++
2018-11-21 13:15:38.204 DEBUG [6064] [testclog@16] Hello Plog
2018-11-21 13:15:38.205 ERROR [6064] [testclog@17] Hello Plog
2018-11-21 13:15:38.205 WARN  [6064] [testclog@18] Hello Clog
2018-11-21 13:15:38.205 INFO  [6064] [testclog@19] Hello Clog
```
<br />
