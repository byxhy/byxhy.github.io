---

layout: post
title: "Hello Unix programming"
author: "Xhy"
categories: Linux
tags: [Embedded]
image: sai-kiran-anagani-Tjbk79TARiE.jpg
---

Photo by sai-kiran-anagani

> Unix环境高级编程学习
>
> 编程环境：Red Hat Enterprise Linux 6
>
> 内核版本：2.6.32-279

<br />



## Table of Contents

001）静态函数库设计

002）动态函数库设计

003）系统调用文件编程

004）库函数文件编程

005）时间函数编程

006）多进程程序设计

007）无名管道通讯编程

008）有名管道通讯编程

009）信号通讯编程

010）信号量互斥编程

011）信号量同步编程

012）共享内存通讯

013）消息队列编程

014）多线程程序设计

015）多线程同步

016）TCP通讯程序设计

017）UDP通讯程序设计

018）网络并发服务器设计

019）守护进程设计

020）Shell脚本高级编程

<br />



### 001）静态函数库设计

目标：设计一个简单求三个数中最大数的静态函数库

<1>代码准备：

max_num.c

    /*
     *****************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : max_num.c
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 04-21-2015
     *      Description:     
     *****************************************************************
     */
     
    #include <stdio.h>
    #include "max.h"
    
     int main(void)
     {
    
     	int max_num = 0;
     	int a,b,c;
     	a = b = c = 0;
    
     	printf("Please input the three number:\n");
     	printf("a = ");
     	scanf("%d",&a);
    
     	printf("b = ");
     	scanf("%d",&b);
    
     	printf("c = ");
     	scanf("%d",&c);
    
     	max_num = max(a, b, c);
    
     	printf("The max number is %d.\n",max_num);
    
     	return 0;
    
     }

max.c

    /*
     *****************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : max.c
     *      Author     : X h y
     *      Version    : 1.0
     *      Date       : 04-21-2015
     *      Description:     
     *****************************************************************
     */
    
    #include <stdio.h>
    
    int max(int a, int b, int c)
    {
    
    	int max_num = 0;
    
    	max_num = (a >= b) ? a : b;
    	max_num = (max_num >= c) ? max_num : c;
    
    	return max_num;
    	
    }
max.h

    extern int max(int a, int b, int c);


<2>制作静态库：

- gcc -c max.c -o max.o
- ar cqs libmax.a max.o
- 将制作好的 libmax.a 复制到 /usr/lib

<3>使用静态库：
- gcc max_num.c -lmax -o max_num
- GCC在链接时，默认只链接C函数库，而对于其他的函数库，则需要使用 -l 选项来指明链接库



### 002）动态函数库设计

### 002）动态函数库设计
目标：设计一个简单求三个数中最大数的动态函数库



<1>代码准备：
- 同静态函数代码

<2>制作动态库：
- gcc -c max.c -o max.o
- gcc -shared -fPIC max.o -o libmax.so
- 将制作好的 libmax.so 复制到 /usr/lib

<3>使用动态库：
- gcc max_num.c -lmax -o max_num

　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

### 003）系统调用文件编程
目标：学习使用linux系统调用下常用文件编程函数

1.打开文件函数：open 

open.c

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : open.c
     *      Author     : X h y
     *      Version    : 1.0
     *      Date       : 04-23-2015
     *      Description:
     ********************************************************************************
     */
     
    /**
     * \file
     * \brief open file
     */
    
    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/stat.h> 
    #include <fcntl.h>
    
    /**
     * \brief main entry
     */
    
    void main(void)
    {
    	int fd = 0;
    
    	/* open the file */
    	fd = open("./tst.txt", O_RDWR | O_CREAT, 0644);
    
    	if (fd < 0) 
    		printf("Failed to open the file !\n");
    	else
    		printf("File already exists !\n");
    }
    
    /* end of file */


2.创建文件函数：creat

creat.c

    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/stat.h> 
    #include <fcntl.h>
    void main(void)
    {
    	int fd = 0;
    
    	/* create the file */
    	fd = creat("./tst.txt", 0664);
    
    	if (fd < 0) 
    		printf("Failed to open the file !\n");
    	else
    		printf("File already exists !\n");
    }


3.关闭文件函数：close

close.c

    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/stat.h> 
    #include <fcntl.h>
    #include <unistd.h>
    void main(void)
    {
    	int fd = 0;
    
    	/* create the file */
    	fd = creat("./tst.txt", 0664);
    
    	if (fd < 0) 
    		printf("Failed to open the file !\n");
    	else
    		printf("File already exists !\n");
    
    	/* close file */
    	close(fd);
    }


4.读文件函数：read

read.c

    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/stat.h> 
    #include <fcntl.h>
    #include <unistd.h>
    void main(void)
    {
    	int fd = 0;
    	
    	char buf[100] = {0};
    
    	/* open the file */
    	fd = open("./read.txt", O_RDWR);
    
    	if (fd < 0) 
    		printf("Failed to open the file !\n");
    	else
    		printf("File is found !\n");
    
    	/* read file*/
       read(fd, buf, 100);
    
       printf("\nWords from read.txt is :\n%s\n", buf);
    
    	/* close file */
    	close(fd);
    }


5.写文件函数：write

write.c

    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/stat.h> 
    #include <fcntl.h>
    #include <unistd.h>
    void main(void)
    {
    	int fd = 0;
    	
    	char *buf = "How to write data to the file ?";
    
    	/* open the file */
    	fd = open("./write.txt", O_RDWR | O_CREAT, 0644);
    
    	if (fd < 0) 
    		printf("Failed to open the file !\n");
    	else
    		printf("File is found !\n");
    
    	/* write to file */
    	write(fd, buf, 31);
    
    	printf("write succuess !\n");
    
    	/* close file */
    	close(fd);
    }


6.定位文件函数：lseek

lssek.c


    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/stat.h> 
    #include <fcntl.h>
    #include <unistd.h>
    void main(void)
    {
    	int  fd = 0;
    
    	char *buf     = "How to write data to the file ?";
    	char buff[40] = {0};
    
    	fd = open("./position.txt", O_RDWR | O_CREAT, 0664);
    
    	if (fd < 0) 
    		printf("Failed to open the file !\n");
    	else
    		printf("File is found !\n");
    
    	/* write to file */
    	write(fd, buf, 31);
    
    	/* set to 0 offset bytes */
    	lseek(fd, 0, SEEK_SET);
    
    	read(fd, buff, 31);
    
    	printf("\nWords from position.txt is :\n%s\n", buff);
    
    	/* close file */
    	close(fd);
    }


7.复制文件描述符函数：dup

dup.c


    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/stat.h> 
    #include <fcntl.h>
    #include <unistd.h>
    void main(void)
    {
    	int fd_old  = 0;
    	int fd      = 0;
    
    	char *buf     = "How to write data to the file ?";
    	char buff[40] = {0};
    
    	fd_old = open("./dup.txt", O_RDWR | O_CREAT, 0664);
    
    	if (fd_old < 0) 
    		printf("Failed to open the file !\n");
    	else
    		printf("File is found !\n");
    
    	/* copy of the file descriptor oldfd */
    	fd = dup(fd_old);
    	
    	write(fd, buf, 31);
    
    	lseek(fd, 0, SEEK_SET);
    
    	read(fd, buff, 36);
    
    	printf("\nWords from dup.txt is:\n%s\n", buff);
    
    	/* close file */
    	close(fd);
    }


8.综合实例：文件复制程序

cp.c

    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/stat.h> 
    #include <fcntl.h>
    #include <unistd.h>
    void main(int argc, char **argv)
    {
    	int fd_s  = 0;
    	int fd_d  = 0;
    
    	int count = 0;
    
    	char buf[512] = {0};
    
        /* 1.open the source file */
        fd_s = open(argv[1], O_RDONLY);
    
        /* 2.open the destination file */
        fd_d = open(argv[2], O_RDWR | O_CREAT, 0664);
    
    	/* 3.read the source file */
    	/* 4.write to destination file */
    	while((count = read(fd_s, buf, 512)) > 0) {
    	/* write count bytes not 512 */
    	write(fd_d, buf, count);
        }
    
    	/* close file */
    	close(fd_s);
    	close(fd_d);
    }


　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

### 004）库函数文件编程
目标：学习使用库函数文件编程函数

1.打开文件函数：fopen 

fopen.c

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : fopen.c
     *      Author     : X h y
     *      Version    : 1.0
     *      Date       : 04-27-2015
     *      Description:
     ********************************************************************************
     */
     
    /**
     * \file
     * \brief open the file
     */
    
    #include <stdio.h>
    
    /**
     * \brief main entry
     */
    
    int main(void)
    {
    	FILE *fp;
    
    	/**
    	 * The file is created if it does not exist,           
    	 * otherwise it is truncated.
    	 */
    	fp = fopen("./tst.txt", "w+");
    
    	if (NULL == fp)
    		printf("Failed to open the file !\n");
    	else
    		printf("File already exists !\n");
    
    	return 0;
    }
    
    /* end of file */



2.关闭文件函数：fclose

fclose.c

    #include <stdio.h>
    int main(void)
    {
    	FILE *fp;
    
    	/**
    	 * The file is created if it does not exist,           
    	 * otherwise it is truncated.
    	 */
    	fp = fopen("./tst.txt", "w+");
    
    	if (NULL == fp)
    		printf("Failed to open the file !\n");
    	else
    		printf("File is found !\n");
    
    	/* close the file */
    	fclose(fp);
    
    	return 0;
    }


3.读文件函数：fread

fread.c


    #include <stdio.h>
    int main(void)
    {
    	FILE *fp;
    
    	int count = 0;
    
    	char c_buf[15] = {0};
    
    	/**
    	 * if the file already exists, 
    	 * can't use the "w+",        
    	 * the file will be truncated.
    	 */
    	/* open the file */
    	fp = fopen("./tst.txt", "r+");
    
    	if (NULL == fp)
    		printf("Failed to open the file !\n");
    	else
    		printf("File is found !\n");
    	
    	/**          
    	 * if 15 < = count -->(c_buf[15]),
    	 * it woule be "Segment fault".
    	 */
    	/* read the file */
    	count = fread(c_buf, 1, 10, fp);
    
    	c_buf[count] = '\0';
    
    	printf("\nText is : %s\n", c_buf);
    
    	/* close the file */
    	fclose(fp);
    
    	return 0;
    }



4.写文件函数：fwrite

fwrite.c


    #include <stdio.h>
    int main(void)
    {
    	FILE *fp;
    
    	char *c_buf = "fwrite";
    	
    	/* open the file */
    	fp = fopen("./tst.txt", "r+");
    
    	if (NULL == fp)
    		printf("Failed to open the file !\n");
    	else
    		printf("File is found !\n");
    
    	/* write to file */
    	fwrite(c_buf, 6, 1, fp);
    
    	/* close the file */
    	fclose(fp);
    
    	return 0;
    }


5.定位文件函数：fseek

fseek.c

    #include <stdio.h>
    int main(void)
    {
    	FILE *fp;
    
    	char *c_buf = "fwrite";
    	
    	/* open the file */
    	fp = fopen("./tst.txt", "r+");
    
    	if (NULL == fp)
    		printf("Failed to open the file !\n");
    	else
    		printf("File is found !\n");
    
    	/* repositions the offset */
    	fseek(fp, 0, SEEK_END);
    
    	/* write to file */
    	fwrite(c_buf, 6, 1, fp);
    
    	/* close the file */
    	fclose(fp);
    
    	return 0;
    }
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

### 005）时间函数编程
目标：学习使用Linux时间编程函数

1.获取日历时间（从1970-01-01 00：00：00到现在经历的秒数函数）：time

time.c

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : time.c
     *      Author     : X h y
     *      Version    : 1.0
     *      Date       : 04-29-2015
     *      Description:
     ********************************************************************************
     */
     
    /**
     * \file
     * \brief get time in seconds
     */
    
    #include <stdio.h>
    #include <time.h>
    
    /**
     * \brief main entry
     */
    
    void main(void)
    {
    	time_t ctime;
    
    	/**
    	 *returns  the  time  since  
    	 *the  Epoch (00:00:00 UTC, January 1,
             *1970), measured in seconds.
             */
    	ctime = time(NULL);
    
    	if (-1 == ctime)
    		printf("Get time failed !\n");
    	else
    		printf("CTime is : %d\n", ctime);
    		printf("CTime is : %d\n", t);	
    }
    
    /* end of file */ 



2.将日历时间转换为格林尼治标准时间函数：gmtime

gmtime.c

    #include <stdio.h>
    #include <time.h>
    void main(void)
    {
    	time_t ctime;
    
    	struct tm *tm;
    	
    	ctime = time(NULL);
    
    	printf("CTime is : %d\n", ctime);
    
    	tm = gmtime(&ctime);
    
    	printf("sec : %d\n", tm->tm_sec);
    	printf("min : %d\n", tm->tm_min);
    	printf("hour: %d\n", tm->tm_hour);
    	
    	printf("mday: %d\n", tm->tm_mday);
    	printf("mon : %d\n", tm->tm_mon);
    	printf("year: %d\n", tm->tm_year);
    
    	printf("wday: %d\n", tm->tm_wday);
    	printf("yday: %d\n", tm->tm_yday);
    	printf("dst : %d\n", tm->tm_isdst);
    }


3.将日历时间转换为本地时间（整型输出）函数：localtime

localtime.c

    #include <stdio.h>
    #include <time.h>
    void main(void)
    {
    	time_t ctime;
    
    	struct tm *localtm;
    	
    	ctime = time(NULL);
    
    	printf("CTime is : %d\n", ctime);
    
    	localtm = localtime(&ctime);
    
    	printf("Now is %d:%d:%d\n", localtm->tm_hour,localtm->tm_min,localtm->tm_sec);
    
    }


4.将日历时间转换为本地时间（字符串输出）函数：asctime

asctime.c

    #include <stdio.h>
    #include <time.h>
    void main(void)
    {
    	time_t ctime;
    
    	struct tm *localtm;
    
    	char *stime;
    	
    	ctime = time(NULL);
    
    	printf("CTime is : %d\n", ctime);
    
    	localtm = localtime(&ctime);
    	printf(" %d:%d:%d\n", localtm->tm_hour,localtm->tm_min,localtm->tm_sec);
    	
    	stime = asctime(localtm);
    	printf("%s\n", stime);	
    }


5.获取精确时间（微秒级）函数：gettimeofday

gettimeofday.c

    #include <stdio.h>
    #include <time.h>
    #include <sys/time.h>
    void main(void)
    {
    	struct timeval tv;
    
    	gettimeofday(&tv, NULL);
    	
    	printf("Sec  is : %d\n", tv.tv_sec);
    	printf("Usec is : %d\n", tv.tv_usec);
    }
### 　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　006）多进程程序设计

目标：学习多进程程序设计

1.创建进程
<1>fork:创建一个子进程

fork.c

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : fork.c
     *      Author     : X h y
     *      Version    : 1.0
     *      Date       : 05-02-2015
     *      Description:
     ********************************************************************************
     */
     
    /**
     * \file
     * \brief create a child process
     */
    
    #include <stdio.h>
    #include <stdlib.h>
    #include <unistd.h>
    
    /**
     * \brief main entry
     */      
       
    void main(void)
    {
    	pid_t pid;
    
    	pid = fork(); /* compare with vfork */
    
    	/* parent process */
    	if (pid > 0) {
    		printf(" This is parent process ! \n");
    		exit(0);
    	}
    
    	/* child process */
    	else if (0 == pid) {
    		printf(" This is child process !\n");
    		exit(0);
    	}	
    }
    
    /* end of file */



<2>vfork:创建一个子进程并阻塞父进程

vfork.c

    #include <stdio.h>
    #include <stdlib.h>
    #include <unistd.h>
    #include <sys/types.h>
    void main(void)
    {
    	pid_t pid;
    
    	int count = 0;
    
    	pid = vfork(); /* compare with fork */
    
    	count ++;
    
    	printf("Count is %d\n", count);
    
    	exit(0); /* compare with return 0 */		
    }


2.进程等待：wait

wait.c

    #include <stdio.h>
    #include <stdlib.h>
    #include <unistd.h>
    #include <sys/types.h>
    #include <sys/wait.h>
    void main(void)
    {
    	pid_t pid;
    
    	pid = fork();
    
    	/* parent process */
    	if (pid > 0) {
    		wait(NULL);
    		printf(" This is parent process ! \n");
    		exit(0);
    	}
    
    	/* child process */
    	else if (0 == pid) {
    		printf(" This is child process !\n");
    		exit(0);
    	}	
    }

3.执行程序：execl

execl.c

    #include <stdio.h>
    #include <stdlib.h>
    #include <unistd.h>
    #include <sys/types.h>
    #include <sys/wait.h>
    void main(void)
    {
    	pid_t pid;
    
    	pid = fork();
    
    	/* parent process */
    	if (pid > 0) {
    		wait(NULL);
    		printf(" This is parent process ! \n");
    		exit(0);
    	}
    
    	/* child process */
    	else if (0 == pid) {
    		execl("/bin/ls", "ls", "/home/xhy/", NULL);
    		printf(" This is child process !\n");
    		exit(0);
    	}	
    }


注意：execl()会将该进程下的代码段清空，并替换为需执行的新代码，
所以代码：printf(" This is child process !\n") 不会被执行。

　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

### 007）无名管道通讯编程

目标：学习无名管道通讯编程

注意：无名管道只能用于父进程与子进程间通信；
　　　尾端写入，头端读取；
　　　读取之后，数据将被清空。

1.无名管道函数：pipe

pipe.c

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : pipe.c
     *      Author     : X h y
     *      Version    : 1.0
     *      Date       : 05-02-2015
     *      Description:
     ********************************************************************************
     */
     
    /**
     * \file
     * \brief create pipe
     */
    
    #include <stdio.h>
    #include <stdlib.h>
    #include <unistd.h>
    #include <sys/types.h>
    
    /**
     * \brief main entry
     */      
    
    void main(void)
    {
    	pid_t pid;
    
    	int pipefd[2];
    
    	char c_buf[10];
    
    	int count = 0;
    
    	/* 2.create pipe */
    	pipe(pipefd);
    
    	/* 1.create child process */
    	pid = fork();
    
    	/* parent process */
    	if (pid > 0) {
    		write(pipefd[1], "Xhy Tech", 8);
    		wait(NULL);
    		close(pipefd[1]);
    
    		printf("This is parent process ! \n");
    		exit(0);
    	}
    
    	/* child process */
    	else if (0 == pid) {
    		count = read(pipefd[0], c_buf, 8);
    		c_buf[count] = '\0';
    		printf("Child read is %s\n", c_buf);
    
    		printf("This is child process  ! \n");
    		close(pipefd[0]);
    
    		exit(0);
    	}	
    }
    
    /* end of file */


记住，必须要先创建管道，再创建进程。

　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

### 008）有名管道通讯编程
目标：学习有名管道通讯编程

注意：有名管道可用于任意两个进程间通信

1.创建有名管道函数：mkfifo

<1>先创建fifo文件，在向fifo文件写入数据

fifo_write.c

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : fifo_write.c
     *      Author     : X h y
     *      Version    : 1.0
     *      Date       : 05-03-2015
     *      Description:
     ********************************************************************************
     */
     
    /**
     * \file
     * \brief make a FIFO special file (a named pipe)
     */
    
    #include <stdio.h>
    #include <fcntl.h>
    #include <sys/stat.h>
    #include <sys/types.h>
    #include <unistd.h>
    
    /**
     * \brief main entry
     */      
       
    void main(void)
    {
    	int fd;
    
    	/* create fifo file */
    	mkfifo("./fifo", 0664);
    
    	/* open the fifo file */
    	fd = open("./fifo", O_WRONLY);
    
    	/* write data to fifo file */
    	write(fd, "Xhy Tech", 15);
    
    	close(fd);
    }
    
    /* end of file */



<1>从fifo文件读出数据

fifo_read.c

    #include <stdio.h>
    #include <sys/types.h>
    #include <fcntl.h>
    #include <sys/stat.h>
    #include <unistd.h>
    void main(void)
    {
    	int fd;
    
    	char c_buf[15] = {0};
    
    	/* open the fifo file */
    	fd = open("./fifo", O_RDONLY);
    
    	/* read the data from fifo file */
    	read(fd, c_buf, 15);
    
    	printf("Read : %s\n", c_buf);
    
    	/* close the fifo file */
    	close(fd);
    
    	/* delete fifo file */
    	unlink("./fifo");
    }

　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

### 009）信号通讯编程    
目标：学习信号通讯编程

1.信号处理
<1>signal:信号处理函数

bprocess.c

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : bprocess.c
     *      Author     : X h y
     *      Version    : 1.0
     *      Date       : 05-04-2015
     *      Description:
     ********************************************************************************
     */
     
    /**
     * \file
     * \brief ANSI C signal handling
     */
    
    #include <stdio.h>
    #include <signal.h>
    #include <unistd.h>
      
    void myfunc(int a)
    {
    	printf("Process B received SIGINT\n");
    }
    /**
     * \brief main entry
     */    
    void main(void)
    {
    	signal(SIGINT, myfunc);
    
    	pause();	
    }
    
    /* end of file */


<2>kill:结束进程同时发送SIGINT信号

aprocess.c
    

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : aprocess.c
     *      Author     : X h y
     *      Version    : 1.0
     *      Date       : 05-04-2015
     *      Description:
     ********************************************************************************
     */
    /**
     * \file
     * \brief terminate a process and sent a SIGINT signal
     */
    #include <stdio.h>
    #include <signal.h>
    #include <unistd.h>
    /**
     * \brief main entry
     */      
    void main(int argc, char *argv[])
    {
    	pid_t pid;
    
    	pid = atoi(argv[1]);
    
    	/* terminate the process */
    	kill(pid, SIGINT);	
    }
    /* end of file */
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

### 010）信号量互斥编程
目标：学习信号量互斥编程

1.创建和获取信号量函数：semget

student1.c

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : student1.c
     *      Author     : X h y
     *      Version    : 1.0
     *      Date       : 05-06-2015
     *      Description:
     ********************************************************************************
     */
     
    /**
     * \file
     * \brief  student1 write data to board.txt
     */
     
    #include <stdio.h>
    #include <unistd.h>
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <sys/ipc.h>
    #include <sys/sem.h>
    
    /**
     * \brief main entry
     */      
     
    void main(void)
    {
    	int fd = 0;
    
    	int semid = 0;	
    	int val   = 0;
    	int ret   = 0;
    	struct sembuf sops;
    
    	key_t key;
    
    	/* create keynum */
    	key = ftok("/home", 1);
    
    	/* 1.create signal */
    	semid = semget(key, 1, IPC_CREAT);
    
    	/* 2.set the init value */
    	semctl(semid, 0, SETVAL, 1);
    	val = semctl(semid, 0, GETVAL);
    	printf("student1 init value is %d\n", val);
    	
    	/* 3.open the board */
    	fd = open("./board.txt", O_RDWR|O_APPEND);
    
    	/* 4.get the signal */
    	sops.sem_num = 0;
    	sops.sem_op  = -1;
    	semop(semid, &sops, 1);
    
    	/* 5.check the value */
    	val = semctl(semid, 0, GETVAL);
    	printf("student1 init value is %d\n", val);
    
    	/* 6.write data to board */
    	write(fd, "class math", 10);
    
    	/* 7.sleep 5 minutes */
    	sleep(5);
    
    	write(fd, " is cancel ", 11);
    
    	/* 8.release the signal */
    	sops.sem_num = 0;
    	sops.sem_op  = 1;
    	sops.sem_flg = SEM_UNDO;
    	ret = semop(semid, &sops, 1);
    	printf("semop = %d\n", ret);
    
    	/* 9.check the value */
    	val = semctl(semid, 0, GETVAL);
    	printf("student1 init value is %d\n", val);
    
    	/* 10.close the board */
    	close(fd);	
    }
    
    /* end of file */



student2.c

    #include <stdio.h>
    #include <unistd.h>
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <sys/ipc.h>
    #include <sys/sem.h>
    void main(void)
    {
    	int fd = 0;
    
    	int semid = 0;    	
    	int val   = 0;
    	int ret   = 0;
        struct sembuf sops;    
    	key_t key;
    
    	/* create keynum */
    	key = ftok("/home", 1);
    
    	/* 1.open signal */
    	semid = semget(key, 1, IPC_CREAT);
    
    	/* 2.check the init value */
    	val = semctl(semid, 0, GETVAL);
    	printf("student2 init value is %d\n", val);
    
    	/* 3.open the board */
    	fd = open("./board.txt", O_RDWR|O_APPEND);
    
    	/* 4.get the signal */
    	sops.sem_num = 0;
    	sops.sem_op  = -1;
    	sops.sem_flg = SEM_UNDO;
    	ret = semop(semid, &sops, 1);
    	printf("semop = %d\n", ret);
    
    	/* 5.write data to board */
    	write(fd, "english exam is cancel", 22);
    
    	/* 6.release the signal */
    	sops.sem_num = 0;
    	sops.sem_op  = 1;
    	semop(semid, &sops, 1);
    
    	/* 7.check the value */
    	val = semctl(semid, 0, GETVAL);
    	printf("student2 init value is %d\n", val);
    
    	/* 8.Close the board */
    	close(fd);	
    } 


2.先运行./student1 再运行./student2，观察 semid的值和./board.txt里的内容 

3.注意以root方式运行，因为这是具有超级用户特权的进程
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

### 011）信号量同步编程
目标：学习信号量同步编程

适用条件：多并发进程按一定顺序执行（有严格次序）

用生产者和消费者经典问题来练习信号量同步编程

1.生产者

producer.c

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : producer.c
     *      Author     : X h y
     *      Version    : 1.0
     *      Date       : 05-07-2015
     *      Description:
     ********************************************************************************
     */
     
    /**
     * \file
     * \brief  bring out products
     */
     
    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <sys/ipc.h>
    #include <sys/sem.h>
    #include <fcntl.h>
    #include <unistd.h>
    
    /**
     * \brief main entry
     */      
     
    void main(void)
    {
    	int fd  = 0;
    	    	
    	int semid = 0;    	
    	int val   = 0;
        int ret   = 0;
        struct sembuf sops;   
    	key_t key;
    
    	/* create keynum */
    	key = ftok("/home", 1);
    
    	/* create the signal */
    	semid = semget(key, 1, IPC_CREAT);
    	semctl(semid, 0, SETVAL, 0);
    
    	/* check the value */
    	val = semctl(semid, 0, GETVAL);
    	printf("value is %d\n", val);
    
    	/* create the product */
    	fd = open("./product.txt", O_RDWR|O_CREAT, 0664);
    
    	/* sleep */
    	sleep(5);
    
    	/* fill in the product */
    	write(fd, "The product is finished !", 25);
    
    	/* close the product */
    	close(fd);
    
    	/* release the signal */
    	sops.sem_num = 0;
    	sops.sem_op  = 1;
    	sops.sem_flg = SEM_UNDO;
    	ret = semop(semid, &sops, 1);
    	printf("semop = %d\n", ret);
    
    	/* check the value */
    	val = semctl(semid, 0, GETVAL);
    	printf("value is %d\n", val);	
    }
    
    /* end of file */


 2.消费者

customer.c 

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : customer.c
     *      Author     : X h y
     *      Version    : 1.0
     *      Date       : 05-07-2015
     *      Description:
     ********************************************************************************
     */
     
    /**
     * \file
     * \brief  get product
     */
     
    #include <stdio.h>
    #include <stdlib.h>
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <sys/ipc.h>
    #include <sys/sem.h>
    #include <fcntl.h>
    #include <unistd.h>
    
    /**
     * \brief main entry
     */      
     
    void main(void)
    {
    	int fd  = 0;
    	
    	int semid = 0;
    	int val   = 0;
    	int ret   = 0;
        struct sembuf sops;    	
        key_t key;
    
    	/* create keynum */
    	key = ftok("/home", 1);
    	semid = semget(key, 1, IPC_CREAT);
    	
    	/* get the signal */
    	sops.sem_num = 0;
    	sops.sem_op  = -1;
    	sops.sem_flg = SEM_UNDO;
    	ret = semop(semid, &sops, 1);
    	printf("semop = %d\n", ret);
    
    	/* check the value */
    	val = semctl(semid, 0, GETVAL);
    	printf("value is %d\n", val);
    
    	/* take away the product */
    	system("cp ./product.txt ./ship/");
    }
    
    /* end of file */

   

### 012）共享内存通讯

目标：学习通过共享内存进行进程间通信

1.创建共享内存并通过键盘向共享内存写入字符串

write.c

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : wtite.c
     *      Author     : X h y
     *      Version    : 1.0
     *      Date       : 05-07-2015
     *      Description:
     ********************************************************************************
     */
     
    /**
     * \file
     * \brief allocates a shared memory segment
     */
    
    #include <stdio.h>
    #include <sys/ipc.h>
    #include <sys/shm.h>
    #include <sys/types.h>
    #include <unistd.h>
    #include <stdlib.h>
    #include <string.h>
    	
    #define TEXT_SIZE  2048	
    
    struct shared_use_at
    {
    	int written_by_you;
    	char some_text[TEXT_SIZE];
    };
    
    /**
     * \brief main entry
     */      
     
    int main(void)
    {
    	int shmid;
    	key_t key;
    
    	int running = 1;
    
    	struct shared_use_at *shared_stuff;
    
    	char buffer[TEXT_SIZE];
    
    	/* allocates a shared memory */
    	key = ftok("/home", 1);
    	shmid = shmget(key, 
    		       sizeof(struct shared_use_at), 
    		       IPC_CREAT | 0664);
    
    	if (-1 == shmid) {
    		printf("Create shared memory failed !\n");
    		exit(EXIT_FAILURE);
    	}
    
    	printf("shmid = %d\n", shmid);
    
    	/* attaches the shared memory segment */
    	shared_stuff = (struct shared_use_at *) shmat(shmid, NULL, 0);
    	
    	/* circulation */
    	while (running) {
    		while (1 == shared_stuff-> written_by_you) {
    			sleep(1);
    			printf("Wait the read process !\n");
    		}
    		
    		printf("input string:");
    
    		/* read data from keyboard */
     		fgets(buffer, TEXT_SIZE, stdin);
    
    		/* write the data to shared memory */
    		strncpy(shared_stuff-> some_text, buffer, TEXT_SIZE);
    		shared_stuff-> written_by_you = 1;
    
    		/* check the exit flag */
    		if (0 == strncmp(buffer, "end", 3)) { 
    			running = 0;
    		}
    	}
    
    	/* detaches the shared memory segment */
    	shmdt((const void *)(shared_stuff));
    
    	return 1;
    
    	exit(EXIT_SUCCESS);
    }
    
    /* end of file */



2.通过共享内存读出字符串

read.c

    #include <stdio.h>
    #include <sys/ipc.h>
    #include <sys/types.h>
    #include <sys/shm.h>
    #include <unistd.h>
    #include <stdlib.h>
    #include <string.h>
    	
    #define TEXT_SIZE  2048	
    
    struct shared_use_at
    {
    	int writen_by_you;
    	char some_text[TEXT_SIZE];
    };
    int main(int argc, char *argv[])
    {
    	int shmid;
    	key_t key;
    
    	int running = 1;
    
    	struct shared_use_at *shared_stuff;
    
    	if (argc < 2) {
    		printf("usage: ./read shmid \n");
    		exit(EXIT_FAILURE);
    	}
    			
    	/* find the shared memory's shmid */
    	shmid = atoi(argv[1]);
    
    	/* attaches the shared memory segment */
    	shared_stuff = (struct shared_use_at *) shmat(shmid, NULL, 0);
    	shared_stuff-> writen_by_you = 0;
    
    	printf("Wait the write process !\n");
    	/* circulation */
    	while (running) {
    		while (1 == shared_stuff-> writen_by_you) {
    			/**
    			 * \brief check the exit flag first
    			 *  if (0 == strncmp(buffer, "end", 3))
    			 *  there is is no buffer       
    			 */        
    			if (0 == strncmp(shared_stuff-> some_text, 
    			    "end", 
    			    3)) {
    				running = 0;
    				printf("end process\n" );
    				break;
    			}
    
    			/* printf the string from write process */
    			printf("Write process write : %s", 
    				shared_stuff-> some_text);
    			shared_stuff-> writen_by_you = 0;
    		}		
    	}
    
    	/* detaches the shared memory segment */
    	shmdt((const void *)(shared_stuff));
    
    	/* destroy the shared memory */
    	shmctl(shmid, IPC_RMID, 0);
    
    	return 1;
    
    	exit(EXIT_SUCCESS);
    }
    
    /* end of file */


​    

### 013）消息队列编程
目标：学习使用消息队列进行进程间通信

1.向消息队列发送消息

send.c

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : send.c
     *      Author     : X h y
     *      Version    : 1.0
     *      Date       : 05-10-2015
     *      Description:
     ********************************************************************************
     */
     
    /**
     * \file
     * \brief send messages to
     */
    
    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/ipc.h>
    #include <sys/msg.h>
    #include <string.h>
    #include <stdlib.h>
    
    struct msgt {
    	long msgtype;
    	char msgtext[1024];
    	};
    
    /**
     * \brief main entry
     */      
     
    int main(void)
    {
    	int msgid;
    
    	char str[1024];
    
    	int running = 1;
    
    	key_t key;
    	struct msgt msgs;
    
    	key = ftok("/home", 1);
    
    	/* create message queue */
    	msgid = msgget(key, IPC_CREAT | 0666);
    
    	if (-1 == msgid) {
    		printf("msgget is failed !\n");
    		exit(EXIT_FAILURE);
    	}
    
    	printf("please input the message.\n");	
    	while (running) {
    		printf("msgsnd:");
    		/* input the message */
    		fgets(str, 1024, stdin);
    		
    		msgs.msgtype = 1;
    		strcpy(msgs.msgtext, str);
    
    		/* send the message */
    		msgsnd(msgid, &msgs, sizeof(struct msgt), 0);
    
    		/* check the exit flag */
    		if (0 == strncmp(str, "end", 3)) { 
    			running = 0;
    		}
    	}
    
    	/* delete the message queue */	
    	msgctl(msgid, IPC_RMID, 0);
    
    	return 0;	
    }
    
    /* end of file */


2.从消息队列读出消息

receive.c

    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/ipc.h>
    #include <sys/msg.h>
    #include <string.h>
    #include <stdlib.h>
    
    struct msgt {
    	long msgtype;
    	char msgtext[1024];
    	};
    
    int main(void)
    {	
    	int msgid;
    
    	char msgtext_last[1024];
    
    	int running = 1;
    
    	key_t key;
    	struct msgt msgs;
    
    	key = ftok("/home", 1);
    
    	/* open message queue */
    	msgid = msgget(key, IPC_EXCL);
    
    	if (-1 == msgid) {
    		printf("Don't have message !\n");
    		exit(EXIT_FAILURE);
    	}
    
    	while (running) {
    		/* check the exit flag */
    		if (0 == strncmp(msgs.msgtext, "end", 3)) { 
    			running = 0;
    			break;
    		}
    
    		/* receive message from message queue */
    		msgrcv(msgid, &msgs, sizeof(struct msgt), 4, 0);
    
    		/* 
    		 *  if the message is same 
    		 *  of the last message,
    		 *  do not print
    		 */
    		if (0 == strncmp(msgs.msgtext, 
    				msgtext_last, 
    				sizeof(msgtext_last))) { 
    			
    		} else {
    			strcpy(msgtext_last, msgs.msgtext);
    
    			printf("msgrcv: %s", msgs.msgtext);
    		}	
    	}
    
    	return 0;	
    }

　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

### 014）多线程程序设计

目标：学习多线程程序设计

注意:编译时一定要链接库 -pthread 

1.thread.c

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : thread.c
     *      Author     : X h y
     *      Version    : 1.0
     *      Date       : 05-12-2015
     *      Description:
     ********************************************************************************
     */
     
    /**
     * \file
     * \brief  create a new thread
     */
    
    #include <stdio.h>
    #include <unistd.h>
    #include <stdlib.h>
    #include <pthread.h>
    
    pthread_t thread[2];
    
    pthread_mutex_t mut;
    
    int num = 0;
    
    void *worker1(void *arg)
    {
    	int i = 0;
    
    	printf("I am worker1 !\n");
    
    	for (i=0; i<10; i++) {
    		/* lock the thread */
    		pthread_mutex_lock(&mut);
    
    		num++;
    
    		/* unlock the thread */
    		pthread_mutex_unlock(&mut);
    
    		printf("worker1 number is %d\n", num);
    		sleep(1);
    	}
    
    	pthread_exit(NULL);
    }
    
    void *worker2(void *arg)
    {
    	int i = 0;
    
    	printf("I am worker2 !\n");
    
    	for (i=0; i<10; i++) {
    		/* lock the thread */
    		pthread_mutex_lock(&mut);
    
    		num++;
    
    		/* unlock the thread */
    		pthread_mutex_unlock(&mut);
    
    		printf("worker2 number is %d\n", num);
    		sleep(1);
    	}
    
    	pthread_exit(NULL);
    }
    
    /**
     * \brief main entry
     */      
     
    int main(void)
    {    	
        /* init the  thread lock */
    	pthread_mutex_init(&mut, NULL);
    
    	/* create worker1 thread */
    	pthread_create(&thread[0], NULL, worker1, NULL);
    
    	/* create worker2 thread */
    	pthread_create(&thread[1], NULL, worker2, NULL);
    
    	/* wait worker1 thread  end */
    	pthread_join(thread[0], NULL);
    
    	/* wait worker2 thread  end */
    	pthread_join(thread[1], NULL);
    
    	return 0;
    }
    
    /* end of file */

　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

### 015）多线程同步设计
目标：学习多线程同步程序设计

注意:编译时一定要链接库 -pthread

1.sync.c

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : sync.c
     *      Author     : X h y
     *      Version    : 2.0
     *      Date       : 05-12-2015
     *      Description:
     ********************************************************************************
     */
     
    /**
     * \file
     * \brief  create a new thread
     */
    
    #include <stdio.h>
    #include <pthread.h>
    #include <unistd.h>
    #include <stdlib.h>
    
    pthread_t thread[2];
    pthread_mutex_t mut;
    pthread_cond_t cond_ready = PTHREAD_COND_INITIALIZER;
    
    int num = 0;
    
    void *student_a(void *arg)
    {
        int i = 0;
    
        for (i=0; i<5; i++) {
            /* sweep the floor one time */
            num++;
    
            printf("Times = %d\n", num);
    
            if (num >= 5) {
                printf("A has finished his work !\n");
                /* send signal to a */
                pthread_cond_signal(&cond_ready);
            }
    
            /* sleep 1 seconds*/
            sleep(1);
        }
    
        /* exit */
        pthread_exit(NULL);
    }
    
    void *student_b(void *arg)
    {  
        pthread_mutex_lock(&mut);
    
        if (num < 5) {
            pthread_cond_wait(&cond_ready, &mut);
        }
        
        /* mop the floor */
        num = 0;
    
        pthread_mutex_unlock(&mut);
        printf("B has finished his work !\n");
    
        /* exit */
        pthread_exit(NULL);
    }
    
    /**
     * \brief main entry
     */      
     
    int main(void)
    {
        /* init the  thread lock*/
        pthread_mutex_init(&mut, NULL);
    
        /* create A thread */
        pthread_create(&thread[0], NULL, student_a, NULL);
        
        /* create B thread */
        pthread_create(&thread[1], NULL, student_b, NULL);
    
        /* wait A thread end */
        pthread_join(thread[0], NULL);
    
        /* wait B thread end */ 
        pthread_join(thread[1], NULL);
    
        return 0;
    }
    
    /* end of file */

　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

### 016）TCP通讯程序设计
目标：学习TCP通讯程序设计

1.服务器程序设计：

tcp_server.c

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : tcp_server.c
     *      Author     : X h y
     *      Version    : 2.0
     *      Date       : 05-13-2015
     *      Description:
     ********************************************************************************
     */
     
    /**
     * \file
     * \brief  tcp server
     */
    
    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <string.h>
    #include <unistd.h>
    #include <stdlib.h>
    
    #define PORTNUM 1234
    #define MSG_SIZE 128
    
    /**
     * \brief main entry
     */      
     
    int main(void)
    {
        int sockfd;
        int newfd;
    
        struct sockaddr_in server_addr;
        struct sockaddr_in client_addr;
    
        char buffer[MSG_SIZE];
        int  nbyte;
        int  addrlen;
    
        char running = 1;
    
        /* create socket */
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
    
        if (-1 == sockfd) {
            printf("Create socket error !\n");
            exit(1);
        }
    
        /* init address */
        bzero(&server_addr, sizeof(struct sockaddr_in));
        server_addr.sin_family      = AF_INET;
        server_addr.sin_port        = htons(PORTNUM);
        server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    
        /* bind address */
        bind(sockfd, 
             (struct sockaddr *)(&server_addr), 
             sizeof(struct sockaddr));
    
        /* listen port */
        listen(sockfd, 5);
    
        /* wait connection request */
        printf("Wait client connection request !\n");
        addrlen = sizeof(struct sockaddr);
        newfd   = accept(sockfd, 
                         (struct sockaddr *)(&client_addr), 
                         &addrlen);
        
        if (-1 == newfd) {
            printf("Accept socket error !\n");
            exit(1);
        }
        
        printf("Server get connection from %s\n", 
               inet_ntoa(client_addr.sin_addr));
        while (running) {
            /* receive data */
            nbyte = recv(newfd, buffer, MSG_SIZE, 0);
            buffer[nbyte] = '\0';
    
            if (0 == strncmp(buffer, "end", 3)) {
                running = 0;
                printf("Server stop !\n");
            } else {
                printf("Server received : %s\n", buffer);
                bzero(buffer, MSG_SIZE);
            }      
        }
        
        /* close the connection */
        close(newfd);
        
        /* close the socket */
        close(sockfd);
    
        return 0;
    }
    
    /* end of file */


2.客户端程序设计：

tcp_client.c

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : tcp_client.c
     *      Author     : X h y
     *      Version    : 2.0
     *      Date       : 05-13-2015
     *      Description:
     ********************************************************************************
     */
     
    /**
     * \file
     * \brief  tcp client
     */
    
    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <string.h>
    #include <unistd.h>
    #include <stdlib.h>
    
    #define PORTNUM 1234
    #define MSG_SIZE 128
    
    /**
     * \brief main entry
     */      
     
    int main(int argc, char *argv[])
    {
        if (2 != argc) {
            printf("Usage: %s server_ip\n", argv[0]);
            exit(1);
        }
        
        int sockfd;
    
        struct sockaddr_in server_addr;
    
        char buffer[MSG_SIZE];
    
        char running = 1;
        
        /* create socket */
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
    
        if (-1 == sockfd) {
            printf("Create socket error !\n");
            exit(1);
        }
            /* init server address */
            bzero(&server_addr, sizeof(struct sockaddr_in));
            server_addr.sin_family      = AF_INET;
            server_addr.sin_port        = htons(PORTNUM);
            server_addr.sin_addr.s_addr = inet_addr(argv[1]);
        

        /* connect the server */
        if (-1 == connect(sockfd, 
                          (struct sockaddr *)(&server_addr), 
                          sizeof(struct sockaddr))) {
            printf("Connect server error !\n");
            exit(1);
        }
    
        while (running) {
            /* send data to server */
            printf("\nPlease input string: ");
            fgets(buffer, 128, stdin);
            send(sockfd, buffer, strlen(buffer), 0);
    
            if (0 == strncmp(buffer, "end", 3)) {
                running = 0;
                printf("\nClient stop !\n");
            } else {
                bzero(buffer, MSG_SIZE);
            }   
        }
     
        /* close the socket */
        close(sockfd);
    
        return 0;
    }
    
    /* end of file */

　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

### 017）UDP通讯程序设计
目标：学习UDP通讯程序设计

1.服务器程序设计：

udp_server.c

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : udp_server.c
     *      Author     : X h y
     *      Version    : 2.0
     *      Date       : 05-14-2015
     *      Description:
     ********************************************************************************
     */
     
    /**
     * \file
     * \brief  udp client
     */
    
    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <string.h>
    #include <unistd.h>
    #include <stdlib.h>
    
    #define PORTNUM 1234
    #define MSG_SIZE 128
    
    /**
     * \brief main entry
     */      
     
    int main(void)
    {   
        int sockfd;
        
        struct sockaddr_in server_addr;
        struct sockaddr_in client_addr;
    
        char buffer[MSG_SIZE];
        int  addr_len;
        int  nbyte;
    
        char running = 1;
    
        /* create socket */
        sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    
        if (-1 == sockfd) {
            printf("Create socket error !\n");
            exit(1);
        }
    
        /* init address */
        bzero(&server_addr, sizeof(struct sockaddr_in));
        server_addr.sin_family      = AF_INET;
        server_addr.sin_port        = htons(PORTNUM);
        server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    
        /* bind address */
        bind(sockfd, 
             (struct sockaddr *)(&server_addr), 
             sizeof(struct sockaddr));
    
        printf("Wait client connection request !\n");
    
        /* receive data */
        while (running) {
            addr_len = sizeof(struct sockaddr);
            bzero(buffer, sizeof(buffer));
            nbyte = recvfrom(sockfd, 
                             buffer, 
                             MSG_SIZE, 
                             0, 
                             (struct sockaddr *)(&client_addr),
                             &addr_len);
            buffer[nbyte] = '\0';
    
            if (0 == strncmp(buffer, "end", 3)) { 
                running = 0;
                printf("Server stop !\n");
            } else {
                printf("Server received : %s\n", buffer);
            }
        }
    
        /* end the socket */
        close(sockfd);
    
        return 0;
    }
    
    /* end of file */


2.客户端程序设计：

udp_client.c

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : udp_client.c
     *      Author     : X h y
     *      Version    : 2.0
     *      Date       : 05-14-2015
     *      Description:
     ********************************************************************************
     */
     
    /**
     * \file
     * \brief  tcp client
     */
    
    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <string.h>
    #include <unistd.h>
    #include <stdlib.h>
    
    #define PORTNUM 1234
    #define MSG_SIZE 128
    
    /**
     * \brief main entry
     */      
     
    int main(int argc, char *argv[])
    {
        if (2 != argc) {
            printf("Usage: %s server_ip\n", argv[0]);
            exit(1);
        }
    
        int sockfd;
        
        struct sockaddr_in server_addr;
        struct sockaddr_in client_addr;
    
        char buffer[MSG_SIZE];
    
        char running = 1;
    
        /* create socket */
        sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    
        if (-1 == sockfd) {
            printf("Create socket error !\n");
            exit(1);
        }
    
        /* init server address */
        bzero(&server_addr, sizeof(struct sockaddr_in));
        server_addr.sin_family = AF_INET;
        server_addr.sin_port   = htons(PORTNUM);
        inet_aton(argv[1], &server_addr.sin_addr);
    
        /* send data */
        while (running)
        {
            printf("\nMessage:");
            fgets(buffer, MSG_SIZE, stdin);
            sendto(sockfd, 
                   buffer, 
                   strlen(buffer), 
                   0, 
                   (struct sockaddr *)(&server_addr),
                   sizeof(struct sockaddr));
    
            /* check the exit flag */
            if(0 == strncmp(buffer, "end", 3)) { 
                running = 0;
                printf("Client stop !\n");
            } else {
                bzero(buffer, MSG_SIZE);
            }   
        }
    
        /* end the socket */
        close(sockfd);
    
        return 0;
    }
    
    /* end of file */


　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

### 018）网络并发服务器设计
目标：学习网络并发服务器程序设计

1.服务器程序设计：

tcp_server.c

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : tcp_server.c
     *      Author     : X h y
     *      Version    : 2.0
     *      Date       : 05-14-2015
     *      Description:
     ********************************************************************************
     */
     
    /**
     * \file
     * \brief  tcp server
     */
    
    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <string.h>
    #include <unistd.h>
    #include <stdlib.h>
    
    #define PORTNUM 1234
    #define MSG_SIZE 128
    
    /**
     * \brief main entry
     */      
     
    int main(void)
    {
        int sockfd;
        int newfd;
    
        pid_t pid;
    
        struct sockaddr_in server_addr;
        struct sockaddr_in client_addr;
    
        char buffer[MSG_SIZE];
        int  nbyte;
        int  addrlen;
    
        char running = 1;
        char flag   = 1;
    
        /* create socket */
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
    
        if (-1 == sockfd) {
            printf("Create socket error !\n");
            exit(1);
        }
    
        /* init address */
        bzero(&server_addr, sizeof(struct sockaddr_in));
        server_addr.sin_family      = AF_INET;
        server_addr.sin_port        = htons(PORTNUM);
        server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    
        /* bind address */
        bind(sockfd, 
             (struct sockaddr *)(&server_addr), 
             sizeof(struct sockaddr));
    
        /* listen port */
        listen(sockfd, 5);
    
        while (running) {
            /* wait connection request */
            printf("Wait client connection request !\n");
            addrlen = sizeof(struct sockaddr);
            newfd   = accept(sockfd, 
                             (struct sockaddr *)(&client_addr), 
                             &addrlen);
            
            if (-1 == newfd) {
                printf("Accept socket error !\n");
                exit(1);
            }
        
            printf("Server get connection from %s\n", 
                   inet_ntoa(client_addr.sin_addr));
    
            /* creat child */
            if (0 == (pid = fork())) {
                flag = 1;
    
                while (flag) {
                /* receive data */
                nbyte = recv(newfd, buffer, MSG_SIZE, 0);
                buffer[nbyte] = '\0';
    
                if (0 == strncmp(buffer, "end", 3)) {
                    flag = 0;
                    printf("Disconnect with client !\n");
    
                    /* close the connection */
                    close(newfd);
                } else {
                    printf("Server received : %s\n", buffer);
                    bzero(buffer, MSG_SIZE);
                }  
                }    
            } else if (pid < 0) {
                printf("Fork error !\n");
            }
        }
        
        /* close the socket */
        close(sockfd);
    
        return 0;
    }
    
    /* end of file */



2.客户端程序设计：

tcp_client.c

    /*
     ********************************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : tcp_client.c
     *      Author     : X h y
     *      Version    : 2.0
     *      Date       : 05-14-2015
     *      Description:
     ********************************************************************************
     */
     
    /**
     * \file
     * \brief  tcp client
     */
    
    #include <stdio.h>
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <string.h>
    #include <unistd.h>
    #include <stdlib.h>
    
    #define PORTNUM 1234
    #define MSG_SIZE 128
    
    /**
     * \brief main entry
     */      
     
    int main(int argc, char *argv[])
    {
        if (2 != argc) {
            printf("Usage: %s server_ip\n", argv[0]);
            exit(1);
        }
        
        int sockfd;
    
        struct sockaddr_in server_addr;
    
        char buffer[MSG_SIZE];
    
        char running = 1;
        
        /* create socket */
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
    
        if (-1 == sockfd) {
            printf("Create socket error !\n");
            exit(1);
        }
            /* init server address */
            bzero(&server_addr, sizeof(struct sockaddr_in));
            server_addr.sin_family      = AF_INET;
            server_addr.sin_port        = htons(PORTNUM);
            server_addr.sin_addr.s_addr = inet_addr(argv[1]);
        

        /* connect the server */
        if (-1 == connect(sockfd, 
                          (struct sockaddr *)(&server_addr), 
                          sizeof(struct sockaddr))) {
            printf("Connect server error !\n");
            exit(1);
        }
    
        while (running) {
            /* send data to server */
            printf("\nPlease input string: ");
            fgets(buffer, 128, stdin);
            send(sockfd, buffer, strlen(buffer), 0);
    
            if (0 == strncmp(buffer, "end", 3)) {
                running = 0;
                printf("\nClient stop !\n");
            } else {
                bzero(buffer, MSG_SIZE);
            }   
        }
     
        /* close the socket */
        close(sockfd);
    
        return 0;
    }
    
    /* end of file */

　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

### 019）守护进程设计  
目标：学习守护进程（daemon）程序设计

1. 守护进程程序设计：

daemon.c
   
   ```
    /*
    ********************************************************************************
    *      Copyright (C), 2015-2115, Xhy Tech. Stu.
    *      FileName   : daemon.c
    *      Author     : X h y
    *      Version    : 2.0
    *      Date       : 05-14-2015
    *
    *      Description:
    ********************************************************************************
    */
    
    /**
    * \file
    * \brief  daemon process
    */
    
    #include <stdio.h>
    #include <unistd.h>
    #include <stdlib.h>
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <string.h>
    
    #define MAXFILE 65535
    
    /**
    * \brief main entry
    */ 
    
    int main(void)
    {
        pid_t pid;
        
        int fd;
        int i;
    
        char flag = 1;
        
        char *buf = "I am daemon !\n";
    
        /* create child process */
        pid = fork();
    
        if (pid < 0) {
            printf("Create child process error !\n");
            exit(1);
        } else if (pid > 0) {
            exit(0);
        } 
    
        /* get away from the teminal */
        setsid();
    
        /* change work directory */
        chdir("/");
    
        /* clear mask */
        umask(0);
    
        /* close file id */
        for (i=0; i<MAXFILE; i++) {
            close(i);
        }
    
        while (1) {
            fd = open("/tmp/daemon.log", 
                      O_CREAT|O_WRONLY|O_APPEND, 
                      0664);
    
            if ((1 == flag) && (fd < 0)) {
                printf("Open file error !\n");
                flag = 0;
                exit(0);
            }
    
            write(fd, buf, strlen(buf));
            close(fd);
            sleep(1);
        }
    
        return 0;
    }
    
    /* end of file */
   ```

　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

### 019）Shell脚本高级编程
目标：学习Shell脚本编程

1. 概念

>脚本是一个包含一系列命令序列的文本文件。
>目前linux上应用最广泛的是bash，它也是事实上
>的shell规范，这里所有的论述都是参照bash的规范。



2.1 基本结构

    #!/bin/bash                                      -----指明脚本使用解释工具
    
    # Filename   : shell.sh                          -----文件头，用#注释
    # Author     : X h y
    # Version    : 1.0
    # Date       : 07-30-2015
    
    ehco "by X h y"                                  -----打印信息
    ……                                               -----命令主体
    
    exit(0)                                          -----退出返回



2.2 变量

    echo '#2.2'
    a="Hello Xhy"
    b=5
    
    echo "a is $a"
    echo "b is $b"
    echo -ne "\n"


注意：变量两边没有空格，使用变量是用$符。

2.3 参数

    echo '#2.3'
    echo '$# is :' $#                                 ---传入脚本命令行参数个数
    echo '$* is :' $*                                 ---命令行参数值
    echo '$0 is :' $0                                 ---shell文件名
    echo '$1 is :' $1                                 ---第1个参数
    echo '$2 is :' $2                                 ---第2个参数
    echo -ne "\n"


2.4 计算

    echo '#2.4'
    var1=8
    var2=4
    var3=`expr $var1 / $var2`
    var4=`expr $var1 - $var2`
    
    echo $var3
    echo $var4
    echo -ne "\n"


注意：计算用expr，表达式用反引号(``)括起来赋值给其他变量。

2.5 流程控制

 1)if语句
    

    echo '#2.5.1'
    var=10
    if [ $1 -gt $var ]
    then
        echo 'the $1 is greater than 10'
    else 
        echo 'the $1 is less than 10'
    fi 
    echo -ne "\n"

 

2)for语句
    

    echo '#2.5.2'
    list="Sun Mon Tue Wed Thur Fri Sat"
    for day in $list
    do
        echo $day
    done
    echo -ne "\n"

 

3)while语句

    echo '#2.5.3'
    var=$2
    while [ $var -gt 0 ]
    do
        echo $var
        var=`expr $var - 1`
    done
    echo -ne "\n"


注意：条件用[]号括起来，[]两边都必须有空格，另外“＝”两边也有。

附：条件比较

* 　-eq　等于
* 　-ne　不等于
* 　-gt　大于
* 　-lt　小于
* 　-le　小于等于
* 　-ge　大于等于
* 　-z 　空串
* 　-n 　非空
* 　=  　两个字符相等
* 　!= 　两个字符不等



2.6 sed工具使用

    echo '#2.6'
    sed -n '3p' tmp.txt
    sed -n '1,3p' tmp.txt
    
    sed '3d' tmp.txt
    sed '1,3d' tmp.txt
    
    sed -n '/root/p' tmp.txt
    
    sed '1c Hi' tmp.txt
>注意：之前的操作对文件本身并没有影响，只是输出有影响。
>但加上 -i选项后就会对文件有实质影响。
>如：sed -i '$a bye' tmp.txt 
>附：
>－ｎ：指定处理后只显示该行
>－ｅ：进行多项编辑任务
>－ｉ：直接修改读取的文件内容，而不是由屏幕输出



2.7 awk工具使用

    echo '#2.7'
    #last -n 5
    last -n 5
    echo -ne "\n"
    
    #last -n 5  |awk '{print $1}'
    last -n 5 | awk '{print $1}'
    echo -ne "\n"
    
    #last -n 5 | awk '{print $2}'
    last -n 5 | awk '{print $2}'
    echo -ne "\n"
    
    #cat /etc/passwd | awk -F ':' '{print $1}'
    cat /etc/passwd | awk -F ':' '{print $1}'
    echo -ne "\n"
    
    #awk -F ':' '$1=="root" {print $0}' /etc/passwd
    awk -F ':' '$1=="root" {print $0}' /etc/passwd
    echo -ne "\n"

　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　**完结**
