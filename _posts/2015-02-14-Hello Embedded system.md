---

layout: post
title: "Hello Embedded system"
author: "Xhy"
categories: Linux
tags: [Embedded]
image: harrison-broadbent-hSHNPyND_dU.jpg
---

Photo by harrison-broadbent-hSHNPyND

> 编程环境：Red Hat Enterprise Linux 6
>
> 内核版本：2.6.32-279 
>
> 实验平台：Tiny6410

<br />



## Table of Contents

* [Samsung S3C6410裸机学习][1]
  * 001）核心初始化
  * 002）初始化外设基地址 & 关闭看门狗和中断
  * 003）关闭MMU和I/D caches
  * 004）点亮LED作测试
* [Linux内核驱动学习][2]
  * 001）U-Boot入门
  * 002）嵌入式Linux内核制作
  * 003）嵌入式Linux文件系统制作
  * 004）内核模块的开发
  * 005）Linux内核子系统
  * 006）字符设备驱动模型
  * 007）LED 驱动程序设计
  * 008）按键驱动程序设计
  * 009）优化按键驱动程序

[1]:	#1
[2]:	#2



<br />



<h3 id="1"> 1. Samsung S3C6410裸机学习</h3>



#### 001）核心初始化

    /*
    ******************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.   
     *      FileName   :   X-Boot
     *      Author     :   X h y       
     *      Version    :   1.0        
     *      Date       :   02-13-2015  
     *      Description:   BootLoader for Tiny6410   
    ******************************************************************
     */
    
    /*
     *****************************************************************
     *      Function   :    Jump vector table(1.1.1) 
     *      Date       :    02-13-2015 
     *****************************************************************
     */
    
    .text
    .global _start
    _start:
    	b	   reset                           
    	ldr	   pc, _undefined_instruction      
    	ldr	   pc, _software_interrupt         
    	ldr	   pc, _prefetch_abort             
    	ldr	   pc, _data_abort                
    	ldr	   pc, _not_used                   @ 0x0000 0014(not_used)
    	ldr	   pc, _irq                        
    	ldr	   pc, _fiq                        
    
    _undefined_instruction:		
    	.word undefined_instruction
    
    _software_interrupt:    	
    	.word software_interrupt
    
    _prefetch_abort:        	
    	.word prefetch_abort
    
    _data_abort:            	
    	.word data_abort
    
    _not_used:             	      
    	.word not_used
    
    _irq:                         
    	.word irq
    
    _fiq:                         
    	.word fiq
       
    undefined_instruction:
    	nop
    
    software_interrupt:
    	nop
    
    prefetch_abort:
    	nop
    
    data_abort:
    	nop
    
    not_used:
    	nop
    
    irq:
    	nop
    	
    fiq:
    	nop


​    
    reset:
    	bl     set_svc               @ Set the cpu to SVC32 mode
    
    /*
     *****************************************************************
     *      Function   :    Set the cpu to SVC32 mode(1.1.2)
     *      Date       :    02-13-2015
     *****************************************************************
     */
    
    set_svc:
    	mrs     r0, cpsr
    	bic     r0, r0, #0x1f 
    	orr     r0, r0, #0xd3        @ 0b1101 0011   (7~0)
    	msr     cpsr,r0              @ I F T M[4:0] Disable IRQ,FIQ | SVC32 mode
    
    	mov     pc, lr
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

#### 002）初始化外设基地址 & 关闭看门狗和中断

       reset:
    	bl	    set_svc               @ Set the cpu to SVC32 mode(1.1.2)
    	bl	    init_peripheral_port  @ Initialize the peripheral port(1.1.3)
    	bl	    disable_watchdog      @ Disable Watchdog(1.1.4)
    	bl	    disable_interrupt     @ Disable all interrupts(1.1.5)
    /*
     *****************************************************************
     *       Function  :    Set peripheral port(1.1.3)  
     *       Date      :    02-14-2015 
     *****************************************************************
     */
    
    set_peripheral_port:
    	ldr	    r0, =0x70000000
    	orr	    r0, r0,#0x13          @[4:0] 0b10011  ~256M
    	mcr	    p15,0,r0,c15,c2,4
    	mov	    pc, lr	
    	
    /*
     *****************************************************************
     *      Function   :    Disabled Watchdog(1.1.4) 
     *      Date       :    02-14-2015
     *****************************************************************
     */
    
    #define pWTCON 0x7e004000        
    disable_watchdog:
    	ldr 	r0, =pWTCON
    	mov 	r1, #0              
    	str 	r1, [r0]             @ WTCON =  0
    
    	mov	    pc, lr
    	
    /*
     *****************************************************************
     *      Function   :    Disable all interrupts(1.1.5)      
     *      Date       :    02-15-2015 
     *****************************************************************
     */
    
    disable_interrupt:
    	mvn 	r1, #0x0             @ 1 = interrupt disabled
    	ldr 	r0, =0x71200014      @ VIC0INCLEAR  0X7120_0014
    	str 	r1, [r0]
    
    	ldr 	r0, =0x71300014      @ VIC1INCLEAR  0X7130_0014
    	str 	r1, [r0]
    	
    	mov	    pc, lr　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

#### 　　　　　　　　

#### 003）关闭MMU和I/D caches

    reset:
    	bl	    set_svc               @ Set the cpu to SVC32 mode(1.1.2)
    	bl	    set_peripheral_port   @ Set peripheral port(1.1.3)
    	bl	    disable_watchdog      @ Disable Watchdog(1.1.4)
    	bl	    disable_interrupt     @ Disable all interrupts(1.1.5)
    	bl	    disable_mmu_caches    @ Disable mmu and caches(1.1.6)
    /*
     *****************************************************************
     *       Function  :    Disabled MMU and I/D caches(1.1.6)     
     *       Date      :    02-15-2015 
     *****************************************************************
     */
    
    disable_mmu_caches:
    	mcr 	p15, 0,r0,c7,c7,0
    	mrc 	p15, 0,r0,c1,c0,0
    	bic 	r0, r0,#0x00000007
    	mcr 	p15, 0,r0,c1,c0,0
    	mov 	pc, lr

#### 　　　　　　　　　　　　　　　　　　　　　　　　　　

#### 004）点亮LED作测试

    reset:
    	bl	    set_svc               @ Set the cpu to SVC32 mode(1.1.2)
    	bl	    set_peripheral_port   @ Set peripheral port(1.1.3)
    	bl	    disable_watchdog      @ Disable Watchdog(1.1.4)
    	bl	    disable_interrupt     @ Disable all interrupts(1.1.5)
    	bl	    disable_mmu_caches    @ Disable mmu and caches(1.1.6)
    	bl	    light_led             @ Light the LEDS(For test 1.2)
    /*
     *****************************************************************
     *       Function  :    Light the LEDS(For test 1.2)    
     *       Date      :    02-16-2015 
     *****************************************************************
     */
    
    #define	GPKCON 0x7f008800
    #define	GPKDAT 0x7f008808
    light_led:
    	ldr	   r0, = GPKCON
    	ldr	   r1, = 0x11110000
    	str	   r1, [r0]
    
    	ldr	   r0, = GPKDAT
    	ldr	   r1, = 0xa0
    	str	   r1, [r0]
    	mov	   pc, lr
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

  **待更新** ……

---





<h3 id="2"> 2. Linux内核驱动学习</h3>

* 001）U-Boot入门
* 002）嵌入式Linux内核制作
* 003）嵌入式Linux文件系统制作
* 004）内核模块的开发
* 005）Linux内核子系统
* 006）字符设备驱动模型
* 007）LED 驱动程序设计
* 008）按键驱动程序设计
* 009）优化按键驱动程序
  
---
#### 001）U-Boot入门
#### 1.linux系统组成

Bootloader、Kernel、Root filesystem

其中U-Boot（Universal Boot Loader）是一种普遍用于嵌入式系统中的Bootloader

#### 2.uboot编译

1)  解压uboot，再make distclean

2)  make tiny6410_config 

3)  make ARCH=arm CROSS_COMPILE=arm-linux-

4)  将当前目录下的uboot.bin下载到开发板

#### 3.uboot常用命令

1)  printenv

2)  setenv 

3)  saveenv

4)  nand erase 400000 500000

5)  nand write 51000000 400000 500000

6)  nand read 51000000 400000 500000

7)  tftp 51000000 uImage

8)  setenv bootcmd nand read 51000000 400000 500000 \; bootm 51000000(斜杠前后都要有空格)

9)  setenv bootcmd tftp 51000000 uImage \; bootm 51000000

---
### 002）嵌入式Linux内核制作
#### 1.linux系统架构

为了保护操作系统，linux系统由用户空间和内核空间构成，用户空间包含用户程序和C库（可视），内核空间包含系统调用接口、狭义内核和体系机构相关代码等（不可视）。（可通过系统调用和硬件中断进行切用户空间和内核空间的转移）

#### 2.linux内核架构

1)  System Call Interface(SCI)

2)  Process Management(PM)

3)  Virtual File System(VFS)

4)  Memory Management(MM)

5)  Network Stack

6)  Arch

7)  Device Drivers(DD)

#### 3.linux内核下载

[www.kernel.org][kernel]

[kernel]: https://www.kernel.org
#### 4.linux内核配置与编译

1)  linux内核配置

① make distclean

② make menuconfig(*为编译进内核，M为模块，动态添加进内存)

③ 参考/boot/config-2.6-32配置文件，在此基础上改进进行配置  

2)  linux内核编译

① make bzImage(编译内核)，另外make zImage用于编译小于512k的内核

② make modules(编译内核模块)

③ make modules_install(将编译好的内核复制到/lib/modules/目录)

④ mkinitrd rd-2.6.39 2.6.39(将编译好的内核模块打包成文件)

⑤ cp arch/x86/boot/bzImage /boot/vmlinux-2.6.39

⑥ cp rd-2.6.39 /boot/

⑦ vim /etc/grub.conf（修改启动项）

⑧ reboot

⑨ make clean vs make distclean （清理）

#### 5.嵌入式linux内核配置与编译

1)  嵌入式linux内核配置

① make distclean

② make menuconfig ARCH=arm（配置内核）

③ 选择相应配置文件载入，改进，保存  

2)  嵌入式linux内核编译

① make uImage ARCH=arm CROSS_COMPILE=arm-linux-(编译内核)

② 若提示"mkimage" command not found，则cp /uboot/tools/mkimage /bin/

③ 再编译，下到开发板就可以启动了（注意没有编译根文件系统进去，启动到一半出错）

### 003）嵌入式Linux文件系统制作

#### 1.制作根文件系统

1)  创建目录

① mkdir rootfs

② mkdir bin dev etc lib proc sbin sys usr mnt tmp var

③ mkdir usr/bin usr/lib usr/sbin lib/modules  

2)  创建设备文件（进入/dev/目录）

① mknod -m 666 console c 5 1

② mknod -m 666 null c 1 2  

3)  加入配置文件(fstab , init.d , inittab , profile) 

① cp .../etc/* ./rootfs/etc  

4)  添加内核模块

① mkdir modules ARCH=arm CROSS_COMPILE=arm-linux-

② mkdir modules_install ARCH=arm INSTALL_MOD_PATH=/home/xhy/rootfs/ 

5)  编译/安装busybox

① make menuconfig（配置busybox）

② make（编译）

③ make install（安装）

#### 2.挂载根文件系统到内核

1)  挂载方式（根据存储设备的硬件特性和系统需求）

① 基于NandFlash的文件系统：yaff2、UbiFS（可读可写）

② 基于NorFlash的文件系统：Jffs2（可读可写）

③ 基于内存的文件系统：Ramdisk（先划分内存大小）--> Initramfs（不划分，按需分配）

④ 基于网络的文件系统：NFS（开发阶段）  

2)  使用Initramfs

① cd .../rootfs/

② ln -s ./bin/busybox init

③ make menuconfig ARCH=arm（配置linux内核，支持initramfs）

④ make uImage ARCH=arm CROSS_COMPILE=arm-linux-（重新编译）

⑤ setenv bootargs noinitrd console=ttySAC0,115200

⑥ saveenv

⑦ tftp下载内核，重新启动

⑧ vim /etc/grub.conf（修改启动项）

⑨ reboot

3)  使用NFS（产品开发阶段）

① make menuconfig ARCH=arm（配置linux内核，支持NFS）

② make uImage ARCH=arm CROSS_COMPILE=arm-linux-（重新编译）

③ setenv bootargs noinitrd console=ttySAC0,115200 init=/init root=/dev/nfs 

　rw nfsroot=192.168.1.130:/home/xhy/rootfs,proto=tcp,nfsvers=3 

　ip=192.168.1.230:192.168.1.130:192.168.1.1:255.255.255.0::eth0:off

④ saveenv

⑤ tftp下载内核，重新启动

### 004）内核模块的开发

### 005）Linux内核子系统
### 006）字符设备驱动模型
### 007）LED 驱动程序设计
目标：设计一个点亮Tiny6410LED的驱动

1)代码准备：
led.c

    /*
     *****************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu. 
     *      FileName   : led.c
     *      Author     : X h y
     *      Version    : 1.0 
     *      Date       : 06-01-2015
     *      Description: for Tiny6410    
     *****************************************************************
     */
    
    #include <linux/module.h> 
    #include <linux/init.h>
    #include <linux/cdev.h>
    #include <linux/fs.h>
    #include <linux/io.h>
    #include "led.h"
    
    unsigned int *led_config;
    unsigned int *led_data;
    
    struct cdev cdev;
    
    dev_t devno;
    
    static int led_open(struct inode *node, struct file *filp)
    {
    	/* mapped the virtual address for GPKCON */
    	led_config = ioremap(GPKCON, 4);
    	writel(0x11110000, led_config);
    
    	led_data = ioremap(GPKDAT, 4);
    
    	return 0;
    }
    
    static long led_ioctl(struct file *filep, unsigned int cmd, unsigned long arg)
    {
        switch (cmd) {
            case LED_ON:
                    writel(0xa0, led_data);
                    return 0;
    
            case LED_OFF:
                    writel(0xff, led_data);
                    return 0;
    
            default:
    
                    return -EINVAL;
        }
    }
    
    static const struct file_operations led_fops =
    {
    	.open = led_open,
    	.unlocked_ioctl = led_ioctl,
    };
    
    static int led_init(void)
    {
    	/* 1. creat the cdev */
    	alloc_chrdev_region(&devno, 0, 1, "myled");
    
    	/* 2. init the cdev */
    	cdev_init(&cdev, &led_fops);
    
    	/* 3. register the cdev */
    	cdev_add(&cdev, devno, 1);
    
    	return 0;
    }
    
    static void led_exit(void)
    {
    	/* 4. delete the cdev */
    	cdev_del(&cdev);
    
    	/* 5. unregister the cdev */
    	unregister_chrdev_region(devno, 1);
    }
    
    module_init(led_init);
    module_exit(led_exit);
    
    MODULE_LICENSE("GPL");
    MODULE_AUTHOR("Xhy Tech. Stu");
    MODULE_DESCRIPTION("led for test");
    MODULE_VERSION("1.0");
led.h

    /* registers for led */
    #define	GPKCON 0x7f008800
    #define	GPKDAT 0x7f008808
    
    /* command for device control */
    #define LED_MAGIC 'l'
    #define LED_ON  _IO(LED_MAGIC, 0)
    #define LED_OFF _IO(LED_MAGIC, 1)

Makefile
    
    obj-m := led.o
    KDIR := /home/xhy/Embedded/s4/last_term/part3/lesson3/linux-tiny6410    
    all:
    	make -C $(KDIR) M=$(PWD) modules CROSS_COMPILE=arm-linux- ARCH=arm
    clean:
    	rm -rf .* *.o *.ko *.mod.o *.mod.c *.symvers *.bak *.order

app_led.c

    /*
     *****************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : app_led.c
     *      Author     : X h y
     *      Version    : 1.0   
     *      Date       : 06-01-2015
     *      Description: for Tiny6410    
     *****************************************************************
     */
    
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <sys/ioctl.h>
    #include <stdlib.h>
    #include "led.h"
    
    int main(int argc, char const *argv[])
    {
    	int fd = 0;
    	int cmd = 0;
    
    	if (argc < 2) {
    		printf("please enter the second parameter !\n");
    		return 0;
    	}
    
    	/* alphanumeric to integer */
    	cmd = atoi(argv[1]);
    
     	fd = open("/dev/myled", O_RDWR);
    
     	if (1 == cmd)
     		ioctl(fd, LED_ON);
    
     	else if (0 == cmd)
     		ioctl(fd, LED_OFF);
    
    	return 0;
    } 



2)编译模块：

- make 
- arm-linux-gcc -static app_led.c -o app_led

3)安装驱动：

- 将生成的led.ko和app_led复制到NFS文件系统里
- insmod led.ko（安装led驱动模块）
- cat /proc/device（查看设备号）
- mknod /dev/myled c 253 0（创建设备文件）
- ./app_led 1（点亮led）
- ./app_led 0（熄灭led）
- .lsmod led（卸载led模块）

### 　　　　　　　　　　　　　　　　　　　　　

### 008）按键驱动程序设计

目标：

设计一个Tiny6410LED的按键驱动，当按下K1是，串口打印:Tiny6410_K1 down !

1）代码准备：
key.c

    /*
     *****************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : key.c
     *      Author     : X h y    
     *      Version    : 1.0       
     *      Date       : 06-02-2015
     *      Description: for Tiny6410   
     *****************************************************************
     */
     
    #include <linux/module.h>
    #include <linux/init.h>
    #include <linux/miscdevice.h>
    #include <linux/interrupt.h>
    #include <linux/fs.h>
    #include <linux/io.h>
    
    #define GPNCON 0x7f008830 
    
    irqreturn_t key_int(int irq, void *dev_id)
    {
    	/* 1.check key interrupt */
    
    	/* 2.clear the interrupt */
    
    	/* 3.print the interrupt */
    	printk("Tiny6410_K1 down !\n");
    
    	return 0;
    }
    
    void key_hw_init(void)
    {
    	unsigned int *gpio_config;
    	unsigned short data;
    
    	gpio_config = ioremap(GPNCON, 4);
    	data = readw(gpio_config);
    	data &= ~0b11;
    	data |= 0b10;
    	writew(data, gpio_config);
    }
    
    int key_open(struct inode *node, struct file *filp)
    {
    
    	return 0;
    }
    
    static const struct file_operations key_fops =
    {
    	.open = key_open,
    };
    
    static const struct miscdevice key_miscdev = 
    {
    	.minor = 200,
    	.name  = "Tiny6410_K1",
    	.fops  = &key_fops,
    };
    
    static int button_init(void)
    {
    	/* register miscdevice */
    	misc_register(&key_miscdev);
    
    	/* init the key */
    	key_hw_init();
    
    	/* register irq */
    	request_irq(IRQ_EINT(0), key_int, IRQF_TRIGGER_FALLING, "Tiny6410_K1", 0);
    
    	return 0;
    }
    
    static void button_exit(void)
    {
    	/* deregister miscdevice */
    	misc_deregister(&key_miscdev);
    
    	/* unregister irq */
    }
    
    module_init(button_init);
    module_exit(button_exit);
    
    MODULE_LICENSE("GPL");
    MODULE_AUTHOR("Xhy Tech.Stu");
    MODULE_VERSION("1.0");
Makefile

    obj-m := key.o    
    KDIR  := /home/xhy/Embedded/s4/last_term/part3/lesson3/linux-tiny6410    
    all:
    	make -C $(KDIR) M=$(PWD) modules CROSS_COMPILE=arm-linux- ARCH=arm
    clean:
    	rm -rf .* *.o *.ko *.mod.o *.mod.c *.symvers *.bak *.order


2）编译模块：

- make 

3）安装驱动：

- 将生成的key.ko复制到NFS文件系统里

- insmod key.ko（安装按键驱动模块）

- 按下K1键，串口屏幕打印：Tiny6410_K1  down !

- lsmod key（卸载key模块）

  　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　

### 009）优化按键驱动程序

目标：

- 中断分层处理，提高系统中断处理效率
- 按键消抖，通过定时器消抖，处理更精确
- 驱动支持多按键处理

1）代码准备：

key.c

    /*
     *****************************************************************
     *      Copyright (C), 2015-2115, Xhy Tech. Stu.
     *      FileName   : key.c
     *      Author     : X h y    
     *      Version    : 1.0  
     *      Date       : 06-04-2015
     *      Description: for Tiny6410   
     *****************************************************************
     */
    
    #include <linux/module.h>
    #include <linux/init.h>
    #include <linux/miscdevice.h>
    #include <linux/interrupt.h>
    #include <linux/fs.h>
    #include <linux/io.h>
    #include <linux/slab.h>
    #include <linux/uaccess.h>
    
    #define GPNCON 0x7f008830 
    #define GPNDAT 0x7f008834 
    
    static struct work_struct *work1;
    
    static struct timer_list key_timer;
    
    static unsigned int *gpio_data;
    
    static unsigned int key_num = 0;
    
    static void work1_func(struct work_struct *work)
    {
    	/* start the timer */
    	mod_timer(&key_timer, jiffies + HZ/100);
    }
    
    static void key_timer_func(unsigned long data)
    {
    	static unsigned char key_val;
    
    	key_val = (readw(gpio_data) & 0x0f);
    
    	/* K1 down */
    	if (0x0e == key_val)
    		key_num = 1;
    
    	/* K2 down */
    	else if (0x0d == key_val)
    		key_num = 2;
    
    	/* K3 down */
    	else if (0x0b == key_val)
    		key_num = 3;
    
    	/* K4 down */
    	else if (0x07 == key_val)
    		key_num = 4;
    }
    
    static irqreturn_t key_int(int irq, void *dev_id)
    {
    	/* schedule the work1 */
    	schedule_work(work1);
    	
    	return 0;
    }
    
    static void key_hw_init(void)
    {
    	unsigned int *gpio_config;
    	unsigned short data;
    
    	gpio_config = ioremap(GPNCON, 4);
    	gpio_data   = ioremap(GPNDAT, 4);
    
    	data  = readw(gpio_config);
    	data &= ~0xff;
    	data |= 0xaa;
    
    	writew(data, gpio_config);
    }
    
    static int key_open(struct inode *node, struct file *filp)
    {
    
    	return 0;
    }
    
    static key_read(struct file *filp, char __user *buf, size_t size, loff_t *pos)
    {
    	copy_to_user(buf, &key_num, 4);
    
    	return 4;
    }
    
    static const struct file_operations key_fops =
    {
    	.open = key_open,
    	.read = key_read,
    };
    
    static const struct miscdevice key_miscdev = 
    {
    	.minor = 200,
    	.name  = "Tiny6410_Key",
    	.fops  = &key_fops,
    };
    
    static int button_init(void)
    {
    	/* register miscdevice */
    	misc_register(&key_miscdev);
    
    	/* init the key */
    	key_hw_init();
    
    	/* register irq */
    	request_irq(IRQ_EINT(0), key_int, IRQF_TRIGGER_FALLING, "Tiny6410_K1", 0);
    	request_irq(IRQ_EINT(1), key_int, IRQF_TRIGGER_FALLING, "Tiny6410_K2", 0);
    	request_irq(IRQ_EINT(2), key_int, IRQF_TRIGGER_FALLING, "Tiny6410_K3", 0);
    	request_irq(IRQ_EINT(3), key_int, IRQF_TRIGGER_FALLING, "Tiny6410_K4", 0);


​    
    	/* creat the work1 */
    	work1 = kmalloc(sizeof(struct work_struct), GFP_KERNEL);
    	INIT_WORK(work1, work1_func);
    
    	/* schedule the work1 in key_int */


​    		
    	/* init the timer */
    	init_timer(&key_timer);
    	key_timer.function = key_timer_func;
    
    	/* register the timer */
    	add_timer(&key_timer);
    
    	/* start the timer int work1_func */

    	return 0;
    }
    
    static void button_exit(void)
    {
    	/* deregister miscdevice */
    	misc_deregister(&key_miscdev);
    }
    
    module_init(button_init);
    module_exit(button_exit);
    
    MODULE_LICENSE("GPL");
    MODULE_AUTHOR("Xhy Tech.Stu");
    MODULE_VERSION("1.0");



Makefile

2）编译模块：

- make 

3）安装驱动：

- 将生成的key.ko复制到NFS文件系统里
- insmod key.ko（安装按键驱动模块
- 按下K1键，串口屏幕打印：Tiny6410_K1  down !
- lsmod key（卸载key模块）





更于06-02-2015

---


待更新 ……