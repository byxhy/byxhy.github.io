---
layout: post
title: "Machine Vision"
author: "Xhy"
categories: MV
tags: [machine vision,sample]
image: shane-hauser.jpg
---


Photo by shane hauser

>When you want to do something? Just do it...

<br />

### 1.My first machine vision code.

```python
open_framegrabber ('File', 1, 1, 0, 0, 0, 0, 'default', -1, 'default', -1, 'false', 'fabrik', 'default', 1, -1, AcqHandle)
grab_image_start (AcqHandle, -1)

dev_close_window ()
dev_open_window (0, 0, 1024, 512, 'black', WindowID)
while (true)
    grab_image_async (Image, AcqHandle, -1)
    * Image Acquisition 01: Do something
    rgb1_to_gray (Image, GrayImage)

    threshold (GrayImage, Regions, 161, 231)

    connection (Regions, ConnectedRegions)

    select_shape (ConnectedRegions, SelectedRegions, 'area', 'and', 5824.07, 8305.56)

    area_center (SelectedRegions, Area, Row, Column)

    disp_message (WindowID, 'Area:' + Area + ' Coordinate:(' + Row + ',' + Column + ')', 'window', 12, 12, 'black', 'true')

endwhile
close_framegrabber (AcqHandle)
```

<br />

### 2.A simple object detection in Halcon.

**1) Problem Finding:**

*How to select the pixels you interested in picture? (Massage ball)*

![](/assets/img/MV-2-1.jpg)

**2) Problem Analysis:**

*A blob analysis mainly consists of three steps:*

* Acquire Image(s):
An image is acquired.
* Segment Image(s):
Isolating the foreground pixels of interest from the image background using preprocessing tools and operations like thresholding and others. This is also called segmentation.
* Extract Features:
Features like area (i.e., the number of pixels), center of gravity, or the orientation of a blob or blobs are calculated.

* ([The above information comes from halcon](https://www.mvtec.com/services-solutions/technologies/blob-analysis/))

**3) Problem Solving:**
```python
* Initialize the program
dev_close_window ()
dev_open_window (0, 0, 1200, 576, 'black', WindowHandle)
dev_set_draw ('margin')

* 1. Acquire Image(s)
read_image (Image, 'E:/M/Halcon/Image/6.jpg')

* 2. Segment the Image(s)
rgb1_to_gray (Image, GrayImage)
dev_display (GrayImage)

threshold (GrayImage, Regions, 113, 255)
dev_display (Regions)

connection (Regions, ConnectedRegions)
dev_display (ConnectedRegions)

* 3. Extract Features
select_shape (ConnectedRegions, SelectedRegions, 'area', 'and', 360952, 515238)
area_center (SelectedRegions, Area, Row, Column)  

* 4. Display
dev_display (Image)
dev_display (SelectedRegions)
disp_message (WindowHandle, 'Area:' + Area + ' Coordinate:(' + Row + ',' + Column + ')', 'window', 12, 12, 'black', 'true')
```

![](/assets/img/MV-2-3.jpg)


**4) Problem Expansion:**
* ([BLOB Analysis (Introduction to Video and Image Processing) Part 1](http://what-when-how.com/introduction-to-video-and-image-processing/blob-analysis-introduction-to-video-and-image-processing-part-1/))


<br />

### 3. Elementary Arithmetic

**1) Problem Finding:**

*What is the difference between the erosion, dilation, opening and closing ?*

![](/assets/img/MV-3-1.jpg)

**2) Problem Analysis:**

*A blob analysis mainly consists of three steps:*

* Acquire Image(s):
An image is acquired.
* Segment Image(s):
Isolating the foreground pixels of interest from the image background using preprocessing tools and operations like thresholding and others. This is also called segmentation.
* Extract Features:
Features like area (i.e., the number of pixels), center of gravity, or the orientation of a blob or blobs are calculated.

* ([The above information comes from halcon](https://www.mvtec.com/services-solutions/technologies/blob-analysis/))

**3) Problem Solving:**
```python
* Initialize the program

dev_update_window ('off')
dev_close_window ()
dev_open_window (0, 0, 1024, 512, 'black', WindowHandle)
*dev_display ('margin')

* 1. Acquire the Image(s)
read_image (Bond, 'die/die_03')
dev_display (Bond)


* 2. Segment the Image(s)
rgb1_to_gray (Bond, GrayImage)
dev_display (GrayImage)

threshold (GrayImage, Regions, 16, 42)
dev_display (Regions)

connection (Regions, ConnectedRegions)
dev_display (ConnectedRegions)


* 3. Extract features

opening_circle (ConnectedRegions, RegionOpening, 16)
dev_display (RegionOpening)

*erosion_circle (ConnectedRegions, RegionOpening, 16)
*dev_display (RegionOpening)

*dilation_circle (ConnectedRegions, RegionOpening, 16)
*dev_display (RegionOpening)

*closing_circle (ConnectedRegions, RegionOpening, 16)
*dev_display (RegionOpening)


* 4. Display
dev_display (Bond)
dev_display (RegionOpening)
```

*opening*
![](/assets/img/MV-3-3-1.jpg)

*erosion*
![](/assets/img/MV-3-3-2.jpg)

*dilation*
![](/assets/img/MV-3-3-3.jpg)

*closing*
![](/assets/img/MV-3-3-4.jpg)


**4) Problem Expansion:**
* ?????

<br />
