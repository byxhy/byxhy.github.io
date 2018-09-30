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



<br />

### 4. Powerful filter

**1) Problem Finding:**
* mean-filtering
* median-filtering
* Gauss-filtering
* template-filtering

**2) Problem Analysis:**
* The operator gauss_filter smoothes images using the discrete Gaussian, a discrete approximation of the Gaussian function.

**3) Problem Solving:**
```python
* Initialize the program
dev_update_off ()
dev_close_window ()

* 1. Acquire Image(s)
read_image (Image, 'datacode/ecc200/ecc200_to_preprocess_004')
dev_open_window_fit_image (Image, 0, 0, -1, -1, WindowHandle)
dev_display (Image)

* 2.Segment the Image(s)

* median_image
median_image (Image, ImageMedian, 'circle', 1.5, 'mirrored')
dev_display (ImageMedian)

* mean_image
dev_display (Image)
mean_image (Image, ImageMean, 9, 9)
dev_display (ImageMean)

* gauss_filter
dev_display (Image)
gauss_filter (Image, ImageGauss, 5)
dev_display (ImageGauss)

```

*original*

![](/assets/img/MV-4-3-1.jpg)

*median_image*

![](/assets/img/MV-4-3-2.jpg)

*mean_image*

![](/assets/img/MV-4-3-3.jpg)

*gauss_filter*

![](/assets/img/MV-4-3-4.jpg)


**4) Problem Expansion:**


<br />


### 5. Affine transformation

**1) Problem Finding:**

*How to rotate the Jordan logo to 90 degrees ?*

![](/assets/img/MV-5-1.jpg)


**2) Problem Analysis:**

* find the angle between the logo and horizontal line
* rotate the angle to 0

**3) Problem Solving:**
```python
* Initialize the program
dev_update_off ()
dev_close_window ()

* 1. Acquire Image(s)
read_image (Image, 'E:/M/Halcon/Image/jordan.jpg')
dev_open_window_fit_image (Image, 0, 0, -1, -1, WindowHandle)
rgb1_to_gray (Image, GrayImage)
dev_display (GrayImage)

* 2.Segment the Image(s)
threshold (GrayImage, Regions, 5, 103)
opening_rectangle1 (Regions, RegionOpening, 10, 10)
shape_trans (RegionOpening, RegionTrans, 'convex')
dev_display (RegionTrans)

* 3. Extract features
orientation_region (RegionTrans, Phi)
area_center (RegionTrans, Area1, Row1, Column1)
vector_angle_to_rigid (Row1, Column1, Phi, Row1, Column1, 1.57, HomMat2D)

affine_trans_image (Image, ImageAffinTrans, HomMat2D, 'constant', 'false')
dev_display (ImageAffinTrans)
```


*The rotating Jordan*

![](/assets/img/MV-5-3-1.jpg)

**4) Problem Expansion:**



<br />


### 6. How to recognition a license number

**1) Problem Finding:**

*How to recognition the license number quickly?*

![](/assets/img/MV-6-1.jpg)

**2) Problem Analysis:**
* Acquire a license number
* Rotate the license number picture to horizontal
* License plate recognition

**3) Problem Solving:**
```python
* Initialize the program
dev_update_off ()
dev_close_window ()

* 1. Acquire Image(s)
read_image (Image, 'E:/M/Halcon/Image/CarlicenseNumber.jpg')
dev_open_window_fit_image (Image, 0, 0, -1, -1, WindowHandle)
rgb1_to_gray (Image, GrayImage)
invert_image (GrayImage, ImageInvert)
dev_display (GrayImage)

* 2. Rotate the lience number picture to horizontal
threshold (GrayImage, Regions, 0, 46)
dilation_rectangle1 (Regions, RegionDilation, 50, 5)
erosion_rectangle1 (RegionDilation, RegionErosion1, 9, 11)
connection (RegionErosion1, ConnectedRegions)
select_shape (ConnectedRegions, SelectedRegions, 'area', 'and', 38588.6, 78028)
opening_rectangle1 (SelectedRegions, RegionOpening, 16, 10)
shape_trans (RegionOpening, RegionTrans, 'convex')
orientation_region (RegionTrans, Phi)
area_center (RegionTrans, Area, Row, Column)

vector_angle_to_rigid (Row, Column, Phi, Row, Column, 3.14, HomMat2D)
affine_trans_image (GrayImage, ImageAffinTrans, HomMat2D, 'constant', 'false')


* 3. License plate recognition
threshold (ImageAffinTrans, Regions1, 1, 52)
opening_circle (Regions1, RegionOpening1, 3.5)
connection (RegionOpening1, ConnectedRegions1)
select_shape (ConnectedRegions1, SelectedRegions1, ['area','height'], 'and', [1914.41,109.41], [4632.13,200])
closing_circle (SelectedRegions1, RegionClosing, 1)
smallest_rectangle1 (RegionClosing, Row1, Column1, Row2, Column2)
sort_region (RegionClosing, SortedRegions, 'character', 'true', 'column')

read_ocr_class_mlp ('Industrial_0-9A-Z_NoRej.omc', OCRHandle)
do_ocr_multi_class_mlp (SortedRegions, ImageAffinTrans, OCRHandle, Class, Confidence)

tuple_max(Row2, RowMax)
tuple_min(Column2, ColumnMin)

dev_display (ImageAffinTrans)
disp_message (WindowHandle, 'License Numbre: '+Class[0]+' '+Class[1]+' '+Class[2]+' '+Class[3]+' '+Class[4]+' '+Class[5]+' '+Class[6], 'window', RowMax+60, ColumnMin-30, 'red', 'true')

```
![](/assets/img/MV-6-3.jpg)

**4) Problem Expansion:**



### 7. Approximate a rigid affine transformation from point correspondences

**1) Problem Finding:**


**2) Problem Analysis:**


**3) Problem Solving:**


**4) Problem Expansion:**
<br />
