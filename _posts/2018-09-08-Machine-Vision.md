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
* Acquire a license number image
* Rotate the license number image to horizontal
* License plate recognition

**3) Problem Solving:**
```python
* Initialize the program
dev_update_off ()
dev_close_window ()

* 1. Acquire Image(s)
read_image (Image, 'E:/M/Halcon/Image/MV-6-1.jpg')
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
tuple_max(Row2, RowMax)
tuple_min(Column2, ColumnMin)

sort_region (RegionClosing, SortedRegions, 'character', 'true', 'column')
read_ocr_class_mlp ('Industrial_0-9A-Z_NoRej.omc', OCRHandle)
do_ocr_multi_class_mlp (SortedRegions, ImageAffinTrans, OCRHandle, Class, Confidence)

strLicenNum := 'License Numbre: '+Class[0]+' '+Class[1]+' '+Class[2]+' '+Class[3]+' '+Class[4]+' '+Class[5]+' '+Class[6]

dev_display (ImageAffinTrans)
disp_message (WindowHandle, strLicenNum, 'window', RowMax+60, ColumnMin-30, 'red', 'true')
```

License number
![](/assets/img/MV-6-3.jpg)

**4) Problem Expansion:**

*another way to select the number region*

```python
* Initialize the program
dev_update_off ()
dev_close_window ()

* 1. Acquire Image(s)
read_image (Image, 'E:/M/Halcon/Image/MV-6-1.jpg')
dev_open_window_fit_image (Image, 0, 0, -1, -1, WindowHandle)
rgb1_to_gray (Image, GrayImage1)
invert_image (Image, ImageInvert)
dev_display (Image)

* 2. Rotate the lience number picture to horizontal
decompose3 (ImageInvert, red, green, bule)
trans_from_rgb (red, red, red, ImageResult1, ImageResult2, ImageResult3, 'hsv')
threshold (ImageResult3, Regions2, 238, 255)
opening_circle (Regions2, RegionOpening2, 4.5)
shape_trans (RegionOpening2, RegionTrans1, 'rectangle2')

area_center (RegionTrans1, Area1, Row3, Column3)
orientation_region (RegionTrans1, Phi1)
vector_angle_to_rigid (Row3, Column3, Phi1, Row3, Column3, 0, HomMat2D1)
affine_trans_image (GrayImage1, ImageAffinTrans, HomMat2D1, 'constant', 'false')
```
![](/assets/img/MV-6-4.jpg)

<br />


### 7. How to measure objects in a photo?

**1) Problem Finding:**

*How long is the book clip?*
![](/assets/img/MV-7-1.jpg)

**2) Problem Analysis:**
* Acquire a DUT image
* Rotate the image to horizontal
* Measure

**3) Problem Solving:**
```python
* Initialize the program
dev_update_off ()
dev_close_window ()

* 1. Acquire the image(s)
read_image (Clip, 'E:/M/Halcon/Image/Clip.jpg')
rgb1_to_gray (Clip, GrayImage)
dev_open_window_fit_image (GrayImage, 0, 0, -1, -1, WindowHandle)
dev_display (GrayImage)

area_center (GrayImage, Area2, Row2, Column2)
disp_message (WindowHandle, 'Pixel Size = ', 'window', Row2, Row2, 'red', 'true')

* 2. Segment the image(s)
threshold (GrayImage, Regions, 1, 106)
erosion_rectangle1 (Regions, RegionOpening, 12, 12)
connection (RegionOpening, ConnectedRegions)

* 3. Extract the features
select_shape (ConnectedRegions, SelectedRegions, 'area', 'and', 145646, 187287)
dilation_rectangle1 (SelectedRegions, RegionDilation, 17, 17)
fill_up (RegionDilation, RegionFillUp)

* 4. horizontally
orientation_region (RegionFillUp, Phi)
area_center (RegionFillUp, Area, Row, Column)
vector_angle_to_rigid (Row, Column, Phi, Row, Column, 0, HomMat2D)
affine_trans_image (GrayImage, ImageAffinTrans, HomMat2D, 'constant', 'false')

* 5. Measure
* Measure 07: Code generated by Measure 07
* Measure 07: Prepare measurement
AmplitudeThreshold := 30
RoiWidthLen2 := 59.5
set_system ('int_zooming', 'true')
* Measure 07: Coordinates for line Measure 07 [0]
LineRowStart_Measure_07_0 := 1546.78
LineColumnStart_Measure_07_0 := 1758.46
LineRowEnd_Measure_07_0 := 1546.78
LineColumnEnd_Measure_07_0 := 2922.7
* Measure 07: Convert coordinates to rectangle2 type
TmpCtrl_Row := 0.5*(LineRowStart_Measure_07_0+LineRowEnd_Measure_07_0)
TmpCtrl_Column := 0.5*(LineColumnStart_Measure_07_0+LineColumnEnd_Measure_07_0)
TmpCtrl_Dr := LineRowStart_Measure_07_0-LineRowEnd_Measure_07_0
TmpCtrl_Dc := LineColumnEnd_Measure_07_0-LineColumnStart_Measure_07_0
TmpCtrl_Phi := atan2(TmpCtrl_Dr, TmpCtrl_Dc)
TmpCtrl_Len1 := 0.5*sqrt(TmpCtrl_Dr*TmpCtrl_Dr + TmpCtrl_Dc*TmpCtrl_Dc)
TmpCtrl_Len2 := RoiWidthLen2
* Measure 07: Create measure for line Measure 07 [0]
* Measure 07: Attention: This assumes all images have the same size!
gen_measure_rectangle2 (TmpCtrl_Row, TmpCtrl_Column, TmpCtrl_Phi, TmpCtrl_Len1, TmpCtrl_Len2, 4032, 3024, 'nearest_neighbor', MsrHandle_Measure_07_0)
* Measure 07: ***************************************************************
* Measure 07: * The code which follows is to be executed once / measurement *
* Measure 07: ***************************************************************
* Measure 07: The image is assumed to be made available in the
* Measure 07: variable last displayed in the graphics window
copy_obj (ImageAffinTrans, Image, 1, 1)
* Measure 07: Execute measurements
measure_pairs (Image, MsrHandle_Measure_07_0, 11.3, 30, 'all', 'all', Row1_Measure_07_0, Column1_Measure_07_0, Amplitude1_Measure_07_0, Row2_Measure_07_0, Column2_Measure_07_0, Amplitude2_Measure_07_0, Width_Measure_07_0, Distance_Measure_07_0)
* Measure 07: Do something with the results
* Measure 07: Clear measure when done
close_measure (MsrHandle_Measure_07_0)


dev_display (ImageAffinTrans)
disp_message (WindowHandle, 'Pixel Size = ' + Width_Measure_07_0, 'window', 440, 360, 'red', 'true')
```
book clip size
![](/assets/img/MV-7-3.jpg)


**4) Problem Expansion:**
* ([How to measure the distance to an object in a photo](http://blog.perunature.com/2013/03/how-to-measure-distance-to-object-in.html))
* ([Measuring distances on pictures](https://forum.image.sc/t/measuring-distances-on-pictures/2178))

<br />


### 8. OCR

**1) Problem Finding:**
* What does OCR stand for
* How to OCR text in image

![](/assets/img/MV-8-1.jpg)

**2) Problem Analysis:**
* Correct and form a separate connected domain
* Associate the region with text
* Training
* OCR

**3) Problem Solving:**
```python
* Initialize the program
dev_update_off ()
dev_close_window ()


* 1. Acquire the Image(s)
read_image (Text, 'E:/M/Halcon/Image/Yamashita-Eiko.jpg')
dev_open_window_fit_image (Text, 0, 0, -1, -1, WindowHandle)
dev_display (Text)

* 2. Segment the Image(s)
rgb1_to_gray (Text, GrayImage)
threshold (GrayImage, Regions, 147, 244)
connection (Regions, ConnectedRegions)
select_shape (ConnectedRegions, SelectedRegions, 'area', 'and', 20, 42542.5)
union1 (SelectedRegions, RegionUnion)
shape_trans (RegionUnion, RegionTrans, 'rectangle2')
orientation_region (RegionTrans, Phi)
area_center (RegionTrans, Area, Row, Column)
vector_angle_to_rigid (Row, Column, Phi, Row, Column, 3.14, HomMat2D)
affine_trans_region (RegionTrans, RegionAffineTrans, HomMat2D, 'nearest_neighbor')
affine_trans_image (GrayImage, ImageAffinTrans, HomMat2D, 'constant', 'false')
reduce_domain (ImageAffinTrans, RegionAffineTrans, ImageReduced)
smallest_rectangle1 (ImageReduced, Row1, Column1, Row2, Column2)
gen_rectangle1 (Rectangle, Row1, Column1+178, Row1+(Row2-Row1)/2, Column2)
reduce_domain (ImageReduced, Rectangle, ImageReduced1)

* 3. Extract features
threshold (ImageReduced1, Regions1, 127, 194)
dilation_rectangle1 (Regions1, RegionDilation, 3, 6)
connection (RegionDilation, ConnectedRegions1)
* Err:sort_region (ConnectedRegions1, SortedRegions, 'character', 'true', column)
sort_region (ConnectedRegions1, SortedRegions, 'character', 'true', 'column')
count_obj (ConnectedRegions1, Number)
for i := 1 to Number by 1
    select_obj (SortedRegions, ObjectSelected, i)
endfor

* 4. Form TRF file
words := ['山', '下', '英', '子', '著']
TrainFile := 'E:/M/Halcon/TrainingsFile/Yamashita-Eiko.trf'
for i := 1 to Number by 1
    select_obj (SortedRegions, ObjectSelected, i)
    append_ocr_trainf (ObjectSelected, ImageReduced1, words[i-1], TrainFile)
endfor

read_ocr_trainf_names (TrainFile, CharacterNames, CharacterCount)

* 5. Start training
ClassFile := 'E:/M/Halcon/TrainingsFile/Yamashita-Eiko.omc'
create_ocr_class_mlp (82, 86, 'constant', 'default', CharacterNames, 80, 'none', 10, 42, OCRHandle)
trainf_ocr_class_mlp (OCRHandle, TrainFile, 200, 1, 0.01, Error, ErrorLog)
write_ocr_class_mlp (OCRHandle, ClassFile)

* 6. Start OCR
read_ocr_class_mlp (ClassFile, OCRHandle1)
do_ocr_multi_class_mlp (SortedRegions, ImageReduced1, OCRHandle1, Class, Confidence)

dev_display (ImageAffinTrans)
disp_message (WindowHandle, Class, 'image', Row2-200, Column2+400, 'red', 'false')
```
Yamashita-Eiko
![](/assets/img/MV-8-3.jpg)

**4) Problem Expansion:**
* ([Optical Character Recognition in Halcon 12](https://multipix.com/supportblog/optical-character-recognition-halcon-12/))

<br />

### 9. Color image segmentation

**1) Problem Finding:**
* Convert a three-channel image into three images
* Regions of interest extraction based on HSV color space
![](/assets/img/MV-9-1.jpg)

**2) Problem Analysis:**
* RGB to HSV
* Get ROI by HSV

**3) Problem Solving:**
```python
* Initialize the program
dev_update_off ()
dev_close_window ()

* 1. Acquire the Image(s)
read_image (Image, 'E:/M/Halcon/Image/jelly-beans.jpg')
dev_open_window_fit_image (Image, 0, 0, -1, -1, WindowHandle)
dev_display (Image)

* 2. Segment the Image(s)
decompose3 (Image, Red, Green, Blue)
trans_from_rgb (Red, Green, Blue, Hue, Saturation, Intensity, 'hsv')
threshold (Saturation, Regions, 58, 255)
reduce_domain (Hue, Regions, ImageReduced)
threshold (ImageReduced, Regions1, 16, 28)
opening_circle (Regions1, RegionOpening, 3.5)
fill_up (RegionOpening, RegionFillUp)
area_center (RegionFillUp, Area, OrangeRow, OrangeColumn)

threshold (ImageReduced, Regions2, 101, 118)
fill_up (Regions2, RegionFillUp1)
area_center (RegionFillUp1, Area1, BlueRow, BlueColumn)

threshold (ImageReduced, Regions3, 45, 51)
opening_circle (Regions3, RegionOpening1, 3.5)
area_center (RegionOpening1, Area2, GreenRow, GreenColumn)

threshold (ImageReduced, Regions4, 0, 15)
dilation_circle (Regions4, RegionDilation, 5)
fill_up (RegionDilation, RegionFillUp2)
connection (RegionFillUp2, ConnectedRegions)
select_shape (ConnectedRegions, SelectedRegions, 'area', 'and', 27752.8, 50000)
area_center (SelectedRegions, Area3, RedRow, RedColumn)

threshold (ImageReduced, Regions5, 36, 43)
opening_circle (Regions5, RegionOpening2, 3.5)
area_center (RegionOpening2, Area4, YellowRow, YellowColumn)

* 3. Extract features
dev_display (Image)
gen_cross_contour_xld (Cross1, OrangeRow, OrangeColumn, 30, 0)
gen_cross_contour_xld (Cross2, BlueRow, BlueColumn, 30, 0)
gen_cross_contour_xld (Cross3, GreenRow, GreenColumn, 30, 0)
gen_cross_contour_xld (Cross4, RedRow, RedColumn, 30, 0)
gen_cross_contour_xld (Cross5, YellowRow, YellowColumn, 30, 0)

disp_message (WindowHandle, 'Orange', 'image', OrangeRow, OrangeColumn, 'black', 'true')
disp_message (WindowHandle, 'Blue', 'image', BlueRow, BlueColumn, 'black', 'true')
disp_message (WindowHandle, 'Green', 'image', GreenRow, GreenColumn, 'black', 'true')
disp_message (WindowHandle, 'Red', 'image', RedRow, RedColumn, 'black', 'true')
disp_message (WindowHandle, 'Yellow', 'image', YellowRow, YellowColumn, 'black', 'true')
```
Jelly beans
![](/assets/img/MV-9-3.jpg)

**4) Problem Expansion:**
* ([How to determine range of HSV values of the image?](https://dsp.stackexchange.com/questions/5922/how-to-determine-range-of-hsv-values-of-the-image))

<br />


### 10. Color Recognition by MLP

**1) Problem Finding:**

*Create a multilayer perceptron for classification or regression.*
![](/assets/img/MV-10-1.jpg)

**2) Problem Analysis:**
* Specify color classes
* Train the specified color classes
* Apply the trained classes

**3) Problem Solving:**
```python
* Initialize the program
dev_update_off ()
dev_close_window ()

Regions := ['red','green','blue','yellow','pink','milky','purple','background']
gen_empty_obj (Classes)

* 1. Acquire the Image(s)
read_image (Image, 'E:/M/Halcon/Image/jelly-bean_train.jpg')
dev_open_window_fit_image (Image, 0, 0, -1, -1, WindowHandle)
dev_display (Image)

* 2. Specify color classes
for I := 1 to |Regions| by 1
    dev_display (Image)
    dev_display (Classes)
    disp_message (WindowHandle, ['Drag rectangle inside ' + Regions[I - 1] + ' color','Click right mouse button to confirm'], 'window', 24, 12, 'black', 'false')
    draw_rectangle1 (WindowHandle, Row1, Column1, Row2, Column2)
    gen_rectangle1 (Rectangle, Row1, Column1, Row2, Column2)
    concat_obj (Rectangle, Classes, Classes)    
endfor

* 3. Train the specified color classes
create_class_mlp (3, 8, 8, 'softmax', 'normalization', 4, 42, MLPHandle)
add_samples_image_class_mlp (Image, Classes, MLPHandle)
disp_message (WindowHandle, 'Training...', 'window', 100, 12, 'black', 'false')
train_class_mlp (MLPHandle, 400, 0.5, 0.01, Error, ErrorLog)

* 4. Apply the trained classes
read_image (Image1, 'E:/M/Halcon/Image/test-1.jpg')
classify_image_class_mlp (Image1, ClassRegions, MLPHandle, 0.5)
count_obj (ClassRegions, Number)
 if (Number != |Regions|)
    disp_message (WindowHandle, 'Not OK', 'window', 0, 0, 'red', 'false')
else
    for Index := 1 to Number by 1
        dev_clear_window ()
        dev_display (Image1)
        copy_obj (ClassRegions, ObjectsSelected, Index, 1)        
        disp_message (WindowHandle, Regions[Number-Index], 'window', 0, 0, 'green', 'true')
    endfor
endif
```
Test image
![](/assets/img/MV-10-3-test.jpg)

![](/assets/img/MV-10-3-red.jpg)

![](/assets/img/MV-10-3-green.jpg)

![](/assets/img/MV-10-3-blue.jpg)

**4) Problem Expansion:**
* ([Colour Recognition in Images Using Neural Networks](http://www.ijircce.com/upload/2016/february/260_73_Colour.pdf))

<br />


### 11. Halcon + VS2013

**1) Problem Finding:**

*Halcon and Visual Studio 2013 midxed programing*

**2) Problem Analysis:**
* .h
  * ..\MVTec\HALCON-12.0\include
  * ..\MVTec\HALCON-12.0\include\halconcpp
* .lib
  * ..\MVTec\HALCON-12.0\lib\x64-win64
  * halconcpp.lib
* dll

**3) Problem Solving:**
```c++
void CGreySkyDlg::OnBnClickedButton1()
{
    //1. Acquire Image(s)
    ReadImage(&ho_Image, "E:/M/Halcon/Image/MV-6-1.jpg");
    GetImageSize(ho_Image, &hv_Width, &hv_Height);  
    SetWindowAttr("background_color", "black");

    HWND hwnd1;
    CRect rect;
    GetDlgItem(IDC_PIC)->GetWindowRect(&rect);
    hwnd1 = GetDlgItem(IDC_PIC)->m_hWnd;
    LONG lWWindowID = (LONG)hwnd1;

    //2. Set the window size
    hv_Width = rect.Width();
    hv_Height = rect.Height();

    OpenWindow(0, 0, hv_Width, hv_Height, lWWindowID, "", "", &hv_WindowHandle);

    HDevWindowStack::Push(hv_WindowHandle);
    if (HDevWindowStack::IsOpen())
        DispObj(ho_Image, HDevWindowStack::GetActive());
}


void CGreySkyDlg::OnBnClickedButton2()
{
    Rgb1ToGray(ho_Image, &ho_GrayImage1);
    if (HDevWindowStack::IsOpen())
        DispObj(ho_GrayImage1, HDevWindowStack::GetActive());
}
```
GreySky
![](/assets/img/MV-11-3.jpg)

**4) Problem Expansion:**
* ([title](https://multipix.com/supportblog/optical-character-recognition-halcon-12/))

<br />


### 12. Title

**1) Problem Finding:**
* 111
* 111
![](/assets/img/MV-8-1.jpg)

**2) Problem Analysis:**
* .h
  * E:\Program Files (x86)\MVTec\HALCON-12.0\include
  * E:\Program Files (x86)\MVTec\HALCON-12.0\include\halconcpp
* .lib
  * E:\Program Files (x86)\MVTec\HALCON-12.0\lib\x86sse2-win32
  * halconcpp.lib
* dll

**3) Problem Solving:**
```c++

```
title
![](/assets/img/MV-8-3.jpg)

**4) Problem Expansion:**
* ([title](https://multipix.com/supportblog/optical-character-recognition-halcon-12/))
