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

How to select the pixels you interested in picture?

**2) Problem Analysis:**

A blob analysis mainly consists of three steps:

* Acquire Image(s):
An image is acquired.
* Segment Image(s):
Isolating the foreground pixels of interest from the image background using preprocessing tools and operations like thresholding and others. This is also called segmentation.
* Extract Features:
Features like area (i.e., the number of pixels), center of gravity, or the orientation of a blob or blobs are calculated.

* ([The above information comes from halcon](https://www.mvtec.com/services-solutions/technologies/blob-analysis/))

**3) Problem Solving:**

**4) Problem Expansion:**
