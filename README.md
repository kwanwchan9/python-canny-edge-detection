# Introduction

Edge Detection is one of the major study in computer vision. By detecting the edge of an image, image segmentation and data extraction can thus be performed which have great benefits in the area of image processing and machine learning for object detection.

In this project, a simple Canny Edge Detector is conducted by using numpy and pillow package with python, which is a multi-stage algorithm to detect a wide range of edges in images.Canny edge detector generally including the following steps:

1. Gaussian Smoothing
2. Sobel Operator
3. Non-max suppression
4. Thresholding
5. Hough transform

The image are first requried to convert into grey scale. Second, Gaussian smoothing should be apply to the image to remove the details and noise of the image. Sopel operator thus is performed on the 2d image for spatial gradient measurement on an image so to emphasizes the regions of correspording edges.The results of the Sopel operator ideally will output all edges with different thickness thay may be connected. Weak edges therefore should be removed and convetering different think edges into lines for visualization which is done by non-max suppression and thresholding.

# Results

Input Image:
![Demo_Img_1](/image/road.jpeg)

RGB to Greyscale image:
![Demo_Img_2](/image/grey.jpg)

Apply Gaussian smoothing:
![Demo_Img_3](/image/gauss.jpg)

x, y gradient after apply sobel operator:
![Demo_Img_4](/image/G_x.jpg)
![Demo_Img_5](/image/G_y.jpg)

Magnitude gradient:
![Demo_Img_6](/image/G.jpg)

Non-max suppression:
![Demo_Img_7](/image/supress.jpg)

Thresholding:
![Demo_Img_8](/image/edgemap.jpg)
