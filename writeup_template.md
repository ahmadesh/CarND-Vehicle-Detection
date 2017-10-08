
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it!

###Reviewing the impo

For training the classification between veheicles and non-vehecles, I used the Udacity's labeled datasets available [here](https://github.com/udacity/self-driving-car/tree/master/annotations). Here are the examples of the images available in the data set.

<img src="./output_images/Samples.png" width="600" alt="Combined Image" />

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the cells 4,5 of the IPython notebook.  I created a function for getting the images and output the HOG feature image. I explored different color spaces and different `skimage.hog()` parameters and I ended up by using (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. Here is an example using the `YUV` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

<img src="./output_images/HOG_example.png" width="600" alt="Combined Image" />


####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and chose the HOG parameters that provided the maximum prediction accuracy on the test-set.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the features from the `extract_features()` function in cell 7 that stacks the HOG, Spatial binning and color histogram features. These features are then scaled using `StandardScaler()` in sklearn library for training. The final parametrs that I used for extracting the features are:
`
HOG orientations = 11
pix_per_cell = (8,8)
cell_per_block = (2,2)
hog_channel = 'ALL'
spatial binning size = (16, 16) 
histogram bins = 32
`
The data are then split into training and test sets with 80% to 20% ratio. For tuning the colorspace and the `C` parameter used in the SVM classifier I used `GridSearchCV` function to search for the best combination. The best comibnation was achieved by `YUV` colorspace and `C=1`. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used a slidng window function in cell 9 of the code that takes the image along with `ystart, ystop, xstart, xstop, scale, cells_per_step` parameters to search for the car in that subimage using the SVM calssifier. I experimented with the sliding window on the image at different scales and locations on a sample image. The final windows that I setteled down are shown in the following:

<img src="./output_images/MultiScale_boxes.png" width="600" alt="Combined Image" />

I only search for vehecles in the lower section of the image. The search boxes are composed of 3 scales of `1.25, 2, 3`. The biger booxes with `2,3` scales are distributed on the two sides of the image as the large scale cars can only be found at these locations. The smaller scale `1.25` boxes are distribute over the length of the image. 

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 3 scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

<img src="./output_images/Sample_cars_detection.png" width="600" alt="Combined Image" />

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
