#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/archzzz/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt tag](https://github.com/archzzz/CarND-Traffic-Sign-Classifier-Project/blob/master/training-data.png)

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because it's faster in training and color is not useful when distinguishing a traffic sign. 

As a last step, I normalized the image data because this increases the training speed. Also some picture are very dark and has low constract, so i set every picture's max value to 1 and min value to 0, and mapped the rest values to [0,1]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook. The training data is shuffled before each EPOCHS and splited into batch of 128. 

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by 

My final training set had 34799 of images. My validation set and test set had 4410 and 12630 of images.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| RELU					|												|
| Dropout					|	keep_prob 0.7											|
| Fully connected		| input 400, output 120        									|
| Dropout					|	keep_prob 0.7											|
| Fully connected		| input 120, output 84        									|
| Dropout					|	keep_prob 0.7											|
| Fully connected		| input 84, output 43        									|
| Dropout					|	keep_prob 0.7											|
| Softmax				| cross entropy 						|
| Reduce mean	|		cost function										|
| AdamOptimizer	|	minimize the cost function				|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an optimizer that implements Adam algorithm. The batch size is 128, learning rate 0.01, epochs 30.  Dropout rate is 0.7, so that the training accuracy doesn't grow too slow, and the overfitting is not very bad. 

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.971
* validation set accuracy of 0.959 
* test set accuracy of 0.939

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? 
The first architecture is the LeNet solution from previous quiz, with 2 convolution layers, RELU layers and fully connected layers. It has many hidden layer and not too complicated to implement. It's a good starting point to give me some knowledge to the data and model.
* What were some problems with the initial architecture?
Validation accuracy stopped at 0.85.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
First i increased the epochs from 10 to 50. Then i found that training accuracy is 0.99 but validation accuracy is 0.92. It looks like overfitting, so i added 3 drop out layer after activation layer, set the dropout rate to 0.5. Then i found that training is very slow, each epoch only adds 0.02, and the training accuracy is only 0.93 after 50 rounds. So i increase the keep_prob to 0.7
* Which parameters were tuned? How were they adjusted and why?
Above answer
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Convolution layer helps identify small component like line and shape. RELU adds nonlinear complexity to the model and helps prevent overfitting. Dropout helps with overfitting. 

If a well known architecture was chosen:
* What architecture was chosen? 
No
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt tag](https://github.com/archzzz/CarND-Traffic-Sign-Classifier-Project/blob/master/12.png) 

The first image might be difficult to classify because it's resolution is very low. The second is not very difficult. The third one is shifting. The forth one is too bright. The fifth sign is blocked at the top. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.


The model was able to correctly guess all of traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.939

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.82  	| Speed limit (120km/h)  									| 
| 0.06  				| Speed limit (80km/h)			|
| 0.03					| Bicycles crossing			|
| 0.027      			| Speed limit (100km/h) 				|
| 0.014	    | Speed limit (50km/h)				|


For the second image 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99  	| Keep right					| 


For the third image 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99  	| Speed limit (50km/h)			| 


For the fourth image 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99  	| No passing									| 


For the fifth image 
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99  	| Priority road							| 
