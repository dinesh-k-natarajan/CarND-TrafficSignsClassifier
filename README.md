# **Traffic Sign Recognition** 

## Traffic Sign Classification based on LeNet-5 using the German Traffic Signs Dataset

Udacity - Self-Driving Car Nanodegree Program

---

**Building a Traffic Sign Recognition Project**

The goals of this project are the following:
* Load the [German Traffic Signs Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
* Explore, summarize and visualize the data set
* Design, train and test a model ConvNet architecture 
* Use the model to make predictions on new unseen images
* Analyze the softmax probabilities of the new images


[//]: # (Image References)

[image1]:  ./screenshots/GermanTrafficSignsSamples.png
[image2]:  ./screenshots/ClassLabelsDistribution.png
[image3]:  ./screenshots/ImagePreprocessing.png
[image4]:  ./screenshots/ImagePreprocessing_1.png
[image5]:  ./screenshots/ImagePreprocessing_2.png
[image6]:  ./screenshots/ImagePreprocessing_3.png
[image7]:  ./screenshots/ImagePreprocessing_4.png
[image8]:  ./screenshots/AccuracyinTraining.png
[image9]:  ./screenshots/NewTestImages.png
[image10]: ./screenshots/NewTestImages_Prediction.png
[image11]: ./screenshots/NewTestImages_Softmax.png
[image12]: ./screenshots/lenet.png

---

### 1. Dataset Summary & Exploration

#### 1.1. About the dataset

The [German Traffic Signs Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) is available in pickled form in the folder 'data' which contains 32x32x3 RGB images in the following files which are loaded using the `pickle` library:

* `train.p` - Training set of 34799 images
* `valid.p` - Validation set of 4410 images
* `test.p`  - Test set of 12630 images

*Note: train.p was too large to be uploaded to GitHub, but the entire dataset can be downloaded from the mentioned link*

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. 
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. These correspond to the original images. Since the images are already resized to 32x32, `'coords'` can be ignored during this project. 

There are 43 different classes/labels in this dataset. The mapping between the class labels (for e.g.: 14) and class names (for e.g.: 'Stop') are available in the file `signnames.csv`. The `pandas` library is used to load the mappings. 


#### 1.2. Visualization of the dataset

The dataset consists of 43 different types of Traffic Signs as shown below. The images are plotted using the `imshow()` function of the `matplotlib` library.  

![alt text][image1]

An ML model performs best when it is trained, validated and tested on data that come from the same probability distribution. Thus, an ideal split of the dataset into training, validation and test sets retains the distribution of the dataset. 

The following image shows the distribution of the whole dataset and the individual datasets. Based on the above reasoning, it can be concluded that the split is done well. 

![alt text][image2]

### 2. Image Preprocessing

The image datasets have the dimensions of (samples, 32, 32, 3), i.e., they are n samples of 32x32 RGB images, where n depends on the type of image dataset. Before building a model architecture, the data is preprocessed using the following techniques to improve the performance of the ConvNet: 

#### 2.1. Grayscale conversion

According to [Sermanet and Lecun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), the performance of their ConvNet improved while using grayscale images instead of color images. Although, color could be a useful feature in classifying traffic signs (for e.g.: some colors have special meaning in traffic signs and obviously traffic lights), it can be argued that using 3 color channels increases the features to be learned, thus increasing the complexity of the model. For this didactic project, grayscale images have been used. The images are converted to grayscale using the `cvtColor` function of the `OpenCV `library. After grayscale conversion, the image datasets have the dimensions of (n, 32, 32). The 3 RGB color channels are reduced to a single grayscale color channel. 

#### 2.2. Contrast Enhancement by Histogram Equalization

The quality of the images were poor and of low contrast. In order to enhance the quality of the images, the contrast could be increased using [Adaptive Histogram Equalization(AHE)](https://cromwell-intl.com/3d/histogram/). A drawback of this contrast enhancement is the resulting amplification of noise. It can be avoided by clipping any histogram bin above the defined clipping limit before equalization. Thus, a special contrast enhancement step called [Contrast Limited Adaptive Histogram Equalization(CLAHE)](http://amroamroamro.github.io/mexopencv/opencv/clahe_demo_gui.html) is applied to the images using the `createCLAHE` function of the `OpenCV` library. 

#### 2.3. Normalization
It is advisable to normalize the image data so that the data has mean zero and standard deviation of one. It can be done using the formula (image - mean(image)) / standard_deviation(image)

Steps 2 and 3 do not alter the dimensions of the image datasets. As a final step, the image datasets are converted to 4 dimensional arrays with dimensions (n, 32, 32, 1) to be compatible with the input layer of the ConvNet.

The improvement of the images as a result of the above preprocessing steps can be observed in the following examples: 

![alt text][image3]

![alt text][image4]

![alt text][image5]

![alt text][image6]

![alt text][image7]


### 3. Designing and Training a Model Architecture

#### 3.1. About LeNet-5

I chose the `LeNet-5` Architecture from [Yann Lecun](http://yann.lecun.com/exdb/lenet/) and challenged myself to achieve > 95% validation accuracy without modifying the LeNet-5 model architecture, but by tuning the preprocessing steps and few hyperparameters like `batch size` and `learning rate`. 

The `LeNet-5` was developed to classify handwritten letters of the English alphabet. [Sermanet and Lecun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) have successfully used an advanced version of LeNet in the [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news) competition. The LeNet-5 architecture can be visualized in the image below: 

![alt text][image12]

#### 3.2. Training the ConvNet

The training of the ConvNet was done in mini-batches. The error function is based on `Cross Entropy` over the softmax probabilties of the final layer of the ConvNet. This error function is then averaged over all samples and the `Adam Optimizer` (with learning rate) is used to minimize the averaged error function. The goal is to achieve >95% accuracy on the validation dataset. 

The following hyperparameters (apart from the model architecture) led to the best results in terms of the accuracy of prediction for the validation dataset: 

* `epochs` = 100
* `batch size` = 48
* `learning rate` = 0.001

The training and validation steps can be summarized as follows: 
* For each epoch:
    1. The images and labels of the dataset are shuffled 
    2. For each mini-batch of `batch size` samples, the forward propagation, backward propagation and optimization steps are done
    3. After looping over all mini-batches, the training and validation accuracies are computed by comparing `logits` (output of the ConvNet) and the labels of the dataset
* The weights and biases of the model are saved at the end of the training `epochs`. 

The training and validation accuracies obtained by the model are visualized in the plot below: 

![alt text][image8]
 
The best accuracies achieved by the model were: 

* On training dataset   = 100% 
* On validation dataset = **96.7%** 
* On test dataset       = 94%

Thus, the goals of this challenge were met by this simple `LeNet-5` model due to good preprocessing techniques and hyperparameter tuning. 

### 4. Testing the Model on New Unseen Images

#### 4.1. Need for testing on new unseen images

The Traffic Sign Classifier is to be used in the real-world on a Self-Driving Car. To test the robustness of the model, the real world is simulated by testing the model on new unseen images outside its dataset and observing its performance. 

The following 5 German traffic signs were found from the Internet:

![alt text][image9]

In general, the images are of high contrast and the signs are prominently visible on the image. The model is expected to perform well on these images. The fourth image might be difficult to classify because the Pedestrian sign at low resolutions could be easily confused with other signs such as General Caution, Right of Way at the Next Intersection, Traffic Signals ahead, etc. 

#### 4.2. Performance of the model on new unseen images

The predictions of the new images of Traffic Signs by the model were: 

![alt text][image10]

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield          		| Yield      									| 
| Speed Limit 60km/h	| Speed Limit 60km/h							|
| No Entry  			| No Entry      								|
| Pedestrians      		| Pedestrians       			 				|
| Stop          		| Stop              							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94%, eventhough the new images set is of a very small size. 

#### 4.3. Certainty of the model in making predictions

The prediction labels were computed by picking out the maximum of the softmax probabilities of the output vector of size `n_classes = 43`. The certainty of the model in its predictions can be observed by looking at the top 5 softmax probabilities of the output vector. This is done using the `tf.nn.top_k` function which takes as input arguments: logits and k, and outputs the top 5 softmax probabilities and associated labels. 

The top 5 softmax probabilties of the predictions for the new images are shown in the image below: 

![alt text][image11]

It is observed that the model is very certain of its prediction of 4 images except the Pedestrians Sign. In the 4 images classified with great certainty, the probabilities of the other 4 labels were very close to machine precision zero. The Pedestrians sign (softmax probability of approx. 80%) was expected to pose some difficulty as the sign is quite similar to other traffic signs such as General Caution, Right of Way at the Next Intersection, Traffic Signals ahead, etc., especially in grayscale.


## Further Improvements

* Other advanced ConvNet architectures could be implemented to achieve better accuracy
* The Image Preprocessing techniques could be further improved by studying extensively about Image Processing and tuning the hyperparameters
* New Unseen Images of lower quality and higher difficulty of prediction could be used to push the ConvNet to its limits and test its robustness. 
* Color Channels could be retained after the Image Preprocessing step to observe the effects on ConvNet performance.
