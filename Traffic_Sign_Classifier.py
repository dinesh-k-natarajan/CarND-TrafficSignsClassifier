#!/usr/bin/env python
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[71]:


# Load pickled data
import pickle
import numpy as np
import pandas as pd

# TODO: Fill this in based on where you saved the training and testing data

training_file = 'data/train.p'
validation_file='data/valid.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[72]:


### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(np.concatenate((y_train, y_valid, y_test))))

# Names of the labels / classes obtained from signnames.csv
classes_df = pd.read_csv('signnames.csv')
name_classes = classes_df['SignName']

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?

# In[73]:


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
get_ipython().run_line_magic('matplotlib', 'inline')

# Unique examples for each label and its corresponding index found from the data labels
unique_labels, unique_indices, unique_counts = np.unique(np.concatenate((y_train,y_valid,y_test)),                                                         return_index=True, return_counts=True)
# Corresponding images of the unique labels from the image data
unique_images = np.concatenate((X_train,X_valid,X_test))[unique_indices,...]

# Plotting traffic sign images
fig, axs = plt.subplots(n_classes//5+np.int(np.ceil((n_classes%5)/5)),5,figsize=(25,25))
axs = axs.reshape(-1)
for idx, ax in enumerate(axs):
    if idx >= n_classes:
        fig.delaxes(ax)
        continue
    ax.imshow(unique_images[idx,...])
    ax.set_title(name_classes[unique_labels[idx]],fontsize=16)
    ax.axis('off')
fig.suptitle('Traffic Sign Images in the German Traffic Signs Dataset', fontsize=20, y=1.02)
fig.subplots_adjust(top=0.8)
fig.tight_layout()

# Plotting the count of each sign
fig, axs = plt.subplots(2,2,figsize=(15,10))
axs = axs.reshape(-1)
# Histogram data
hist_data = [[y_train,y_valid,y_test], y_train, y_valid, y_test];
stacked   = [True, False, False, False]
colors    = [['blue','green','red'], 'blue', 'green','red']
labels    = [['Training','Validation','Test'], None, None, None]
titles    = ['In all 3 datasets','In training dataset','In validation dataset','In test dataset']

# Loop over subplots
for idx, ax in enumerate(axs):
    ax.hist(hist_data[idx], bins=n_classes, stacked=stacked[idx], color= colors[idx], label=labels[idx])
    ax.legend()
    ax.set_title(titles[idx])
    ax.set_xlabel('Class Labels')
    ax.set_ylabel('Number of Samples')
    
fig.suptitle('Distribution of samples from each class', fontsize=20, y=1.02)
fig.subplots_adjust(top=0.8)
fig.tight_layout(pad=2.5)


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 
# 
# With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture (is the network over or underfitting?)
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 
# 
# Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 
# 
# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[74]:


### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
from sklearn.utils import shuffle
import cv2

idx = np.random.randint(0,len(X_train))

print('Size of RGB image:', X_train.shape[1:])
fig, axs = plt.subplots(1,3,figsize=(32,32))
axs = axs.flatten()
axs[0].imshow(X_train[idx,...])
axs[0].axis('off')
axs[0].set_title('Original RGB Image',fontsize=30)

# Is it advisable to use grayscale conversion as the color might be a useful feature in classifying traffic signs? (TODO)
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

X_train = np.array(list(map(grayscale, X_train)))
X_valid = np.array(list(map(grayscale, X_valid)))
X_test  = np.array(list(map(grayscale, X_test)))
print('Size of grayscale image:', X_train[500,...].shape)
axs[1].imshow(X_train[idx,...],cmap='gray')
axs[1].axis('off')
axs[1].set_title('After Grayscale Conversion',fontsize=30)

# Enhance contrast of the grayscale image by histogram equivalization
def enhance_contrast(image):
    image = np.uint8(cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    image = clahe.apply(image)
    return image

X_train = np.array(list(map(enhance_contrast, X_train)))
X_valid = np.array(list(map(enhance_contrast, X_valid)))
X_test  = np.array(list(map(enhance_contrast, X_test)))

# Normalization of Images to have mean = 0, and standard deviation = 1
def normalization(image):
    mean_image = np.array(image.mean(axis=(1,2))).reshape(-1,1,1)
    std_image  = np.array(image.std(axis=(1,2))).reshape(-1,1,1)
    return np.divide(image - mean_image, std_image)

X_train = normalization(X_train)
X_valid = normalization(X_valid)
X_test  = normalization(X_test)

print('Normalized min and max values: ', X_train.min(), X_train.max())
print('Normalized mean and variance: ', X_train.mean(), X_train.var())
axs[2].imshow(X_train[idx,...],cmap='gray')
axs[2].axis('off')
axs[2].set_title('After Normalization',fontsize=30)
fig.tight_layout()

# Adding 4th dimension to image data to be compatible with Conv Layer Input
X_train = X_train[...,np.newaxis]
X_valid = X_valid[...,np.newaxis]
X_test  = X_test[...,np.newaxis]
print('Size of pre-processed image:', X_train[idx,...].shape)

# Shuffling the datasets
X_train, y_train = shuffle(X_train, y_train)
X_valid, y_valid = shuffle(X_valid, y_valid)
X_test , y_test  = shuffle(X_test,  y_test)


# ### Model Architecture

# In[75]:


### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from tensorflow.contrib.layers import flatten

def MyModel(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Architecture is based on LeNet-5
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Activation.
    fc1    = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation.
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 43 (n_classes).
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits

## Features and Labels
# Defining placeholders for x and y
x = tf.placeholder(tf.float32, (None, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

## Training Pipeline
EPOCHS = 100
BATCH_SIZE = 48
rate = 0.001

logits = MyModel(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

## Model Evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

# In[18]:


### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

## Training and Validating the Model
training_acc   = []
validation_acc = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        training_accuracy   = evaluate(X_train, y_train)  
        training_acc.append(training_accuracy)
        validation_accuracy = evaluate(X_valid, y_valid)
        validation_acc.append(validation_accuracy)
        print("EPOCH {} ...".format(i+1))
        print("Training Accuracy = {:.3f}".format(training_accuracy))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        
        
    # Saving the model
    saver.save(sess, './mymodel')
    print("Model saved")
    
    # Plot the training and validation accuracies
    plt.figure()
    plt.plot(np.arange(EPOCHS), training_acc, label='Training')
    plt.plot(np.arange(EPOCHS), validation_acc, label='Validation')
    plt.title('Accuracy over training epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# ### Testing the Model on test dataset

# In[76]:


with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[81]:


### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import os

new_test_images = []
new_prep_images = []
path = 'data/NewTestImages/'
for image in os.listdir(path):
    img = cv2.imread(path + image)
    img = cv2.resize(img, (32,32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    new_test_images.append(img)
    # Creating preprocessed images
    img_pp = grayscale(img)
    img_pp = enhance_contrast(img_pp)
    img_pp = (img_pp - img_pp.mean()) / img_pp.std()
    img_pp = img_pp[...,np.newaxis]
    new_prep_images.append(img_pp)
    
new_labels = [13, 3, 17, 27, 14]
print("Number of new test images: ", len(new_test_images))

## Displaying the images
fig, axs = plt.subplots(1,5,figsize=(32,32))
axs = axs.flatten()
for idx, ax in enumerate(axs):
    ax.imshow(new_test_images[idx])
    ax.set_title(name_classes[new_labels[idx]],fontsize=20)
    ax.axis('off')


# ### Predict the Sign Type for Each Image

# In[83]:


### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

# Prediction for the new images
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    y_prob, y_pred  = sess.run(tf.nn.top_k(tf.nn.softmax(logits),k=5),feed_dict={x:new_prep_images})

## Displaying the preprocessed images with predictions
print('Pre-processed images with the predicted labels')
fig, axs = plt.subplots(1,5,figsize=(32,32))
axs = axs.flatten()
for idx, ax in enumerate(axs):
    ax.imshow(new_prep_images[idx].squeeze(), plt.gray())
    ax.set_title(name_classes[y_pred[idx][0]],fontsize=20)
    ax.axis('off')


# ### Analyze Performance

# In[84]:


### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.

# Accuracy on new images
correct = 0
for idx in range(len(new_prep_images)):
    if y_pred[idx][0] == new_labels[idx]:
        correct += 1
new_accuracy = correct / len(new_prep_images)
print('Accuracy of the model on new images = {:.0%}'.format(new_accuracy))
    


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# In[85]:


### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.

# Plotting the images and their corresponding softmax probabilities in a histogram
fig, axs = plt.subplots(len(new_prep_images),4, figsize=(15,20),gridspec_kw={"width_ratios" : [1,1,.55,1]})
fig.suptitle("Class Label, Predicted Label and Softmax Probabilties for the New Images",y=0.92,fontsize=20)
for idx in range(len(new_prep_images)):
    # Image
    axs[idx,0].imshow(new_test_images[idx])
    axs[idx,0].axis('off')
    axs[idx,0].set_title(name_classes[new_labels[idx]])
    # Softmax Probabilities
    axs[idx,3].barh(np.arange(len(y_pred[idx])), y_prob[idx], align='center')
    axs[idx,3].set_yticks(np.arange(len(y_pred[idx])))
    axs[idx,3].set_yticklabels(name_classes[y_pred[idx]])
    axs[idx,3].set_xlabel('Softmax Probabilties')
    # Predicted class example
    uniq_idx = unique_labels.tolist().index(y_pred[idx][0])
    axs[idx,1].imshow(unique_images[uniq_idx])
    axs[idx,1].axis('off')
    axs[idx,1].set_title(name_classes[y_pred[idx][0]])
    # Dummy plot for spacing
    axs[idx,2].axis('off')


# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

# ---
# 
# ## Step 4 (Optional): Visualize the Neural Network's State with Test Images
# 
#  This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.
# 
#  Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.
# 
# For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.
# 
# <figure>
#  <img src="visualize_cnn.png" width="380" alt="Combined Image" />
#  <figcaption>
#  <p></p> 
#  <p style="text-align: center;"> Your output should look something like this (above)</p> 
#  </figcaption>
# </figure>
#  <p></p> 
# 

# In[24]:


### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")

