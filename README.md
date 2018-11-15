# Speech_Recognition

This repo contains the code for a speech recognition neural network using the dataset from [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge).

## Gathering Data
****
The notebook which gathers and converts the data is Convert_to_Spec.ipynb (Data_spec_pt_2.ipynb is an alternate and unused method). This notebook is capable of reading the data and converting it into images or csv files of pixel data in either spectrograms or mfccs. It is currently configured to convert the data to csv files of mfcc data. Additionally, some audio files are note quite 1 second in length and this notebook will pad the data with 0's so all entries are exactly the same size.

![alt-text](/mfcc.png)

## Tensorflow CNN Model
****
The first model I have built is a CNN built with tensorflow and is in the notebook 11_class_CNN.ipynb. This model is designed to work easily on local computers as it never imports the whole dataset at once, and only the files it needs at any one time.

The basic architecture is two convolutional layers, the first with 56 7x7 filters and the second with 112 7x7 filters, each with 2x2 max pooling. This is followed by a flattening layer and a dropout layer.

Next is two fully connected layers with 256 neurons each followed by another dropout layer and finally the output layer.

### Performance
****
This fairly basic model performs quite well with an accuracy on the validation set of 83%


Confusion Matrix:
[[ 251    3    1    2    2    0    1    0    0    1    0]
 [   2  230    6    3    0    1    0    1    0   21    6]
 [   0    0  246    1    0    1    3    1    1    3    4]
 [   1   11    6  234    0    0    2    0    1    8    1]
 [   6    5    4    0  222    3    0    1    0    1    5]
 [   0    1    2    0    4  236    1    0    0    0   12]
 [   0    0    8    1    0    0  237    7    0    2    2]
 [   0    0   25    0    0    1    9  220    0    1    0]
 [   0    2   14    4    0    0    0    2  220    1    3]
 [   0   16    7    9    1    2    5    5    0  215    0]
 [  28   94   90  211   53  109  299   81   71  211 2974]]
 
 ![alt-text](/confusion.png)
 
 
## Keras Models
****
### Motivation
There is another avenue to look at to improve accuracy. That is in regards to the 'unknown' class. The above model is trained using a number of other words all categorized as 'unknown' and then downsampled so it doesnt interfere with accuracy. However, we may be able to improve the model by only training the known words and classifying any word whose softmax amount doesnt reach a certain threshold as 'unknown.'

Implementing this directly in tensorflow is very difficult and I was not able to find examples of this, so instead we will do so in Keras. So ... step 1:

### Initial model
The first model in the notebook Prelim_Keras.ipynb does not do this. This model is just a recreation of the above tensorflow model.
