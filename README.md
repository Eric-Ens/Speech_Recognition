# Speech_Recognition

This repo contains the code for a speech recognition neural network using the dataset from [TensorFlow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge).

The model performs with 76% accuracy on the Kaggle test dataset.

## Gathering Data
****
The notebook which gathers and converts the data is Convert_to_Spec.ipynb. This notebook is capable of reading the data and converting it into images or csv files of pixel data in either spectrograms or mfccs. It is currently configured to convert the data to csv files of mfcc data. Additionally, some audio files are note quite 1 second in length and this notebook will pad the data with 0's so all entries are exactly the same size.

![alt-text](/mfcc.png)

Another challenge is including silence in the dataset. The dataset, as provided, only includes 6 longer files of background noise. In the notebook Dealing_with_Silence.ipynb we chop up these files into 1 second pieces to use as silence files. These files were manually added in the OS to a created "Silence" folder.

## Tensorflow CNN Model
****
The model I built is a CNN built with tensorflow and is in the notebook 11_class_CNN.ipynb. This model is designed to work easily on local computers as it never imports the whole dataset at once, and only the files it needs at any one time.

The basic architecture is two convolutional layers, the first with 72 6x6 filters and the second with 112 6x6 filters, each with 2x2 max pooling. This is followed by a flattening layer and a dropout layer.

Next is two fully connected layers with 256 neurons each with dropout layers after each and finally the output layer.

![alt-text](/architecture.png)

### Performance
****
Including Silence, the model performs adequetly at 80% on the validation set. In the notebook there are some clips of sample errors. Some of which are obviously wrong, but some are also ones which are likely mislabelled or indistinguishable. This initially trained model then performed at 76% on the test set according to the Kaggle Submission.

Confusion Matrix:
 ![alt-text](/confusion.png)
 
The next step is in the notebook Predictor_CNN.ipynb. This model uses precisely the same architecture as the one in 11_class_CNN but instead trains on the entire training dataset (which includes the validation set for training). This is the model which I ultimately use to make my final prediction to submit to Kaggle.
 
 
## Possible ways to improve
****
### The Unknown Class
The way in which I could improve the performance the most is to change how the model deals with the unknown label. Currently, the Kaggle dataset comes with much more than the 10 words included in the target class, but these words are all labelled. I have combined those words (and then downsampled) to create a trainable "unknown" class. A better, but more complicated way to deal with this is to exclude this synthetic unknown class from training and then use a confidence threshold in the softmax layer to determine if an entry is unknown. In other words, if the model is not confident enough that the entry is any one of the 10 words, it labels it as unknown.

I currently don't know how to implement this in Tensorflow.

### Silence

The silence data is substantially different from any of the other data, so it can be dealt with a possible better way. One way is to use an enemble model, where an initial model is built to determine if an audio clip is a clip of silence, or not, a simple binary classifier. Another model is then used on the clips which were not classified as silence, this would be very similar to the model trained above.

### Keras
Both of the above issues could possibly be done by continuing with the notebook Prelim_Keras.ipynb. This notebook was written to run on an aws cluster. Due to frustrations with free aws accounts, I had mostly given up on this notebook. It may have some small bugs with the data.

