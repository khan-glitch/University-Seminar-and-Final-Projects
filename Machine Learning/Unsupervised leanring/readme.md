I applelid a convolutional neural network to classify images from the TensorFlow Flowers dataset. The work was completed in Google Colab using TensorFlow, Keras, and TensorFlow Datasets. The aim is to design, train, and evaluate an image-classification model with clear steps and reproducible results.

## Dataset

* TensorFlow Flowers dataset
* Five classes: daisy, dandelion, rose, sunflower, tulip
* Automatically downloaded using TFDS
* Split into training, validation, and testing subsets

## Environment and Tools

* Google Colab
* TensorFlow / Keras
* TensorFlow Datasets
* NumPy
* Matplotlib
* scikit-learn metrics
* GPU runtime enabled

## Preprocessing

* Image resizing to 180×180
* Normalisation of pixel values
* Batching
* Caching and prefetching
* Basic augmentation (flip, rotation, zoom)

## Model Architecture

* Data augmentation layer
* Three Conv2D + MaxPooling blocks
* Flatten layer
* Dense layer (128 units)
* Dropout layer
* Final Dense softmax output (5 classes)

## Training

* Trained for up to 20 epochs
* EarlyStopping callback
* ModelCheckpoint callback
* Accuracy and loss tracked each epoch

## Evaluation

* Test accuracy around 73%
* Confusion matrix
* Classification report
* Performance consistent with validation results

## External Image Testing

* Four external flower images tested
* All correctly classified
* Confidence remained high across classes

## Additional Feature

* Probability distribution visualisation for each class

## How to Run

* Open notebook in Google Colab
* Enable GPU runtime
* Run all cells sequentially
* Dataset downloads automatically
