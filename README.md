# Emotion Recognition using deep learning

## Introduction

This project emotion recognition. Address 2 problems face detection and emotion classification. Model detection trained on **UTKFace Large Scale Face Dataset** dataset which which was published on [here](https://analyticsindiamag.com/10-face-datasets-to-start-facial-recognition-projects/).The size of the dataset is 10GB, and it includes approximately 1293 videos with consecutive frames of up to 240 frames for each original video. The overall single image frames are a total of 155,560 images. The model emotion classification trained with **three categories**--"happy,neutral,sad" on the **FER-2013** dataset which was published on International Conference on Machine Learning (ICML). This dataset consists of 35887 grayscale, 48x48 sized face images with **seven emotions** - angry, disgusted, fearful, happy, neutral, sad and surprised.
## Requirements

* Python 3.6
* Fastai 1.0.61
* OpenCV
* Skimage
* Json
* Pandas
* Matplotlib

## Basic Usage
  * Download UTKFace Large Scale Face Dataset from [here](https://drive.google.com/drive/folders/0BxYys69jI14kSVdWWllDMWhnN2c)
  * Download the FER-2013 dataset from [here](https://drive.google.com/file/d/1X60B-uR3NtqPd4oosdotpbDgy8KOfUdr/view?usp=sharing)
  * Label
  ```bash
  Labeldataforface.ipynb -- Use OpenCV to label and convert to json file.
  ```
  * Training model face detection
  ```bash
  facedetection.ipynb 
  ```
  * Training model emotion classification
  ```bash
  Emotion.ipynb
  ```
  * Test model
  ```bash
  api.ipynb
  ````
