# Importing the necessary modules:

from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage.io import imread

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imutils.object_detection import non_max_suppression
import joblib as jb
import imutils
import numpy as np
import argparse
import cv2
import os
import glob
from PIL import Image 
from numpy import *


# define parameters of HOG feature extraction
orientations = 8
pixels_per_cell = (32, 32)
cells_per_block = (1, 1)
threshold = .3
# define path to images:
pos_im_path = "archive/1" # This is the path of our positive input dataset
# define the same for negatives
neg_im_path= "archive/0"

# read the image files:
pos_im_listing = os.listdir(pos_im_path) # it will read all the files in the positive image path (so all the required images)
neg_im_listing = os.listdir(neg_im_path)
num_pos_samples = size(pos_im_listing) # simply states the total no. of images
num_neg_samples = size(neg_im_listing)
print(num_pos_samples) # prints the number value of the no.of samples in positive dataset
print(num_neg_samples)
data= []
labels = []

# compute HOG features and label them:

for file in pos_im_listing:
    img = Image.open(pos_im_path + '\\' + file) 
    img = img.resize((64,128))
    gray = img.convert('L') 
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)
    data.append(fd)
    labels.append(1)
    

for file in neg_im_listing:
    img= Image.open(neg_im_path + '\\' + file)
    img = img.resize((64,128))
    gray= img.convert('L')

    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 
    data.append(fd)
    labels.append(0)

print(data)
print(labels)

print(data[0].shape)
print(len(data))

le = LabelEncoder()
labels = le.fit_transform(labels)
print(" Constructing training/testing split...")

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data), labels, test_size=0.20, random_state=42)
print(trainData.shape)
print(testData.shape)
print(trainLabels.shape)
print(testLabels.shape)

print(" Training Linear SVM classifier...")
model = LinearSVC(max_iter=10000, random_state=42, tol=1e-3, verbose=1)
model.fit(trainData, trainLabels)
print(" Evaluating classifier on test data ...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))
jb.dump(model, 'model_name.npy')
