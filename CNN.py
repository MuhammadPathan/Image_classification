import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

#var
batch_size = 32
img_height = 224
img_width = 224
#3 colour channels
channels=3
input_shape=(img_height,img_width,channels)
#data path
wd=os.path.abspath(os.getcwd())
data_folder="data"
data_dir=os.path.join(wd,data_folder)

#load training set
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
#print class num and names
class_names = train_ds.class_names
print(class_names)
num_of_classes=len(class_names)
print(num_of_classes)
