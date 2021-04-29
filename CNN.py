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