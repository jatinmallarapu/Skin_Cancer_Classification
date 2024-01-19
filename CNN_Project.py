# Ensure the Kaggle API credentials are set up and dataset is downloaded
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets download fanconic/skin-cancer-malignant-vs-benign
! mkdir cancerDataset
! unzip skin-cancer-malignant-vs-benign.zip -d cancerDataset

# Import necessary libraries
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import os
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Activation, Flatten
import tensorflow as tf

# Define dataset directories
benign_train = '/content/cancerDataset/data/train/benign'
malignant_train = '/content/cancerDataset/data/train/malignant'
benign_test = '/content/cancerDataset/data/test/benign'
malignant_test = '/content/cancerDataset/data/test/malignant'

# Define function to resize images
def resize_image(src_img, size=(64,64), bg_color="white"): 
    src_img.thumbnail(size, Image.ANTIALIAS)
    new_image = Image.new("RGB", size, bg_color)
    new_image.paste(src_img, (int((size[0] - src_img.size[0]) / 2), int((size[1] - src_img.size[1]) / 2)))
    return new_image

# List image files in train and test folders
benign_train_files = os.listdir(benign_train)
benign_test_files = os.listdir(benign_test)
malignant_train_files = os.listdir(malignant_train)
malignant_test_files = os.listdir(malignant_test)

# Initialize arrays to store resized images
image_arrays = []
image_arrays2 = []
image_arrays3 = []
image_arrays4 = []
size = (64,64)
background_color="white"

# Load and resize images, then append to arrays
for file_idx in range(len(benign_train_files)):
    img = Image.open(os.path.join(benign_train, benign_train_files[file_idx]))
    resized_img = np.array(resize_image(img, size, background_color))
    image_arrays.append(resized_img)

for file_idx in range(len(malignant_train_files)):
    img = Image.open(os.path.join(malignant_train_folder, malignant_train_files[file_idx]))
    resized_img = np.array(resize_image(img, size, background_color))
    image_arrays2.append(resized_img)

for file_idx in range(len(benign_test_files)):
    img = Image.open(os.path.join(benign_test_folder, benign_test_files[file_idx]))
    resized_img = np.array(resize_image(img, size, background_color))
    image_arrays3.append(resized_img)

for file_idx in range(len(malignant_test_files)):
    img = Image.open(os.path.join(malignant_test_folder, malignant_test_files[file_idx]))
    resized_img = np.array(resize_image(img, size, background_color))
    image_arrays4.append(resized_img)

# Convert image arrays to NumPy arrays
X_benign = np.array(image_arrays)
X_malignant = np.array(image_arrays2)
X_benign_test = np.array(image_arrays3)
X_malignant_test = np.array(image_arrays4)

# Create labels for the data
y_benign = np.zeros(X_benign.shape[0])
y_malignant = np.ones(X_malignant.shape[0])
y_benign_test = np.zeros(X_benign_test.shape[0])
y_malignant_test = np.ones(X_malignant_test.shape[0])

# Concatenate data and labels
X_train = np.concatenate((X_benign, X_malignant), axis=0)
y_train = np.concatenate((y_benign, y_malignant), axis=0)
X_test = np.concatenate((X_benign_test, X_malignant_test), axis=0)
y_test = np.concatenate((y_benign_test, y_malignant_test), axis=0)

# Shuffle the data
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
y_train = y_train[s]
s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
y_test = y_test[s]

# Display an example image
plt.imshow(X_test[1], interpolation='nearest')
plt.show()

# Normalize pixel values
X_train = X_train/255
X_test = X_test/255

# Further normalize using TensorFlow's normalize function
X_Train = tf.keras.utils.normalize(X_train)
X_Test = tf.keras.utils.normalize(X_test)

# Build a simple convolutional neural network model using TensorFlow
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(128, (3,3), input_shape=X_Train.shape[1:], activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=None))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Set up early stopping
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, restore_best_weights=True)
callbacks = [earlystop]

# Train the model
model.fit(X_Train, y_train, epochs=50, batch_size=50, validation_split=0.1)

# Make predictions on the test set and evaluate accuracy
y_pred = model.predict(X_Test)
yp = []
for i in range(0, 660):
    if y_pred[i][0] >= 0.5:
        yp.append(0)
    else:
        yp.append(1)
print(accuracy_score(y_test, yp))
