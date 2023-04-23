# Set the paths to the training, validation, and test data
import os
import random

import numpy as np
from keras import Model
from keras.applications.inception_v3 import preprocess_input, InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from params import img_width, img_height, batch_size
from preprocess import prepare_image_file

image_dir = 'photos'
train_dir = 'data/train'
val_dir = 'data/val'
test_dir = 'data/test'

# Set the percentage of images to use for training, validation, and testing
train_pct = 0.8
val_pct = 0.1
test_pct = 0.1
num_epochs = 10


# Get a list of all image files in the directory and its subdirectories
image_files = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith('.jpg'):
            image_files.append(os.path.join(root, file))

# Shuffle the list of image files
random.shuffle(image_files)

# Split the image files into training, validation, and test sets
train_files, test_val_files = train_test_split(image_files, test_size=val_pct + test_pct)
val_files, test_files = train_test_split(test_val_files, test_size=test_pct / (test_pct + val_pct))

# Create temporary directories for the training, validation, and test sets
if not os.path.exists(train_dir):
    # Copy the training files to the training directory
    for file in train_files:
        prepare_image_file(file, img_width, img_height, train_dir)
        print(f"prepared {file} for train")
if not os.path.exists(val_dir):
    # Copy the validation files to the validation directory
    for file in val_files:
        prepare_image_file(file, img_width, img_height, val_dir)
        print(f"prepared {file} for val")
if not os.path.exists(test_dir):
    # Copy the test files to the test directory
    for file in test_files:
        prepare_image_file(file, img_width, img_height, test_dir)
        print(f"prepared {file} for test")

# Define the data generators
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load the pre-trained Inception v3 model
base_model = InceptionV3(weights='imagenet', include_top=False)

# Add a global average pooling layer and a dense output layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# compute class weights
total_samples = len(train_generator.labels)
class_counts = np.bincount(train_generator.labels)
num_classes = class_counts.size
class_weights = {i: 1 / count * total_samples / num_classes for i, count in enumerate(class_counts)}

# Train the model
history = model.fit(
    train_generator,
    epochs=num_epochs,
    validation_data=val_generator,
    class_weight=class_weights,
)
model.save("model.h5")

# Evaluate the model on the test set
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print(f'Test loss = {test_loss}, test accuracy = {test_acc}')
