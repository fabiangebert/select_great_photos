import os
import shutil

import numpy as np
from keras.models import load_model

from params import img_width, img_height, min_class_thres
from preprocess import prepare_image

model = load_model('model.h5')

# Create a list of file paths for the unlabeled images
image_dir = 'classified_photos'
file_paths = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.lower().endswith('.jpg'):
            file_paths.append(os.path.join(root, file))

# Create a dictionary to map class indices to class names
class_names = ['bad', 'great']
class_dict = {i: name for i, name in enumerate(class_names)}

# Make predictions for each image and move it to a subfolder named after the predicted class
for file_path in file_paths:
    # Preprocess the image
    img = prepare_image(file_path, img_width, img_height)
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Make a prediction
    predictions = model.predict(img)
    class_index = np.argmax(predictions, axis=1)[0]
    class_name = class_dict[class_index]

    if predictions.max() < min_class_thres:
        class_name = 'undecided'

    # Move the image to the corresponding class folder
    class_dir = os.path.join(image_dir, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    file_name = os.path.basename(file_path)
    new_file_path = os.path.join(class_dir, file_name)
    shutil.move(file_path, new_file_path)
