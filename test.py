import math

import numpy as np
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model

# Load the model
from keras.preprocessing.image import ImageDataGenerator

from params import img_height, img_width, batch_size

model = load_model('model.h5')

test_dir = 'test_photos'

# Set up the test data generator
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)
test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print('Test accuracy:', test_acc)

# Use the model to predict the classes of the test images
test_steps = math.ceil(test_generator.samples / batch_size)
preds = model.predict(test_generator, steps=test_steps)

# Get the predicted classes
predicted_classes = np.argmax(preds, axis=1)

# Get the true classes
true_classes = test_generator.classes

# Print the classification report
from sklearn.metrics import classification_report

print(classification_report(true_classes, predicted_classes, target_names=test_generator.class_indices.keys()))

# Get the filenames from the generator
filenames = test_generator.filenames

# Print the predicted classes for each file
for i, filename in enumerate(filenames):
    pred_class = np.argmax(preds[i])
    pred_score = preds[i][pred_class]
    print(f"{filename} --> Predicted class: {pred_class}, Predicted score: {pred_score}")