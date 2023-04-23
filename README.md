# How to use this

## Overview

* select some great and some bad photos from your library
* train a classification model using `train.py`
* classify a bunch of photos from your library using the trained model with `classify.py`
* improve trained model

## Setup

```
pip install tensorflow
```

* on macOS you need `tensorflow-macos`

## Selecting photos for training

* copy randomly selected photos from your library to the `photos` folder
```shell
find /path/to/photo/library -type f -iname "*.jpg" -print0 | shuf -z -n 100 | xargs -0 -I{} cp -v {} ./photos
```
* manually move great ones to the `./photos/great` folder
* manually move bad ones to the `./photos/bad` folder

## Train model

* make sure you have at least 100 photos in the `photos` folder (the more, the merrier)
* launch training
```shell
python3 train.py
```
* model will be saved to `model.h5`

## Classify photos

* make sure you have some photos in the `classified_photos` folder

```shell
find /path/to/photo/library -type f -iname "*.jpg" -print0 | shuf -z -n 100 | xargs -0 -I{} cp -v {} ./classified_photos
```

* launch classification
```shell
python3 classify.py
```

* verify if photos in `classified_photos/bad` folder are actually bad and `classified_photos/great` photos are actually great.
* incorrectly classified photos should be added as photos for training to the respective folder in `photos` 

# Contributing

* looking forward to your PR!

# Troubleshooting

* on macOS, you should use macOS built-in Python 3 and not the brew version for tensorflow to work