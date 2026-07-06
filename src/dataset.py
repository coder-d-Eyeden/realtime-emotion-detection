from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

from .config import (
    TRAIN_DIR,
    TEST_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    BATCH_SIZE,
    VALIDATION_SPLIT,
    SEED
)

#Making Datasets

def create_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        shuffle=True,
        seed=SEED,
        validation_split=VALIDATION_SPLIT,
        subset="training"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        shuffle=True,
        seed=SEED,
        validation_split=VALIDATION_SPLIT,
        subset="validation"
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=TEST_DIR,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        shuffle=False
    )

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds

# Data Augmentation Part

def get_data_augmentation():

    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

# Compute Class weights

def compute_class_weights(train_ds):

    labels = []

    for _, batch_labels in train_ds:
        
        labels.extend(batch_labels.numpy())

    labels = np.array(labels)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )

    class_weights = {
        i: float(weight)
        for i, weight in enumerate(weights)
    }

    return class_weights



# Get Class

def get_class_names(train_ds):

    return train_ds.class_names