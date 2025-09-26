import os
import json
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE
NUM_CLASSES = 10
IMG_SIZE = 96  # MobileNetV2 default
BATCH_SIZE = 128
EPOCHS_STAGE1 = 2   # train head
EPOCHS_STAGE2 = 4   # fine-tune backbone
MODEL_DIR = "saved_model"
MODEL_FILE = os.path.join(MODEL_DIR, "classifier.keras")


def build_datasets():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    val_count = 5000
    x_val, y_val = x_train[-val_count:], y_train[-val_count:]
    x_train, y_train = x_train[:-val_count], y_train[:-val_count]

    def preprocess(x, y, train=True):
        x = tf.image.resize(x, (IMG_SIZE, IMG_SIZE))
        x = tf.cast(x, tf.float32) / 255.0
        if train:
            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_brightness(x, 0.15)
            x = tf.image.random_contrast(x, 0.8, 1.2)
            x = tf.image.random_saturation(x, 0.8, 1.2)
        return x, tf.squeeze(tf.cast(y, tf.int32), axis=-1)

    def make_ds(x, y, train):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if train:
            ds = ds.shuffle(10000, reshuffle_each_iteration=True)
        ds = ds.map(lambda a, b: preprocess(a, b, train),
                    num_parallel_calls=AUTOTUNE)
        ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
        return ds

    train_ds = make_ds(x_train, y_train, True)
    val_ds   = make_ds(x_val,   y_val,   False)
    test_ds  = make_ds(x_test,  y_test,  False)

    steps_per_epoch = math.ceil(len(x_train) / BATCH_SIZE)
    val_steps = math.ceil(len(x_val) / BATCH_SIZE)
    print(f"[INFO] train samples={len(x_train)}, val samples={len(x_val)}, test samples={len(x_test)}")
    print(f"[INFO] steps_per_epoch={steps_per_epoch}, val_steps={val_steps}")

    return train_ds, val_ds, test_ds, steps_per_epoch, val_steps


def build_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    base = keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", pooling="avg"
    )
    x = base(x, training=False)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    base.trainable = False
    return model, base


def fine_tune(model, base):
    base.trainable = True
    fine_tune_at = int(len(base.layers) * 0.9)   # unfreeze t*
