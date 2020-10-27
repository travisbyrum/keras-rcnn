#!/usr/bin/env python

"""
Created October 20, 2020

@author: Travis Byrum
"""

import argparse
import json
import math
import sys
from os import path

import keras_rcnn.datasets as ds
import numpy
from keras_rcnn.models import RCNN
from keras_rcnn.preprocessing import ObjectDetectionGenerator
from tensorflow.keras import backend
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam


def schedule(epoch_index):
    """Schedule learning rate."""

    return 0.1 * numpy.power(0.5, numpy.floor((1 + epoch_index) / 1.0))


def write_json(filename: str, data: dict):
    """Write json to file."""

    with open(filename, "w") as json_file:
        json.dump(data, json_file)


def read_json(filename: str):
    """Read json from file."""

    data = None

    with open(filename) as json_file:
        data = json.load(json_file)

    return data


def write_model(directory: str, model: Any, history):
    """Write model and training history to file."""

    model_json = model.to_json()

    with open(os.path.join(directory, "model.json"), "w") as json_file:
        json_file.write(model_json)

    model.save_weights(os.path.join(directory, "model.h5"))

    with open(os.path.join(directory, "history.json"), "w") as f:
        json.dump(history.history, f)


def main():
    """Entrypoint for training execution."""

    parser = argparse.ArgumentParser(description="Train image detection model")
    parser.add_argument(
        "--input", type=str, help="Input image dataset", default="malaria_phenotypes"
    )
    parser.add_argument("--target", type=int, help="Target image size", default=224)
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs", default=1
    )
    parser.add_argument(
        "--learning-rate", type=float, help="Model learning rate", default=0.1
    )
    parser.add_argument(
        "--data-dir", type=str, help="Data directory for model output", default="data"
    )

    args = parser.parse_args()

    training_dictionary, test_dictionary = ds.load_data(args.input)

    categories = {
        "red blood cell": 1,
        "schizont": 2,
        "difficult": 3,
        "leukocyte": 4,
        "ring": 5,
        "trophozoite": 6,
        "gametocyte": 7,
    }

    generator = ObjectDetectionGenerator()
    generator = generator.flow_from_dictionary(
        dictionary=training_dictionary,
        categories=categories,
        target_size=(args.target, args.target),
    )

    validation_data = ObjectDetectionGenerator()
    validation_data = validation_data.flow_from_dictionary(
        dictionary=test_dictionary,
        categories=categories,
        target_size=(args.target, args.target),
    )

    backend.set_learning_phase(1)

    model = RCNN(
        categories=categories.keys(),
        dense_units=512,
        input_shape=(args.target, args.target),
    )

    optimizer = Adam(args.learning_rate)
    model.compile(optimizer)

    history = model.fit_generator(
        epochs=args.epochs,
        generator=generator,
        validation_data=validation_data,
        callbacks=[LearningRateScheduler(schedule)],
    )

    write_model(args.data_dir, model, history)

    return model.summary()


if __name__ == "__main__":
    main()
