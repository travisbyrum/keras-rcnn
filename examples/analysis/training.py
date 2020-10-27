#!/usr/bin/env python

"""
Created October 20, 2020

@author: Travis Byrum
"""

import argparse
import json
import sys
from os import path

import keras
import keras_rcnn.datasets as ds
from keras_rcnn.models import RCNN
from keras_rcnn.preprocessing import ObjectDetectionGenerator


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


def main():
    """Entrypoint for training execution."""

    parser = argparse.ArgumentParser(description="Train image detection model")
    parser.add_argument("--input", type=str, help="Input image dataset")
    parser.add_argument("--target", type=int, help="Target image size", default=224)
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs", default=1
    )
    parser.add_argument(
        "--learning-rate", type=float, help="Model learning rate", default=0.1
    )
    parser.add_argument(
        "--training-path",
        type=str,
        help="Path to training json file",
        default="./data/training.json",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        help="Path to test json file",
        default="./data/test.json",
    )

    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_usage()
        sys.exit(1)

    try:
        training_dictionary = read_json(args.training_path)
        test_dictionary = read_json(args.test_path)
    except FileNotFoundError:
        training_dictionary, test_dictionary = ds.load_data(args.input)
        write_json(TRAINING_FILE, training_dictionary)
        write_json(TEST_FILE, test_dictionary)

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

    keras.backend.set_learning_phase(1)

    model = RCNN(
        categories=categories.keys(),
        dense_units=512,
        input_shape=(args.target, args.target, 3),
    )

    # optimizer = keras.optimizers.Adam(args.learning_rate)
    # model.compile(optimizer)

    # # model.fit_generator(
    # #     epochs=args.epoch,
    # #     generator=generator,
    # #     validation_data=validation_data
    # # )


if __name__ == "__main__":
    main()
