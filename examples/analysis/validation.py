#!/usr/bin/env python

"""
Created October 27, 2020

@author: Travis Byrum
"""

import argparse
import json

import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json


def read_model(filename: str):
    """Read json from file."""

    model = None

    with open(filename) as json_file:
        model = model_from_json(json_file.read())

    if model is None:
        raise FileNotFoundError

    return model


def read_json(filename: str):
    """Read json from file."""

    data = None

    with open(filename) as json_file:
        data = json.load(json_file)

    if data is None:
        raise FileNotFoundError

    return data


def main():
    """Entrypoint for validation execution."""

    parser = argparse.ArgumentParser(description="Validate image detection model")
    parser.add_argument(
        "--model", type=str, help="Input image model", default="data/model.json"
    )
    parser.add_argument(
        "--weights", type=str, help="Weights file path", default="data/model.h5"
    )
    parser.add_argument(
        "--history", type=str, help="History file path", default="data/history.json"
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs", default=1
    )
    parser.add_argument(
        "--output-image", type=str, help="Output image file", default="output.png"
    )

    args = parser.parse_args()

    model = read_model(args.model)
    model.load_weights(args.weights)
    history = read_json(args.history)

    acc = history["accuracy"]
    val_acc = history["val_accuracy"]

    loss = history["loss"]
    val_loss = history["val_loss"]

    epochs_range = range(args.epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.savefig(args.output_image)


if __name__ == "__main__":
    main()
