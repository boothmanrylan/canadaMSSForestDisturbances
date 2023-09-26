""" Methods to help assess a trained model.
"""

import tensorflow as tf


def build_confusion_matrix(model, parsed_dataset, num_classes, subset=25):
    """ Run dataset through model and return a confusion matrix of the results.

    **NOTE** In remote sensing the standard is to put the True/Reference values
    as the columns and the Predicted/Map values as the rows, but in ML the
    standard is reversed. Because this is a remote sensing first project we
    return the transposed result of tf.math.confusion_matrix and follow the
    remote sensing standard.

    Args:
        model: tf.keras.Model
        parsed_dataset: tf.data.Dataset
        num_classes: int, how many classes are in the dataset
        subset: int, how many batches to use if the dataset repeats infinitely

    Returns:
        Tensor with shape [num_classes, num_classes]
    """
    complete_confusion_matrix = tf.zeros(
        (num_classes, num_classes),
        dtype=tf.int32,
    )

    if parsed_dataset.cardinality() == tf.data.INFINITE_CARDINALITY:
        print(f"Dataset repeats infinitely; taking a subset of {subset} batches")
        parsed_dataset = parsed_dataset.take(subset)

    for x, y in parsed_dataset:
        y_prime = model(x)

        current_confusion_matrix = tf.math.confusion_matrix(
            labels=tf.reshape(tf.argmax(y, -1), [-1]),
            predictions=tf.reshape(tf.argmax(y_prime, -1), [-1]),
            num_classes=num_classes,
        )

        complete_confusion_matrix += current_confusion_matrix

    # follow the remote sensing standard of true values as columns
    return tf.transpose(complete_confusion_matrix)
