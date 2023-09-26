""" Methods for visualizing model input/output.
"""

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from . import constants, dataset


CMAP = ListedColormap(constants.CLASS_PALETTE, constants.NUM_CLASSES)
NORM = BoundaryNorm(np.arange(constants.NUM_CLASSES + 1), constants.NUM_CLASSES)


def get_vmin_vmax(arr):
    """Compute reasonable vmin and vmax to display arr.

    Args:
        arr: ndarray, image data

    Returns:
        vmin, vmax, floats that works to pass as vmin and vmax in plt.imshow()
    """
    mean = np.mean(arr)
    std = np.std(arr)
    vmin = mean - (0.5 * std)
    vmax = mean + (0.5 * std)
    return vmin, vmax


def visualize(
    parsed_dataset,
    model=None,
    count=10,
    rgb_indices=None,
    historical_rgb_indices=None,
    vmin=None,
    vmax=None,
):
    """Display model inputs and corresponding outputs.

    Args:
        parsed_dataset: tf.data.Dataset
        model: tf.keras.Model, optional, if given the model input and output
            are both plotted.
        count: how many samples from the dataset to plot.
        rgb_indices: 3-tuple of ints, which bands to display as RGB channels.
            If not given (0, 1, 2) is used as a default.
        historical_rgb_indices: 3-tuple of ints, which bands to display as the
            historical RGB channels. If not given (7, 8, 9) is used as a
            default.
        vmin: float, optional used as vmin in plt.imshow, if not given, a
            reasonable value will be calculated for each image.
        vmax: float, optional used as vmax in plt.imshow, if not given, a
            reasonable value will be calculated for each image.

    Returns:
        None
    """
    if rgb_indices is None:
        rgb_indices = (0, 1, 2)
    if historical_rgb_indices is None:
        historical_rgb_indices = (7, 8, 9)

    num_cols = 3 if model is None else 4
    size = 10
    figsize = (num_cols * size, count * size)
    fig, axes = plt.subplots(count, num_cols, figsize=figsize)

    def plot_row(x, historical_x, y, model_output, index):
        if vmin is None or vmax is None:
            vmin_x, vmax_x = get_vmin_vmax(x)
            vmin_historical_x, vmax_historical_x = get_vmin_vmax(historical_x)
        else:
            vmin_x, vmax_x = vmin, vmax
            vmin_historical_x, vmax_historical_x = vmin, vmax

        axes[index, 0].imshow(
            historical_x,
            vmin=vmin_historical_x,
            vmax=vmax_historical_x,
        )
        axes[index, 1].imshow(x, vmin=vmin_x, vmax=vmax_x)

        y = np.argmax(y, axis=-1)
        axes[index, 2].imshow(y, cmap=CMAP, norm=NORM)

        if model_output is not None:
            model_output = np.squeeze(np.argmax(model_output, axis=-1))

    # unbatch so we dont get groups then rebatch with 1 to avoid errors
    parsed_dataset = parsed_dataset.unbatch().batch(1)
    parsed_dataset = parsed_dataset.shuffle(100).take(count)
    for i, (x, y) in enumerate(parsed_dataset):
        model_output = None if model is None else model(x)

        # get the first/only element from the batch
        x = x[0]
        y = y[0]

        if isinstance(x, (tuple, list)):
            x = x[0]  # drop metadata from input

        x_rgb = tf.gather(x, rgb_indices, axis=-1)
        historical_x_rgb = tf.gather(x, historical_rgb_indices, axis=-1)

        plot_row(x_rgb, historical_x_rgb, y, model_output, i)


def plot_x(x, axes, i, j, rgb_indices):
    """Helper function for crop_visualizer and data_augmentation_visualizer"""
    x = tf.gather(x, rgb_indices, axis=-1)
    vmin, vmax = get_vmin_vmax(x)
    axes[i, j].imshow(x, vmin=vmin, vmax=vmax)


def plot_y(y, axes, i, j):
    """Helper function for crop_visualizer and data_augmentation_visualizer"""
    y = np.squeeze(np.argmax(y, axis=-1))
    axes[i, j].imshow(y, cmap=CMAP, norm=NORM)


def crop_visualizer(
    tfrecord_pattern,
    crop_size,
    parse_options,
    count=10,
    random_crop=True,
    rgb_indices=None,
):
    """Method to verify that non_overlapping_crop and random_crop function.

    Args:
        tfrecord_pattern: str, unix style file pattern to create a dataset from
        crop_size: int, the size of patches to crop x and y to
        parse_options: dict, passed to dataset.parse()
        count: int, number of examples to plot
        random_crop: bool, if True visualize random_crop, if False visualize
            non_overlapping_crop
        rgb_indices: 3tuple of ints, indicating which bands to use as the RGB
            channels in the visualization, if not given (0, 1, 2) is used as
            the default.

    Returns:
        None
    """
    if rgb_indices is None:
        rgb_indices = (0, 1, 2)

    # create our own dataset without using dataset.build_dataset() because we
    # don't want it to be already cropped
    files = tf.data.Dataset.list_files(tfrecord_pattern, shuffle=False)
    raw_dataset = tf.data.TFRecordDataset(files, compression_type="GZIP")
    parsed_dataset = raw_dataset.map(lambda x: dataset.parse(x, **parse_options))

    size = 6

    if random_crop:
        num_blocks = 1
        cropped_dataset = parsed_dataset.map(dataset.random_crop_wrapper)
        cropped_dataset = cropped_dataset.take(count)
        fig, axes = plt.subplots(count, 4, figsize=(4 * size, count * size))
    else:
        num_blocks = (parse_options["size"] // crop_size) ** 2
        cropped_dataset = parsed_dataset.flat_map(dataset.non_overlapping_crop)
        cropped_dataset = cropped_dataset.take(num_blocks * count)
        cols = 2 + (2 * num_blocks)
        fig, axes = plt.subplots(count, cols, figsize=(cols * size, count * size))

    parsed_dataset = parsed_dataset.take(count)

    for i, (x, y) in enumerate(parsed_dataset):
        if isinstance(x, (tuple, list)):
            x = x[0]  # drop the metadata from x
        plot_x(x, axes, i, 0, rgb_indices)
        plot_y(y, axes, i, 1 + num_blocks)

    for i, (x, y) in enumerate(cropped_dataset):
        if isinstance(x, (tuple, list)):
            x = x[0]  # drop the metadata from x
        plot_x(x, axes, i // num_blocks, num_blocks + (i % num_blocks), rgb_indices)
        plot_y(y, axes, i // num_blocks, 2 + num_blocks + (i % num_blocks))


def data_augmentation_visualizer(
    tfrecord_pattern,
    parse_options,
    count=10,
    rgb_indices=None,
):
    """Method to verify that data augmentation is working properly.

    Args:
        tfrecord_pattern: str, unix style file pattern to create a dataset from
        parse_options: dict, passed to dataset.parse()
        count: int, number of examples to plot
            non_overlapping_crop
        rgb_indices: 3tuple of ints, indicating which bands to use as the RGB
            channels in the visualization, if not given (0, 1, 2) is used as
            the default.

    Returns:
        None
    """
    if rgb_indices is None:
        rgb_indices = (0, 1, 2)

    # create our own dataset without using dataset.build_dataset() because we
    # don't want it to be already have been augmented
    files = tf.data.Dataset.list_files(tfrecord_pattern, shuffle=False)
    raw_dataset = tf.data.TFRecordDataset(files, compression_type="GZIP")
    parsed_dataset = raw_dataset.map(lambda x: dataset.parse(x, **parse_options))
    parsed_dataset = parsed_dataset.take(count)

    include_metadata = parse_options["float_metadata"] is not None
    augmented_dataset = parsed_dataset.map(
        lambda x, y: dataset.apply_data_augmentation(x, y, include_metadata)
    )

    size = 6

    fig, axes = plt.subplots(count, 4, figsize=(4 * size, count * size))

    for i, (x, y) in enumerate(parsed_dataset):
        if isinstance(x, (tuple, list)):
            x = x[0]  # drop metadata from x
        plot_x(x, axes, i, 0, rgb_indices)
        plot_y(y, axes, i, 1)

    for i, (x, y) in enumerate(augmented_dataset):
        if isinstance(x, (tuple, list)):
            x = x[0]
        plot_x(x, axes, i, 2, rgb_indices)
        plot_y(y, axes, i, 3)
