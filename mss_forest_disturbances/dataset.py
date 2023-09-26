""" Methods to build TFRecord datasets.
"""

import tensorflow as tf


RNG = tf.random.Generator.from_seed(42, alg="philox")
AUGMENTATION = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2, "reflect"),
    ]
)


def parse(
    example,
    size,
    bands,
    label,
    num_classes,
    integer_metadata=None,
    float_metadata=None,
    label_smoothing_matrix=None,
):
    """Parse a tfrecord example.

    Args:
        example: Example
        size: int, the height/width of the image in the example
        bands: list[str], the band names to create x with
        label: str, the name of the label contained in the example
        num_classes: int, how many possible classes there are
        integer_metadata: list[str], list of names of integer metadata features
        float_metadata: list[str], list of names of float metadata features
        label_smoothing_matrix: tf.Tensor, optional, used to apply label
            smoothing so that some types of errors are penalized more/less than
            others

    Returns:
        x: Tensor of inputs, (a tuple if metadata are output)
        y: Tensor of labels (one hot encoded)
    """
    image_features = {
        b: tf.io.FixedLenFeature(shape=(size, size), dtype=tf.float32) for b in bands
    }
    label_features = {label: tf.io.FixedLenFeature(shape=(size, size), dtype=tf.int64)}

    x = tf.io.parse_single_example(example, image_features)
    x = tf.stack([x[b] for b in bands], axis=-1)

    y = tf.io.parse_single_example(example, label_features)[label]

    if label_smoothing_matrix is not None:
        y = tf.reshape(
            tf.gather(label_smoothing_matrix, tf.reshape(y, (-1,))),
            (size, size, num_classes),
        )
    else:
        y = tf.one_hot(y, num_classes)

    if integer_metadata is not None or float_metadata is not None:
        # if one is given, the other must be given as well
        assert integer_metadata is not None and float_metadata is not None
        metadata_features = {
            m: tf.io.FixedLenFeature(shape=1, dtype=tf.float32)
            for m in integer_metadata + float_metadata
        }
        metadata_inputs = tf.io.parse_single_example(example, metadata_features)
        integer_metadata_inputs = [
            tf.cast(metadata_inputs[m], tf.int64) for m in integer_metadata
        ]
        float_metadata_inputs = [metadata_inputs[m] for m in float_metadata]
        metadata_inputs = integer_metadata_inputs + float_metadata_inputs
        x = (x, *metadata_inputs)

    return x, y


def non_overlapping_crop(x, y, size, include_metadata=False):
    """Crops x and y into a grid of patches with shape (size, size).

    Args:
        x: Tensor or tuple if include_metadata is True
        y: Tensor
        size: int, the size of patches to crop x and y into
        include_metadata: bool, if True indicates that x contains an image as
            well as metadata.

    Returns:
        tf.data.Dataset containing each of the patches from x and y
    """
    initial_size = x.shape[0]

    def crop(tensor):
        """based on https://stackoverflow.com/a/31530106"""
        tensor = tf.reshape(
            tensor, (initial_size // size, size, initial_size // size, size, -1)
        )
        cropped = tf.experimental.numpy.swapaxes(tensor, 1, 2)

        num_blocks = (initial_size // size) ** 2
        cropped = tf.reshape(cropped, (num_blocks, size, size, -1))

        return tf.data.Dataset.from_tensor_slices(cropped)

    if include_metadata:
        metadata = [tf.data.Dataset.from_tensor_slices(m).repeat() for m in x[1:]]
        x = x[0]

    x = crop(x)
    y = crop(y)

    if include_metadata:
        x = tf.data.Dataset.zip((x, *metadata))

    return tf.data.Dataset.zip((x, y))


def _apply_fn_to_xy(x, y, fn, include_metadata=False):
    """Helper function to apply an identical random transform to x and y.

    Args:
        x: Tensor
        y: Tensor
        fn: function Tensor -> Tensor, the transform to apply to x and y
        include_metadata: bool, if True indicates that x contains an image as
            well as metadata.

    Returns:
        x, y, the input tensors after apply fn
    """
    if include_metadata:
        metadata = x[1:]
        x = x[0]

    y_shape = tf.shape(y)
    if len(y_shape) == 2:  # add temp channel dimension
        y = tf.reshape(y, y_shape + (1,))
        num_y_bands = 1
    else:
        num_y_bands = y_shape[-1]

    y_type = y.dtype
    desired_type = x.dtype
    y = tf.cast(y, desired_type)

    xy = tf.concat([x, y], -1)

    xy = fn(xy)

    x = xy[:, :, :-num_y_bands]
    y = tf.squeeze(tf.cast(xy[:, :, -num_y_bands:], y_type))

    if include_metadata:
        x = (x, *metadata)

    return x, y


def random_crop(x, y, size, seed, include_metadata=False):
    """Applies the same random crop to x and y.

    Args:
        x: Tensor
        y: Tensor
        size: int, the height/width to crop x and y to.
        seed: (int, int) random seed for crop
        include_metadata: bool, if True indicates that x contains an image as
            well as metadata.

    Returns:
        x, y the input tensor after applying the random crop
    """
    y_shape = tf.shape(y)
    if len(y_shape) == 2:  # add temp channel dimension
        y = tf.reshape(y, y_shape + (1,))
        num_y_bands = 1
    else:
        num_y_bands = y_shape[-1]

    if include_metadata:
        num_x_bands = tf.shape(x[0])[-1]
    else:
        num_x_bands = tf.shape(x)[-1]

    target_shape = (size, size, num_x_bands + num_y_bands)

    def fn(xy):
        return tf.image.stateless_random_crop(xy, target_shape, seed=seed)

    return _apply_fn_to_xy(x, y, fn, include_metadata=include_metadata)


def random_crop_wrapper(x, y, size, include_metadata=False):
    """Generates random seed when called, calls random_crop with the seed

    See random_crop()

    Args:
        x: Tensor
        y: Tensor
        size: int, the height/width to crop x and y to.
        include_metadata: bool, if True indicates that x contains an image as
            well as metadata.

    Returns:
        x, y, the input tensors after applying random_crop
    """
    seed = RNG.make_seeds(2)[0]
    x, y = random_crop(x, y, size, seed, include_metadata=include_metadata)


def apply_data_augmentation(x, y, include_metadata=False):
    """Applies the same random data augmentation to x and y.

    Args:
        x: Tensor
        y: Tensor
        include_metadata: bool, if True indicates that x contains an image as
            well as metadata.
    """

    def fn(xy):
        return AUGMENTATION(xy, training=True)

    return _apply_fn_to_xy(x, y, fn, include_metadata=include_metadata)


def build_dataset(
    tfrecord_pattern,
    parse_options,
    training=True,
    data_augmentation=True,
    cycle_length=30,
    block_length=8,
    shuffle_buffer=100,
    batch_size=32,
    normalization_subset_size=100,
):
    """Builds a TFRecord dataset from files indicated by tfrecord_pattern.

    Args:
        tfrecord_pattern: str, unix style file pattern
        parse_options: dict, passed to parse, see parse for explanation
        training: bool, if True apply training only transforms to the dataset,
            e.g., shuffle, data augmentation, and repeat
        data_augmentation: bool, if True and training is also True, apply data
            augmentation
        cycle_length: int, passed to interleave
        block_length: int, passed to interleave
        shuffle_buffer: int, size of buffer to use when shuffling
        batch_size: int, batch size of dataset
        normalization_subset_size: int, if training is True a random subset of
            the dataset with this many elements is also returned to be used to
            adapt a normalization layer in the model.

    Returns:
        tf.data.Dataset
    """
    include_metadata = parse_options["integer_metadata"] is not None
    size = parse_options["size"]

    def interleave_fn(filename):
        raw_dataset = tf.data.TFRecordDataset(filename, compression_type="GZIP")
        parsed_dataset = raw_dataset.map(
            lambda x: parse(x, **parse_options), num_parallel_calls=1
        )
        return parsed_dataset

    tfrecords = tf.data.Dataset.list_files(tfrecord_pattern, shuffle=training)
    parsed_dataset = tfrecords.interleave(
        interleave_fn,
        cycle_length=cycle_length,
        block_length=block_length,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not training,
    )

    parsed_dataset = parsed_dataset.cache()

    if training:
        parsed_dataset = parsed_dataset.shuffle(shuffle_buffer)
        parsed_dataset = parsed_dataset.map(
            lambda x, y: random_crop_wrapper(x, y, size, include_metadata),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        subset = []
        for x, y in parsed_dataset.take(normalization_subset_size):
            if include_metadata:
                subset.append(x[0])
            else:
                subset.append(x)
        subset = tf.concat(subset, axis=0)

        if data_augmentation:
            parsed_dataset = parsed_dataset.map(
                lambda x, y: apply_data_augmentation(x, y, include_metadata)
            )

        parsed_dataset = parsed_dataset.repeat()
    else:
        parsed_dataset = parsed_dataset.flat_map(
            lambda x, y: non_overlapping_crop(x, y, size, include_metadata)
        )

    parsed_dataset = parsed_dataset.batch(batch_size)
    parsed_dataset = parsed_dataset.prefetch(tf.data.AUTOTUNE)

    if training:
        return parsed_dataset, subset
    else:
        return parsed_dataset
