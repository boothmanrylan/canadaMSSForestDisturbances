""" Spatio Temporal UNet model.
"""

import tensorflow as tf


class TemporalFusion(tf.keras.layers.Layer):
    """Change detection layer.

    Based on Late Fusion from Matetto et al. 2021 10.1109/LGRS.2020.298407
    """

    def __init__(self, filters, **kwargs):
        """
        filters: int, the number of filters in the Conv2D layer
        **kwargs: dict, passed to super().__init
        """
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            padding="same",
            activation="relu",
        )

    def call(self, input1, input2):
        """Forward pass.

        Args:
            input1: 4D tensor
            input2: 4D tensor

        Returns:
            4D tensor, the inputs after being concatenated and passed through a
            Conv2D layer.
        """
        x = tf.concat([input1, input2], -1)
        x = self.conv(x)
        return x


class DownSample(tf.keras.layers.Layer):
    """Down sample layer used in UNet."""

    def __init__(self, filters, kernel_size, dilation_rate, **kwargs):
        super().__init__(**kwargs)
        """
        filters: int, passed to SeparableConv2D
        kernel_size: int, passed to SeparableConv2D
        dilation_rate: int, passed to SeparabelConv2D
        **kwargs: dict, passed to super().__init__
        """
        self.separable_conv2d = tf.keras.layers.SeparableConv2D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="same",
            activation="relu",
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, x):
        """Forward pass.

        Args:
            x: 4D tensor

        Returns:
            4D tensor, the input after being passed through a SeparableConv2D layer
            and batch normalization.
        """
        x = self.separable_conv2d(x)
        x = self.batch_norm(x)
        return x


class UpSample(tf.keras.layers.Layer):
    """Up sample layer used in UNet."""

    def __init__(self, filters, kernel_size, **kwargs):
        """
        filters: int, passed to Conv2DTranspose
        kernel_size: int, passed to Conv2DTranspose
        **kwargs: dict, passed to super().__init__
        """
        super().__init__(**kwargs)
        self.transposed_conv2d = tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, x):
        """Forward pass.

        Args:
            x: 4D tensor

        Returns:
            4D tensor, the input after being passed to through a Conv2DTranspose
            layer and batch normalization.
        """
        x = self.transposed_conv2d(x)
        x = self.batch_norm(x)
        return x


class MetadataBias(tf.keras.layers.Layer):
    """Layer to include scalar metadata in a fully convolutional network.

    Based on LSENet from Xie, Guo, and Dong 2022 10.1109/TGRS.2022.3176635
    """

    def __init__(
        self, num_int_inputs, max_int_inputs, num_float_inputs, num_outputs, **kwargs
    ):
        """
        num_int_inputs: int, how many integer metadata inputs there are.
        max_int_inputs: list[int], the max value of each integer input.
        num_float_inputs: int, how many float metadata inputs there are.
        num_outputs: int, dimensionality of the output space.
        **kwargs: dict, passed to super().__init__
        """
        super().__init__(**kwargs)

        message = "Must pass a max value for each integer metadata input."
        assert len(max_int_inputs) == num_int_inputs, message

        self.num_int_inputs = num_int_inputs
        self.num_inputs = num_float_inputs + num_int_inputs
        self.num_outputs = num_outputs

        integer_embeddors = [
            tf.keras.layers.Embedding(x, self.num_outputs // self.num_inputs)
            for x in max_int_inputs
        ]

        float_embeddors = [
            tf.keras.layers.Dense(self.num_outputs // self.num_inputs)
            for _ in range(num_float_inputs)
        ]
        self.embeddors = integer_embeddors + float_embeddors

        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(self.num_outputs // self.num_inputs)
        self.dense2 = tf.keras.layers.Dense(self.num_outputs)

    def call(self, x, *metadata):
        """Forward pass.

        Embeds each metadata value individual using a TF Embedding layer for
        integer values and a Dense layer for float values. Concatenates the
        embedded layers with a global average pooling of the input image,
        passing the result through a dense layer and then adding it back to the
        input.

        The input metadata **MUST** be ordered with all integer values first in
        the same order as max_int_inputs was given to __init__ followed by all
        float values.

        TODO: find a better way to do this so that we do not have to assume
        that the ordering has remained consistent (named inputs?)

        Args:
            x: 4D tensor
            *metadata: list of scalar tensors

        Returns:
            tensor, x + the metadata bias.
        """
        msg = f"Must pass {self.num_inputs} metadata values got {len(metadata)}"
        assert len(metadata) == self.num_inputs, msg

        embedded = []
        for i, val in enumerate(metadata):
            if i < self.num_int_inputs:
                embedded.append(self.embeddors[i](val)[:, 0])
            else:
                embedded.append(self.embeddors[i](val))

        pooled_x = self.pool(x)
        pooled_x = self.dense1(pooled_x)

        metadata = tf.concat([pooled_x, *embedded], axis=-1)
        metadata = self.dense2(metadata)
        metadata = tf.reshape(metadata, (-1, 1, 1, self.num_outputs))

        return x + metadata


class Preprocessing(tf.keras.layers.Layer):
    """Layer to convert an input features dict of bands into an image.

    Based on:
    https://github.com/google/earthengine-community/blob/master/guides/linked/Earth_Engine_TensorFlow_Vertex_AI.ipynb

    Necessary for hosting a model in Google cloud and accessing it through GEE
    """

    def __init__(self, band_names, metadata=None, **kwargs):
        """
        band_names: list[str], the names of all input bands
        metadata: list[str], the names of all input metadata
        kwargs: dict, based to super().__init__
        """
        super().__init__(**kwargs)
        self.band_names = band_names
        self.metadata = metadata

    def call(self, features_dict):
        """Forward pass

        Converts a dict of Tensors with shape (None, 1, 1, 1) to one Tensor
        with shape (None, 1, 1, P)

        Args:
            features_dict: dict

        Returns:
            Tensor or tuple of (Tensor, *metadata) if include_metadata is True
        """
        image = tf.concat(
            [features_dict[b] for b in self.band_names], axis=-1, name="image"
        )

        if self.metadata is not None:
            metadata = [features_dict[m] for m in self.metadata]
            return (image, *metadata)
        else:
            return image

    def get_config(self):
        """Gets config from super()"""
        config = super().get_config()
        return config


class WrappedModel(tf.keras.Model):
    """Wraps an input model with a Preprocessing layer.

    Based on:
    https://github.com/google/earthengine-community/blob/master/guides/linked/Earth_Engine_TensorFlow_Vertex_AI.ipynb

    Necessary for hosting a model in Google cloud and accessing it through GEE
    """

    def __init__(self, model, band_names, metadata, **kwargs):
        """
        model: tf.keras.Model
        band_names: list[str], the names of all input bands
        metadata: list[str], the names of all input metadata
        kwargs: dict, based to super().__init__
        """
        super().__init__(**kwargs)
        self.preprocessing = Preprocessing(band_names, metadata)
        self.model = model

    def call(self, features_dict):
        """Passes features dict through Preprocessing then through model.

        Args:
            features_dict: dict

        Returns:
            same as the output of the given model
        """
        x = self.preprocessing(features_dict)
        return self.model(x)

    def get_config(self):
        """Gets config from super()"""
        config = super().get_config()
        return config


class DeSerializeInput(tf.keras.layers.Layer):
    """Decodes base64 input and preps it for input to a model.

    Based on:
    https://github.com/google/earthengine-community/blob/master/guides/linked/Earth_Engine_TensorFlow_Vertex_AI.ipynb

    Necessary for hosting a model in Google cloud and accessing it through GEE
    """

    def __init__(self, band_names, integer_metadata, float_metadata, **kwargs):
        """
        band_names: list[str], the name of all input bands
        integer_metadata: list[str], list of all names of scalar integer inputs
        float_metadata: list[str], list of all names of scalar float inputs
        kwargs: dict, passed to super().__init__
        """
        super().__init__(**kwargs)
        self.all_inputs = band_names + integer_metadata + float_metadata
        self.integer_metadata = integer_metadata

    def call(self, inputs_dict):
        """Converts input base64 string to dict mapping band names -> Tensor.

        Args:
            inputs_dict: dictionary mapping input names to encoded tf.strings

        Returns:
            dict mapping input names to float32 Tensors
        """

        def get_output_type(key):
            if key in self.integer_metadata:
                return tf.int64
            else:
                return tf.float32

        serialized_dict = {
            k: tf.map_fn(
                lambda x: tf.io.parse_tensor(x, tf.float32),
                tf.io.decode_base64(v),
                fn_output_signature=get_output_type(k),
            )
            for (k, v) in inputs_dict.items()
            if k in self.all_inputs
        }
        return serialized_dict

    def get_config(self):
        """Gets config from super()"""
        config = super().get_config()
        return config


class ReSerializeOutput(tf.keras.layers.Layer):
    """Encode model output as base64 and preps it to be interpreted by GEE.
    Based on:
    https://github.com/google/earthengine-community/blob/master/guides/linked/Earth_Engine_TensorFlow_Vertex_AI.ipynb

    Necessary for hosting a model in Google cloud and accessing it through GEE
    """

    def __init__(self, **kwargs):
        """
        kwargs: dict, passed to supe().__init__
        """
        super().__init__(**kwargs)

    def call(self, output_tensor):
        """Encodes modle output as base64 string.

        Args:
            output_tensor: Tensor, output from model.

        Returns:
            The input tensor encoded as a tf.string
        """
        return tf.map_fn(
            lambda x: tf.io.encode_base64(tf.io.serialize_tensor(x)),
            output_tensor,
            fn_output_signature=tf.string,
        )

    def get_config(self):
        """Gets config from super()"""
        config = super().get_config()
        return config


def build_model(
    input_shape,
    filters,
    kernels,
    dilation_rates,
    normalization_subset,
    upsample_filters=3,
    metadata_filters=32,
    output_kernel=3,
    num_outputs=10,
    two_downstacks=True,
    include_metadata=True,
    first_downstack_inputs=None,
    integer_metadata=None,
    max_integer_metadata_values=None,
    float_metadata=None,
):
    """Builds and returns a Spatio Temporal Unet Model.

    Does not compile the model.

    Args:
        input_shape: (int, int, int) height, width, and number of channels of
            an input patch.
        filters: list[int], number of filters to use in each DownSample layer.
        kernels: list[int], kernel size to use in each DownSample layer.
        diltation_rates list[int], dilation rate to use in each DownSample
            layer.
        normalization_subset: tf.data.Dataset, dataset to adapt a Normalization
            layer with at the beginning of the model.
        upsample_filters; int, number of filters to use in each UpSample layer.
        metadata_filters: int, number of filters to use in the MetadatBias
            layer, only used if include_metadata is True.
        output_kernel: int, kernel size of final Conv2D layer.
        two_downstacks: bool, if True split the input and use two downstacks,
            useful if we are including the current image alongside historical data
            as inputs.
        include_metadata: bool, if True include scalar metadata as inputs to
            the model.
        first_downstack_inputs: int, the cutoff point to determine how many of
            the input channels should be given to the first downstack and how many
            should be given to the second downstack, must be set if two_downstacks
            is True.
        integer_metadata: list[str], optional, the names of all integer
            metadata values to be included, must be given if include_metadata
            is True.
        max_integer_metadata_values: list[int], optional, the maximum value
            that each of the integer metadata inputs can take, must be given if
            include_metadata is True.
        float_metadata: list[str], optional, the names of all float metadata
            values to be included, must be given if include_metadata is True.

    Returns:
        tf.keras.Model
    """
    normalizer = tf.keras.layers.Normalization()
    normalizer.adapt(normalization_subset)

    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = normalizer(input_layer)

    model_config = list(zip(filters, kernels, dilation_rates))
    skips = []

    if two_downstacks:
        x1 = x[:, :, :, :first_downstack_inputs]
        x2 = x[:, :, :, first_downstack_inputs:]

        downstack1 = [DownSample(*config) for config in model_config]
        downstack2 = [DownSample(*config) for config in model_config]

        for i, (down1, down2) in enumerate(zip(downstack1, downstack2)):
            x1 = down1(x1)
            x2 = down2(x2)
            x = TemporalFusion(filters[i])(x1, x2)
            skips.append(x)
    else:
        downstack = [DownSample(*config) for config in model_config]

        for i, down in enumerate(downstack):
            x = down(x)
            skips.append(x)

    if include_metadata:
        integer_metadata_inputs = [
            tf.keras.layers.Input(shape=1, dtype=tf.int64, name=m)
            for m in integer_metadata
        ]
        float_metadata_inputs = [
            tf.keras.layers.Input(shape=1, dtype=tf.float32, name=m)
            for m in float_metadata
        ]
        metadata_inputs = integer_metadata_inputs + float_metadata_inputs

        metadata_bias = MetadataBias(
            len(integer_metadata),
            max_integer_metadata_values,
            len(float_metadata),
            filters[-1],
        )

        x = metadata_bias(x, *metadata_inputs)

    skips = reversed(skips[:-1])

    upstack = [UpSample(f, upsample_filters) for f in reversed(filters)]

    for up, skip in zip(upstack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = tf.keras.layers.Conv2DTranspose(
        num_outputs,
        kernel_size=output_kernel,
        padding="same",
        activation="softmax",
    )(x)

    if include_metadata:
        inputs = [input_layer, *metadata_inputs]
    else:
        inputs = input_layer

    model = tf.keras.Model(inputs, x)
    return model


def train_model(
    model,
    dataset,
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    steps=50,
    epochs=20,
    checkpoint_path=None,
    class_weight=None,
):
    """Compiles and trains a model on the given dataset.

    Args:
        model: tf.keras.Model
        dataset: tf.data.Dataset, dataset the model is trained on
        loss: tf.keras.losses.Loss, used to compile the model
        optimizer: tf.keras.optimizers.Optimizer, used to compile the model
        steps: int, how many steps per epoch to train the dataset
        epochs: int, how many epochs to train the dataset
        checkpoint_path: str, if given, path where the Model weights are
            checkpointed during training.
        class_weight: dict(int -> float), optional if given passed to fit to
            weight classes differently in the loss function (for handling class
            imbalance e.g.)

    Returns:
        tf.keras.callbacks.History, the History returned by calling fit on the
        model.
    """
    if checkpoint_path is not None:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
        )
        callbacks = [checkpoint]
    else:
        callbacks = None

    model.compile(loss=loss, optimizer=optimizer)

    history = model.fit(
        dataset,
        steps_per_epoch=steps,
        epochs=epochs,
        callbacks=callbacks,
    )

    return history


def prepare_model_for_hosting(
    model,
    band_names,
    integer_metadata,
    float_metadata,
    model_path,
    checkpoint_path=None,
):
    """Prepares a model to be hosted with VertexAI and accessed through GEE.

    Args:
        model: tf.keras.Model, e.g., output from build_model()
        band_names: list[str], the name of all input bands
        integer_metadata: list[str], list of all names of scalar integer inputs
        float_metadata: list[str], list of all names of scalar float inputs
        model_path: str, location to store the full model after wrapping in
            preprocessing/de/serialization layers.
        checkpoint_path: str, if given model weights are loaded from here.

    Returns:
        None
    """
    if checkpoint_path is not None:
        model.load_weights(checkpoint_path)

    deserializer = DeSerializeInput(
        band_names,
        integer_metadata,
        float_metadata,
    )
    wrapped_model = WrappedModel(
        model,
        band_names,
        integer_metadata + float_metadata,
    )
    reserializer = ReSerializeOutput()

    serialized_inputs = {
        x: tf.keras.Input(shape=[], dtype="string", name=x)
        for x in band_names + integer_metadata + float_metadata
    }

    deserialized_inputs = deserializer(serialized_inputs)
    model_output = wrapped_model(deserialized_inputs)
    serialized_output = reserializer(model_output)

    prepared_model = tf.keras.model(serialized_inputs, serialized_output)

    prepared_model.save(model_path)
