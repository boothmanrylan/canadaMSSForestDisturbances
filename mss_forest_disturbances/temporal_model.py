import tensorflow as tf


class RecurrentBlock(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().___init__(**kwargs)
        self.lstm1 = tf.keras.layers.LSTM(units, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(units, return_sequences=True)
        self.lstm3 = tf.keras.layers.LSTM(
            units, return_sequences=False, return_state=True
        )

    def call(self, x, initial_state=None):
        initial_state = [None] * 3 if initial_state is None else initial_state
        x, state1 = self.lstm1(x, initial_state[0])
        x, state2 = self.lstm2(x, initial_state[1])
        x, state3 = self.lstm3(x, initial_state[2])
        return x, [state1, state2, state3]


def build_temporal_model(units, num_inputs, num_outputs):
    lookback_input = tf.keras.layers.Input(shape=(None, num_inputs))
    target_input = tf.keras.layers.Input(shape=(None, num_inputs))
    lookahead_input = tf.keras.layers.Input(shape=(None, num_inputs))

    lookback, state = RecurrentBlock(units)(lookback_input)
    target, state = RecurrentBlock(units)(target_input, initial_state=state)
    lookahead, _ = RecurrentBlock(units)(lookahead_input, initial_state=state)

    x = tf.concat([lookback, target, lookahead])
    x = tf.Dense(num_outputs, activation="softmax" if num_outputs > 1 else "sigmoid")(x)

    model = tf.keras.Model(
        inputs=[lookback_input, target_input, lookahead_input], outputs=x
    )
    return model


temporal_model = build_temporal_model(64, 16, 3)
