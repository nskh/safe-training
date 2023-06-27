import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from safe_train import propagate_interval
from interval import interval, inf


class SafeModel(keras.Model):
    def __init__(self, inputs, outputs, input_interval, desired_interval):
        super().__init__(inputs, outputs)
        self.input_interval = input_interval
        self.desired_interval = desired_interval

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # TODO fix hardcoding of intervals somehow
        output_interval, _ = propagate_interval(self.input_interval, self, graph=True)
        if type(output_interval) is list:
            if len(output_interval) == 1:
                output_interval = output_interval[0]
            else:
                raise NotImplementedError("Output interval was interval of length > 1")
        if output_interval not in self.desired_interval:
            print(f"safe region test FAILED, interval was {output_interval}")
            print(self.layers[-1].weights)
        else:
            print(f"safe region test passed, interval was {output_interval}")

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


def train_safe_single_node_nn(x, y):
    normalizer = layers.Normalization(
        input_shape=[
            1,
        ],
        axis=None,
    )
    normalizer.adapt(x)
    inputs = tf.keras.Input(shape=(1,))
    # input -> normalizer -> dense1 -> dense1
    # outputs = layers.Dense(units=1)(layers.Dense(units=1)(normalizer(inputs)))
    # input -> dense1
    outputs = layers.Dense(units=1)(inputs)
    regression_model = SafeModel(inputs, outputs, interval[20, 40], interval[10, 30])
    regression_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss="mean_squared_error",
        run_eagerly=True,
    )
    history = regression_model.fit(
        x,
        y,
        epochs=100,
        # Suppress logging.
        verbose=0,
        # Calculate validation results on 20% of the training data.
        validation_split=0.2,
    )
    return regression_model, history
