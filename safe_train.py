import tensorflow as tf
from tensorflow.keras import layers

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy

from interval import interval, inf

# from safe_model import SafeModel


def generate_data(NOISE_STD=2, M=0.5, B=5, xmin=5, xmax=55, n=30):
    x = np.linspace(xmin, xmax, n)
    y_func = lambda x: M * x + B
    y_noisy = lambda x: y_func(x) + np.random.normal(0, NOISE_STD, np.shape(x))
    # y = np.array([5, 20, 14, 32, 22, 38])
    y = y_noisy(x)
    return x, y


def train_single_node_nn(x, y):
    normalizer = layers.Normalization(
        input_shape=[
            1,
        ],
        axis=None,
    )
    normalizer.adapt(x)
    regression_model = tf.keras.Sequential(
        [normalizer, layers.Dense(units=1), layers.Dense(units=1)]
    )
    regression_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss="mean_squared_error"
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


def plot_loss(history):
    plt.figure()
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_predictions(model, xs, ys, xlim=[0, 60], ylim=[0, 60]):
    # neural network values
    y_predict = model.predict(xs)
    # scipy values
    popt, _ = scipy.optimize.curve_fit(lambda x, b0, b1: b0 + b1 * x, xs, ys)
    y_scipy = xs * popt[1] + popt[0]

    plt.figure()
    plt.plot(xs, y_predict)
    plt.scatter(xs, ys, color="C1")
    plt.plot(xs, y_scipy, color="C2")
    plt.legend(["predictions", "data", "scipy"])
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.show()


def propagate_interval(input_interval, model, graph=False):
    # TODO check relu handling for multiple intervals
    # TODO change inputs to lists?

    # Complain if input_interval is not a list
    if type(input_interval) is not list:
        print("Warning! Input interval was not a list")

    num_layers = len(model.layers)
    current_interval = input_interval
    for layer_idx, layer in enumerate(model.layers):
        # print(current_interval)
        if layer_idx == num_layers - 1:
            penultimate_interval = current_interval
        config = layer.get_config()
        if "normalization" in config["name"]:
            # print(f"on normalization layer {layer_idx}")
            if graph:
                norm_mean, norm_var, _ = layer.weights
            else:
                norm_mean, norm_var, _ = layer.get_weights()
            norm_std = np.sqrt(norm_var)
            if type(current_interval) == list:
                num_intervals = len(current_interval)
                if num_intervals == 1:
                    current_interval = [
                        (current_interval[0] - float(norm_mean)) / float(norm_std)
                    ]
                else:
                    assert len(norm_std) == len(current_interval)
                    intervals = [0] * num_intervals
                    for i in range(num_intervals):
                        if current_interval[i] is not None:
                            intervals[i] += (
                                current_interval[i] - norm_mean[i]
                            ) / norm_std[i]
                    current_interval = intervals
            else:
                current_interval = [
                    (current_interval - float(norm_mean)) / float(norm_std)
                ]
        elif "dense" in config["name"]:
            if graph:
                weight, bias = layer.weights
            else:
                weight, bias = layer.get_weights()
            num_combinations = weight.shape[0]
            num_intervals = weight.shape[1]
            # print(
            #     f"on dense layer {layer_idx} of dim ({num_combinations}x{num_intervals})"
            # )
            if num_combinations == 1 and num_intervals == 1:
                if type(current_interval) == list:
                    assert (
                        len(current_interval) == 1
                    ), f"Expected only one interval, got {len(current_interval)}"
                    current_interval = current_interval[0]
                current_interval = [current_interval * float(weight) + float(bias)]
            # elif num_combinations == 1 and num_intervals > 1:
            #     # Make multiple intervals
            #     intervals = []
            #     assert type(current_interval) is list
            #     assert len(current_interval) == 1
            #     for i in range(num_intervals):
            #         intervals.append(current_interval[0] * weight[0, i] + bias[i])
            #     current_interval = intervals
            #     assert (
            #         type(current_interval) is list
            #     ), "Current interval was not type list"
            #     assert (
            #         len(current_interval) == num_intervals
            #     ), "Length of intervals was wrong"
            # elif num_combinations > 1 and num_intervals == 1:
            #     assert (
            #         type(current_interval) == list
            #     ), "Current interval was not type list"
            #     # start at 0
            #     interval = 0
            #     for i in range(num_combinations):
            #         interval += current_interval[i] * weight[i, 0]
            #     interval += bias
            #     current_interval = interval
            else:
                intervals = [0] * num_intervals
                for i in range(num_combinations):
                    # print(f"comb {i}")
                    for j in range(num_intervals):
                        # print(f"interval {j}")
                        if current_interval[i] is not None:
                            # print(
                            #     f"current interval is {current_interval[i]} with type {type(current_interval[i])}"
                            # )
                            # print(f"this weight is {weight[i, j]}")
                            intervals[j] += current_interval[i] * float(weight[i, j])
                for j in range(num_intervals):
                    intervals[j] += float(bias[j])
                current_interval = intervals
            if config["activation"] == "relu":
                # can't do a for-in here since that does a copy
                for interval_idx in range(len(current_interval)):
                    current_interval[interval_idx] &= interval[0, inf]
                    if current_interval[interval_idx] == interval():
                        current_interval[interval_idx] = interval[0, 0]
            elif config["activation"] == "linear":
                # Do nothing, just pass interval through
                pass
            else:
                raise NotImplementedError(
                    f"Activation type {config['activation']} is not handled"
                )
        elif "input" in config["name"]:
            # Do nothing, just pass interval through
            pass
        else:
            raise NotImplementedError(f"Layer type {config['name']} is not handled")
    return current_interval, penultimate_interval


def plot_intervals(
    input_interval,
    output_interval,
    xs=None,
    ys=None,
    y_predict=None,
    y_scipy=None,
    xlim=[0, 60],
    ylim=[0, 60],
):
    fig = plt.figure()
    ax = fig.gca()

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if xs is not None:
        legend = []
        if ys is not None:
            plt.scatter(xs, ys, color="C2")
            legend.append("data")
        if y_predict is not None:
            plt.plot(xs, y_predict)
            legend.append("NN")
        if y_scipy is not None:
            plt.plot(xs, y_scipy, color="C1")
            legend.append("OLS")
        plt.legend(legend)

    input_width = input_interval[0].sup - input_interval[0].inf
    if type(output_interval[0]) == interval:
        output_width = output_interval[0][0].sup - output_interval[0][0].inf
    elif type(output_interval[0]) == list:
        output_width = output_interval[0][1] - output_interval[0][0]
    interval_rect = matplotlib.patches.Rectangle(
        (input_interval[0].inf, output_interval[0][0].inf), input_width, output_width
    )
    ax.add_collection(
        matplotlib.collections.PatchCollection(
            [interval_rect], facecolor="k", alpha=0.1, edgecolor="k"
        )
    )

    plt.show()


class SafeRegionLoss(tf.keras.losses.Loss):
    """Mean squared loss plus a penalty for safe regions"""

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.math.square(y_pred - y_true), axis=-1)


def safe_training_loop():
    x, y = generate_data()

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
    input_interval, desired_interval = interval[20, 40], interval[10, 30]
    EXPLORATION_BUDGET = 10
    regression_model = tf.keras.Model(inputs, outputs)
    regression_model.compile(
        # run_eagerly=True,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # init to None but write on epoch 0; weights empty before first grad application
    last_safe_weights = None
    last_safe_epoch = 0
    num_unsafe_epochs = 0

    epochs = 40
    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch}")

        with tf.GradientTape() as tape:
            y_pred = regression_model(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = loss_fn(y, y_pred)

        # Compute gradients
        trainable_vars = regression_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        optimizer.apply_gradients(zip(gradients, trainable_vars))

        # first time only, store weights to avoid empty weights
        if epoch == 0:
            last_safe_weights = regression_model.get_weights()

        output_interval, _ = propagate_interval(
            input_interval, regression_model, graph=False
        )
        if type(output_interval) is list:
            if len(output_interval) == 1:
                output_interval = output_interval[0]
            else:
                raise NotImplementedError("Output interval was interval of length > 1")
        if output_interval not in desired_interval:
            print(f"safe region test FAILED, interval was {output_interval}")
            print(regression_model.layers[-1].weights)
            num_unsafe_epochs += 1
        else:
            print(f"safe region test passed, interval was {output_interval}")
            last_safe_weights = regression_model.get_weights()
            last_safe_epoch = epoch
            num_unsafe_epochs = 0

        if num_unsafe_epochs == EXPLORATION_BUDGET:
            print(
                f"Restarting training from last known safe set of weights, "
                f"as unsafe epoch tolerance {EXPLORATION_BUDGET} was reached. "
                f"Weights are {last_safe_weights}"
            )
            regression_model.set_weights(last_safe_weights)
        else:
            # Update metrics (includes the metric that tracks the loss)
            regression_model.compiled_metrics.update_state(y, y_pred)
            # Return a dict mapping metric names to current value

    return regression_model


def safe_training_loop_large_network():
    x = np.linspace(-10, 10, 100)
    y = x**2

    normalizer = layers.Normalization(
        input_shape=[
            1,
        ],
        axis=None,
    )
    normalizer.adapt(x)
    inputs = tf.keras.Input(shape=(1,))
    # input -> normalizer -> dense1linear -> dense64 relu --x3--> dense1linear
    outputs = layers.Dense(units=1, activation="linear")(
        layers.Dense(units=64, activation="relu")(
            layers.Dense(units=64, activation="relu")(
                layers.Dense(units=64, activation="relu")(
                    layers.Dense(units=1)(normalizer(inputs))
                )
            )
        )
    )
    input_interval, desired_interval = interval[-8, -5], interval[15, 70]
    EXPLORATION_BUDGET = 10
    regression_model = tf.keras.Model(inputs, outputs)
    regression_model.compile(
        # run_eagerly=True,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # init to None but write on epoch 0; weights empty before first grad application
    last_safe_weights = None
    last_safe_epoch = 0
    num_unsafe_epochs = 0

    epochs = 40
    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch}")

        with tf.GradientTape() as tape:
            y_pred = regression_model(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = loss_fn(y, y_pred)

        # Compute gradients
        trainable_vars = regression_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        optimizer.apply_gradients(zip(gradients, trainable_vars))

        # first time only, store weights to avoid empty weights
        if epoch == 0:
            last_safe_weights = optimizer.get_weights()

        # print("propagating interval")
        output_interval, _ = propagate_interval(
            input_interval, regression_model, graph=False
        )
        if type(output_interval) is list:
            if len(output_interval) == 1:
                output_interval = output_interval[0]
            else:
                raise NotImplementedError("Output interval was interval of length > 1")
        if output_interval not in desired_interval:
            print(f"safe region test FAILED, interval was {output_interval}")
            # print(regression_model.layers[-1].weights)
            num_unsafe_epochs += 1
        else:
            print(f"safe region test passed, interval was {output_interval}")
            # TODO: use weights from network, NOT optimizer
            last_safe_weights = optimizer.get_weights()
            last_safe_epoch = epoch
            num_unsafe_epochs = 0

        if num_unsafe_epochs == EXPLORATION_BUDGET:
            print(
                f"Restarting training from last known safe set of weights, "
                f"as unsafe epoch tolerance {EXPLORATION_BUDGET} was reached. "
                f"Weights are {last_safe_weights}"
            )
            optimizer.set_weights(last_safe_weights)
        else:
            # Update metrics (includes the metric that tracks the loss)
            regression_model.compiled_metrics.update_state(y, y_pred)
            # Return a dict mapping metric names to current value

    return regression_model


def projection_training_loop():
    x, y = generate_data()

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
    input_interval, desired_interval = interval[20, 40], interval[10, 30]
    EXPLORATION_BUDGET = 10
    regression_model = tf.keras.Model(inputs, outputs)
    regression_model.compile(
        # run_eagerly=True,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # don't project every epoch
    EPOCH_TO_PROJECT = 5

    epochs = 40
    # epochs = 5
    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch}")

        with tf.GradientTape() as tape:
            y_pred = regression_model(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = loss_fn(y, y_pred)

        # Compute gradients
        trainable_vars = regression_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        print(gradients)
        print(trainable_vars)
        # Update weights
        optimizer.apply_gradients(zip(gradients, trainable_vars))

        output_interval, penultimate_interval = propagate_interval(
            input_interval, regression_model, graph=False
        )
        if type(output_interval) is list:
            if len(output_interval) == 1:
                output_interval = output_interval[0]
            else:
                raise NotImplementedError("Output interval was interval of length > 1")
        if output_interval not in desired_interval:
            print(f"safe region test FAILED, interval was {output_interval}")
            print(regression_model.layers[-1].weights)
            if epoch % EPOCH_TO_PROJECT == 0:
                print(f"\nProjecting weights at epoch {epoch}.")
                weights = regression_model.layers[-1].weights
                print(
                    f"Old weights: {np.squeeze(np.array([weight.numpy() for weight in weights]))}"
                )
                projected_weights = project_weights(
                    desired_interval,
                    penultimate_interval,
                    np.squeeze(np.array(weights)),
                )
                print(
                    f"Projected weights: {projected_weights} yield new interval: "
                    f"{penultimate_interval * projected_weights[0] + projected_weights[1]}"
                )
                proj_weight, proj_bias = projected_weights
                regression_model.layers[-1].set_weights(
                    [np.array([[proj_weight]]), np.array([proj_bias])]
                )
                # NOTE: assume positive weights
                # TODO: handle both signs of weights

                # print(optimizer.get_weights())
                # optimizer.set_weights(last_safe_weights)
        else:
            print(f"safe region test passed, interval was {output_interval}")

        # Update metrics (includes the metric that tracks the loss)
        regression_model.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value

    return regression_model


def projection_training_loop_larger():
    x, y = generate_data()

    normalizer = layers.Normalization(
        input_shape=[
            1,
        ],
        axis=None,
    )
    normalizer.adapt(x)
    inputs = tf.keras.Input(shape=(1,))
    # input -> normalizer -> dense1 -> dense1
    outputs = layers.Dense(units=1)(layers.Dense(units=1)(normalizer(inputs)))
    # input -> dense1
    # outputs = layers.Dense(units=1)(inputs)
    input_interval, desired_interval = interval[20, 40], interval[10, 30]
    EXPLORATION_BUDGET = 10
    regression_model = tf.keras.Model(inputs, outputs)
    regression_model.compile(
        # run_eagerly=True,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    loss_fn = tf.keras.losses.MeanSquaredError()

    # don't project every epoch
    EPOCH_TO_PROJECT = 5

    epochs = 40
    # epochs = 5
    for epoch in range(epochs):
        print(f"\nStart of epoch {epoch}")

        with tf.GradientTape() as tape:
            y_pred = regression_model(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = loss_fn(y, y_pred)

        # Compute gradients
        trainable_vars = regression_model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        print(gradients)
        print(trainable_vars)
        # Update weights
        optimizer.apply_gradients(zip(gradients, trainable_vars))

        output_interval, penultimate_interval = propagate_interval(
            input_interval, regression_model, graph=False
        )
        if type(output_interval) is list:
            if len(output_interval) == 1:
                output_interval = output_interval[0]
            else:
                raise NotImplementedError("Output interval was interval of length > 1")
        if output_interval not in desired_interval:
            print(f"safe region test FAILED, interval was {output_interval}")
            print(regression_model.layers[-1].weights)
            if epoch % EPOCH_TO_PROJECT == 0:
                print(f"\nProjecting weights at epoch {epoch}.")
                weights = regression_model.layers[-1].weights
                print(
                    f"Old weights: {np.squeeze(np.array([weight.numpy() for weight in weights]))}"
                )
                projected_weights = project_weights(
                    desired_interval,
                    penultimate_interval,
                    np.squeeze(np.array(weights)),
                )
                print(
                    f"Projected weights: {projected_weights} yield new interval: "
                    f"{penultimate_interval * projected_weights[0] + projected_weights[1]}"
                )
                proj_weight, proj_bias = projected_weights
                regression_model.layers[-1].set_weights(
                    [np.array([[proj_weight]]), np.array([proj_bias])]
                )
                # NOTE: assume positive weights
                # TODO: handle both signs of weights

                # print(optimizer.get_weights())
                # optimizer.set_weights(last_safe_weights)
        else:
            print(f"safe region test passed, interval was {output_interval}")

        # Update metrics (includes the metric that tracks the loss)
        regression_model.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value

    return regression_model


def projection_training_loop_multivariate():
    return None


def project_weights(goal_interval, input_intervals, theta):
    shift_lower = np.array([0, goal_interval[0].inf])
    print(f"input interval: {input_intervals}")
    direction_lower = np.array([1, -input_intervals[0].inf])
    project_lower = (
        (direction_lower @ (theta - shift_lower))
        / (direction_lower @ direction_lower)
        * direction_lower
    )
    param_lower = project_lower + shift_lower

    shift_upper = np.array([0, goal_interval[0].sup])
    direction_upper = np.array([1, -input_intervals[0].sup])
    project_upper = (
        (direction_upper @ (theta - shift_upper))
        / (direction_upper @ direction_upper)
        * direction_upper
    )
    param_upper = project_upper + shift_upper

    return min(
        [param_upper, param_lower], key=lambda param: np.linalg.norm(theta - param)
    )
