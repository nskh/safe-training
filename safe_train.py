import tensorflow as tf
from tensorflow.keras import layers

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy

from interval import interval, inf


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
        epochs=1000,
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


def propagate_interval(input_interval, model):
    # TODO check relu handling for multiple intervals
    # TODO change inputs to lists?
    current_interval = input_interval
    for layer in model.layers:
        config = layer.get_config()
        if "normalization" in config["name"]:
            norm_mean, norm_var, _ = layer.get_weights()
            norm_std = np.sqrt(norm_var)
            # this *should* generalize to vectors directly
            # TODO check
            current_interval = [(current_interval - norm_mean) / norm_std]
        elif "dense" in config["name"]:
            weight, bias = layer.get_weights()
            num_combinations = weight.shape[0]
            num_intervals = weight.shape[1]
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
                # TODO test larger dim handling
                intervals = [0] * num_intervals
                for i in range(num_combinations):
                    for j in range(num_intervals):
                        intervals[j] += current_interval[i] * float(weight[i, j])
                for j in range(num_intervals):
                    intervals[j] += float(bias[j])
                current_interval = intervals
            if config["activation"] == "relu":
                # intersect with [0, +inf] to clip like a relu
                # TODO test this handling inside loop!!
                for node_interval in current_interval:
                    node_interval &= interval[0, inf]
            elif config["activation"] == "linear":
                # everything OK, don't modify
                pass
            else:
                raise NotImplementedError(
                    f"Activation type {config['activation']} is not handled"
                )
        else:
            raise NotImplementedError(f"Layer type {config['name']} is not handled")
    return current_interval


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

    input_width = input_interval[0].sup - input_interval[0].inf
    output_width = output_interval[0][1] - output_interval[0][0]
    interval_rect = matplotlib.patches.Rectangle(
        (input_interval[0].inf, output_interval[0][0]), input_width, output_width
    )
    ax.add_collection(
        matplotlib.collections.PatchCollection(
            [interval_rect], facecolor="k", alpha=0.1, edgecolor="k"
        )
    )

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
    plt.show()
