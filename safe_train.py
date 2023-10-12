import tensorflow as tf
from tensorflow.keras import layers

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from maraboupy import Marabou, MarabouUtils, MarabouCore
import cvxpy as cp
import itertools

from interval import interval, inf

import matplotlib.pyplot as plt

# from safe_model import SafeModel

# Constants
ra1 = (0.9, 0.9, 0.9)  # white
ra2 = (0.0, 1.0, 1.0)  # cyan
ra3 = (144.0 / 255.0, 238.0 / 255.0, 144.0 / 255.0)  # lightgreen
ra4 = (30.0 / 255.0, 144.0 / 255.0, 1.0)  # dodgerblue
ra5 = (0.0, 1.0, 0.0)  # lime
ra6 = (0.0, 0.0, 1.0)  # blue
ra7 = (34.0 / 255.0, 139.0 / 255.0, 34.0 / 255.0)  # forestgreen
ra8 = (0.0, 0.0, 128.0 / 255.0)  # navy
ra9 = (0.0, 100.0 / 255.0, 0.0)  # darkgreen
colors = [ra1, ra2, ra3, ra4, ra5, ra6, ra7, ra8, ra9]
bg_colors = [(1.0, 1.0, 1.0)]
action_names = [
    "COC",
    "DNC",
    "DND",
    "DES1500",
    "CL1500",
    "SDES1500",
    "SCL1500",
    "SDES2500",
    "SCL2500",
]


def generate_data(NOISE_STD=2, M=0.5, B=5, xmin=5, xmax=55, n=30):
    x = np.linspace(xmin, xmax, n)
    y_func = lambda x: M * x + B
    y_noisy = lambda x: y_func(x) + np.random.normal(0, NOISE_STD, np.shape(x))
    # y = np.array([5, 20, 14, 32, 22, 38])
    y = y_noisy(x)
    return x, y


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


def propagate_interval(input_interval, model, graph=False, verbose=False):
    # TODO check relu handling for multiple intervals
    # TODO change inputs to lists?

    # Complain if input_interval is not a list
    if type(input_interval) is not list:
        if verbose:
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
    if type(penultimate_interval) is not list:
        penultimate_interval = [penultimate_interval]
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
    desired_interval=None,
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
    if desired_interval is not None:
        out_rect = matplotlib.patches.Rectangle(
            (-60, desired_interval[0].inf),
            120,
            desired_interval[0].sup - desired_interval[0].inf,
        )
        ax.add_collection(
            matplotlib.collections.PatchCollection(
                [out_rect], facecolor="r", alpha=0.1, edgecolor="r"
            )
        )

    plt.show()


class SafeRegionLoss(tf.keras.losses.Loss):
    """Mean squared loss plus a penalty for safe regions"""

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.math.square(y_pred - y_true), axis=-1)


def generate_constraints(input_intervals, goal_interval, x, theta, verbose=False):
    lowers = []
    uppers = []
    for i, ivl in enumerate(input_intervals):
        if theta[i] < 0:
            # if weight is negative
            # swap order of interval bounds used in constraints
            lowers.append(ivl[0][1])
            uppers.append(ivl[0][0])
        else:
            lowers.append(ivl[0][0])
            uppers.append(ivl[0][1])

    interval_combinations = [lowers, uppers]
    constraint_vectors = [np.hstack([elem, 1]) for elem in interval_combinations]
    constraints = []
    if verbose:
        print(f"Generated {len(constraint_vectors)} constraint orderings")
        print("Generating constraints:")
    for constraint_vector in constraint_vectors:
        constraints.append(constraint_vector @ x >= goal_interval[0][0])
        constraints.append(constraint_vector @ x <= goal_interval[0][1])
        if verbose:
            print(f"{constraint_vector} @ x >= {goal_interval[0][0]}")
            print(f"{constraint_vector} @ x <= {goal_interval[0][1]}")

    for i in range(len(theta) - 1):
        constraint_row = np.zeros(theta.shape)
        np.put(constraint_row, i, 1)
        if theta[i] >= 0:
            # enforce weight stays positive during optimization
            constraints.append(constraint_row @ x >= 0)
        else:
            # enforce weight stays negative optimization
            constraints.append(constraint_row @ x <= 0)

    return constraints


def project_weights(
    goal_interval, input_intervals, theta, verbose=False, multiple_intervals=False
):
    x = cp.Variable(theta.shape)
    constraints = []
    if multiple_intervals:
        for goal_interval_single in goal_interval:
            print(goal_interval_single[0])
            print(goal_interval_single[0][0][0])
            print(goal_interval_single[0][0][1])
            these_constraints = generate_constraints(
                input_intervals, goal_interval_single[0], x, theta, verbose
            )
        constraints += these_constraints
    else:
        constraints = generate_constraints(
            input_intervals, goal_interval_single, x, theta, verbose
        )
    obj = cp.Minimize(cp.norm(x - theta))
    prob = cp.Problem(obj, constraints)
    prob.solve()  # Returns the optimal value.
    return x.value


def project_weights_vector(goal_interval, input_intervals, theta, verbose=False):
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


def check_intervals(output_interval, goal_interval):
    assert type(output_interval) == type(goal_interval)
    if type(output_interval) is list:
        assert len(output_interval) == len(goal_interval)
        for i in range(len(output_interval)):
            if (
                goal_interval[i] is not None
                and output_interval[i] not in goal_interval[i]
            ):
                return False
        return True
    else:
        return output_interval in goal_interval


def plot_policy(model, filename=None, zoom=False, vo=0, vi=0, use_sisl_colors=False):
    x_grid = None
    taus = np.linspace(0, 40, 81)
    hs = np.hstack(
        [
            np.linspace(-5000, -2000, 20),
            np.linspace(-2000, 2000, 40),
            np.linspace(2000, 5000, 20),
        ]
    )
    for tau in taus:
        grid_component = np.vstack(
            [
                hs,
                np.ones(hs.shape) * vo,
                np.ones(hs.shape) * vi,
                np.ones(hs.shape) * tau,
            ]
        ).T
        if x_grid is not None:
            x_grid = np.vstack([x_grid, grid_component])
        else:
            x_grid = grid_component

    y_pred = model.predict(x_grid)
    advisory_idxs = np.argmax(y_pred, axis=1)

    # dict indexed by color/advisory of all points
    xs = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
    ys = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}

    for i, advisory_idx in enumerate(advisory_idxs):
        scatter_x = x_grid[i, 3]  # tau
        scatter_y = x_grid[i, 0]  # h
        xs[advisory_idx].append(scatter_x)
        ys[advisory_idx].append(scatter_y)

    plt.figure()
    plt.tight_layout()
    for i in range(len(colors)):
        if use_sisl_colors:
            plt.scatter(xs[i], ys[i], s=10, c=[colors[i]])
        else:
            plt.scatter(xs[i], ys[i], s=10)
    plt.legend(action_names)
    plt.xlabel("Tau (sec)")
    plt.ylabel("h (ft)")
    plt.title(f"Policy for vo:{vo} and vi:{vi}")
    if filename is None:
        plt.savefig(f"viz_policy_vo{vo}_vi{vi}.pdf")
    else:
        if filename[-4:] == ".pdf":
            plt.savefig(filename)
        else:
            plt.savefig(filename + ".pdf")
    plt.show()

    if zoom:
        plt.figure()
        plt.tight_layout()
        for i in range(len(colors)):
            if use_sisl_colors:
                plt.scatter(xs[i], ys[i], s=10, c=[colors[i]])
            else:
                plt.scatter(xs[i], ys[i], s=10)
        plt.legend(action_names)
        plt.xlabel("Tau (sec)")
        plt.ylabel("h (ft)")
        plt.title(f"Policy for vo:{vo} and vi:{vi}")
        plt.ylim([-2100, 2100])
        if filename is None:
            plt.savefig(f"viz_policy_vo{vo}_vi{vi}_zoomed.pdf")
        else:
            if filename[-4:] == ".pdf":
                plt.savefig(filename[:-4] + "_zoomed.pdf")
            else:
                plt.savefig(filename + "_zoomed.pdf")
        plt.show()
