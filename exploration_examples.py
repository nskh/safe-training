from safe_train import *


def safe_training_loop(input_interval, desired_interval, x, y, verbose=False):
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
        if verbose:
            print("*" * 20)
            print(f"\nStart of epoch {epoch}")

        with tf.GradientTape() as tape:
            y_pred = regression_model(x, training=True)  # Forward pass
            y_pred = tf.squeeze(y_pred)
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
        print(input_interval)
        print(output_interval)
        if type(output_interval) is list:
            if len(output_interval) == 1:
                output_interval = output_interval[0]
            else:
                raise NotImplementedError("Output interval was interval of length > 1")

        print("With Marabou:\n")
        tf.saved_model.save(regression_model, "tmp")
        network = Marabou.read_tf("tmp", modelType="savedModel_v2")
        inputVars = network.inputVars[0][0]
        outputVars = network.outputVars[0][0]

        print("adding constraints")
        print(inputVars[0], ">", input_interval[0].inf)
        network.setLowerBound(inputVars[0], input_interval[0].inf)

        print(inputVars[0], "<", input_interval[0].sup)
        network.setUpperBound(inputVars[0], input_interval[0].sup)

        print(outputVars[0], ">", desired_interval[0].inf)
        print(outputVars[0], "<", desired_interval[0].sup)

        # network.setLowerBound(outputVars[0], desired_interval[0].inf)
        # network.setUpperBound(outputVars[0], desired_interval[0].sup)

        ineq1 = MarabouCore.Equation(MarabouCore.Equation.LE)
        ineq1.addAddend(outputVars[0], 1)
        ineq1.setScalar(1.0)

        ineq2 = MarabouCore.Equation(MarabouCore.Equation.GE)
        ineq2.addAddend(outputVars[0], 1)
        ineq2.setScalar(4.0)
        disjunction = [[ineq1], [ineq2]]
        network.addDisjunctionConstraint(disjunction)

        _, vals, stats = network.solve("marabou.log")
        if vals is None:
            print("UNSAT, so we are safe?")
        else:
            print(f"vals are {vals}")

        print("\n\nWithout:\n")
        if output_interval not in desired_interval:
            if verbose:
                print(f"safe region test FAILED, interval was {output_interval}")
                print(regression_model.layers[-1].weights)
                print("output interval", output_interval)
            num_unsafe_epochs += 1
        else:
            if verbose:
                print(f"safe region test passed, interval was {output_interval}")
                print("output interval", output_interval)
            last_safe_weights = regression_model.get_weights()
            last_safe_epoch = epoch
            num_unsafe_epochs = 0

        if num_unsafe_epochs == EXPLORATION_BUDGET:
            if verbose:
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
