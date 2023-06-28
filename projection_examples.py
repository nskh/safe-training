from safe_train import *


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
        if type(penultimate_interval) is not list:
            penultimate_interval = [penultimate_interval]
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
                if type(penultimate_interval) is list:
                    print(
                        f"Projected weights: {projected_weights} yield new interval: "
                        f"{penultimate_interval[0] * projected_weights[0] + projected_weights[1]}"
                    )
                else:
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
            [input_interval], regression_model, graph=False
        )
        if type(penultimate_interval) is not list:
            penultimate_interval = [penultimate_interval]
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
                if type(penultimate_interval) is list:
                    print(
                        f"Projected weights: {projected_weights} yield new interval: "
                        f"{penultimate_interval[0] * projected_weights[0] + projected_weights[1]}"
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
