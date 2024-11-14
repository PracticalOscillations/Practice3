import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Parameters of the harmonic oscillator
omega_0 = 1.0  # Natural frequency
omega = 1.0    # Forcing frequency
x0 = 1.0       # Initial position
v0 = 0.0       # Initial velocity

# Time domain
t_start = 0.0
t_end = 10.0
N_t = 1000
t = np.linspace(t_start, t_end, N_t).reshape(-1, 1)

# Function to create training data
def create_training_data(N_f):
    t_f = np.linspace(t_start, t_end, N_f).reshape(-1, 1)
    return t_f

# Neural Network Model for Formulation 1
def pinn_formulation1():
    # Define the neural network
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(1,)),
        tf.keras.layers.Dense(20, activation='tanh'),
        tf.keras.layers.Dense(20, activation='tanh'),
        tf.keras.layers.Dense(1)
    ])

    # Define the custom loss function
    def loss(model, t):
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(t)
            with tf.GradientTape() as tape1:
                tape1.watch(t)
                x = model(t)
            dx_dt = tape1.gradient(x, t)
        d2x_dt2 = tape2.gradient(dx_dt, t)
        # Differential equation residual
        f = d2x_dt2 + omega_0**2 * x - tf.cos(omega * t)
        # Initial conditions
        x0_pred = model(tf.zeros((1, 1)))
        dx_dt0_pred = tape1.gradient(x0_pred, t)
        # Loss terms
        loss_f = tf.reduce_mean(tf.square(f))
        loss_ic = tf.reduce_mean(tf.square(x0_pred - x0)) + tf.reduce_mean(tf.square(dx_dt0_pred - v0))
        return loss_f + loss_ic

    # Training
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    epochs = 5000
    t_f = create_training_data(1000)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            current_loss = loss(model, t_f)
        grads = tape.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if epoch % 500 == 0:
            print(f"Formulation 1 - Epoch {epoch}, Loss: {current_loss.numpy()}")

    return model

# Neural Network Model for Formulation 2
def pinn_formulation2():
    # Define the neural network
    input_layer = tf.keras.Input(shape=(1,))
    hidden = tf.keras.layers.Dense(20, activation='tanh')(input_layer)
    hidden = tf.keras.layers.Dense(20, activation='tanh')(hidden)
    x_output = tf.keras.layers.Dense(1)(hidden)
    y_output = tf.keras.layers.Dense(1)(hidden)
    model = tf.keras.Model(inputs=input_layer, outputs=[x_output, y_output])

    # Define the custom loss function
    def loss(model, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            x, y = model(t)
            dx_dt = tape.gradient(x, t)
            dy_dt = tape.gradient(y, t)
        # Differential equations residuals
        f1 = dx_dt - y
        f2 = dy_dt + omega_0**2 * x - tf.cos(omega * t)
        # Initial conditions
        x0_pred, y0_pred = model(tf.zeros((1, 1)))
        # Loss terms
        loss_f = tf.reduce_mean(tf.square(f1)) + tf.reduce_mean(tf.square(f2))
        loss_ic = tf.reduce_mean(tf.square(x0_pred - x0)) + tf.reduce_mean(tf.square(y0_pred - v0))
        return loss_f + loss_ic

    # Training
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    epochs = 5000
    t_f = create_training_data(1000)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            current_loss = loss(model, t_f)
        grads = tape.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if epoch % 500 == 0:
            print(f"Formulation 2 - Epoch {epoch}, Loss: {current_loss.numpy()}")

    return model

# Neural Network Model for Formulation 3
def pinn_formulation3():
    # Define the neural network
    input_layer = tf.keras.Input(shape=(1,))
    hidden = tf.keras.layers.Dense(20, activation='tanh')(input_layer)
    hidden = tf.keras.layers.Dense(20, activation='tanh')(hidden)
    x_output = tf.keras.layers.Dense(1)(hidden)
    y_output = tf.keras.layers.Dense(1)(hidden)
    model = tf.keras.Model(inputs=input_layer, outputs=[x_output, y_output])

    # Define the custom loss function
    def loss(model, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            x, y = model(t)
            dx_dt = tape.gradient(x, t)
            dy_dt = tape.gradient(y, t)
        # Differential equations residuals
        f1 = dx_dt - omega * y - (1 / omega) * tf.sin(omega * t)
        f2 = dy_dt + omega_0 * x
        # Initial conditions
        x0_pred, y0_pred = model(tf.zeros((1, 1)))
        # Loss terms
        loss_f = tf.reduce_mean(tf.square(f1)) + tf.reduce_mean(tf.square(f2))
        loss_ic = tf.reduce_mean(tf.square(x0_pred - x0)) + tf.reduce_mean(tf.square(y0_pred - v0 / omega_0))
        return loss_f + loss_ic

    # Training
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    epochs = 5000
    t_f = create_training_data(1000)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            current_loss = loss(model, t_f)
        grads = tape.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        if epoch % 500 == 0:
            print(f"Formulation 3 - Epoch {epoch}, Loss: {current_loss.numpy()}")

    return model

# Main function to run the PINNs
def main():
    # Solve using Formulation 1
    model1 = pinn_formulation1()
    x_pred1 = model1(t).numpy()

    # Solve using Formulation 2
    model2 = pinn_formulation2()
    x_pred2, y_pred2 = model2(t)
    x_pred2 = x_pred2.numpy()

    # Solve using Formulation 3
    model3 = pinn_formulation3()
    x_pred3, y_pred3 = model3(t)
    x_pred3 = x_pred3.numpy()

    # Exact solution for comparison (if known)
    # For the harmonic oscillator with external forcing, the analytical solution is complex.
    # Here, we can use numerical integration as the "exact" solution.
    from scipy.integrate import odeint

    def exact_solution(t):
        def ode_system(state, t):
            x, v = state
            dxdt = v
            dvdt = -omega_0**2 * x + np.cos(omega * t)
            return [dxdt, dvdt]
        state0 = [x0, v0]
        sol = odeint(ode_system, state0, t.flatten())
        return sol[:, 0]

    x_exact = exact_solution(t)

    # Plot the results
    plt.figure(figsize=(12, 8))
    plt.plot(t, x_exact, 'k-', label='Exact Solution')
    plt.plot(t, x_pred1, 'r--', label='Formulation 1')
    plt.plot(t, x_pred2, 'b--', label='Formulation 2')
    plt.plot(t, x_pred3, 'g--', label='Formulation 3')
    plt.xlabel('Time t')
    plt.ylabel('Displacement x(t)')
    plt.title('Harmonic Oscillator with External Forcing - PINN Solutions')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
