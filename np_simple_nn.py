import numpy as np
import os
import cv2

def load_data(data_dir):
    X, y = [], []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.jpg'):
            img = cv2.imread(os.path.join(data_dir, file_name))
            img = cv2.resize(img, (64, 64))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.array(img).flatten()
            X.append(img)
            if 'cat' in file_name:
                y.append(1)
            else:
                y.append(0)
    y = np.array(y)
    y = y.reshape(-1, 1)
    X = np.array(X)
    return X, y

data_dir = '/home/salahuddin/projects/datasets/catsanddogs/train/mix'
X, y = load_data(data_dir)
# np.squeeze(X)
print(X.shape, y.shape)
# split data into train, validation and test sets
n_samples = len(X)
n_train = int(1 * n_samples)
n_val = int(0 * n_samples)
n_test = n_samples - n_train - n_val

indices = np.random.permutation(n_samples)
X_train = X[indices[:n_train]]
y_train = y[indices[:n_train]]
X_val = X[indices[n_train:n_train+n_val]]
y_val = y[indices[n_train:n_train+n_val]]
X_test = X[indices[n_train+n_val:]]
y_test = y[indices[n_train+n_val:]]

# Define the sigmoid activation function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Define the derivative of the sigmoid function
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Define the forward pass function
def forward_pass(X, W1, b1, W2, b2):
    # Compute the activation of the hidden layer
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)

    # Compute the activation of the output layer
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Return the activations of the hidden and output layers
    return a1, a2

# Define the backward pass function
def backward_pass(X, y, a1, a2, W1, b1, W2, b2, learning_rate):
    # Compute the error in the output layer
    delta2 = (a2 - y) * sigmoid_prime(a2)

    # Compute the error in the hidden layer
    delta1 = np.dot(delta2, W2.T) * sigmoid_prime(a1)

    # Compute the gradients of the weights and biases
    dW2 = np.dot(a1.T, delta2)
    db2 = np.sum(delta2, axis=0, keepdims=True)
    dW1 = np.dot(X.T, delta1)
    db1 = np.sum(delta1, axis=0, keepdims=True)

    # Update the weights and biases
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Return the updated weights and biases
    return W1, b1, W2, b2


a1_ = 0
a2_ = 0
# Define the training loop
def train(X_train, y_train, n_epochs, learning_rate):
    # Initialize the weights and biases
    input_size = X_train.shape[1]
    hidden_size = 16
    output_size = 1
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    # Train the model fo1r the specified number of epochs
    bs = 0
    for i in range(n_epochs):
        # Perform a forward pass on the training data
        a1, a2 = forward_pass(X_train, W1, b1, W2, b2)
        if (not bs):
            a1_ = a1
            a2_ = a2
            bs = 1
        # Compute the loss
        loss = np.mean((a2 - y_train) ** 2)

        # Print the loss every 100 epochs
        if i % 100 == 0:
            print('Epoch', i, 'loss:', loss)

        # Perform a backward pass to update the weights and biases
        W1, b1, W2, b2 = backward_pass(X_train, y_train, a1, a2, W1, b1, W2, b2, learning_rate)

    # Return the trained weights and biases
    return W1, b1, W2, b2


def plot_error(a1, a2, y_train) :
    loss = np.mean((a2 - y_train) ** 2)
    print("initial loss")
    print(loss)

W1, b1, W2, b2 = train(X_train, y_train, n_epochs=500, learning_rate=1e-3)
plot_error(a1_, a2_, y_train)
# Train the model
a1, a2 = forward_pass(X_train, W1, b1, W2, b2)
loss = np.mean((a2 - y_train) ** 2)
print("Loss after traning")
print(loss)
