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
print(X.shape, y.shape)

class NeuralNetwork:
    def __init__(self, X, y, batch = 64, lr = 1e-3,  epochs = 50):
        self.input = X 
        self.target = y
        self.batch = batch
        self.epochs = epochs
        self.lr = lr
        
        self.x = self.input[:self.batch] # batch input 
        self.y = self.target[:self.batch] # batch target value
        self.loss = []
        self.acc = []
        self.a1_ = 0
        self.a2_ = 0

        
        self.init_weights()
      
    def init_weights(self):
        # self.W1 = np.random.randn(self.input.shape[1],256)
        # self.W2 = np.random.randn(self.W1.shape[1],128)
        # self.W3 = np.random.randn(self.W2.shape[1],self.y.shape[1])

        # self.b1 = np.random.randn(self.W1.shape[1],)
        # self.b2 = np.random.randn(self.W2.shape[1],)
        # self.b3 = np.random.randn(self.W3.shape[1],)

        input_size = X.shape[1]
        hidden_size = 16
        output_size = 1
        W1 = np.random.randn(input_size, hidden_size) * 0.01
        b1 = np.zeros((1, hidden_size))
        W2 = np.random.randn(hidden_size, output_size) * 0.01
        b2 = np.zeros((1, output_size))

    # Define the sigmoid activation function
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    # Define the derivative of the sigmoid function
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))


    def ReLU(self, x):
        return np.maximum(0,x)

    def dReLU(self,x):
        return 1 * (x > 0) 
    
    def softmax(self, z):
        z = z - np.max(z, axis = 1).reshape(z.shape[0],1)
        return np.exp(z) / np.sum(np.exp(z), axis = 1).reshape(z.shape[0],1)
    
    def shuffle(self):
        idx = [i for i in range(self.input.shape[0])]
        np.random.shuffle(idx)
        self.input = self.input[idx]
        self.target = self.target[idx]

    # Define the forward pass function
    def forward_pass(self, X, W1, b1, W2, b2):
        # Compute the activation of the hidden layer
        z1 = np.dot(X, W1) + b1
        a1 = self.sigmoid(z1)

        # Compute the activation of the output layer
        z2 = np.dot(a1, W2) + b2
        a2 = self.sigmoid(z2)

        # Return the activations of the hidden and output layers
        return a1, a2
    
        # Define the backward pass function
    def backward_pass(self, X, y, a1, a2, W1, b1, W2, b2, learning_rate):
        # Compute the error in the output layer
        delta2 = (a2 - y) * self.sigmoid_prime(a2)

        # Compute the error in the hidden layer
        delta1 = np.dot(delta2, W2.T) * self.sigmoid_prime(a1)

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
    
        # Initialize the weights and biases
    def init(self, X_train, y_train) :
        input_size = X_train.shape[1]
        hidden_size = 16
        output_size = 1
        W1 = np.random.randn(input_size, hidden_size) * 0.01
        b1 = np.zeros((1, hidden_size))
        W2 = np.random.randn(hidden_size, output_size) * 0.01
        b2 = np.zeros((1, output_size))
        return W1, W2, b2, b1

# Define the training loop
    def train(self, X_train, y_train, n_epochs, learning_rate):
        # Train the model fo1r the specified number of epochs
        W1, W2, b2, b1 = self.init(X_train, y_train)
        bs = 0
        for i in range(n_epochs):
            # Perform a forward pass on the training data
            a1, a2 = self.forward_pass(X, W1, b1, W2, b2)
            if (not bs):
                self.a1_ = a1
                self.a2_ = a2
                bs = 1
            # Compute the loss
            loss = np.mean((a2 - y_train) ** 2)

            # Print the loss every 100 epochs
            if i % 100 == 0:
                print('Epoch', i, 'loss:', loss)

            # Perform a backward pass to update the weights and biases
            W1, b1, W2, b2 = self.backward_pass(X_train, y_train, a1, a2, W1, b1, W2, b2, learning_rate)

        # Return the trained weights and biases
        return W1, b1, W2, b2

nn = NeuralNetwork(X, y, 64, lr = 0.001)

def plot_error(a1, a2, y_train) :
    loss = np.mean((a2 - y_train) ** 2)
    print("initial loss")
    print(loss)

W1, b1, W2, b2 = nn.train(X, y, n_epochs=500, learning_rate=1e-3)
plot_error(nn.a1_, nn.a2_, y)
# Train the model
a1, a2 = nn.forward_pass(X, W1, b1, W2, b2)
loss = np.mean((a2 - y) ** 2)
print("Loss after traning")
print(loss)
