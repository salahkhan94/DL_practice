def backward_pass(self, X, y, a1, a2, a3, W1, b1, W2, b2, W3, b3, learning_rate):
    # Compute the error in the output layer
    delta3 = (a3 - y) * self.sigmoid_prime(a3)

    # Compute the error in the second hidden layer
    delta2 = np.dot(delta3, W3.T) * self.sigmoid_prime(a2)

    # Compute the error in the first hidden layer
    delta1 = np.dot(delta2, W2.T) * self.sigmoid_prime(a1)

    # Compute the gradients of the weights and biases
    dW3 = np.dot(a2.T, delta3)
    db3 = np.sum(delta3, axis=0, keepdims=True)
    dW2 = np.dot(a1.T, delta2)
    db2 = np.sum(delta2, axis=0, keepdims=True)
    dW1 = np.dot(X.T, delta1)
    db1 = np.sum(delta1, axis=0, keepdims=True)

    # Update the weights and biases
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Return the updated weights and biases
    return W1, b1, W2, b2, W3, b3
