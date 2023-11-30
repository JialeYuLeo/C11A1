import numpy as np

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Gradient descent function
def gradient_descent(X, y, theta, learning_rate, iterations):

    cost_history = np.zeros(iterations)

    for i in range(iterations):
        p = sigmoid(np.dot(X, theta))
        gradient = np.dot(X.T, (p - y))
        theta -= learning_rate * gradient

    return theta

# Example usage:
X = np.array([[4, 4,1],[6,4,1] ,[6,5,1], [6,8,1], [6,10,1], [8, 8,1], [8, 10,1]])  # feature matrix
y = np.array([1,1,1,0,0,1,0])  # labels
theta = np.array([0.3, -0.2, 0.7])  # initial parameters
learning_rate = 0.01
iterations = 3

theta = gradient_descent(X, y, theta, learning_rate, iterations)
print(theta)
y_pred = np.round(sigmoid(np.dot(X, theta)))
correct_predictions = np.sum(y_pred == y)
print(correct_predictions)

P = np.array([[3,3,1],[4,10,1],[9,8,1],[9,10,1]])
y_pred = np.round(sigmoid(np.dot(P, theta)))
print(y_pred)