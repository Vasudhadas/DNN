import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Data for AND, OR, XOR gates
data = {
    'AND': (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1])),
    'OR':  (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 1])),
    'XOR': (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 0])),
}

# Classify AND, OR, XOR gates
for gate, (X, y) in data.items():
    perceptron = Perceptron(max_iter=10, eta0=1, random_state=42)
    perceptron.fit(X, y)
    y_pred = perceptron.predict(X)
    acc = accuracy_score(y, y_pred) * 100

    print(f"{gate} Gate: Accuracy = {acc:.2f}%")
    print(f"Predictions: {y_pred}")
    print(f"True Labels: {y}")

    # Plot decision boundary
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
    plt.title(f"{gate} Gate")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.show()
