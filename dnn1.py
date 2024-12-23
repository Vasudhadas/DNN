import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoid(x): return 1 / (1 + np.exp(-x))
def tanh(x): return np.tanh(x)
def relu(x): return np.maximum(0, x)
def softmax(x): return np.exp(x) / np.sum(np.exp(x))

# Input values
x = np.linspace(-10, 10, 400)

# Plotting
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x), label='Sigmoid', color='blue')
plt.title("Sigmoid")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(x, tanh(x), label='Tanh', color='orange')
plt.title("Tanh")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(x, relu(x), label='ReLU', color='green')
plt.title("ReLU")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(x, softmax(x), label='Softmax', color='red')
plt.title("Softmax")
plt.grid(True)

plt.tight_layout()
plt.show()