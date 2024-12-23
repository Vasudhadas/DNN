import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist

# Load and preprocess Fashion MNIST dataset

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train[..., None] / 255.0, x_test[..., None] / 255.0  # Normalize and add channel dimension
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)  # Convert labels to one-hot encoding

# Define a simpler VGG-like model
model = Sequential([
    Input(shape=(28, 28, 1)),  # Input layer for 28x28 grayscale images
    Conv2D(32, (3, 3), activation='relu'),  # Convolutional layer
    MaxPooling2D((2, 2)),  # Max pooling layer
    Conv2D(64, (3, 3), activation='relu'),  # Another convolutional layer
    MaxPooling2D((2, 2)),  # Another max pooling layer
    Flatten(),  # Flatten layer to convert 2D features into 1D vector
    Dense(128, activation='relu'),  # Fully connected layer
    Dropout(0.5),  # Dropout layer to prevent overfitting
    Dense(10, activation='softmax')  # Output layer with 10 neurons for classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=128)
# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")