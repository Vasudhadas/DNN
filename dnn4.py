import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.regularizers import l1, l2
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images, test_images = train_images[..., None] / 255.0, test_images[..., None] / 255.0
 # Function to build and evaluate a model
def build_and_evaluate(name, regularizer=None, dropout_rate=None):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=regularizer),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizer),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizer),
        layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizer),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizer),
        layers.Dense(10, activation='softmax')
    ])
    if dropout_rate:
        model.add(layers.Dropout(dropout_rate))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2, verbose=1)
    loss, accuracy = model.evaluate(test_images, test_labels)
    print(f'Test Accuracy: {accuracy * 100:.2f}%\n Loss:{loss:.2f}')
build_and_evaluate("Base Model")
build_and_evaluate("L1 Regularization", regularizer=l1(0.001))
build_and_evaluate("L2 Regularization", regularizer=l2(0.001))
build_and_evaluate("Dropout", dropout_rate=0.5)