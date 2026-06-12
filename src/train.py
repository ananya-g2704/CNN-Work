import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Input
)

# Load Dataset
(X_train, y_train), (X_test, y_test) = \
    tf.keras.datasets.mnist.load_data()

# Normalize
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# CNN expects channels
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# CNN Model
model = Sequential([

    Input(shape=(28, 28, 1)),

    Conv2D(
        32,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D((2,2)),

    Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D((2,2)),

    Flatten(),

    Dense(
        128,
        activation='relu'
    ),

    Dropout(0.3),

    Dense(
        10,
        activation='softmax'
    )
])
model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1,
    verbose=2
)

loss, acc = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\nTraining Complete")
print(f"Final Test Accuracy: {acc:.4f}")

model.save("models/cnn_model.keras")

print("CNN Model Saved!")