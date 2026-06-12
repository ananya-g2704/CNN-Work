import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model(
    "../models/cnn_model.keras"
)

(_, _), (X_test, y_test) = \
    tf.keras.datasets.mnist.load_data()

X_test = X_test.astype("float32") / 255.0

sample = X_test[0]

prediction = model.predict(
    sample.reshape(1,28,28,1),
    verbose=0
)

print(
    "Predicted:",
    np.argmax(prediction)
)

print(
    "Actual:",
    y_test[0]
)