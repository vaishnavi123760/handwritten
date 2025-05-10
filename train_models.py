import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Data augmentation to improve robustness
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(x_train)

# Function to create a CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train three models with different initializations
models = []
for i in range(3):
    print(f"\nTraining Model {i+1}...")
    model = create_model()
    model.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        epochs=15,
        validation_data=(x_test, y_test),
        verbose=1
    )
    # Evaluate model
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Model {i+1} test accuracy: {accuracy:.4f}")
    models.append(model)
    # Save model
    model.save(f'D:\\ML\\model{i+1}.keras')

# Save a fallback model (single model for backup)
print("\nTraining fallback model...")
fallback_model = create_model()
fallback_model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    epochs=15,
    validation_data=(x_test, y_test),
    verbose=1
)
loss, accuracy = fallback_model.evaluate(x_test, y_test, verbose=0)
print(f"Fallback model test accuracy: {accuracy:.4f}")
fallback_model.save('D:\\ML\\mnist_model.keras')

print("\nAll models trained and saved successfully!")