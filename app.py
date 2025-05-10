import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import base64
from PIL import Image
import io

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app, resources={r"/*": {"origins": ["http://localhost:5000", "http://localhost:5500", "*"]}})

# Load and preprocess MNIST dataset
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print("MNIST dataset loaded and preprocessed.")

# Define model filenames
MODEL_FILES = ['model1.keras', 'model2.keras', 'model3.keras']
FINAL_MODEL = 'mnist_model.keras'

# Data augmentation for better generalization
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    fill_mode='nearest'
)

# Build and train a model
def build_model():
    print("Building new model...")
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Load or train individual models
models = []
for model_file in MODEL_FILES:
    if os.path.exists(model_file):
        print(f"Loading existing model from {model_file}...")
        try:
            model = load_model(model_file)
            print(f"Successfully loaded model from {model_file}")
        except Exception as e:
            print(f"Error loading model from {model_file}: {str(e)}")
            print(f"Training a new model for {model_file}...")
            model = build_model()
            datagen.fit(x_train)
            model.fit(datagen.flow(x_train, y_train, batch_size=64), 
                      epochs=10, 
                      validation_data=(x_test, y_test))
            model.save(model_file)
            print(f"Trained and saved new model as {model_file}")
    else:
        print(f"Model file {model_file} not found. Training a new model...")
        model = build_model()
        datagen.fit(x_train)
        model.fit(datagen.flow(x_train, y_train, batch_size=64), 
                  epochs=10, 
                  validation_data=(x_test, y_test))
        model.save(model_file)
        print(f"Trained and saved new model as {model_file}")
    models.append(model)

# Load or train the final ensemble model
if os.path.exists(FINAL_MODEL):
    print(f"Loading final model from {FINAL_MODEL}...")
    try:
        final_model = load_model(FINAL_MODEL)
        print(f"Successfully loaded final model from {FINAL_MODEL}")
    except Exception as e:
        print(f"Error loading final model from {FINAL_MODEL}: {str(e)}")
        print(f"Training a new final model for {FINAL_MODEL}...")
        final_model = build_model()
        datagen.fit(x_train)
        final_model.fit(datagen.flow(x_train, y_train, batch_size=64), 
                        epochs=15, 
                        validation_data=(x_test, y_test))
        final_model.save(FINAL_MODEL)
        print(f"Trained and saved final model as {FINAL_MODEL}")
else:
    print(f"Final model {FINAL_MODEL} not found. Training a new final model...")
    final_model = build_model()
    datagen.fit(x_train)
    final_model.fit(datagen.flow(x_train, y_train, batch_size=64), 
                    epochs=15, 
                    validation_data=(x_test, y_test))
    final_model.save(FINAL_MODEL)
    print(f"Trained and saved final model as {FINAL_MODEL}")

# Evaluate the final model
test_loss, test_accuracy = final_model.evaluate(x_test, y_test, verbose=0)
print(f"Final model test accuracy: {test_accuracy}")

@app.route('/')
def serve_index():
    print("Serving index.html from / endpoint")
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(f"Received {request.method} request at /predict endpoint")
        if request.method != 'POST':
            print("Method not allowed: Expected POST, received", request.method)
            return jsonify({'error': 'Method not allowed, please use POST'}), 405

        data = request.get_json()
        if not data or 'image' not in data:
            print("Error: No image data provided")
            return jsonify({'error': 'No image data provided'}), 400

        # Extract base64 image data
        image_data = data['image']
        if not image_data.startswith('data:image/png;base64,'):
            print("Error: Invalid image format")
            return jsonify({'error': 'Invalid image format'}), 400

        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28
        image_array = np.array(image).astype('float32') / 255.0
        image_array = image_array.reshape(1, 28, 28, 1)

        # Invert the image to match MNIST (white digit on black background)
        image_array = 1.0 - image_array  # Invert: white (1) becomes black (0), black (0) becomes white (1)

        # Make prediction
        print("Making prediction with image shape:", image_array.shape)
        prediction = final_model.predict(image_array)
        digit = np.argmax(prediction, axis=1)[0]
        probabilities = prediction[0].tolist()

        print(f"Predicted digit: {digit}, Probabilities: {probabilities}")
        return jsonify({'digit': int(digit), 'probabilities': probabilities})
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)