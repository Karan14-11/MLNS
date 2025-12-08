import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Load the dataset (replace with the actual path to load_data.py script)

def load_data():
    # Load training data and labels
    train_data1 = np.load('data0.npy')
    train_lab1 = np.load('lab0.npy')

    # Stack additional datasets if needed (example for multiple files)
    # train_data2 = np.load('data1.npy')
    # train_lab2 = np.load('lab1.npy')
    # data = np.concatenate([train_data1, train_data2], axis=0)
    # labels = np.concatenate([train_lab1, train_lab2], axis=0)

    # For now, use a single dataset
    data = train_data1
    labels = train_lab1

    return data, labels

# Load data and labels
data, labels = load_data()  
# Load data and labels
data, labels = load_data()  # Assuming load_data returns numpy arrays for images and labels

# Normalize data and preprocess
# Assuming data is of shape (num_samples, height, width)
data = data / 255.0  # Normalize pixel values to [0, 1]
data = np.expand_dims(data, axis=-1)  # Add channel dimension (e.g., for grayscale images)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define the baseline CNN model
def create_baseline_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(data.shape[1], data.shape[2], 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')  # Linear activation for regression
    ])
    return model

# Compile the model
model = create_baseline_model()
model.compile(optimizer=Adam(learning_rate=0.001), loss='cross_entopy_loss', metrics=['cross_entopy_loss'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=32,
    verbose=1
)

# Evaluate the model on the validation set
val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}")

# Save the model
model.save("baseline_cnn_model.h5")

# Plot training and validation loss/metrics
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('MAE vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()

plt.show()

# Example prediction
example_image = X_val[0:1]  # Take a single example
predicted_sum = model.predict(example_image)
print(f"Predicted Sum: {predicted_sum[0][0]}, Actual Sum: {y_val[0]}")
