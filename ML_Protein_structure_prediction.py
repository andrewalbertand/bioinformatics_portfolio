# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

## The data can be found at the following link: https://archive.ics.uci.edu/dataset/866/9mers+from+cullpdb ##
# Define the file paths
parquet_path = '/path/to/file/9mers.parquet'
txt_path = '/path/to/file/amino_acid_order.txt'

# Load the dataset from Parquet file
data = pd.read_parquet(parquet_path)

# Read the amino acid order file
with open(txt_path, 'r') as file:
    amino_acid_order = file.read()

# Preliminary EDA
print(data.head())
print(data.columns)
print(len(data))
print(data.shape)

# Helper function to process each row individually
def process_row(row, tuple_size):
    try:
        # Initialize an empty list to store the normalized and reshaped arrays
        processed_data = []
        for sub_row in row:
            # Convert list to numpy array and flatten
            array_data = np.array(sub_row, dtype=np.float32).flatten()
            # Check if the total elements in array is divisible by tuple_size
            if len(array_data) % tuple_size != 0:
                continue  # Skip this sub-row if it's not divisible
            # Reshape and scale
            reshaped_data = array_data.reshape(-1, tuple_size)
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(reshaped_data).flatten()  # Flatten back after scaling
            processed_data.append(normalized_data)
        return processed_data if processed_data else None  # Return None if no valid data found
    except Exception as e:
        print(f"Error processing row: {e}")
        return None
        
# Sample a subset of the data for faster demonstration
subset_data = data.sample(n=500, random_state=42)  # Adjust n to preferred sample size

# Apply processing to each row of the subset DataFrame
subset_data['phi_psi_angles'] = subset_data['phi_psi_angles'].apply(lambda x: process_row(x, 2))
subset_data['3d_coordinates'] = subset_data['3d_coordinates'].apply(lambda x: process_row(x, 3))

# Flatten the target arrays for training
y_train_angles_flat = y_train_angles_subset.reshape(y_train_angles_subset.shape[0], -1)
y_test_angles_flat = y_test_angles_subset.reshape(y_test_angles_subset.shape[0], -1)

# Define the model
model_subset = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(9,)),  # Adjust the input shape based on your data
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(y_train_angles_flat.shape[1])  # Set the output layer to match the flattened shape
])

# Compile the model
model_subset.compile(optimizer='adam', loss='mse')

# Train the model
history_subset = model_subset.fit(X_train_subset, y_train_angles_flat, epochs=10, validation_split=0.1)

# Evaluate the model on the test set
test_loss_subset = model_subset.evaluate(X_test_subset, y_test_angles_flat)
print('Test Loss:', test_loss_subset)

# Plotting the training and validation loss
plt.plot(history_subset.history['loss'], label='Train Loss')
plt.plot(history_subset.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Histogram of predictions
predictions = model_subset.predict(X_test_subset).flatten()
plt.figure(figsize=(10, 6))
plt.hist(predictions, bins=20, color='blue', alpha=0.7, rwidth=0.85)
plt.title('Distribution of Predicted Values', fontsize=16)
plt.xlabel('Predicted Value')
plt.ylabel('Frequency')
plt.show()