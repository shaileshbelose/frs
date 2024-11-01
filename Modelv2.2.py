import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.utils import shuffle
import json

# Specify the path where LFW dataset is located
lfw_pairs_path = 'cropped'
pairs_file_path = '.'  # Update this if needed

# Function to read pairs from the txt file
def read_pairs(filename):
    pairs = []
    with open(filename, 'r') as f:
        next(f)  # Skip the header
        for line in f:
            pair = line.strip().split()
            if len(pair) == 3:
                pairs.append((pair[0], int(pair[1]), pair[0], int(pair[2])))
            elif len(pair) == 4:
                pairs.append((pair[0], int(pair[1]), pair[2], int(pair[3])))
    return pairs

# Function to load images based on name and index
# Size we have kept as 250X250
def load_image(name, index):
    image_path = os.path.join(lfw_pairs_path, name, f"{name}_{index:04d}.png")
    print(f"Trying to load image from path: {image_path}")  # Debug print
    if os.path.exists(image_path):
        img = cv2.imread(image_path)  # Load color image
        if img is not None:
            return cv2.resize(img, (250, 250))  # Resize to match the input shape
    return None

# Load the pairs from the train and test files
train_pairs = read_pairs(os.path.join(pairs_file_path, 'pairsDevTrain.txt'))
test_pairs = read_pairs(os.path.join(pairs_file_path, 'pairsDevTest.txt'))

# Prepare training data
X = []
y = []
class_names = []

for pair in train_pairs:
    img1 = load_image(pair[0], pair[1])
    img2 = load_image(pair[2], pair[3])


    # if img1 is None or img2 is None:
    #     print(f"Error loading images: {pair[0]}_{pair[1]}, {pair[2]}_{pair[3]}")
    #     continue

    if img1 is not None and img2 is not None:
        combined_img = np.hstack([img1, img2])
        X.append(combined_img)
        y.append(1 if pair[0] == pair[2] else 0)
        if pair[0] not in class_names:
            class_names.append(pair[0])
        if pair[2] not in class_names:
            class_names.append(pair[2])

# Convert to NumPy arrays and preprocess
X = np.array(X) / 255.0  # Normalize
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Shuffle the data for random testing
X, y = shuffle(X, y, random_state=42)

num_classes = len(class_names)

# Build the CNN model with multiple layers
model = models.Sequential([
    layers.InputLayer(input_shape=(250, 500, 3)),  # Combined image size (250x500) and 3 channels
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.BatchNormalization(),
    layers.Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.001))
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with 40 epochs, batch_size=32 and validation test data used as 20%
history = model.fit(X, y, epochs=40, batch_size=32, validation_split=0.2)

# Save the model and the class names
model.save('FRSv_croppedv1.1.keras')
with open('class_names.json', 'w') as f:
    json.dump(class_names, f)
