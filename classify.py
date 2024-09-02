import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Function to load and preprocess images
def load_images(data_dir):
    images = []
    labels = []

    # List all files in the directory
    filenames = os.listdir(data_dir)

    for filename in filenames:
        img_path = os.path.join(data_dir, filename)
        image = cv2.imread(img_path)

        if image is not None:  # Ensure the image is loaded successfully
            image = cv2.resize(image, (64, 64))  # Resize to 64x64 pixels
            images.append(image.flatten())  # Flatten the image for SVM
            labels.append(1 if 'dog' in filename else 0)  # Create labels (1 for dogs, 0 for cats)

    return np.array(images), np.array(labels)

# Path to our dataset
data_dir = 'C:\\Users\\acer\\OneDrive\\Desktop\\Classify_images\\train'

# Load and preprocess images
X, y = load_images(data_dir)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with StandardScaler and SVC
model = make_pipeline(StandardScaler(), SVC(kernel='linear'))

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))