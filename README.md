# =====================================
# AI Tools & Frameworks Assignment
# Consolidated Practical Implementation
# =====================================

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Scikit-learn (Classical ML)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# TensorFlow (Deep Learning)
import tensorflow as tf
from tensorflow.keras import layers, models

# spaCy (NLP)
import spacy

# =====================================================
# Task 1: Classical ML with Scikit-learn (Iris Dataset)
# =====================================================
print("\n===== TASK 1: Iris Species Prediction =====")

# Load the Iris dataset
iris = load_iris()
X = iris.data      # Feature matrix
y = iris.target    # Labels (species)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")

# =====================================================
# Task 2: Deep Learning with TensorFlow (MNIST)
# =====================================================
print("\n===== TASK 2: MNIST Handwritten Digit Classification =====")

# Load MNIST dataset
(X_train_dl, y_train_dl), (X_test_dl, y_test_dl) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to [0,1]
X_train_dl = X_train_dl / 255.0
X_test_dl = X_test_dl / 255.0

# Reshape images to add channel dimension for CNN input
X_train_dl = X_train_dl.reshape(-1,28,28,1)
X_test_dl = X_test_dl.reshape(-1,28,28,1)

# Build Convolutional Neural Network (CNN)
cnn_model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)), # Conv layer
    layers.MaxPooling2D((2,2)),                                         # Max pooling
    layers.Conv2D(64, (3,3), activation='relu'),                        # Conv layer
    layers.MaxPooling2D((2,2)),                                         # Max pooling
    layers.Flatten(),                                                    # Flatten for dense layers
    layers.Dense(64, activation='relu'),                                 # Fully connected layer
    layers.Dense(10, activation='softmax')                               # Output layer for 10 classes
])

# Compile the CNN model
cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train the model for 5 epochs
history = cnn_model.fit(X_train_dl, y_train_dl, epochs=5, validation_split=0.1)

# Evaluate the model on test data
test_loss, test_acc = cnn_model.evaluate(X_test_dl, y_test_dl)
print(f"Test Accuracy: {test_acc:.2f}")

# Visualize predictions on 5 random test images
sample_idx = random.sample(range(len(X_test_dl)), 5)
for i in sample_idx:
    img = X_test_dl[i].reshape(28,28)
    pred = np.argmax(cnn_model.predict(X_test_dl[i].reshape(1,28,28,1)))
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {pred}, Actual: {y_test_dl[i]}")
    plt.show()

# =====================================================
# Task 3: NLP with spaCy (Amazon Reviews)
# =====================================================
print("\n===== TASK 3: NLP - Named Entity Recognition and Sentiment =====")

# Load spaCy English model
nlp = spacy.load('en_core_web_sm')

# Sample Amazon reviews
reviews = [
    "I love the Samsung Galaxy phone, it works perfectly.",
    "The Sony headphones are terrible and broke after 2 days.",
    "The Amazon Echo Dot is excellent and very useful."
]

# Perform Named Entity Recognition (NER) and rule-based sentiment analysis
for review in reviews:
    doc = nlp(review)
    print(f"\nReview: {review}")
    
    # Extract Named Entities
    print("Entities:")
    for ent in doc.ents:
        print(f" - {ent.text} ({ent.label_})")
    
    # Rule-based sentiment analysis
    positive_words = ['love', 'perfect', 'great', 'excellent', 'useful']
    negative_words = ['terrible', 'broke', 'bad', 'awful']
    
    if any(word in review.lower() for word in positive_words):
        sentiment = "Positive"
    elif any(word in review.lower() for word in negative_words):
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    print(f"Sentiment: {sentiment}")

# =====================================================
# End of Consolidated Practical Implementation
# =====================================================
print("\nAll tasks completed successfully!")
