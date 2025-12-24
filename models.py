# =====================================
# AI Tools & Frameworks Assignment
# Consolidated Practical Implementation
# =====================================

# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluate
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

# Normalize images
X_train_dl = X_train_dl / 255.0
X_test_dl = X_test_dl / 255.0

# Reshape for CNN
X_train_dl = X_train_dl.reshape(-1,28,28,1)
X_test_dl = X_test_dl.reshape(-1,28,28,1)

# Build CNN model
cnn_model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Train model
history = cnn_model.fit(X_train_dl, y_train_dl, epochs=5, validation_split=0.1)

# Evaluate
test_loss, test_acc = cnn_model.evaluate(X_test_dl, y_test_dl)
print(f"Test Accuracy: {test_acc:.2f}")

# Visualize 5 sample predictions
import random
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

# Sample reviews
reviews = [
    "I love the Samsung Galaxy phone, it works perfectly.",
    "The Sony headphones are terrible and broke after 2 days.",
    "The Amazon Echo Dot is excellent and very useful."
]

for review in reviews:
    doc = nlp(review)
    print(f"\nReview: {review}")
    
    # Named Entities
    print("Entities:")
    for ent in doc.ents:
        print(f" - {ent.text} ({ent.label_})")
    
    # Simple rule-based sentiment analysis
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
