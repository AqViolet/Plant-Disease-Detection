# train_plant_disease.py

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

# -----------------------------
# 1. Dataset paths
# -----------------------------
train_dir = "data/train"
valid_dir = "data/valid"
test_dir = "data/test"

# -----------------------------
# 2. Load datasets
# -----------------------------
batch_size = 32
img_size = (128, 128)

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="categorical",
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    valid_dir,
    labels="inferred",
    label_mode="categorical",
    batch_size=batch_size,
    image_size=img_size,
    shuffle=True
)

# test_ds = tf.keras.utils.image_dataset_from_directory(
#     test_dir,
#     labels="inferred",
#     label_mode="categorical",
#     batch_size=1,
#     image_size=img_size,
#     shuffle=False
# )

num_classes = len(train_ds.class_names)
print(f"Number of classes: {num_classes}")
print("Classes:", train_ds.class_names)

# -----------------------------
# 3. Build CNN Model (Keras 3)
# -----------------------------
cnn = tf.keras.models.Sequential([
    tf.keras.Input(shape=(128, 128, 3)),  # Input layer
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),

    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),

    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),

    tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(256, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),

    tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu'),
    tf.keras.layers.Conv2D(512, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),

    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1500, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn.summary()

# -----------------------------
# 4. Train the model
# -----------------------------
epochs = 5
history = cnn.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=epochs
)

# -----------------------------
# 5. Evaluate
# -----------------------------
train_loss, train_acc = cnn.evaluate(train_ds)
val_loss, val_acc = cnn.evaluate(valid_ds)
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

# -----------------------------
# 6. Save the model
# -----------------------------
cnn.save("trained_plant_disease_model.keras")
print("Model saved as trained_plant_disease_model.keras")

# -----------------------------
# 7. Plot Accuracy
# -----------------------------
plt.figure(figsize=(8,6))
plt.plot(range(1, epochs+1), history.history['accuracy'], label="Train Accuracy", color='red')
plt.plot(range(1, epochs+1), history.history['val_accuracy'], label="Validation Accuracy", color='blue')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()

# -----------------------------
# 8. Confusion Matrix & Classification Report
# -----------------------------
# y_pred = cnn.predict(test_ds)
predicted_classes = tf.argmax(y_pred, axis=1)
# true_classes = tf.concat([y for _, y in test_ds], axis=0)
true_classes = tf.argmax(true_classes, axis=1)

cm = confusion_matrix(true_classes, predicted_classes)
print(classification_report(true_classes, predicted_classes, target_names=train_ds.class_names))

plt.figure(figsize=(20,20))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_ds.class_names, yticklabels=train_ds.class_names)
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
plt.show()
