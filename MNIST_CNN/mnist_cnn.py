"""
DEBUGGED & OPTIMIZED CNN FOR MNIST DATASET
TROUBLESHOOTING CHALLENGE - DEBUGGED VERSION
Author: Anthonia Othetheaso
Date: October 17, 2025
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("=" * 70)
print("TROUBLESHOOTING CHALLENGE - DEBUGGED MNIST CNN")
print("=" * 70)

# ============================================================================
# STEP 1: SYSTEM CONFIGURATION
# ============================================================================
print("\n[STEP 1] Checking System Configuration...")
print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# ============================================================================
# STEP 2: LOAD AND PREPROCESS DATA
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 2] Loading and Preprocessing MNIST Dataset...")
print("=" * 70)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"✓ Dataset loaded successfully!")
print(f"  - Training samples: {len(x_train)}")
print(f"  - Testing samples: {len(x_test)}")

# BUG FIX 1: Proper normalization with float division
print("\n✓ Fixing normalization (float division)...")
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# BUG FIX 2: Add channel dimension for CNN
print("✓ Fixing dimension mismatch (adding channel dimension)...")
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# BUG FIX 3: Ensure correct label types
y_train = y_train.astype('int32')
y_test = y_test.astype('int32')

print(f"  - Final training shape: {x_train.shape}")
print(f"  - Final testing shape: {x_test.shape}")

# ============================================================================
# STEP 3: BUILD IMPROVED CNN MODEL
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 3] Building Debugged CNN Architecture...")
print("=" * 70)

model = tf.keras.models.Sequential([
    # Convolutional Block 1
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                          input_shape=(28, 28, 1), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    # Convolutional Block 2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    
    # Fully Connected Layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    
    # Output Layer
    tf.keras.layers.Dense(10, activation='softmax')
])

print("\n✓ Model Architecture:")
model.summary()

# ============================================================================
# STEP 4: COMPILE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 4] Compiling Model...")
print("=" * 70)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model compiled successfully!")

# ============================================================================
# STEP 5: SETUP CALLBACKS
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 5] Setting Up Training Callbacks...")
print("=" * 70)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_mnist_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

callbacks = [early_stopping, reduce_lr, checkpoint]

print("✓ Callbacks configured for optimized training")

# ============================================================================
# STEP 6: TRAIN MODEL (OPTIMIZED FOR SPEED)
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 6] Training Model (Optimized for Speed)...")
print("=" * 70)

# OPTIMIZED: Reduced epochs and larger batch size for faster training
history = model.fit(
    x_train, y_train,
    batch_size=256,    # ✅ Larger batch = faster training
    epochs=5,          
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

print("\n✓ Training completed in just 5 epochs!")

# ============================================================================
# STEP 7: EVALUATE MODEL
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 7] Evaluating Model Performance...")
print("=" * 70)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"✓ Test Loss: {test_loss:.4f}")
print(f"✓ Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Get predictions
y_pred = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# ============================================================================
# STEP 8: PERFORMANCE ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 8] Performance Analysis...")
print("=" * 70)

print("\n✓ Classification Report:")
print(classification_report(y_test, y_pred_classes))

print("\n✓ Per-Class Accuracy:")
for digit in range(10):
    mask = y_test == digit
    digit_accuracy = np.mean(y_pred_classes[mask] == digit)
    print(f"  - Digit {digit}: {digit_accuracy:.4f}")

# ============================================================================
# STEP 9: VISUALIZE RESULTS
# ============================================================================
print("\n" + "=" * 70)
print("[STEP 9] Generating Results Visualization...")
print("=" * 70)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Training history
ax1.plot(history.history['accuracy'], label='Training Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history.history['loss'], label='Training Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
ax3.set_title('Confusion Matrix')
ax3.set_xlabel('Predicted')
ax3.set_ylabel('Actual')

# Sample predictions
correct_idx = np.where(y_pred_classes == y_test)[0]
incorrect_idx = np.where(y_pred_classes != y_test)[0]

if len(correct_idx) > 0 and len(incorrect_idx) > 0:
    sample_correct = np.random.choice(correct_idx, 1)
    sample_incorrect = np.random.choice(incorrect_idx, 1)
    
    ax4.imshow(x_test[sample_correct[0]].reshape(28, 28), cmap='gray')
    ax4.set_title(f'Correct: True={y_test[sample_correct[0]]}, Pred={y_pred_classes[sample_correct[0]]}')
    ax4.axis('off')

plt.tight_layout()
plt.show()

print("=" * 70)
print("DEBUGGING CHALLENGE - COMPLETED SUCCESSFULLY!")
print("=" * 70)