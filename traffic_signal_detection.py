# ============================================================
#   Traffic Signal Detection Using Deep Learning
#   CNN + MobileNetV2 Transfer Learning
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── Configuration ────────────────────────────────────────────
IMG_SIZE    = (96, 96)
BATCH_SIZE  = 32
EPOCHS      = 20
DATA_DIR    = "dataset/"       # Subfolders: Red/, Yellow/, Green/
TEST_DIR    = "test_dataset/"
MODEL_PATH  = "best_model.h5"

# ── Step 1: Data Loading & Augmentation ─────────────────────
print("\n[1] Loading dataset...")

train_gen = ImageDataGenerator(
    rescale           = 1./255,
    rotation_range    = 15,
    zoom_range        = 0.15,
    width_shift_range = 0.1,
    height_shift_range= 0.1,
    validation_split  = 0.1
)

train_ds = train_gen.flow_from_directory(
    DATA_DIR,
    target_size  = IMG_SIZE,
    batch_size   = BATCH_SIZE,
    class_mode   = 'categorical',
    subset       = 'training',
    seed         = 42
)

val_ds = train_gen.flow_from_directory(
    DATA_DIR,
    target_size  = IMG_SIZE,
    batch_size   = BATCH_SIZE,
    class_mode   = 'categorical',
    subset       = 'validation',
    seed         = 42
)

test_gen = ImageDataGenerator(rescale=1./255)
test_ds  = test_gen.flow_from_directory(
    TEST_DIR,
    target_size = IMG_SIZE,
    batch_size  = BATCH_SIZE,
    class_mode  = 'categorical',
    shuffle     = False
)

print("Classes:", train_ds.class_indices)

# ── Step 2: Build Model ──────────────────────────────────────
print("\n[2] Building model...")

base = MobileNetV2(
    input_shape = (*IMG_SIZE, 3),
    include_top = False,
    weights     = 'imagenet'
)
base.trainable = False          # Freeze pretrained weights

inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
x       = base(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.5)(x)
outputs = layers.Dense(3, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer = tf.keras.optimizers.Adam(1e-4),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

model.summary()

# ── Step 3: Train ────────────────────────────────────────────
print("\n[3] Training model...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor            = 'val_accuracy',
        patience           = 5,
        restore_best_weights = True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        save_best_only = True,
        monitor        = 'val_accuracy'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.5,
        patience = 3,
        verbose  = 1
    )
]

history = model.fit(
    train_ds,
    epochs          = EPOCHS,
    validation_data = val_ds,
    callbacks       = callbacks,
    verbose         = 1
)

# Fine-tuning: unfreeze last 30 layers
print("\n[3b] Fine-tuning last 30 layers...")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer = tf.keras.optimizers.Adam(1e-5),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

history_ft = model.fit(
    train_ds,
    epochs          = 10,
    validation_data = val_ds,
    callbacks       = callbacks,
    verbose         = 1
)

# ── Step 4: Evaluate ─────────────────────────────────────────
print("\n[4] Evaluating on test set...")

loss, acc = model.evaluate(test_ds)
print(f"Test Accuracy : {acc * 100:.2f}%")
print(f"Test Loss     : {loss:.4f}")

y_pred = np.argmax(model.predict(test_ds), axis=1)
y_true = test_ds.classes
class_names = list(test_ds.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# ── Step 5: Plot Results ─────────────────────────────────────
print("\n[5] Plotting results...")

# Merge histories
all_acc     = history.history['accuracy']     + history_ft.history['accuracy']
all_val_acc = history.history['val_accuracy'] + history_ft.history['val_accuracy']
all_loss    = history.history['loss']         + history_ft.history['loss']
all_val_loss= history.history['val_loss']     + history_ft.history['val_loss']

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(all_acc,     label='Train Accuracy')
axes[0].plot(all_val_acc, label='Val Accuracy')
axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].grid(True)

axes[1].plot(all_loss,     label='Train Loss')
axes[1].plot(all_val_loss, label='Val Loss')
axes[1].set_title('Loss'); axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=130)
plt.show()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=130)
plt.show()

# ── Step 6: Predict on Single Image ─────────────────────────
from tensorflow.keras.preprocessing import image as kimage

def predict_signal(img_path, model=model):
    img  = kimage.load_img(img_path, target_size=IMG_SIZE)
    x    = kimage.img_to_array(img) / 255.0
    x    = np.expand_dims(x, axis=0)
    pred = model.predict(x)[0]
    cls  = list(test_ds.class_indices.keys())
    idx  = np.argmax(pred)
    print(f"Predicted : {cls[idx]}  ({pred[idx]*100:.1f}%)")
    plt.imshow(img)
    plt.title(f"Signal: {cls[idx]} ({pred[idx]*100:.1f}%)")
    plt.axis('off')
    plt.show()

# Example usage:
# predict_signal("test_images/red_signal.jpg")
# predict_signal("test_images/green_signal.jpg")
# predict_signal("test_images/yellow_signal.jpg")

# ── Step 7: Save Final Model ─────────────────────────────────
model.save("traffic_signal_model_final.h5")
print("\nModel saved to traffic_signal_model_final.h5")
print("Done!")
