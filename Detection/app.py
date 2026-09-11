import cv2
import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# 1. SETTINGS
# ============================================================

DATASET_PATH = "Detection/dataset"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 40

labels = {
    "b0": 0,   # 0 bikes
    "b1": 1,   # 1 bike
    "b2": 2,   # 2 bikes
    "b3": 3,   # 3 bikes
    "b4": 4    # 4 bikes
}


# ============================================================
# 2. LOAD DATASET
# ============================================================

X = []
y = []

print("\nLoading dataset...\n")

for folder_name, label in labels.items():

    folder_path = os.path.join(DATASET_PATH, folder_name)

    if not os.path.exists(folder_path):
        print("Folder not found:", folder_path)
        continue

    count = 0

    for file_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, file_name)

        img = cv2.imread(image_path)

        if img is None:
            continue

        # Resize image
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Convert BGR → RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        X.append(img)
        y.append(label)

        count += 1

    print(f"{folder_name}: {count} images")


X = np.array(X, dtype="float32")
y = np.array(y)

print("\nTotal images:", len(X))


# ============================================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First: 80% train, 20% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Second: temporary → validation + test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDataset split:")
print("Training   :", len(X_train))
print("Validation :", len(X_val))
print("Testing    :", len(X_test))


# ============================================================
# 4. PREPROCESS IMAGES
# ============================================================

X_train = preprocess_input(X_train)
X_val = preprocess_input(X_val)
X_test = preprocess_input(X_test)


# ============================================================
# 5. DATA AUGMENTATION
# ============================================================

datagen = ImageDataGenerator(
    rotation_range=8,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.15,
    brightness_range=[0.75, 1.25],
    horizontal_flip=True,
    fill_mode="nearest"
)


# ============================================================
# 6. LOAD PRETRAINED MOBILENETV2
# ============================================================

print("\nLoading MobileNetV2...\n")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze pretrained layers
base_model.trainable = False


# ============================================================
# 7. BUILD MODEL
# ============================================================

model = Sequential([
    base_model,

    GlobalAveragePooling2D(),

    Dense(128, activation="relu"),

    Dropout(0.4),

    Dense(5, activation="softmax")
])


# ============================================================
# 8. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 9. DISPLAY MODEL
# ============================================================

model.summary()


# ============================================================
# 10. CALLBACKS
# ============================================================

# Stops only if validation loss stops improving
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

# Automatically reduces learning rate
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\n========================================")
print("STARTING TRAINING")
print("========================================\n")

history = model.fit(

    datagen.flow(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        shuffle=True
    ),

    epochs=EPOCHS,

    validation_data=(
        X_val,
        y_val
    ),

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# 12. FINAL TESTING
# ============================================================

print("\n========================================")
print("FINAL TEST RESULTS")
print("========================================\n")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ============================================================
# 13. SAVE MODEL
# ============================================================

model.save("bike_detection_model.keras")

print("\n========================================")
print("MODEL SAVED")
print("========================================")

print("\nFile:")
print("bike_detection_model.keras")


# ============================================================
# 14. PREDICTION FUNCTION
# ============================================================

def predict_empty_slots(image_path):

    img = cv2.imread(image_path)

    if img is None:
        print("Image not found!")
        return None

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Float
    img = img.astype("float32")

    # MobileNetV2 preprocessing
    img = preprocess_input(img)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img, verbose=0)

    predicted_class = np.argmax(prediction[0])

    confidence = prediction[0][predicted_class]

    # Number of empty slots
    total_slots = 4

    empty_slots = total_slots - predicted_class

    return predicted_class, empty_slots, confidence


# ============================================================
# 15. OPTIONAL TEST IMAGE
# ============================================================

# Change this path if you want to test an image after training.

# test_image = "Detection/dataset/b2/example.jpg"
#
# result = predict_empty_slots(test_image)
#
# if result is not None:
#     predicted_class, empty_slots, confidence = result
#
#     print("\nPrediction:")
#     print("Bikes detected:", predicted_class)
#     print("Empty slots:", empty_slots)
#     print("Confidence:", round(confidence * 100, 2), "%")