import os
import csv
import json
import random
import numpy as np
import cv2
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint



ORIGINAL_DATASET = "Detection/original_dataset"
CSV_FILE = "Detection/slot_labels.csv"
COORDINATES_FILE = "Detection/slot_coordinates_for_lap_training.json"

MODEL_FILE = "Detection/parking_slot_binary_model.keras"

IMAGE_SIZE = 224
BATCH_SIZE = 16

RANDOM_SEED = 42

# Dataset split
TRAIN_SIZE = 0.80
VAL_SIZE = 0.10
TEST_SIZE = 0.10

# Training
INITIAL_EPOCHS = 20


# ============================================================
# 2. RANDOM SEEDS
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ============================================================
# 3. LOAD SLOT COORDINATES
# ============================================================

print("\nLoading slot coordinates...")

with open(COORDINATES_FILE, "r") as f:
    slot_coordinates = json.load(f)

print("Slot coordinates loaded successfully.")



# 4. CROP SLOT FROM IMAGE

def crop_slot(image, points):
    """
    Crop a polygon-shaped parking slot from the image.

    The area outside the parking-slot polygon is masked.
    """

    points = np.array(points, dtype=np.int32)

    # Create mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    cv2.fillPoly(mask, [points], 255)

    # Apply mask
    masked = cv2.bitwise_and(image, image, mask=mask)

    # Bounding rectangle around polygon
    x, y, w, h = cv2.boundingRect(points)

    crop = masked[y:y+h, x:x+w]

    return crop



# 5. LOAD LABELS FROM CSV


print("\nLoading labels from:", CSV_FILE)

records = []

with open(CSV_FILE, "r", newline="", encoding="utf-8-sig") as f:

    reader = csv.DictReader(f)

    required_columns = [
        "image_path",
        "slot1",
        "slot2",
        "slot3",
        "slot4"
    ]

    for column in required_columns:

        if column not in reader.fieldnames:
            raise ValueError(
                f"Missing column '{column}' in {CSV_FILE}"
            )

    for row in reader:

        image_path = row["image_path"].strip()

        labels = [
            int(row["slot1"]),
            int(row["slot2"]),
            int(row["slot3"]),
            int(row["slot4"])
        ]

        # Validate labels
        if any(label not in [0, 1] for label in labels):
            raise ValueError(
                f"Invalid label found in: {image_path}"
            )

        records.append({
            "image_path": image_path,
            "labels": labels
        })


print("Total labeled images:", len(records))


# 6. CHECK LABEL DISTRIBUTION

empty_count = 0
occupied_count = 0

for record in records:

    for label in record["labels"]:

        if label == 0:
            empty_count += 1
        else:
            occupied_count += 1


total_slots = empty_count + occupied_count


print("LABEL DISTRIBUTION")


print("Total slot samples :", total_slots)
print("EMPTY samples      :", empty_count)
print("OCCUPIED samples   :", occupied_count)

print(
    "EMPTY percentage    : "
    f"{empty_count / total_slots * 100:.2f}%"
)

print(
    "OCCUPIED percentage : "
    f"{occupied_count / total_slots * 100:.2f}%"
)


# 7. SPLIT ORIGINAL IMAGES


print("SPLITTING DATA")

image_paths = [
    record["image_path"]
    for record in records
]

train_paths, temp_paths = train_test_split(
    image_paths,
    test_size=(VAL_SIZE + TEST_SIZE),
    random_state=RANDOM_SEED,
    shuffle=True
)

val_paths, test_paths = train_test_split(
    temp_paths,
    test_size=0.50,
    random_state=RANDOM_SEED,
    shuffle=True
)

train_paths = set(train_paths)
val_paths = set(val_paths)
test_paths = set(test_paths)

print("Training images   :", len(train_paths))
print("Validation images :", len(val_paths))
print("Testing images    :", len(test_paths))

print(
    "\nTotal:",
    len(train_paths) + len(val_paths) + len(test_paths)
)


# ============================================================
# 8. CREATE LOOKUP TABLE FOR LABELS
# ============================================================

label_lookup = {}

for record in records:

    label_lookup[record["image_path"]] = record["labels"]


# ============================================================
# 9. LOAD SLOT CROPS
# ============================================================

def find_actual_file(relative_path):
    """
    Find the actual image file even if the CSV filename
    has small spacing differences such as:

    2.47.39PM
    2.47.39 PM

    Returns the actual path if found.
    """

    full_path = os.path.join(
        ORIGINAL_DATASET,
        relative_path
    )

    # First try exact path
    if os.path.isfile(full_path):
        return full_path

    # Split folder and filename
    folder = os.path.dirname(relative_path)
    filename = os.path.basename(relative_path)

    folder_path = os.path.join(
        ORIGINAL_DATASET,
        folder
    )

    if not os.path.isdir(folder_path):
        return None

    # Normalize spaces for comparison
    def normalize_name(name):
        return (
            name.lower()
            .replace(" ", "")
            .replace("_", "")
        )

    target = normalize_name(filename)

    # Search folder
    for actual_filename in os.listdir(folder_path):

        actual_normalized = normalize_name(
            actual_filename
        )

        if actual_normalized == target:

            return os.path.join(
                folder_path,
                actual_filename
            )

    return None


def load_dataset(image_path_list):

    images = []
    labels = []

    failed_images = []

    for image_path in image_path_list:

        # Find actual image file
        full_path = find_actual_file(
            image_path
        )

        if full_path is None:

            print(
                "WARNING: Could not find:",
                image_path
            )

            failed_images.append(
                image_path
            )

            continue

        image = cv2.imread(full_path)

        if image is None:

            print(
                "WARNING: Could not read:",
                full_path
            )

            failed_images.append(
                full_path
            )

            continue

        # Convert BGR -> RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image_labels = label_lookup[
            image_path
        ]

        # ----------------------------------------
        # Process all 4 parking slots
        # ----------------------------------------

        for slot_number in range(1, 5):

            slot_key = f"slot{slot_number}"

            points = slot_coordinates[
                slot_key
            ]

            crop = crop_slot(
                image,
                points
            )

            if crop is None or crop.size == 0:

                print(
                    "WARNING: Empty crop:",
                    image_path,
                    slot_key
                )

                continue

            # Resize
            crop = cv2.resize(
                crop,
                (
                    IMAGE_SIZE,
                    IMAGE_SIZE
                ),
                interpolation=cv2.INTER_AREA
            )

            # Convert to float32
            crop = crop.astype(
                np.float32
            )

            # MobileNetV2 preprocessing
            crop = preprocess_input(
                crop
            )

            images.append(crop)

            # Actual label from CSV
            #
            # 0 = EMPTY
            # 1 = OCCUPIED
            labels.append(
                image_labels[
                    slot_number - 1
                ]
            )

    images = np.array(
        images,
        dtype=np.float32
    )

    labels = np.array(
        labels,
        dtype=np.int32
    )

    return (
        images,
        labels,
        failed_images
    )


# ============================================================
# 10. LOAD TRAINING DATA
# ============================================================

print("\nLoading training data...")

X_train, y_train, failed_train = load_dataset(
    list(train_paths)
)

print(
    "Training slot samples:",
    len(X_train)
)


# ============================================================
# 11. LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation data...")

X_val, y_val, failed_val = load_dataset(
    list(val_paths)
)

print(
    "Validation slot samples:",
    len(X_val)
)


# ============================================================
# 12. LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

X_test, y_test, failed_test = load_dataset(
    list(test_paths)
)

print(
    "Test slot samples:",
    len(X_test)
)


# ============================================================
# 13. DISPLAY DATASET DISTRIBUTION
# ============================================================

print("\n==========================================")
print("DATASET SPLIT")
print("==========================================")

print(
    "TRAIN:",
    len(X_train),
    "samples"
)

print(
    "VAL  :",
    len(X_val),
    "samples"
)

print(
    "TEST :",
    len(X_test),
    "samples"
)

print(
    "TOTAL:",
    len(X_train) +
    len(X_val) +
    len(X_test),
    "samples"
)


print("\nTraining labels:")

print(
    "EMPTY    :",
    np.sum(y_train == 0)
)

print(
    "OCCUPIED :",
    np.sum(y_train == 1)
)


# ============================================================
# 14. CLASS WEIGHTS
# ============================================================

classes = np.unique(y_train)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(
        classes,
        class_weights_array
    )
}

print("\n==========================================")
print("CLASS WEIGHTS")
print("==========================================")

print(
    "EMPTY (0)    :",
    class_weights.get(0, 1.0)
)

print(
    "OCCUPIED (1) :",
    class_weights.get(1, 1.0)
)


# ============================================================
# 15. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomRotation(
            0.03
        ),

        layers.RandomTranslation(
            height_factor=0.03,
            width_factor=0.03
        ),

        layers.RandomZoom(
            height_factor=0.08,
            width_factor=0.08
        ),

        layers.RandomContrast(
            0.10
        )
    ],
    name="data_augmentation"
)


# ============================================================
# 16. CREATE MOBILENETV2 MODEL
# ============================================================

print("\n==========================================")
print("CREATING MODEL")
print("==========================================")

base_model = MobileNetV2(
    input_shape=(
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    ),
    include_top=False,
    weights="imagenet"
)

# Freeze base model initially
base_model.trainable = False


inputs = layers.Input(
    shape=(
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    )
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(
    0.35
)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)


model = models.Model(
    inputs,
    outputs
)


# ============================================================
# 17. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy",

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
)


model.summary()


# ============================================================
# 18. CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(
    MODEL_FILE,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_lr=1e-7,
    verbose=1
)




history1 = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=INITIAL_EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ],

    shuffle=True,

    verbose=1
)


# 21. LOAD BEST MODEL

print("\nLoading best saved model...")

model = tf.keras.models.load_model(
    MODEL_FILE
)


# ============================================================
# 22. TEST SET EVALUATION
# ============================================================

print("\n==========================================")
print("FINAL TEST EVALUATION")
print("==========================================")

test_results = model.evaluate(
    X_test,
    y_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

print("\nTest results:")

for name, value in zip(
    model.metrics_names,
    test_results
):

    print(
        f"{name:12s}: {value:.4f}"
    )


# ============================================================
# 23. PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
).ravel()


# Threshold
predictions = (
    probabilities >= 0.5
).astype(int)


# ============================================================
# 24. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n==========================================")
print("TEST ACCURACY")
print("==========================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 25. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1]
)

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(
    "                 Predicted"
)

print(
    "              EMPTY  OCCUPIED"
)

print(
    f"Actual EMPTY   {cm[0][0]:5d}  {cm[0][1]:8d}"
)

print(
    f"Actual OCCUPIED{cm[1][0]:5d}  {cm[1][1]:8d}"
)


# ============================================================
# 26. CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "EMPTY",
            "OCCUPIED"
        ],
        digits=4
    )
)


# ============================================================
# 27. INDIVIDUAL CLASS ACCURACY
# ============================================================

empty_mask = (
    y_test == 0
)

occupied_mask = (
    y_test == 1
)


if np.sum(empty_mask) > 0:

    empty_accuracy = np.mean(
        predictions[empty_mask] == 0
    )

else:

    empty_accuracy = 0


if np.sum(occupied_mask) > 0:

    occupied_accuracy = np.mean(
        predictions[occupied_mask] == 1
    )

else:

    occupied_accuracy = 0


print("\n==========================================")
print("CLASS-WISE ACCURACY")
print("==========================================")

print(
    f"EMPTY accuracy    : "
    f"{empty_accuracy * 100:.2f}%"
)

print(
    f"OCCUPIED accuracy : "
    f"{occupied_accuracy * 100:.2f}%"
)


# ============================================================
# 28. SAVE FINAL MODEL
# ============================================================

model.save(
    MODEL_FILE
)

print("\n==========================================")
print("TRAINING COMPLETE")
print("==========================================")

print(
    "Model saved as:",
    MODEL_FILE
)

print("\nModel purpose:")

print(
    "0 = EMPTY"
)

print(
    "1 = OCCUPIED"
)

print("\nThe model can now be used")
print("for individual parking-slot prediction.")