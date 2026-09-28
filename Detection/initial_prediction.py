import cv2
import json
import numpy as np
import tensorflow as tf
from collections import deque

# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "parking_slot_binary_model.keras"
COORDINATES_PATH = "Detection/slot_coordinates.json"

# Android IP Camera
CAMERA_URL = "http://10.161.43.10:8080/video"

IMG_SIZE = 224

# Prediction smoothing
SMOOTHING_FRAMES = 7

# ============================================================
# IMPORTANT THRESHOLDS
# ============================================================

# A slot will be shown as EMPTY/AVAILABLE
# ONLY when empty confidence is >= 90%
EMPTY_CONFIDENCE_THRESHOLD = 0.90

# A slot is considered occupied when occupied confidence
# reaches 50% or more.
OCCUPIED_CONFIDENCE_THRESHOLD = 0.50


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# LOAD SLOT COORDINATES
# ============================================================

with open(COORDINATES_PATH, "r") as f:
    slot_coordinates = json.load(f)

print("Slot coordinates loaded successfully!")


# ============================================================
# CONNECT TO ANDROID CAMERA
# ============================================================

print("Connecting to Android camera...")

cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():

    print("❌ Could not connect to Android camera")
    print()
    print("Check:")
    print("1. Phone camera server is running")
    print("2. Phone and laptop are on same Wi-Fi")
    print("3. IP address is correct")
    print("4. /video works in Chrome")

    exit()

print("✅ Android camera connected!")


# ============================================================
# PREDICTION HISTORY
# ============================================================

prediction_history = {

    "slot1": deque(maxlen=SMOOTHING_FRAMES),
    "slot2": deque(maxlen=SMOOTHING_FRAMES),
    "slot3": deque(maxlen=SMOOTHING_FRAMES),
    "slot4": deque(maxlen=SMOOTHING_FRAMES)

}


# ============================================================
# PREDICT SLOT
# ============================================================

def predict_slot(frame, points):

    points = np.array(
        points,
        dtype=np.int32
    )

    # --------------------------------------------------------
    # Create mask
    # --------------------------------------------------------

    mask = np.zeros(
        frame.shape[:2],
        dtype=np.uint8
    )

    cv2.fillPoly(
        mask,
        [points],
        255
    )

    # --------------------------------------------------------
    # Bounding rectangle
    # --------------------------------------------------------

    x, y, w, h = cv2.boundingRect(points)

    cropped = frame[
        y:y+h,
        x:x+w
    ]

    cropped_mask = mask[
        y:y+h,
        x:x+w
    ]

    # --------------------------------------------------------
    # Apply polygon mask
    # --------------------------------------------------------

    cropped = cv2.bitwise_and(
        cropped,
        cropped,
        mask=cropped_mask
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    resized = cv2.resize(
        cropped,
        (IMG_SIZE, IMG_SIZE)
    )

    # BGR → RGB
    resized = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    # Normalize
    resized = resized.astype(
        np.float32
    ) / 255.0

    # Add batch dimension
    resized = np.expand_dims(
        resized,
        axis=0
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    prediction = model.predict(
        resized,
        verbose=0
    )[0][0]

    occupied_probability = float(
        prediction
    )

    empty_probability = 1.0 - occupied_probability

    return empty_probability, occupied_probability


# ============================================================
# WINDOW
# ============================================================

WINDOW_NAME = "Smart Parking - Android Camera"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print(
            "❌ Failed to receive frame from Android camera"
        )

        break

    display_frame = frame.copy()

    available_count = 0
    occupied_count = 0
    uncertain_count = 0


    # ========================================================
    # PROCESS EACH SLOT
    # ========================================================

    for slot_name, points in slot_coordinates.items():

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        empty_probability, occupied_probability = predict_slot(
            frame,
            points
        )

        # ----------------------------------------------------
        # Store occupied probability for smoothing
        # ----------------------------------------------------

        prediction_history[
            slot_name
        ].append(
            occupied_probability
        )

        # ----------------------------------------------------
        # Smoothed probability
        # ----------------------------------------------------

        smoothed_occupied = np.mean(
            prediction_history[
                slot_name
            ]
        )

        smoothed_empty = 1.0 - smoothed_occupied


        # ====================================================
        # STATUS DECISION
        # ====================================================

        # EMPTY ONLY IF EMPTY CONFIDENCE >= 90%
        if smoothed_empty >= EMPTY_CONFIDENCE_THRESHOLD:

            status = "EMPTY"
            available_count += 1

            polygon_color = (
                0,
                255,
                0
            )

            confidence = smoothed_empty * 100


        # OCCUPIED
        elif smoothed_occupied >= OCCUPIED_CONFIDENCE_THRESHOLD:

            status = "OCCUPIED"
            occupied_count += 1

            polygon_color = (
                0,
                0,
                255
            )

            confidence = smoothed_occupied * 100


        # UNCERTAIN
        else:

            status = "CHECK"
            uncertain_count += 1

            polygon_color = (
                0,
                165,
                255
            )

            # Show the stronger probability
            confidence = max(
                smoothed_empty,
                smoothed_occupied
            ) * 100


        # ----------------------------------------------------
        # Slot polygon
        # ----------------------------------------------------

        points_array = np.array(
            points,
            dtype=np.int32
        )

        cv2.polylines(
            display_frame,
            [points_array],
            True,
            polygon_color,
            4
        )


        # ----------------------------------------------------
        # Find center
        # ----------------------------------------------------

        center = np.mean(
            points_array,
            axis=0
        ).astype(int)

        text_x = int(center[0])
        text_y = int(center[1])


        # ----------------------------------------------------
        # Slot name
        # ----------------------------------------------------

        slot_number = slot_name.replace(
            "slot",
            "Slot "
        )

        cv2.putText(
            display_frame,
            slot_number,
            (
                text_x - 45,
                text_y - 25
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            polygon_color,
            2,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            status,
            (
                text_x - 50,
                text_y + 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            polygon_color,
            2,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        cv2.putText(
            display_frame,
            f"{confidence:.1f}%",
            (
                text_x - 35,
                text_y + 30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            polygon_color,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    cv2.rectangle(
        display_frame,
        (0, 0),
        (470, 120),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        display_frame,
        f"Available: {available_count} / 4",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        display_frame,
        f"Occupied: {occupied_count} / 4",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        display_frame,
        f"Check: {uncertain_count} / 4",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 165, 255),
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # SHOW FRAME
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        display_frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    # Q = Quit
    if key == ord("q"):
        break

    # S = Save screenshot
    elif key == ord("s"):

        filename = "parking_detection.jpg"

        cv2.imwrite(
            filename,
            display_frame
        )

        print(
            f"Screenshot saved: {filename}"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("Camera detection stopped.")