import cv2
import json
import numpy as np

# ============================================================
# SETTINGS
# ============================================================

CAMERA_URL = "http://10.161.43.10:8080/video"

OUTPUT_FILE = "Detection/slot_coordinates.json"

WINDOW_NAME = "Mark Parking Slots"


# ============================================================
# VARIABLES
# ============================================================

points = []
current_slot = 1
slot_coordinates = {}

frozen = False
frame = None
display_frame = None

# Display scaling
scale = 1.0


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global points

    if not frozen:
        return

    if event == cv2.EVENT_LBUTTONDOWN:

        # Convert displayed coordinates
        # back to original image coordinates
        original_x = int(x / scale)
        original_y = int(y / scale)

        points.append([original_x, original_y])

        print(
            f"Slot {current_slot} - Point {len(points)}: "
            f"({original_x}, {original_y})"
        )


# ============================================================
# CAMERA
# ============================================================

print("Connecting to Android camera...")

cap = cv2.VideoCapture(CAMERA_URL)

if not cap.isOpened():
    print("❌ Could not connect to Android camera")
    exit()

print("✅ Android camera connected!")

print()
print("================================================")
print("CONTROLS")
print("================================================")
print("S     → Freeze current camera frame")
print("Click → Select slot corners")
print("ENTER → Save current slot")
print("R     → Reset current slot")
print("Q     → Quit")
print("================================================")
print()


# ============================================================
# WINDOW
# ============================================================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # LIVE CAMERA
    # --------------------------------------------------------

    if not frozen:

        ret, frame = cap.read()

        if not ret:
            print("❌ Failed to receive camera frame")
            break

        display_frame = frame.copy()

        # Add instructions
        cv2.rectangle(
            display_frame,
            (0, 0),
            (700, 80),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            display_frame,
            "LIVE CAMERA - Press S to freeze",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_frame,
            "Position phone correctly before pressing S",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    h, w = display_frame.shape[:2]

    # Fit image inside approximately 1200 x 700
    max_width = 1200
    max_height = 700

    scale = min(
        max_width / w,
        max_height / h,
        1.0
    )

    display_width = int(w * scale)
    display_height = int(h * scale)

    resized_display = cv2.resize(
        display_frame,
        (display_width, display_height)
    )

    # --------------------------------------------------------
    # DRAW SELECTED POINTS
    # --------------------------------------------------------

    if frozen:

        for i, point in enumerate(points):

            x = int(point[0] * scale)
            y = int(point[1] * scale)

            cv2.circle(
                resized_display,
                (x, y),
                7,
                (0, 255, 255),
                -1
            )

            cv2.putText(
                resized_display,
                str(i + 1),
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        # Draw lines between selected points

        if len(points) >= 2:

            for i in range(len(points) - 1):

                p1 = (
                    int(points[i][0] * scale),
                    int(points[i][1] * scale)
                )

                p2 = (
                    int(points[i + 1][0] * scale),
                    int(points[i + 1][1] * scale)
                )

                cv2.line(
                    resized_display,
                    p1,
                    p2,
                    (0, 255, 0),
                    3
                )

        # Close polygon after 4 points

        if len(points) == 4:

            p1 = (
                int(points[3][0] * scale),
                int(points[3][1] * scale)
            )

            p2 = (
                int(points[0][0] * scale),
                int(points[0][1] * scale)
            )

            cv2.line(
                resized_display,
                p1,
                p2,
                (0, 255, 0),
                3
            )

        # Instruction panel

        cv2.rectangle(
            resized_display,
            (0, 0),
            (500, 75),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            resized_display,
            f"Slot {current_slot}: {len(points)}/4 points",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            resized_display,
            "Click 4 corners, then ENTER",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

    cv2.imshow(
        WINDOW_NAME,
        resized_display
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # S = FREEZE
    # --------------------------------------------------------

    if key == ord("s") and not frozen:

        frozen = True
        points = []

        print()
        print("====================================")
        print("Frame frozen!")
        print(f"Now marking SLOT {current_slot}")
        print("Click 4 corners of the slot.")
        print("Then press ENTER.")
        print("====================================")


    # --------------------------------------------------------
    # ENTER = SAVE CURRENT SLOT
    # --------------------------------------------------------

    elif key == 13 and frozen:

        if len(points) != 4:

            print(
                f"❌ Slot {current_slot}: "
                f"You selected {len(points)} points."
            )

            print("Please select exactly 4 points.")

            continue


        # Save slot

        slot_name = f"slot{current_slot}"

        slot_coordinates[slot_name] = points.copy()

        print()
        print(
            f"✅ {slot_name} saved:"
        )

        print(points)


        # Move to next slot

        if current_slot < 4:

            current_slot += 1

            points = []

            print()
            print(
                f"Now mark SLOT {current_slot}"
            )

        else:

            # =================================================
            # ALL 4 SLOTS COMPLETE
            # =================================================

            with open(
                OUTPUT_FILE,
                "w"
            ) as f:

                json.dump(
                    slot_coordinates,
                    f,
                    indent=4
                )

            print()
            print("====================================")
            print("✅ ALL 4 SLOTS SAVED!")
            print("====================================")
            print()
            print(
                f"Saved to: {OUTPUT_FILE}"
            )
            print()
            print(json.dumps(
                slot_coordinates,
                indent=4
            ))

            break


    # --------------------------------------------------------
    # R = RESET CURRENT SLOT
    # --------------------------------------------------------

    elif key == ord("r") and frozen:

        points = []

        print(
            f"🔄 Slot {current_slot} reset."
        )


    # --------------------------------------------------------
    # Q = QUIT
    # --------------------------------------------------------

    elif key == ord("q"):

        print("Quitting...")
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("Program finished.")