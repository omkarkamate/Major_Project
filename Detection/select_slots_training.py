import cv2
import os
import json


DATASET_PATH = "Detection/original_dataset"
REFERENCE_FOLDER = os.path.join(DATASET_PATH, "b1")

files = os.listdir(REFERENCE_FOLDER)

# Select the first image from b1
image_file = None

for file in files:
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        image_file = file
        break

if image_file is None:
    print("No image found in b1 folder.")
    exit()

image_path = os.path.join(
    REFERENCE_FOLDER,
    image_file
)

image = cv2.imread(image_path)

if image is None:
    print("Could not read image.")
    exit()

print("\nReference image:")
print(image_path)

print("PARKING SLOT SELECTION")

print("""
For each slot, click 4 corners in this order:

""")


# VARIABLES


current_points = []
all_slots = []
current_slot = 1

display_image = image.copy()


def mouse_callback(event, x, y, flags, param):

    global current_points
    global all_slots
    global current_slot
    global display_image

    if event == cv2.EVENT_LBUTTONDOWN:

        # Add point
        current_points.append([x, y])

        print(
            f"Slot {current_slot} "
            f"- Point {len(current_points)}: "
            f"({x}, {y})"
        )

        # Draw point
        cv2.circle(
            display_image,
            (x, y),
            6,
            (0, 0, 255),
            -1
        )

        # Draw line between points
        if len(current_points) > 1:

            p1 = current_points[-2]
            p2 = current_points[-1]

            cv2.line(
                display_image,
                tuple(p1),
                tuple(p2),
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "Select Parking Slots",
            display_image
        )


        if len(current_points) == 4:

            # Close polygon
            cv2.line(
                display_image,
                tuple(current_points[3]),
                tuple(current_points[0]),
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "Select Parking Slots",
                display_image
            )

            # Save slot
            all_slots.append(
                current_points.copy()
            )

            print(
                f"\nSlot {current_slot} completed!"
            )

            current_points = []

            # Move to next slot
            current_slot += 1



            if current_slot > 4:

                print("ALL 4 SLOTS SELECTED!")

                save_coordinates()

            else:

                print(
                    f"\nNow select Slot {current_slot}"
                )


def save_coordinates():

    data = {

        "slot1": all_slots[0],
        "slot2": all_slots[1],
        "slot3": all_slots[2],
        "slot4": all_slots[3]

    }

    output_file = "slot_coordinates_for_lap_training.json"

    with open(output_file, "w") as f:

        json.dump(
            data,
            f,
            indent=4
        )

    print("\nCoordinates saved successfully!")

    print(
        "File:",
        output_file
    )

    print("\nCoordinates:")

    for i, slot in enumerate(all_slots, start=1):

        print(
            f"Slot {i}: {slot}"
        )

    print("\nYou can close the image window.")

    cv2.destroyAllWindows()


# OPEN WINDOW


cv2.namedWindow("Select Parking Slots", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Select Parking Slots", 1200, 800)

cv2.setMouseCallback(
    "Select Parking Slots",
    mouse_callback
)

print("\nSelect Slot 1 now...")

while True:

    cv2.imshow(
        "Select Parking Slots",
        display_image
    )

    key = cv2.waitKey(1) & 0xFF

    # ESC = exit
    if key == 27:

        print("\nSelection cancelled.")

        cv2.destroyAllWindows()

        break

    # If all slots completed, exit loop
    if current_slot > 4:
        break