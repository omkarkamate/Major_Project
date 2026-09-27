import os
import csv
import cv2
import json
import numpy as np


ORIGINAL_DATASET = "Detection/original_dataset"
COORDINATES_FILE = "Detection/slot_coordinates_for_lap_training.json"
OUTPUT_CSV = "Detection/slot_labels.csv"

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

if not os.path.exists(COORDINATES_FILE):
    print("\nERROR: slot_coordinates.json not found!")
    exit()

with open(COORDINATES_FILE, "r") as file:
    slot_coordinates = json.load(file)


classes = ["b1", "b2", "b3", "b4"]

for class_name in classes:

    folder = os.path.join(
        ORIGINAL_DATASET,
        class_name
    )

    if not os.path.exists(folder):

        print(f"\nERROR: Folder not found: {folder}")
        exit()

images = []

for class_name in classes:

    folder = os.path.join(
        ORIGINAL_DATASET,
        class_name
    )

    for filename in sorted(os.listdir(folder)):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):

            relative_path = os.path.join(
                class_name,
                filename
            )

            full_path = os.path.join(
                ORIGINAL_DATASET,
                relative_path
            )

            images.append(
                (relative_path, full_path)
            )

# LOAD EXISTING LABELS

labeled_images = set()

if os.path.exists(OUTPUT_CSV):

    with open(
        OUTPUT_CSV,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            labeled_images.add(
                row["image_path"]
            )


# ============================================================
# CREATE CSV
# ============================================================

if not os.path.exists(OUTPUT_CSV):

    with open(
        OUTPUT_CSV,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "image_path",
                "slot1",
                "slot2",
                "slot3",
                "slot4"
            ]
        )

# REMOVE ALREADY LABELED

remaining_images = []

for relative_path, full_path in images:

    if relative_path not in labeled_images:

        remaining_images.append(
            (relative_path, full_path)
        )

# INFORMATION

print("        PARKING SLOT LABELING TOOL")

print(
    f"\nTotal images    : {len(images)}"
)

print(
    f"Already labeled : {len(labeled_images)}"
)

print(
    f"Remaining       : {len(remaining_images)}"
)

print("\n----------------------------------------------")

print("""
LABEL FORMAT

0 = EMPTY
1 = OCCUPIED


COMMANDS:

s = Skip image
q = Quit
""")


if len(remaining_images) == 0:

    print("\nAll images are already labeled.")
    print("File:", OUTPUT_CSV)
    exit()

# MAIN LOOP


for index, (relative_path, full_path) in enumerate(
    remaining_images
):

    print("\n==============================================")

    print(
        f"IMAGE {index + 1} / {len(remaining_images)}"
    )

    print(
        f"File: {relative_path}"
    )

    print("==============================================")


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = cv2.imread(full_path)

    if image is None:

        print(
            "ERROR: Could not load image:"
        )

        print(full_path)

        continue

    # DRAW SLOT POLYGONS


    display_image = image.copy()


    for slot_number in range(1, 5):

        slot_key = f"slot{slot_number}"

        points = np.array(
            slot_coordinates[slot_key],
            dtype=np.int32
        )


        # Draw polygon

        cv2.polylines(
            display_image,
            [points],
            isClosed=True,
            color=(0, 255, 255),
            thickness=5
        )

        # FIND CENTER

        center = np.mean(
            points,
            axis=0
        ).astype(int)

        center_x = int(center[0])
        center_y = int(center[1])


        # DRAW SLOT NUMBER

        text = f"SLOT {slot_number}"

        text_size = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            3
        )[0]

        text_x = center_x - (
            text_size[0] // 2
        )

        text_y = center_y


        # Black background for text

        cv2.rectangle(
            display_image,

            (
                text_x - 5,
                text_y - text_size[1] - 10
            ),

            (
                text_x + text_size[0] + 5,
                text_y + 5
            ),

            (0, 0, 0),

            -1
        )


        cv2.putText(
            display_image,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
            cv2.LINE_AA
        )


    # RESIZE IMAGE


    height, width = display_image.shape[:2]

    scale = min(
        WINDOW_WIDTH / width,
        WINDOW_HEIGHT / height
    )

    new_width = int(
        width * scale
    )

    new_height = int(
        height * scale
    )


    display_image = cv2.resize(
        display_image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # SHOW IMAGE

    window_name = "Parking Slot Labeling"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window_name,
        new_width,
        new_height
    )

    cv2.imshow(
        window_name,
        display_image
    )

    cv2.waitKey(100)

    # WAIT UNTIL USER PRESSES A KEY

    print("\nImage displayed.")

    print(
        "Look at the image window."
    )

    print(
        "Press ANY KEY inside the image window to continue."
    )

    print(
        "Then enter the 4-digit label in this terminal."
    )


    cv2.waitKey(0)

    cv2.destroyWindow(
        window_name
    )

    # GET LABEL

    while True:

        answer = input(
            "\nEnter 4 digits "
            "(0=Empty, 1=Occupied): "
        ).strip().lower()


        # QUIT

        if answer == "q":

            cv2.destroyAllWindows()

            print(
                "\nProgress saved."
            )

            print(
                "Labels file:",
                OUTPUT_CSV
            )

            exit()

        # SKIP

        if answer == "s":

            print(
                "\nImage skipped."
            )

            break

        # VALID LABEL

        if (
            len(answer) == 4
            and all(
                character in "01"
                for character in answer
            )
        ):

            slot1 = int(answer[0])
            slot2 = int(answer[1])
            slot3 = int(answer[2])
            slot4 = int(answer[3])

            # SAVE

            with open(
                OUTPUT_CSV,
                "a",
                newline=""
            ) as file:

                writer = csv.writer(
                    file
                )

                writer.writerow(
                    [
                        relative_path,
                        slot1,
                        slot2,
                        slot3,
                        slot4
                    ]
                )


            print("\nSaved successfully!")

            print(
                f"Slot 1: "
                f"{'OCCUPIED' if slot1 else 'EMPTY'}"
            )

            print(
                f"Slot 2: "
                f"{'OCCUPIED' if slot2 else 'EMPTY'}"
            )

            print(
                f"Slot 3: "
                f"{'OCCUPIED' if slot3 else 'EMPTY'}"
            )

            print(
                f"Slot 4: "
                f"{'OCCUPIED' if slot4 else 'EMPTY'}"
            )

            break

        # INVALID


        print(
            "\nInvalid input."
        )

        print(
            "Enter exactly 4 digits."
        )

        print(
            "Example: 0101"
        )

# FINISHED

cv2.destroyAllWindows()

print("\n==============================================")
print("       LABELING COMPLETED")
print("==============================================")

print(
    "\nAll available images have been processed."
)

print(
    "Labels saved to:",
    OUTPUT_CSV
)
