"""
Object detection script.
It detects one instance and returns "prints" out the coordinates in a box like manner.
Coordiantes start at the top left of the screen at (0,0) and increase on the x axis when going right and on the y axis when going down.
Format: x_top_left y_top_left x_bottom_right y_bottom_right
Example: 321 43 421 54
So this would have detected an object that is 100px wide and 11px high.
"""
from enum import Enum, IntEnum

import cv2
import numpy as np
from PIL import ImageGrab
import time
import argparse

from PIL.Image import Image


SUPER_FAST_TRIES: int = 2
FAST_TRIES: int = 5
AVERAGE_TRIES: int = 240
SLOW_TRIES: int = 600


class Sensitivity(float, Enum):
    LOW: float = 0.70
    MEDIUM: float = 0.90
    HIGH: float = 0.98


def get_parser() -> argparse.ArgumentParser:
    """Retrieves an argument parser for object detection."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='Process arguments related to object detection.'
    )
    parser.add_argument("-i", "--image", help='image path', default="images/present_trader.png")
    parser.add_argument("-m", "--max-tries", type=int, default=FAST_TRIES)
    parser.add_argument("-s", "--sensitivity", type=int, default=Sensitivity.MEDIUM)
    parser.add_argument("-g", "--grayscale", action='store_true', help='Grayscales both the image and screenshot if True.')
    parser.add_argument("-cr", "--crop", action='store_true')

    return parser

def get_main_image() -> cv2.typing.MatLike:
    """Takes a screenshot and returns it as a MatLike object in RGB form."""
    screenshot: Image = ImageGrab.grab()
    return cv2.cvtColor(
        src=np.array(screenshot),
        code=cv2.COLOR_BGR2RGB
    )


if __name__ == '__main__':
    parser: argparse.ArgumentParser = get_parser()
    args: argparse.Namespace = parser.parse_args()

    image_name: str = args.image

    limit: float = args.sensitivity
    max_tries: int = args.max_tries

    grayscale: bool = args.grayscale
    crop: bool = args.crop

    tries: int = 0
    max_val: float = 0.00


    while max_val < limit:
        # If it has tried for 4 minutes then break
        if tries > max_tries:
            break

        main_image: cv2.typing.MatLike = get_main_image()
        template: cv2.typing.MatLike = cv2.imread(image_name, cv2.IMREAD_COLOR)


        if grayscale:
            main_image = cv2.cvtColor(main_image, cv2.COLOR_RGB2GRAY)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        if crop:
            # The shape of an opencv image follows this format: Height (Rows), Width (Columns), Channels
            height, width = template.shape

            # 0.05 means 5% of the image will be cut off from every side of the image.
            # This can be recalibrated.
            # Smaller images should be cropped more drasticly as the in-game borders around the item do not change but only the size of the item
            # Bigger items will get scaled too far.
            if height < 70:
                crop_height = int(height * 0.15)
            else:
                crop_height = int(height * 0.05)

            if width < 70:
                crop_width = int(width * 0.15)
            else:
                crop_width = int(width * 0.05)

            # Format of cropping is: [Start height:End Height, Start width:End width]
            template = template[crop_height:-crop_height, crop_width:-crop_width]

        # Use template matching
        result = cv2.matchTemplate(main_image, template, cv2.TM_CCOEFF_NORMED)
        min_val: float
        max_val: float
        min_loc: cv2.typing.Point
        max_loc: cv2.typing.Point
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        tries += 1
        time.sleep(1)
        # print(f"Certainty Score: {max_val:.2f}")


    if tries < max_tries:
        # Get the top-left corner of the matched area
        top_left = max_loc
        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

        # Draw a rectangle around the matched object
        cv2.rectangle(main_image, top_left, bottom_right, (0, 255, 0), 2)

        # Display the result in a named window
        # window_name = "Detected Object"
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        # cv2.imshow(window_name, main_image)

        # Print the coordinates of the detected object directly
        print(top_left[0], top_left[1], bottom_right[0], bottom_right[1])

        # cv2.destroyAllWindows()
    else:
        print("Could not detect")


