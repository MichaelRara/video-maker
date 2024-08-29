from typing import List

import os.path
import cv2
import numpy as np

from PIL import Image as img
from PIL.Image import Image


def build_video(image_folder: str,
                prefix_of_image_name: str,
                image_format: str,
                frames_per_second: int,
                name_of_output_video: str) -> None:
    """This method can build a video from a set of images. Images must start with the same prefix, have the same data
    format and be sorted by sequence of indexes. Output is a video created from provided images and save in avi format.

    Args:
        image_folder (str): Name of folder where images are stored.
        prefix_of_image_name (str): Prefix of names of images.
        image_format (str): Data format of stored images. Example: jpg, png, bmp etc...
        frames_per_second (int): Defines amount of frames per second in output video.
        name_of_output_video (str): Name of output video. Example: my_output_video
    """
    index = 0
    frame = cv2.imread(os.path.join(image_folder, prefix_of_image_name + str(index) + '.' + image_format))
    height, width, _ = frame.shape
    video = cv2.VideoWriter(name_of_output_video + ".avi", 0, frames_per_second, (width, height))
    while os.path.isfile(image_folder + "/" + prefix_of_image_name + str(index) + '.' + image_format):
        video.write(cv2.imread(os.path.join(image_folder, prefix_of_image_name + str(index) + '.' + image_format)))
        index += 1
    cv2.destroyAllWindows()
    video.release()


def calculate_grayscale_gradient_in_pixel(image: Image, width_position: int, height_position: int) -> np.ndarray[float, float]:
    """Calculate gradient of provided image in pixel specified by width_position, height_position.

    Args:
        image (Image): Input image to calculate gradient in.
        width_position (int): Width coordinate of selected pixel.
        height_position (int): Height coordinate of selected pixel.

    Returns:
        np.ndarray[float, float]: Gradient of brightness in selected pixel of provided image.
    """
    if len(image.shape) == 3:
        image = np.asarray(convert_rgb_image_to_grayscale_image(image))
    # Derivation in horizontal direction
    if width_position == 0:
        next_brightness = image[width_position + 1, height_position]
        actual_brightness = image[width_position, height_position]
        horizontal_derivation = next_brightness - actual_brightness
    elif width_position == (image.width - 1):
        actual_brightness = image[width_position, height_position]
        previous_brightness = image[width_position - 1, height_position]
        horizontal_derivation = actual_brightness - previous_brightness
    else:
        next_brightness = image[width_position + 1, height_position]
        previous_brightness = image[width_position - 1, height_position]
        horizontal_derivation = (next_brightness - previous_brightness)/2
    # Derivation in vertical direction
    if height_position == 0:
        next_brightness = image[width_position, height_position + 1]
        actual_brightness = image[width_position, height_position]
        vertical_derivation = next_brightness - actual_brightness
    elif height_position == (image.height - 1):
        actual_brightness = image[width_position, height_position]
        previous_brightness = image[width_position, height_position - 1]
        vertical_derivation = actual_brightness - previous_brightness
    else:
        next_brightness = image[width_position, height_position + 1]
        previous_brightness = image[width_position, height_position - 1]
        vertical_derivation = (next_brightness - previous_brightness)/2
    return np.array([horizontal_derivation, vertical_derivation])


def calculate_gradient_of_grayscale_image(image: Image) -> Image:
    """Calculate gradient image of provided image. Edges are visualized in this way. 

    Args:
        image (Image): Grayscale image to detect edges in.

    Returns:
        Image: Grayscale image where edges are visualized.
    """
    image = np.asarray(image)

    # Horizontal derivation
    first_column = image[:, 0]
    last_column = image[:, -1]
    frame_right = np.column_stack((image * 0.5, last_column))[:, 1:]
    frame_left = np.column_stack((first_column, image * 0.5))[:, :-1]
    horizontal_derivation = frame_right - frame_left

    # Vertical derivation
    first_row = image[:][0]
    last_row = image[:][-1]
    frame_up = np.vstack([image * 0.5, last_row])[1:, :]
    frame_down = np.vstack([first_row, image * 0.5])[:-1, :]
    vertical_derivation = frame_up - frame_down

    # Calculate norm of gradient
    return img.fromarray(np.sqrt(horizontal_derivation**2 + vertical_derivation**2))


def convert_rgb_image_to_grayscale_image(image: Image) -> Image:
    """Convert rgb image to grayscale image.

    Args:
        image (Image): Input RGB image of class Image from PIL.

    Returns:
        Image: Grayscale version of input image.
    """
    rgb_pixels = np.asarray(image).astype("float")
    grayscale_pixels = rgb_pixels.mean(axis=2).astype("uint8")
    return img.fromarray(grayscale_pixels)


def create_rgb_image(width: int, height: int) -> Image:
    """Create white RGB image.

    Args:
        width (int): Width of created image.
        height (int): Height of created image.

    Returns:
        Image: White RGB image.
    """
    return img.new("RGB", (width, height), "white")


def get_brightness_of_pixel(image: Image, width_position: int, height_position: int) -> int:
    """Get brightness of a pixel from provided RGB image.

    Args:
        image (Image): Image to find a pixel in.
        width_position (int): Width position of selected pixel.
        height_position (int): Height position of selected pixel.s

    Returns:
        int: Brightness of a selected pixel in provided image.
    """
    R, G, B = image.getpixel((width_position, height_position))
    return int(sum([R, G, B])/3)


def interpolate_images(start_image: Image, end_image: Image, amount_of_interpolated_images: int = 100) -> List[Image]:
    """Create list of PIL.Image files which are created by linear interpolation of input start_image and end_image.

    Args:
        start_image (Image): Image which works as starting point of linear interpolation.
        end_image (Image): Image which works as ending point of linear interpolation.
        amount_of_interpolated_images (int, optional): Amount of interpolated images to create. Defaults to 100.

    Returns:
        List[Image]: Interpolated images.
    """
    pixels_of_start_image = np.asarray(start_image).astype("float")
    pixels_of_end_image = np.asarray(end_image).astype("float")
    pixels_of_difference_image = pixels_of_end_image - pixels_of_start_image
    interpolation_step = pixels_of_difference_image/(amount_of_interpolated_images+1)
    interpolated_images = []
    for amount_of_steps in range(1, amount_of_interpolated_images+1):
        interpolated_image = img.fromarray((pixels_of_start_image + amount_of_steps*interpolation_step).astype("uint8"),
                                           'RGB')
        interpolated_images.append(interpolated_image)
    return interpolated_images


def make_grayscale_image_sharper(image: Image) -> Image:
    """Sharper input grayscale image by standard kernel.

    Args:
        image (Image): Input image to sharp.

    Returns:
        Image: Sharpened input image.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    input_image_pixels = np.asarray(image)
    sharpen_image_pixels = np.array(image).astype("float")
    for i in range(1, image.width - 1):
        for j in range(1, image.height - 1):
            window_to_sharp = input_image_pixels[j-1:j+2, i-1:i+2]
            sharpen_image_pixels[j, i] = np.sum(window_to_sharp*kernel).item()
    output_sharpen_image = np.clip(sharpen_image_pixels, 0, 255)
    return img.fromarray(output_sharpen_image.astype("uint8"))


def split_video_to_frames(frame_output_format: str, path_to_video: str) -> None:
    """Split video into individual frames.

    Args:
        frame_output_format (str): Format of output frames. (jpg, png, bmp etc...)
        path_to_video (str): Absolute path to input video.
    """
    cap = cv2.VideoCapture(path_to_video)
    frame_index = 0
    success, frame = cap.read()
    while success:  # Capture frame-by-frame
        cv2.imwrite('./frame_' + str(frame_index) + "." + frame_output_format, frame)
        frame_index += 1
        success, frame = cap.read()


def main() -> None:
    split_video_to_frames(frame_output_format="jpg",
                          path_to_video='C:/Users/A200179575/Python_Projects/personal/video_maker/chessboard_moves.avi')

    build_video(image_folder='C:/Users/A200179575/Python_Projects/personal/video_maker',
                prefix_of_image_name='Chessboard_',
                image_format="jpg",
                frames_per_second=1,
                name_of_output_video="chessboard_moves")
    interpolated_images = interpolate_images(img.open("Chessboard_0.png"), img.open("Chessboard_1.png"), 10)
    for img_index, image in enumerate(interpolated_images):
        image.save("interpolated_img_" + str(img_index) + ".png")


if __name__ == "__main__":
    a = convert_rgb_image_to_grayscale_image(img.open("lachtan.jpg"))
    #a.show()
    b = make_grayscale_image_sharper(a)
    b.show()
    main()
