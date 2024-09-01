from typing import List

import os.path
import os

import cv2
import numpy as np

from PIL import Image as img
from PIL.Image import Image


def build_video(image_folder: str,
                prefix_of_image_name: str,
                image_format: str,
                frames_per_second: int,
                name_of_output_video: str,
                start_index: int) -> None:
    """This method can build a video from a set of images. Images must start with the same prefix, have the same data
    format and be sorted by sequence of indexes. Output is a video created from provided images and save in avi format.

    Args:
        image_folder (str): Name of folder where images are stored.
        prefix_of_image_name (str): Prefix of names of images.
        image_format (str): Data format of stored images. Example: jpg, png, bmp etc...
        frames_per_second (int): Defines amount of frames per second in output video.
        name_of_output_video (str): Name of output video. Example: my_output_video.
        start_index (int): Index of the first frame.
    """
    index = start_index
    frame = cv2.imread(os.path.join(image_folder, prefix_of_image_name + str(index) + '.' + image_format))
    height, width, _ = frame.shape
    video = cv2.VideoWriter(name_of_output_video + ".avi", 0, frames_per_second, (width, height))
    while os.path.isfile(os.path.join(image_folder, prefix_of_image_name + str(index) + '.' + image_format)):
        video.write(cv2.imread(os.path.join(image_folder, prefix_of_image_name + str(index) + '.' + image_format)))
        index += 1
    cv2.destroyAllWindows()
    video.release()


def calculate_grayscale_gradient_in_pixel(image: Image,
                                          width_position: int, height_position: int) -> np.ndarray[float, float]:
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


def calculate_gradient_of_grayscale_image(image: Image, threshold_of_significance: float = 1) -> Image:
    """Calculate gradient image of provided image. Edges are visualized in this way.

    Args:
        image (Image): Grayscale image to detect edges in.
        threshold_of_significance (float): Defines what percentage of maximum value of gradient in image will be set to
                                max brightness value 255. Must be higher than 0 and lower or equal to one. Default to 1.

    Returns:
        Image: Grayscale image where edges are visualized.
    """
    if threshold_of_significance > 1 or threshold_of_significance <= 0:
        raise ValueError("Parameter threshold_of_significance must be higher than 0 and smaller or equal to 1.")

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
    grad_image = np.sqrt(horizontal_derivation**2 + vertical_derivation**2)

    # Recalculate values of brightness with a help of threshold_of_significance
    max_gradient = np.max(grad_image)
    max_brightness = threshold_of_significance * max_gradient
    recalculated_brightness = 255/max_brightness * grad_image
    edges = np.clip(recalculated_brightness, 0, 255)
    return img.fromarray(edges).convert("L")


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


def create_sequence_of_frames_of_edge_detection(name_of_input_folder: str,
                                                prefix_of_input_image: str,
                                                name_of_output_folder: str,
                                                prefix_of_output_image: str,
                                                amount_of_original_frames: int,
                                                threshold_of_significance: float):
    """Create a sequence of frames where edges are visualized.

    Args:
        name_of_input_folder (str): Name of folder with input images.
        prefix_of_input_image (str): Prefix of input images.
        name_of_output_folder (str): Name of folder where results are stored.
        prefix_of_output_image (str): Prefix of output images.
        amount_of_original_frames (int): Amount of original frames.
        threshold_of_significance (float): Defines what percentage of maximum value of gradient in image will be set to
                                           max brightness value 255. Must be higher than 0 and lower or equal to one.
    """
    os.mkdir(name_of_output_folder)
    for i in range(0, amount_of_original_frames):
        original_frame = img.open(name_of_input_folder + "/" + prefix_of_input_image + str(i) + ".png")
        original_frame_grayscale = convert_rgb_image_to_grayscale_image(original_frame)
        edges_frame = calculate_gradient_of_grayscale_image(original_frame_grayscale, threshold_of_significance)
        edges_frame.save(name_of_output_folder + "/" + prefix_of_output_image + str(i) + ".png")


def denoise_video(path_to_video: str, directory_for_results: str, image_prefix_of_denoised_frames: str) -> None:
    """Iterates through every frame of input video and suppress noise in it.
    First two frames are not returned and last two frames are not returned.

    This method is used for noise removing.
    #https://docs.opencv.org/4.x/d1/d79/group__photo__denoise.html#ga723ffde1969430fede9241402e198151

    Args:
        path_to_video (str): Path to video to remove noise from.
        directory_for_results (str): Name of folder where frames with suppress noise are stored.
        image_prefix_of_denoised_frames (str): Prefix of denoised frames.
    """
    cap = cv2.VideoCapture(path_to_video)
    success, frame = cap.read()
    images_of_input_video = []
    while success:  # Capture frame-by-frame
        images_of_input_video.append(frame)
        success, frame = cap.read()
    os.mkdir(directory_for_results)
    for i in range(2, len(images_of_input_video) - 2):
        denoised_frame = cv2.fastNlMeansDenoisingMulti(srcImgs=images_of_input_video,
                                                       imgToDenoiseIndex=i,
                                                       temporalWindowSize=5,
                                                       dst=None,
                                                       templateWindowSize=5,
                                                       searchWindowSize=7,
                                                       h=35)
        cv2.imwrite(directory_for_results + "/" + image_prefix_of_denoised_frames + str(i) + ".png", denoised_frame)


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
        interpolated_image = img.fromarray((pixels_of_start_image + amount_of_steps*interpolation_step).astype("uint8"))
        interpolated_images.append(interpolated_image)
    return interpolated_images


def interpolate_sequence_of_images(directory_for_results: str,
                                   folder_of_input_frames: str,
                                   input_image_prefix: str,
                                   output_image_prefix: str,
                                   amount_of_interpolated_images: int,
                                   amount_of_original_frames: int) -> None:
    """Create smoother sequence of provided images by interpolating them.

    Args:
        directory_for_results (str): Directory where output are stored.
        folder_of_input_frames (str): Directory where input files are read from.
        input_image_prefix (str): Prefix of input images.
        output_image_prefix (str): Prefix of output images.
        amount_of_interpolated_images (int): Amount of interpolated frames.
        amount_of_original_frames (int): Amount of original frames to use for interpolation.
    """
    os.mkdir(directory_for_results)
    output_image_sequence = []
    start_index = 2
    end_index = amount_of_original_frames - 3
    for i in range(start_index, end_index):
        actual_frame = img.open(folder_of_input_frames + "/" + input_image_prefix + str(i) + ".png")
        next_frame = img.open(folder_of_input_frames + "/" + input_image_prefix + str(i+1) + ".png")
        output_image_sequence += [actual_frame]
        interpolated_images = interpolate_images(start_image=actual_frame,
                                                 end_image=next_frame,
                                                 amount_of_interpolated_images=amount_of_interpolated_images)
        output_image_sequence += interpolated_images
    output_image_sequence += [img.open(folder_of_input_frames + "/" + input_image_prefix + str(end_index) + ".png")]

    for i, image in enumerate(output_image_sequence):
        image.save(directory_for_results + "/" + output_image_prefix + str(i+start_index) + ".png")


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


def make_grayscale_image_sharper_in_selected_pixels(image: Image,
                                                    pixels_to_sharper: List[List[int]]) -> Image:
    """Sharper input image in selected pixels if it is possible.

    Args:
        image (Image): Input image to sharper in selected pixels.
        pixels_to_sharper (List): List of pixels where to make input image sharper.

    Returns:
        Image: Input image which was sharpened in selected pixels.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    input_image_pixels = np.asarray(image)
    sharpen_image_pixels = np.array(image).astype("float")
    for j, i in pixels_to_sharper:
        if j < 1 or j > image.height - 2 or i < 1 or i > image.width - 2:
            continue
        window_to_sharp = input_image_pixels[j-1:j+2, i-1:i+2]
        sharpen_image_pixels[j, i] = np.sum(window_to_sharp*kernel).item()
    output_sharpen_image = np.clip(sharpen_image_pixels, 0, 255)
    return img.fromarray(output_sharpen_image.astype("uint8"))


def put_images_side_by_side(image_folder_for_original_images: str,
                            image_folder_for_improved_images: str,
                            name_of_left_frame: str,
                            name_of_right_frame: str,
                            start_index: int,
                            amount_of_interpolated_frames: int,
                            directory_for_results: str,
                            image_prefix: str) -> None:
    """Concatenate two set of images side by side. The images should have the same height and width and amount of
    channels. Results are stored into current working directory.

    Args:
        image_folder_for_original_images (str): Folder where original frames are stored.
        image_folder_for_improved_images (str): Folder where improved frames are stored.
        name_of_left_frame (str): Name of image to placed on a left side of resulting image.
        name_of_right_frame (str): Name of image to placed on a right side of resulting image.
        start_index (int): Starting index of image.
        directory_for_results (str): Create directory where output frames are stored.
        image_prefix (str): Image prefix of output frames.
    """
    os.mkdir(directory_for_results)
    index_improved_image = start_index
    index_original_image = start_index
    while os.path.isfile(image_folder_for_improved_images
                         + "/"
                         + name_of_right_frame
                         + str(index_improved_image)
                         + ".png"):
        original_frame = cv2.imread(os.path.join(image_folder_for_original_images,
                                                 name_of_left_frame + str(index_original_image) + ".png"))
        upgraded_frame = cv2.imread(os.path.join(image_folder_for_improved_images,
                                                 name_of_right_frame + str(index_improved_image) + ".png"))
        concatenated_frame = np.concatenate((original_frame, upgraded_frame), axis=1)
        cv2.imwrite(directory_for_results + "/" + image_prefix + str(index_improved_image) + ".png",
                    concatenated_frame)
        index_improved_image += 1
        if index_improved_image % (amount_of_interpolated_frames+start_index) == 0:
            index_original_image += 1


def put_images_side_by_side_initial(image_folder: str,
                            name_of_left_frame: str,
                            name_of_right_frame: str,
                            index: int,
                            img_format: str,
                            directory_for_results: str) -> None:
    """Concatenate two set of images side by side. The images should have the same height and width and amount of
    channels. Results are stored into current working directory.

    Args:
        image_folder (str): Name of folder where images are stored.
        name_of_left_frame (str): Name of image to placed on a left side of resulting image.
        name_of_right_frame (str): Name of image to placed on a right side of resulting image.
        index (int): Starting index of image.
        img_format (str): Format of input images. Must be same for left and right image.
        directory_for_results (str): Create directory where output frames are stored.
    """
    os.mkdir(directory_for_results)
    while os.path.isfile(image_folder + "/" + name_of_left_frame + str(index) + "." + img_format):
        original_frame = cv2.imread(os.path.join(image_folder, name_of_left_frame + str(index) + "." + img_format))
        upgraded_frame = cv2.imread(os.path.join(image_folder, name_of_right_frame + str(index) + "." + img_format))
        concatenated_frame = np.concatenate((original_frame, upgraded_frame), axis=1)
        cv2.imwrite(directory_for_results + "/" + 'parallel_frames_' + str(index) + "." + img_format,
                    concatenated_frame)
        index += 1


def sharper_frames_along_most_significant_edges(directory_for_results: str,
                                                folder_of_input_frames: str,
                                                image_prefix: str,
                                                threshold_of_significance: float,
                                                min_value_of_significant_edge: int,
                                                amount_of_original_frames: int):
    """Iterates over frames. Pixels of significant edges are detected in every frame. Image is sharpened in positions of
    these pixels.

    Args:
        directory_for_results (str): Folder where output of this method are stored.
        folder_of_input_frames (str): Folder where inputs are uploaded.
        image_prefix (str): Prefix name of image to process.
        threshold_of_significance (float): Defines the percentage of value of max gradient. Pixels with this value of
            brightness or higher will have maximum value of brightness 255.
        min_value_of_significant_edge (int): Edges with brightness with value higher than is specified by this parameter
            will be considered significant.
        amount_of_original_frames (int): Amount of original frames.
    """
    os.mkdir(directory_for_results)
    for i in range(2, amount_of_original_frames-2):
        original_frame = img.open(folder_of_input_frames + "/" + image_prefix + str(i) + ".png")
        original_frame_grayscale = convert_rgb_image_to_grayscale_image(original_frame)
        frame_of_edges = calculate_gradient_of_grayscale_image(original_frame_grayscale, threshold_of_significance)
        significant_edges = np.argwhere(np.asarray(frame_of_edges) > min_value_of_significant_edge).tolist()
        enhanced_image = make_grayscale_image_sharper_in_selected_pixels(original_frame_grayscale, significant_edges)
        enhanced_image.save(directory_for_results + "/" + "enhanced_image_" + str(i) + ".png")


def split_video_to_frames(directory_for_results: str,
                          path_to_video: str,
                          image_prefix: str) -> int:
    """Split video into individual frames.

    Args:
        directory_for_results (str): Create directory where output frames are stored.
        path_to_video (str): Absolute path to input video.
        image_prefix (str): Prefix of image from provided video.

    Returns:
        int: Amount of frames in input video.
    """
    cap = cv2.VideoCapture(path_to_video)
    frame_index = 0
    success, frame = cap.read()
    os.mkdir(directory_for_results)
    while success:  # Capture frame-by-frame
        cv2.imwrite(directory_for_results + '/' + image_prefix + str(frame_index) + ".png", frame)
        frame_index += 1
        success, frame = cap.read()
    return frame_index


def main() -> None:
    path_to_input_video = "lock_video.avi"
    dir_for_frames_of_input_video = "lock_video_original_frames"
    dir_for_denoised_frames = "denoised_frames"
    dir_for_sharpened_images = "enhanced_images"
    dir_for_final_frames = "final_frames"
    dir_for_parallel_frames = "parallel_frames"
    dir_for_frames_edges = "frames_edges"

    image_prefix_of_original_frames = "original_frame_"
    image_prefix_of_denoised_frames = "denoised_frame_"
    image_prefix_of_enhanced_image = "enhanced_image_"
    image_prefix_of_parallel_frames = "parallel_frames_"
    image_prefix_of_final_image = "final_frame_"
    image_prefix_of_edges_images = "edges_frame_"

    amount_of_interpolated_frames = 150
    """
    amount_of_original_frames = split_video_to_frames(directory_for_results=dir_for_frames_of_input_video,
                                                      path_to_video=path_to_input_video,
                                                      image_prefix=image_prefix_of_original_frames)

    denoise_video(path_to_video=path_to_input_video,
                  directory_for_results=dir_for_denoised_frames,
                  image_prefix_of_denoised_frames=image_prefix_of_denoised_frames)

    sharper_frames_along_most_significant_edges(directory_for_results=dir_for_sharpened_images,
                                                folder_of_input_frames=dir_for_denoised_frames,
                                                image_prefix=image_prefix_of_denoised_frames,
                                                threshold_of_significance=0.6,
                                                min_value_of_significant_edge=100,
                                                amount_of_original_frames=amount_of_original_frames)

    interpolate_sequence_of_images(directory_for_results=dir_for_final_frames,
                                   folder_of_input_frames=dir_for_sharpened_images,
                                   input_image_prefix=image_prefix_of_enhanced_image,
                                   output_image_prefix=image_prefix_of_final_image,
                                   amount_of_interpolated_images=amount_of_interpolated_frames,
                                   amount_of_original_frames=amount_of_original_frames)

    put_images_side_by_side(image_folder_for_original_images=dir_for_frames_of_input_video,
                            image_folder_for_improved_images=dir_for_final_frames,
                            name_of_left_frame=image_prefix_of_original_frames,
                            name_of_right_frame=image_prefix_of_final_image,
                            start_index=2,
                            amount_of_interpolated_frames=amount_of_interpolated_frames,
                            directory_for_results=dir_for_parallel_frames,
                            image_prefix=image_prefix_of_parallel_frames)
    build_video(image_folder=dir_for_parallel_frames,
                prefix_of_image_name=image_prefix_of_parallel_frames,
                image_format="png",
                frames_per_second=int(17667/24),
                name_of_output_video="input_output_video",
                start_index=2)
    """
    """
    create_sequence_of_frames_of_edge_detection(name_of_input_folder=dir_for_frames_of_input_video,
                                                prefix_of_input_image=image_prefix_of_original_frames,
                                                name_of_output_folder=dir_for_frames_edges,
                                                prefix_of_output_image=image_prefix_of_edges_images,
                                                amount_of_original_frames=122,
                                                threshold_of_significance=0.4)
    build_video(image_folder=dir_for_frames_edges,
                prefix_of_image_name=image_prefix_of_edges_images,
                image_format="png",
                frames_per_second=5,
                name_of_output_video="video_of_edges",
                start_index=0)
    """
    build_video(image_folder=dir_for_final_frames,
                prefix_of_image_name=image_prefix_of_final_image,
                image_format="png",
                frames_per_second=int(17667/24),
                name_of_output_video="output_video",
                start_index=2)

if __name__ == "__main__":
    main()
