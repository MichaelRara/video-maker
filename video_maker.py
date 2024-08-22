import os.path
import cv2


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


build_video('C:/Users/A200179575/Python_Projects/personal/video_maker', 'Chessboard_', "jpg", 1, "chessboard_moves")
