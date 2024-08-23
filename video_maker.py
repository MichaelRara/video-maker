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


if __name__ == "__main__":
    split_video_to_frames(frame_output_format="jpg",
                          path_to_video='C:/Users/A200179575/Python_Projects/personal/video_maker/chessboard_moves.avi')

    build_video(image_folder='C:/Users/A200179575/Python_Projects/personal/video_maker',
                prefix_of_image_name='Chessboard_',
                image_format="jpg",
                frames_per_second=1,
                name_of_output_video="chessboard_moves")
