import cv2
import os.path

# Works with version opencv-python == 3.4.8.29
# Write: pip install opencv-python==3.4.8.29

def build_video(image_folder, name_of_image):
    index = 0
    frame = cv2.imread(os.path.join(image_folder, name_of_image + str(index) + '.bmp'))
    height, width, layers = frame.shape
    video = cv2.VideoWriter("A_star_Map_Points.avi", 0, 40, (width,height))
    while os.path.isfile(image_folder + name_of_image + str(index) + '.bmp'):    
        video.write(cv2.imread(os.path.join(image_folder, name_of_image + str(index) + '.bmp')))
        index += 1
    cv2.destroyAllWindows()
    video.release()

build_video('C:/Navmatix/git/d-lite/output/', 'Mapa_iterace_')
