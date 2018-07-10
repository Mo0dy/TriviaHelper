import numpy as np
import cv2 as cv
import os, os.path
from PIL import ImageGrab

save_path = r"Images"

# the begginning end end coordinates around the square with the important information
beginning = 702, 220
end = 1222, 700


# this will extract images from youtube and store them to be then used to train the algorithm / create training data


# this will count the images already saved
def count_images():
    return len([name for name in os.listdir(save_path)])


def take_screenshot():
    printscreen_pil = ImageGrab.grab()
    return np.array(printscreen_pil.getdata(), dtype='uint8').reshape((printscreen_pil.size[1], printscreen_pil.size[0], 3))


def save_image(img, num):
    cv.imwrite(save_path + '\\' + str(num) + ".png", img)


def run_capture():
    # create a window to listen to keys
    cv.imshow('test_window', np.ones((300, 300)).astype(np.uint8) * 100)
    curr_images = count_images()
    print(curr_images)
    while True:
        k = cv.waitKey(0)
        if k == 111: # o
            # take screenshot and save the image
            img = take_screenshot()[beginning[1]: end[1], beginning[0]: end[0]]
            cv.imshow('test_window', img)
            save_image(img, curr_images)
            curr_images += 1
        elif k == 27: # esc
            break

if __name__ == "__main__":
    run_capture()
