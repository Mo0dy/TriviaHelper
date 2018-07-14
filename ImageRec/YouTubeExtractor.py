import numpy as np
import cv2 as cv
import os, os.path
import TriviaHelper.ImageRec.Settings as set
from PIL import ImageGrab
import time

save_path = r"New_Images"

# the font for displaying feedback to the user
font = cv.FONT_HERSHEY_SIMPLEX


# the channel: https://www.youtube.com/channel/UCjWzbKvt5F2pA3th5__N9JA

# this will extract images from youtube and store them to be then used to train the algorithm / create training data


# this will count the images already saved
def count_images(path):
    return len([name for name in os.listdir(path)])


def take_screenshot(crop=None):
    # make screenshot
    printscreen_pil = ImageGrab.grab()
    # cropping before converting is a HUGE tame save (about 1.4 seconds for this crop size)
    if crop != None:
        printscreen_pil = printscreen_pil.crop(crop[0] + crop[1])

    return np.array(printscreen_pil.getdata(), dtype='uint8').reshape((printscreen_pil.size[1], printscreen_pil.size[0], 3))


def save_image(img, num):
    cv.imwrite(save_path + '\\' + str(num) + ".png", img)


def clipped_screenshot():
    # return take_screenshot()[set.beginning[1]: set.end[1], set.beginning[0]: set.end[0]]
    return take_screenshot((set.beginning, set.end))

curr_images = 0


def run_capture():
    global curr_images
    # take screenshot and save the image
    img = clipped_screenshot()
    cv.imshow('youtube_extractor', img)
    save_image(img, curr_images)
    curr_images += 1


def remove_last_image():
    global curr_images
    if curr_images > 0:
        curr_images -= 1
        os.remove(save_path + '\\' + str(curr_images) + ".png")
        img = np.ones((500, 500)).astype(np.uint8) * 50
        cv.putText(img, 'REMOVED!', (50, 200), font, 3, 255, 2, cv.LINE_AA)
        cv.imshow('img_removed', img)
        cv.waitKey(1000)
        cv.destroyWindow('img_removed')



# the callback function
def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        run_capture()
    elif event == cv.EVENT_RBUTTONDOWN:
        remove_last_image()


if __name__ == "__main__":
    # create a window to listen to keys
    cv.imshow('youtube_extractor', np.ones((300, 300)).astype(np.uint8) * 100)
    cv.setMouseCallback('youtube_extractor', mouse_callback)
    curr_images = count_images(save_path)
    while True:
        k = cv.waitKey(0)
        print(k)
        if k == 111 or k == 79:  # o, O
            run_capture()
        elif k == 27:  # esc
            break