from TriviaHelper.Question import Question
from TriviaHelper.ImageRec import ImageRec
from TriviaHelper.SearchAlg import SearchAlg
from TriviaHelper.ImageRec.YouTubeExtractor import clipped_screenshot
from TriviaHelper import Settings
import cv2 as cv
import numpy as np

# the main starting point of the program

# create the image
img = None

# the font for displaying feedback to the user
font = cv.FONT_HERSHEY_SIMPLEX

# train the knn-Algorithm with the example data
knn = ImageRec.train_knn()


def analyze():
    # take screenshot and save the image (converted to grayscale)
    img = cv.cvtColor(clipped_screenshot(), cv.COLOR_BGR2GRAY)
    # display image as example
    try:
        # use image recognition to construct question object
        quest = ImageRec.image_rec(knn, img)
        q_img = quest.get_quest_img(img.shape)
        cv.imshow('main_window', np.hstack((img, q_img)))
        # use the search algorithm to find the correct answer
        answer = SearchAlg.search_alg(quest)
        print(answer)
    except:
        err_img = np.ones(img.shape).astype(np.uint8) * 50
        cv.putText(err_img, 'ERROR!', (50, 200), font, 3, 255, 2, cv.LINE_AA)
        cv.imshow('main_window', np.hstack((img, err_img)))


# the callback function
def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        analyze()


# create a window to listen to keys
cv.imshow('main_window', np.ones((300, 300)).astype(np.uint8) * 100)
cv.setMouseCallback('main_window', mouse_callback)


while True:
    # indefinitely wait for a key
    k = cv.waitKey(0)
    if k == Settings.screenshot_key:  # o
        analyze()
    elif k == 27:  # esc
        break

