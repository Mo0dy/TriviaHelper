from TriviaHelper.Question import Question
from TriviaHelper.ImageRec import ImageRec
from TriviaHelper.SearchAlg import SearchAlg
from TriviaHelper.ImageRec.YouTubeExtractor import clipped_screenshot
import cv2 as cv
import numpy as np

# the main starting point of the program

# create the image
img = None

# create a window to listen to keys
cv.imshow('test_window', np.ones((300, 300)).astype(np.uint8) * 100)
while True:
    k = cv.waitKey(0)
    if k == 111:  # o
        # take screenshot and save the image
        img = cv.cvtColor(clipped_screenshot(), cv.COLOR_BGR2GRAY)
        cv.imshow('test_window', img)
    elif k == 27:  # esc
        break

    quest = ImageRec.image_rec(img)
# answer = SearchAlg.search_alg(quest)
# print(answer)

