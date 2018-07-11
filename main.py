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

# train the knn-Algorithm with the example data
ImageRec.train_knn()


# create a window to listen to keys
cv.imshow('test_window', np.ones((300, 300)).astype(np.uint8) * 100)
while True:
    # indefinitely wait for a key
    k = cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imshow('test_window', np.ones((300, 300)).astype(np.uint8) * 100)
    if k == Settings.screenshot_key:  # o
        # take screenshot and save the image (converted to grayscale)
        img = cv.cvtColor(clipped_screenshot(), cv.COLOR_BGR2GRAY)
        # display image as example
        cv.imshow('test_window', img)

        # use image recognition to construct question object
        quest = ImageRec.image_rec(img)
        quest.show()
        # use the search algorithm to find the correct answer
        answer = SearchAlg.search_alg(quest)
        print(answer)
    elif k == 27:  # esc
        break

