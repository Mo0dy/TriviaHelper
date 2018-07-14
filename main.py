from TriviaHelper.Question import Question
from TriviaHelper.ImageRec import ImageRec
from TriviaHelper.SearchAlg import SearchAlg
from TriviaHelper.ImageRec.YouTubeExtractor import clipped_screenshot
from TriviaHelper import Settings
import time
import cv2 as cv
import numpy as np

# the main starting point of the program

# create the image
img = None

# the font for displaying feedback to the user
font = cv.FONT_HERSHEY_SIMPLEX

# train the knn-Algorithm with the example data
knn = ImageRec.train_knn()


states = {
    'default': 0,
    'analyzed': 1,
    'change_answer': 2,
}


state = 0
m_x, m_y = None, None


def analyze():
    global state
    # take screenshot and save the image (converted to grayscale)
    img = cv.cvtColor(clipped_screenshot(), cv.COLOR_BGR2GRAY)
    # display image as example
    try:
        # use image recognition to construct question object
        quest = ImageRec.image_rec(knn, img)
        q_img, answer_fields = quest.get_quest_img(img.shape)

        # debug answer fields
        # for a in answer_fields:
        #     q_img[a[1]:a[3], a[0]:a[2]] = 100

        # use the search algorithm to find the correct answer
        answer = SearchAlg.search_alg(quest)
        cv.putText(q_img, "A: " + answer, (10, img.shape[0] - 70), font, 2, 255, 2)
        cv.imshow('main_window', np.hstack((img, q_img)))
        print(answer)
        state = states['analyzed']
        while state == states['analyzed'] or state == states['change_answer']:
            k = cv.waitKey(10)
            if k == 27: # esc
                state = states['default']
            elif state == states['change_answer']:
                # answer fields and the answers in question have the same order
                for i in range(len(answer_fields)):
                    a = answer_fields[i]
                    # check if in box m_x needs to be transformed to the right
                    if a[0] < (m_x - img.shape[1]) < a[2] and a[1] < m_y < a[3]:
                        cv.namedWindow('new_answer')
                        text_img = np.zeros((100, 500)).astype(np.uint8)
                        new_answer = ''
                        while True:
                            k = cv.waitKey(0)
                            if k == 13: #enter
                                break
                            elif k == 8: # return
                                new_answer = new_answer[:-1]
                            else:
                                new_answer += chr(k)
                            text_img[:, :] = 0
                            cv.putText(text_img, new_answer, (10, 50), font, 1, 255)
                            cv.imshow('new_answer', text_img)
                        cv.destroyWindow('new_answer')
                        quest.answers[i] = new_answer
                        answer = SearchAlg.search_alg(quest)
                        q_img, answer_fields = quest.get_quest_img(img.shape)
                        cv.putText(q_img, "A: " + answer, (10, img.shape[0] - 70), font, 2, 255, 2)
                        cv.imshow('main_window', np.hstack((img, q_img)))
                        print("new answer: " + answer)
                        break
                state = states['analyzed']
        cv.imshow('main_window', np.ones((300, 300)).astype(np.uint8) * 100)
    except:
        err_img = np.ones(img.shape).astype(np.uint8) * 50
        cv.putText(err_img, 'ERROR!', (50, 200), font, 3, 255, 2, cv.LINE_AA)
        cv.imshow('main_window', np.hstack((img, err_img)))


# the callback function
def mouse_callback(event, x, y, flags, param):
    global state, m_x, m_y
    if event == cv.EVENT_LBUTTONDOWN:
        if state == states['default']:
            analyze()
        elif state == states['analyzed']:
            state = states['change_answer']
            m_x, m_y = x, y



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

