import cv2 as cv
import numpy as np

# the font for displaying feedback to the user
font = cv.FONT_HERSHEY_SIMPLEX

# this class stores the information extracted from the Trivia HQ Question
class Question(object):
    def __init__(self, question=None, answers=None):
        # a string storing the question asked
        self.question = question
        # a list of the possible answers (strings)
        self.answers = answers

    def get_quest_img(self, window_size):
        # the maximum amount of chars per line
        img = np.ones(window_size).astype(np.uint8)
        max_char = 30

        start = 0
        curr_line = 0
        answer_counter = 0
        curr_info = self.question + ' '
        while True:
            while True:
                end = start + max_char
                if end >= len(curr_info):
                    end = len(curr_info) - 1
                else:
                    # find last space
                    while curr_info[end] != ' ':
                        end -= 1
                    end += 1

                cv.putText(img, curr_info[start:end], (10, 40 * curr_line + 50), font, 1, 255, 2, cv.LINE_AA)
                start = end
                curr_line += 1
                if start == len(curr_info) - 1:
                    break
            # next answer
            if answer_counter >= len(self.answers):
                break
            curr_info = '> ' + self.answers[answer_counter] + ' '
            start = 0
            answer_counter += 1

        return img


