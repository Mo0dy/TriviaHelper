from TriviaHelper.Question import Question
from TriviaHelper.ImageRec.TrainDataPrep import load_train_data
import TriviaHelper.ImageRec.imgprep as imprep
import TriviaHelper.ImageRec.Settings as settings
import re
import cv2 as cv
import numpy as np


# this changes the recognized strings to correct for expected errors i.e ('' -> ")
def post_process(input_string):
    # fix '' -> "
    new_string = input_string.replace("''", '"')
    return new_string


def train_knn(different_paths=False):
    # train knn
    if different_paths:
        images, lables = load_train_data('TrainingData\data.npy', 'TrainingData\labels.txt')
    else:
        images, lables = load_train_data()
    train_lables = np.array(lables)
    train_lables.reshape((train_lables.shape[0], 1))

    knn = cv.ml.KNearest_create()
    knn.train(images.astype(np.float32), cv.ml.ROW_SAMPLE, train_lables.astype(np.float32))
    return knn


# gets an opencv image (np array) and returns a question object
def image_rec(knn, img):
    # analyze image
    split_images = imprep.prep_img(img, settings.youtube_areas)


    # result, img, lines, charpos will be returned
    split_image_info = [imprep.split_to_chars(i) for i in split_images]
    solution = []

    # for every single interpreted image combine string and add spaces if distance between chars is big enough
    for s in split_image_info:
        # split the information from the split_to_chars
        image = s[0]
        row_info = s[2]
        char_info = s[3]
        # analyze image
        ret, results, neighbours, dist = knn.findNearest(image.astype(np.float32), settings.k_nearest)

        # the string that will be generated
        string = ''
        # an interator to keep track of which char we are on
        iterator = 0
        # for every character in every row except the last one
        for r in range(len(row_info)):
            for c in range(len(char_info[r])):
                # add the char
                string += (chr(results[iterator]))
                iterator += 1
                if c != len(char_info[r]) - 1: # not the last char:
                    # check the distance to the next one
                    dist = char_info[r][c + 1][0] - char_info[r][c][1]
                    # if big enough add space
                    if dist > 6:
                        string += ' '

        solution.append(string)

    print(solution)

    return Question(solution[0], solution[1:])


if __name__ == "__main__":
    iterator = 0
    train_knn(True)
    questions = []
    while True:
        # try:
        img = cv.imread('Old_Images\\' + str(iterator) + '.png', cv.IMREAD_GRAYSCALE)
        questions.append(image_rec(img))
        iterator += 1
        # except:
        #     break
        questions[-1].show()
    for q in questions:
        print(str(q.question) + str(q.answers))