from TriviaHelper.Question import Question
from TriviaHelper.ImageRec.TrainDataPrep import load_train_data
import TriviaHelper.ImageRec.imgprep as imprep
from TriviaHelper.ImageRec.Settings import youtube_areas
import cv2 as cv
import numpy as np


knn = None


def train_knn():
    global knn
    # train knn
    images, lables = load_train_data()
    train_lables = np.array(lables)
    train_lables.reshape((train_lables.shape[0], 1))

    knn = cv.ml.KNearest_create()
    knn.train(images.astype(np.float32), cv.ml.ROW_SAMPLE, train_lables.astype(np.float32))


# gets an opencv image (np array) and returns a question object
def image_rec(img):
    # analyze image
    split_images = imprep.prep_img(img, youtube_areas)


    # result, img, lines, charpos will be returned
    split_character_images = [imprep.split_to_chars(i)[0] for i in split_images]
    solution = []

    for c in split_character_images:
        ret, results, neighbours, dist = knn.findNearest(c.astype(np.float32), 3)
        solution.append(''.join(chr(r) for r in results))

    print(solution)

    return Question(solution[0], solution[1:])


if __name__ == "__main__":
    img = cv.imread('TestImage.png', cv.IMREAD_GRAYSCALE)
    image_rec(img)
