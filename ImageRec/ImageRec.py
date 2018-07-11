from TriviaHelper.Question import Question
from TriviaHelper.ImageRec.TrainDataPrep import load_train_data
import TriviaHelper.ImageRec.imgprep as imprep
import TriviaHelper.ImageRec.Settings as settings
import cv2 as cv
import numpy as np


knn = None


def train_knn(different_paths=False):
    global knn
    # train knn
    if different_paths:
        images, lables = load_train_data('TrainingData\data.npy', 'TrainingData\labels.txt')
    else:
        images, lables = load_train_data()
    train_lables = np.array(lables)
    train_lables.reshape((train_lables.shape[0], 1))

    knn = cv.ml.KNearest_create()
    knn.train(images.astype(np.float32), cv.ml.ROW_SAMPLE, train_lables.astype(np.float32))


# gets an opencv image (np array) and returns a question object
def image_rec(img):
    # analyze image
    split_images = imprep.prep_img(img, settings.youtube_areas)


    # result, img, lines, charpos will be returned
    split_character_images = [imprep.split_to_chars(i)[0] for i in split_images]
    solution = []

    for c in split_character_images:
        ret, results, neighbours, dist = knn.findNearest(c.astype(np.float32), settings.k_nearest)
        solution.append(''.join(chr(r) for r in results))

    print(solution)

    return Question(solution[0], solution[1:])


if __name__ == "__main__":
    iterator = 0
    train_knn(True)
    questions = []
    while True:
        try:
            img = cv.imread('Old_Images\\' + str(iterator) + '.png', cv.IMREAD_GRAYSCALE)
            questions.append(image_rec(img))
            iterator += 1
        except:
            break

    print("done")
    all = ''
    for q in questions:
        all += q.question
        for a in q.answers:
            all += a

    images, lables = load_train_data('TrainingData\data.npy', 'TrainingData\labels.txt')
    correct_question = 0
    for i in range(len(all)):
        if all[i] == chr(int(lables[i])):
            correct_question += 1

    # this percentage only has meaning if tested with a different set of characters!!!
    print(str(correct_question / len(all) * 100) + '% are correctly identified')

