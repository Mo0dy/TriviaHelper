import numpy as np
import cv2 as cv
from TriviaHelper.ImageRec.imgprep import prep_img, split_to_chars
import TriviaHelper.ImageRec.Settings as Settings
from TriviaHelper.ImageRec.YouTubeExtractor import count_images
import os

# this module takes an image and queries the user for the correct interpretation. This will generate Training data it will later learn from

# the font for displaying feedback to the user
font = cv.FONT_HERSHEY_SIMPLEX

# the paths of the save files for the generated data. New data will be appended to these files
path_labels = r'ImageRec\TrainingData\labels.txt'
path_data = r'ImageRec\TrainingData\data.npy'

new_image_path = r'New_Images'
old_image_path = r'Old_Images'

# areas: the first one ist the question the others the answer they go [p1y, p1x, p2y, p2x]
youtube_areas = Settings.youtube_areas

# UTILITY =======================================================================================
# this loads the training data
def load_train_data():
    # exeptions in chase this is the first time data gets entered
    try:
        images = np.load(path_data)
        print("loaded images")
    except:
        print("coult not load")
        images = None
    # labels get converted to lists because it is easier to append more information
    try:
        labels = np.loadtxt(path_labels)
        print(labels.shape)
        labels = [labels[i] for i in range(labels.shape[0])]
    except:
        labels = []

    return images, labels


# this function appends the newly generated data to the save files
def save_train_data(images, labels):
    np.savetxt(path_labels, labels)
    # save images
    np.save(path_data, images)


def append_new_data(images, lables):
    # first load old data
    old_images, old_labels = load_train_data()
    # append and save labels as numpy array

    lables_to_save = np.array(old_labels + lables, dtype=np.int)

    # check if old_images is None
    if np.any(old_images):
        # append new data (stored as a n x flattenedImageSize matrix) where n is the number of characters stored
        images = np.vstack((old_images, images))
    save_train_data(images, lables_to_save)


# draws the rectangle around the char that needs to be input according to it's row start and end (y) and char start and end (x)
def draw_rec(img, row, char):
    # copy the image so the original one doesnt get changed
    nimg = img.copy()
    # cv.rectangle(nimg, (char[0] - 5, row[0] - 5), (char[1] + 5, row[1] + 5), [0, 0, 255], 2)
    region = nimg[row[0] - 5: row[1] + 10, char[0]: char[1]]
    mask = region[:, :, 1] > 0
    region[:, :, :] = 0
    region[mask, 2] = 255
    return nimg


# displayes the currently obtained characters on a dedicated window
def display_curr_text(characters, window_name, window_size):
    offset = 0
    if len(characters) > 50:
        offset = 15 * (len(characters) - 50)
    img = np.zeros(window_size)
    cv.putText(img, ''.join(chr(c) for c in characters), (10 - offset, 50), font, 1, 255, 2, cv.LINE_AA)
    cv.imshow(window_name, img)


# this function will view the currently generated training data. i.e. the cars and the associated lables
def view_training_data():
    padding_size = 10
    view_size = 800 # the height that gets viewed
    view_pos = 0
    scrollspeed = 100

    # load data
    images, lables = load_train_data()

    # generate character image
    # make character images out of the flattened chars + padding
    char_images = [np.vstack((images[i].reshape(Settings.char_shape), np.zeros((padding_size, Settings.char_shape[1])))) for i in range(images.shape[0])]
    # the image of all chars stacked on top of each other
    char_images = np.vstack(char_images).astype(np.uint8)

    char_images = np.abs(char_images.astype(np.int) - 255)

    # generate interpreted text
    generated_text = []
    text_shape = Settings.char_shape[0] + padding_size, Settings.char_shape[1]
    for l in lables:
        lable_image = np.ones(text_shape) * 255
        cv.putText(lable_image, chr(int(l)), (5, text_shape[1] - 8), font, 1, 50, 2, cv.LINE_AA)
        generated_text.append(lable_image)

    stacked_lables = np.vstack(generated_text)

    # combine images and add padding
    half_horizontal_padding = 3
    padding = np.ones((char_images.shape[0], half_horizontal_padding)) * 255
    view_img = np.hstack((char_images, padding, np.ones((char_images.shape[0], 4)) * 100, padding, stacked_lables)).astype(np.uint8)
    # add seperation lines
    for i in range(len(generated_text) - 1):
        cv.line(view_img, (0, (i + 1) * text_shape[0] - 5), (view_img.shape[1], (i + 1) * text_shape[0] - 5), 100, 3)

    # stack image in parallel
    rows = 15
    pixles_per_row = view_img.shape[0] / rows
    chars_per_row = int(np.ceil(pixles_per_row / (Settings.char_shape[0] + padding_size)))
    max_height = chars_per_row * (Settings.char_shape[0] + padding_size)
    width = view_img.shape[1]
    padding_size = 15
    padding = np.ones((max_height, padding_size)).astype(np.uint8) * 30

    rows = [padding]
    beginning = 0
    while True:
        end = beginning + max_height
        if end >= view_img.shape[0]:
            # extended the end now we need to pad this
            last_bit = view_img[beginning:-1, :]
            rows.append(np.vstack((last_bit, np.zeros((max_height - last_bit.shape[0], width)))))
            break
        else:
            rows.append(np.hstack((view_img[beginning:end], padding)))
            beginning += max_height

    rows.append(padding)

    view_img = np.hstack(rows).astype(np.uint8)

    while True:
        # calculate view image
        if view_pos < 0:
            view_pos = 0
        elif view_pos > view_img.shape[0] - view_size * 3 / 4:
            view_pos = int(view_img.shape[0] - view_size * 3 / 4)
        end = view_pos + view_size
        if end >= view_img.shape[0]:
            end = view_img.shape[0] - 1

        cv.imshow('training Data', view_img[view_pos:end, :])
        k = cv.waitKey(0)
        if k == 119: # w for up
            view_pos -= scrollspeed
        elif k == 115: # s for down
            view_pos += scrollspeed
        elif k == 27: # esc. for quit
            print(quit)
            break

    cv.destroyWindow('training Data')


def remove_data(beginning, end):
    images, labels = load_train_data()
    save_train_data(images[beginning:end], labels[beginning:end])


# ALGORITHMS =======================================================================================
# generates training data from an image
def train_on_img(img, debug=True, threshold=Settings.threshold):
    # passes image to the image prep. this will return the data (flattened characters) and the positions of the characters for input feedback (red box)
    prepped_data, clipped_img, row_pos, char_pos = split_to_chars(img, debug=debug, threshold=threshold)

    # this list will store the obtained labeling
    train_labels = []

    # changes the clipped image to color so we can have a red border
    color_img = np.stack([clipped_img for i in range(3)], 2)
    # this window will display the typed chars as feedback
    cv.namedWindow('display_chars')
    # this window will show what chars need to be input
    cv.imshow('type_window', clipped_img)
    # keeps track which row / char needs to be input
    iterator = 0
    row = 0

    display_image = color_img.copy()
    # the image to display will have add. information about the analysis of the characters
    # draw in the lines:
    for i in range(len(row_pos)):
        for j in range(len(char_pos[i])):
            cv.line(display_image, (char_pos[i][j][0] + 2, row_pos[i][1] + 5), (char_pos[i][j][1] - 2, row_pos[i][1] + 5), (255, 255, 255), 2)

    # the loop will end if we are on the last row / character (i.e. the last character can't be changed)
    while True:
        # display the character that needs to be entered
        cv.imshow('type_window', draw_rec(display_image, row_pos[row], char_pos[row][iterator]))
        # wait for a key to be pressed (maybe we should add an abort option)
        k = cv.waitKey(0)
        # return has been pressed i.e. delete the last input character
        if k == 8:
            if iterator == 0:
                # the row before
                if row == 0:
                    break
                else:
                    # last row
                    row -= 1
                    # the last character of the row
                    iterator = len(char_pos[row]) - 1
            else:
                # the last character
                iterator -= 1
            # remove the last lable added
            del train_labels[-1]
            display_curr_text(train_labels, 'display_chars', (100, 1000))
            continue
        elif k == 9: # tab for adjusting threshold
            # adjust threshold
            cv.destroyAllWindows()
            while True:
                cv.imshow('threshold_image', cv.threshold(img, threshold, 255, cv.THRESH_BINARY_INV)[1])
                print("Threshold:" + str(threshold) + "adjust threshold [w] for higher, [s] for lower and [Tab] again to end")
                k = cv.waitKey(0)
                if k == 9:
                    # found new threshold redo image
                    cv.destroyAllWindows()
                    # recursive call to self and results get saved function will be left early
                    train_on_img(img, debug=debug, threshold=threshold)
                    return
                elif k == 119:  # w for up
                    threshold += 10
                elif k == 115:  # s for down
                    threshold -= 10
                # display new image

        # add pressed character to training lable and advance the loop. if neccecarry skip to next row or end the program
        train_labels.append(k)
        iterator += 1
        if iterator >= len(char_pos[row]):
            print("next row")
            iterator = 0
            row += 1
            if row >= len(row_pos):
                print("done")
                break
        # display the currently entered characters
        display_curr_text(train_labels, 'display_chars', (100, 1000))

    # save data at the end
    append_new_data(prepped_data, train_labels)
    # close windows
    cv.destroyAllWindows()


# goes through all images in the New_Images folder. uses them for training and then moves them to a new folder
def train_on_new_images(debug=False):
    iterator = 0
    # brute force try all file names
    while True:
        curr_path = new_image_path + '\\' + str(iterator) + '.png'
        img = cv.imread(curr_path, cv.IMREAD_GRAYSCALE)
        if not np.any(img):
            if iterator > 50:
                break
            iterator += 1
            continue
        if debug:
            cv.imshow('orig_image', img)

        # split image into components
        split_images = prep_img(img, youtube_areas)

        if debug:
            for i in range(len(split_images)):
                cv.imshow('split_image: ' + str(i), split_images[i])

            k = cv.waitKey(0)
            cv.destroyAllWindows()
            if k == 27: #escape
                break

        # start training
        for s in split_images:
            train_on_img(s, debug=False)

        # preview the current data if wanted
        if debug:
            view_training_data()

        # move the current file
        # count amount of files in old_image path
        amount = count_images(old_image_path)
        os.rename(curr_path, old_image_path + "\\" + str(amount) + ".png")
        iterator += 1


if __name__ == '__main__':
    # train_on_img(cv.imread('TestImage.png', cv.IMREAD_GRAYSCALE))
    # remove_data(-31 * 23, -1)
    view_training_data()
    train_on_new_images()
