import numpy as np
import cv2 as cv
from TriviaHelper.ImageRec.imgprep import prep_img
import TriviaHelper.ImageRec.Settings as Settings

# this module takes an image and queries the user for the correct interpretation. This will generate Training data it will later learn from

# the font for displaying feedback to the user
font = cv.FONT_HERSHEY_SIMPLEX

# the paths of the save files for the generated data. New data will be appended to these files
path_labels = r'TrainingData\labels.txt'
path_data = r'TrainingData\data.npy'

# UTILITY =======================================================================================
# this loads the training data
def load_train_data():
    # exeptions in chase this is the first time data gets entered
    try:
        images = np.load(path_data)
        print("loaded images")
    except:
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
    # first load old data
    old_images, old_labels = load_train_data()
    # append and save labels as numpy array
    np.savetxt(path_labels, np.array(old_labels + labels, dtype=np.int))

    # check if old_images is None
    if np.any(old_images):
        # append new data (stored as a n x flattenedImageSize matrix) where n is the number of characters stored
        images = np.vstack((old_images, images))
    # save images
    np.save(path_data, images)


# draws the rectangle around the char that needs to be input according to it's row start and end (y) and char start and end (x)
def draw_rec(img, row, char):
    # copy the image so the original one doesnt get changed
    nimg = img.copy()
    cv.rectangle(nimg, (char[0] - 5, row[0] - 5), (char[1] + 5, row[1] + 5), [0, 0, 255], 2)
    return nimg


# displayes the currently obtained characters on a dedicated window
def display_curr_text(characters, window_name, window_size):
    img = np.zeros(window_size)
    cv.putText(img, ''.join(chr(c) for c in characters), (10, 50), font, 1, 255, 2, cv.LINE_AA)
    cv.imshow(window_name, img)


# this function will view the currently generated training data. i.e. the cars and the associated lables
def view_training_data():
    padding_size = 10
    view_size = 1000 # the height that gets viewed
    view_pos = 0
    scrollspeed = 100

    # load data
    images, lables = load_train_data()

    # generate character image
    # make character images out of the flattened chars + padding
    char_images = [np.vstack((images[i].reshape(Settings.char_shape), np.zeros((padding_size, Settings.char_shape[1])))) for i in range(images.shape[0])]
    # the image of all chars stacked on top of each other
    char_images = np.vstack(char_images).astype(np.uint8)

    # generate interpreted text
    generated_text = []
    text_shape = Settings.char_shape[0] + padding_size, Settings.char_shape[1]
    for l in lables:
        lable_image = np.zeros(text_shape)
        cv.putText(lable_image, chr(int(l)), (5, text_shape[1] - 13), font, 1, 200, 2, cv.LINE_AA)
        generated_text.append(lable_image)

    stacked_lables = np.vstack(generated_text)

    # combine images and add padding
    half_horizontal_padding = 5
    view_img = np.hstack((char_images, np.zeros((char_images.shape[0], half_horizontal_padding)), np.ones((char_images.shape[0], 2)) * 100, np.zeros((char_images.shape[0], half_horizontal_padding)), stacked_lables)).astype(np.uint8)
    # add seperation lines
    for i in range(len(generated_text) - 1):
        cv.line(view_img, (0, (i + 1) * text_shape[0] - 5), (view_img.shape[1], (i + 1) * text_shape[0] - 5), 100, 2)

    while True:
        # calculate view image
        if view_pos < 0:
            view_pos = 0
        elif view_pos > view_img.shape[0] - view_size / 2:
            view_pos = int(view_img.shape[0] - view_size / 2)
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


# ALGORITHMS =======================================================================================
# generates training data from an image
def train_on_img(img):
    # passes image to the image prep. this will return the data (flattened characters) and the positions of the characters for input feedback (red box)
    prepped_data, clipped_img, row_pos, char_pos = prep_img(img, debug=True)

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

    # the loop will end if we are on the last row / character (i.e. the last character can't be changed)
    while True:
        # display the character that needs to be entered
        cv.imshow('type_window', draw_rec(color_img, row_pos[row], char_pos[row][iterator]))
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
            display_curr_text(train_labels, 'display_chars', (100, 800))
            continue

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
    save_train_data(prepped_data, train_labels)
    # close windows
    cv.destroyAllWindows()


if __name__ == '__main__':
    train_on_img(cv.imread('TestImage.png', cv.IMREAD_GRAYSCALE))
    view_training_data()
