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
def load_train_data(path_data=path_data, path_labels=path_labels):
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
def save_train_data(images, labels, local_paths=True):
    if local_paths:
        np.savetxt(r'TrainingData\labels.txt', labels)
        # save images
        np.save('TrainingData\data.npy', images)
    else:
        np.savetxt(path_labels, labels)
        np.save(path_data, images)


def append_new_data(images, lables):
    # first load old data
    old_images, old_labels = load_train_data('TrainingData\data.npy', 'TrainingData\labels.txt')
    # append and save labels as numpy array

    lables_to_save = np.array(old_labels + lables, dtype=np.int)

    # check if old_images is None
    if np.any(old_images):
        # append new data (stored as a n x flattenedImageSize matrix) where n is the number of characters stored
        images = np.vstack((old_images, images))
    save_train_data(images, lables_to_save)


# draws the rectangle around the char that needs to be input according to it's row start and end (y) and char start and end (x)
def color_char(img, row, char, color=2):
    # copy the image so the original one doesnt get changed
    nimg = img.copy()
    # cv.rectangle(nimg, (char[0] - 5, row[0] - 5), (char[1] + 5, row[1] + 5), [0, 0, 255], 2)
    region = nimg[row[0] - 5: row[1] + 10, char[0]: char[1]]
    mask = region[:, :, 1] > 0
    region[:, :, :] = 0
    region[mask, color] = 255
    return nimg


# displayes the currently obtained characters on a dedicated window
def display_curr_text(characters, window_name, window_size, wrong_chars):
    offset = 0
    if len(characters) > 50:
        offset = 15 * (len(characters) - 50)
    img = np.zeros(window_size)
    cv.putText(img, ''.join(chr(c) for c in characters), (10 - offset, 50), font, 1, 255, 2, cv.LINE_AA)

    # display the chars guessed wrong
    cv.putText(img, 'input:   ' + (''.join(chr(c) + ', ' for c in wrong_chars[0][:])), (10, 130), font, 1, 200, 2, cv.LINE_AA)
    cv.putText(img, 'guessed: ' + (''.join(chr(c) + ', ' for c in wrong_chars[1][:])), (10, 170), font, 1, 200, 2, cv.LINE_AA)
    cv.imshow(window_name, img)


# this function will view the currently generated training data. i.e. the cars and the associated lables the color is a boolean array of fields that will be colored initially
def view_training_data(color=None):
    from TriviaHelper.ImageRec.ImageRec import train_knn

    padding_size = 10
    view_size = 800 # the height that gets viewed
    view_pos = 0
    scrollspeed = 100

    # load data
    images, lables = load_train_data('TrainingData\data.npy', 'TrainingData\labels.txt')
    if not np.any(images):
        print("no data")
        return


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
    chars_per_col = int(np.ceil(pixles_per_row / (Settings.char_shape[0] + padding_size)))
    max_height = chars_per_col * (Settings.char_shape[0] + padding_size)
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

    # second padding for the scroll bar
    rows.append(padding)
    rows.append(padding)


    # Actual viewing ===================================================================================================
    rgb_padding = np.stack([padding for i in range(3)], 2)
    view_img = np.hstack(rows).astype(np.uint8)
    scroll_bar_height = 50
    inter_height = view_size - scroll_bar_height
    scroll_max_height = max_height - view_size

    # a copy so that the image can be modified
    display_image = np.stack([view_img for i in range(3)], 2)

    # calculate square_size for the mouse_listener to figure out the correct square that was pressed
    square_x = width + padding_size
    square_y = text_shape[0]

    # gets activated if scrollbar is used
    update_pos = False

    cv.namedWindow('training Data')

    # convert the lable text to one string
    all_lables = [''.join(chr(int(l))) for l in lables]

    # the size of the square that will be viewed
    vs_x = square_x - padding_size
    vs_y = square_y - half_horizontal_padding * 2 - 1

    info_window_size = 200, 700


    # color form boolean list
    # draws AROUND every entry that is in the list with a certain color
    def col_from_bool(arr, color):
        nonlocal display_image
        for i in range(arr.shape[0]):
            if not arr[i]:
                # calculate coordinates on the window
                column = int(i / chars_per_col)
                row = i - column * chars_per_col

                print(row, column)

                # the whole square
                selected_img_pos = [column * square_x + padding_size, row * square_y - 2,
                                    (column + 1) * square_x,
                                    (row + 1) * square_y - half_horizontal_padding * 2 - 1]

                region = display_image[selected_img_pos[1]:selected_img_pos[3], selected_img_pos[0]:selected_img_pos[2]]
                mask = np.logical_or(region[:, :, 1] > 200, region[:, :, 0] > 200, region[:, :, 2] > 200)
                region[mask, 0] = color[0]
                region[mask, 1] = color[1]
                region[mask, 2] = color[2]

    # an array containing the information about the analysis
    correct, results = None, None
    # trains the knn and analyzes the training data. highlights the entries that might be wrong
    def train_and_analyze():
        nonlocal correct, results
        knn = train_knn(True)
        ret, results, neighbours, dist = knn.findNearest(images.astype(np.float32), Settings.k_nearest)
        training_lables = np.array(lables)
        correct = training_lables == results.reshape(-1)
        # every wrong one needs to be colored
        col_from_bool(correct, (200, 150, 50))

    # this function gets called on mouse events and will allow the use to do actions with the mouse
    def mouse_callback(event, x, y, flags, param):
        nonlocal view_pos, update_pos, display_image, correct, results
        if event == cv.EVENT_LBUTTONUP:
            update_pos = False
        elif update_pos:
            view_pos = int((y / inter_height) * scroll_max_height)
        elif event == cv.EVENT_LBUTTONDOWN: # cant do rmb if update pos is active
            # check if in scrollbar if so set scrollbar
            if x > view_img.shape[1] - padding_size:
                update_pos = True
            else:
                curr_x_square = int(x / square_x)
                curr_y_square = int((y + view_pos) / square_y)

                # calculate index of current character
                index = curr_x_square * chars_per_col + curr_y_square

                # the beginning and end coordinates of the current square
                selected_img_pos = [curr_x_square * square_x + padding_size, curr_y_square * square_y, (curr_x_square + 1) * square_x, (curr_y_square + 1) * square_y - half_horizontal_padding * 2 - 1]
                selected_img = display_image[selected_img_pos[1]:selected_img_pos[3], selected_img_pos[0]:selected_img_pos[2], 0]

                dimg = np.ones(info_window_size).astype(np.uint8) * 50
                cv.putText(dimg, 'what do you want to do? Selected: ', (10, 40), font, 1, (255, 255, 255), 1, cv.LINE_AA)
                dimg[10:10+vs_y, 580:580+vs_x] = selected_img
                cv.putText(dimg, 'currently identified as: "' + (chr(results[index])) + '"', (10, 80), font, 1, (255, 255, 255), 1, cv.LINE_AA)
                cv.putText(dimg, 'modify [m], remove[r], return[esc]', (10, 120), font, 1, (255, 255, 255), 1, cv.LINE_AA)
                cv.putText(dimg, 'position: ' + str(index), (10, 160), font, 1,
                           (255, 255, 255), 1, cv.LINE_AA)
                cv.imshow('options', dimg)
                while True:
                    k = cv.waitKey(0)
                    if k == 109: # m
                        dimg = np.ones(info_window_size).astype(np.uint8) * 50
                        cv.putText(dimg, 'press new button:', (10, 50), font, 1, (255, 255, 255), 1, cv.LINE_AA)
                        cv.imshow('options', dimg)

                        while True:
                            k = cv.waitKey(0)
                            stored_key = k
                            char = ''.join(chr(k))
                            dimg = np.ones(info_window_size).astype(np.uint8) * 50
                            cv.putText(dimg, 'do you want to change: "' + str(all_lables[index]) + '" to: "' + char + '"?', (10, 50), font, 1, (255, 255, 255), 1, cv.LINE_AA)
                            cv.putText(dimg, 'yes [y], no [n] return [esc]', (10, 120), font, 1, (255, 255, 255),
                                       1, cv.LINE_AA)
                            cv.imshow('options', dimg)
                            k = cv.waitKey(0)
                            if k == 121: # y
                                # change stored item and restart program
                                lables[index] = stored_key
                                save_train_data(images, lables)

                                # make square white:
                                middle_x = int((selected_img_pos[0] + selected_img_pos[2]) / 2) + 2
                                display_image[selected_img_pos[1]:selected_img_pos[3], middle_x:selected_img_pos[2]] = 255

                                # make everything white background again
                                correct[index] = False
                                col_from_bool(correct, (255, 255, 255))
                                train_and_analyze()
                                cv.putText(display_image, chr(int(stored_key)), (middle_x + 7, selected_img_pos[3] - 14), font, 1, (50, 200, 200), 2, cv.LINE_AA)
                                break
                            elif k == 110: # n
                                continue
                            elif k == 27: # esc
                                break
                    elif k == 114: # r
                        print("functionality not yet supported")
                        print("r")
                    elif k != 27: # esc
                        continue
                    break
                cv.destroyWindow('options')

        elif event == cv.EVENT_MOUSEWHEEL:
            if flags > 0: # scroll up
                view_pos -= scrollspeed
            else: # scroll down
                view_pos += scrollspeed

    cv.setMouseCallback('training Data', mouse_callback)

    train_and_analyze()
    if np.any(color):
        col_from_bool(color, (0, 100, 200))

    while True:
        # calculate view image
        if view_pos < 0:
            view_pos = 0
        elif view_pos > view_img.shape[0] - view_size:
            view_pos = int(view_img.shape[0] - view_size)
        end = view_pos + view_size
        if end >= view_img.shape[0]:
            end = view_img.shape[0] - 1

        # calculate scroll_bar_position (linear interpolation)
        scroll_height = view_pos + int(view_pos / scroll_max_height * inter_height)
        display_image[:, -padding_size:] = rgb_padding
        display_image[scroll_height:scroll_height + scroll_bar_height, -padding_size:] = 200

        cv.imshow('training Data', display_image[view_pos:end, :])
        # this allows the mouse function to skip the keyboard input
        k = cv.waitKey(20)
        if k == 119: # w for up
            view_pos -= scrollspeed
        elif k == 115: # s for down
            view_pos += scrollspeed
        elif k == 27: # esc. for quit
            print(quit)
            break

    cv.destroyWindow('training Data')


def remove_data(beginning, end):
    images, labels = load_train_data('TrainingData\data.npy', 'TrainingData\labels.txt')
    images = np.vstack((images[:beginning], images[end:])).copy()
    del labels[beginning:end]
    save_train_data(images, labels)


# ALGORITHMS =======================================================================================
# generates training data from an image
def train_on_img(img, knn, debug=True, threshold=Settings.threshold):
    # passes image to the image prep. this will return the data (flattened characters) and the positions of the characters for input feedback (red box)
    prepped_data, clipped_img, row_pos, char_pos = split_to_chars(img, debug=debug, threshold=threshold)
    ret, results, neighbours, dist = knn.findNearest(prepped_data.astype(np.float32), Settings.k_nearest)

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
            cv.line(display_image, (char_pos[i][j][0] + 2, row_pos[i][1] + 5), (char_pos[i][j][1] - 2, row_pos[i][1] + 5), (200, 200, 50), 2)

    # the loop will end if we are on the last row / character (i.e. the last character can't be changed)
    while True:
        # display the character that needs to be entered
        dimg = color_char(display_image, row_pos[row], char_pos[row][iterator])

        # this could only be done once and remembered!
        # color all chars that are interpreted differend:
        # wrong chars store the thought char and input char
        wrong_chars = [[], []]
        running = True
        iter = 0
        for i in range(len(row_pos)):
            if not running:
                break
            for j in range(len(char_pos[i])):
                if i < row or j < iterator:
                    if train_labels[iter] != results[iter]:
                        # thinks the char is wrong
                        dimg = color_char(dimg, row_pos[i], char_pos[i][j], 1)
                        wrong_chars[0].append(train_labels[iter])
                        wrong_chars[1].append(results[iter])
                else:
                    running = False
                    break
                iter += 1

        cv.imshow('type_window', dimg)
        # wait for a key to be pressed (maybe we should add an abort option)
        k = cv.waitKey(0)
        # return has been pressed i.e. delete the last input character
        if k == 8: # return
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
            display_curr_text(train_labels, 'display_chars', (200, 1000), wrong_chars)
            continue
        elif k == 9: # tab for adjusting threshold
            # adjust threshold
            cv.destroyAllWindows()
            while True:
                cv.imshow('threshold_image', cv.threshold(img, threshold, 255, cv.THRESH_BINARY_INV)[1])
                print("showing")
                print("Threshold:" + str(threshold) + "adjust threshold [w] for higher, [s] for lower and [Tab] again to end")
                k = cv.waitKey(0)
                if k == 9:
                    # found new threshold redo image
                    cv.destroyAllWindows()
                    # recursive call to self and results get saved function will be left early
                    train_on_img(img, knn, debug=debug, threshold=threshold)
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
        display_curr_text(train_labels, 'display_chars', (200, 1000), wrong_chars)

    # save data at the end
    append_new_data(prepped_data, train_labels)
    # close windows
    cv.destroyAllWindows()


# goes through all images in the New_Images folder. uses them for training and then moves them to a new folder
def train_on_new_images(debug=False):
    from TriviaHelper.ImageRec.ImageRec import train_knn
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
            knn = train_knn(True)
            train_on_img(s, knn, debug=False)

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
    # remove_data(3586, -1)
    view_training_data()
    images, lables = load_train_data('TrainingData\data.npy', 'TrainingData\labels.txt')
    curr_count = len(lables)
    train_on_new_images()
    images, lables = load_train_data('TrainingData\data.npy', 'TrainingData\labels.txt')
    new_count = len(lables)
    arr = np.ones(new_count)
    arr[curr_count:] = 0
    view_training_data(arr.astype(bool))
