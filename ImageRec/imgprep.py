import numpy as np
import cv2 as cv
import TriviaHelper.ImageRec.Settings as Settings

# settings
# the threshold for the boolean image (i.e. color to black / white) difficult because of AA-Characters


# the path for a test image
path = 'TestImage.png'


# this function splits the image (plain text) into chars (flattened) and returns some extra information i.e.
# the coordinates of the chars etc.
# if debug=True it will generate windows that show the process
def split_to_chars(img, debug):
    if debug:
        cv.imshow('beginning', img)

    # Thresholding (Color -> black and white)
    ret, img = cv.threshold(img, Settings.threshold, 255, cv.THRESH_BINARY_INV)

    # find index of lines beginnings
    # 1d array that is true if there is any white pixel on the row (this will build up stripes for character rows)
    rows = np.any(img, 1)
    rows_inverted = np.logical_not(rows)
    if debug:
        viewimg = np.tile(rows.astype(np.uint8) * 255, (500, 1)).transpose()
        cv.imshow('line analysis', viewimg)

    # this algorithm gets used twice and could be it's own function (never touch a running system)
    # the coordinates of the lines
    lines = []
    # the index of the current beginning of a row
    first = 0
    # first lines cant be the beginning of a line because argmax will return 0 if there arent any lines anymore
    while True:
        # find the next beginning of a row relative to the current first
        next = np.argmax(rows[first:])
        # if no beginning is found or it begins right at the first one (potential bug) end
        if next == 0:
            break
        # advance the current beginning of the row and search for the end (the beginning of the inverted row data)
        first += next
        # calculate the end position of the current row
        end = np.argmax(rows_inverted[first:]) + first
        # memorize the data and repeat for next row
        lines.append([first, end])
        first = end

    # create images that contain one row each
    img_lines = [img[l[0]:l[1], :] for l in lines]
    if debug:
        for i in range(len(img_lines)):
            cv.imshow(str(i), img_lines[i])

    # repeat the algorithm for every row (split chars)
    chars = []
    charpos = [[]]
    current_row = 0

    # split lines into characters
    for i in range(len(img_lines)):
        l = img_lines[i]
        # find index of char beginnings
        line_summed = np.any(l, 0)
        line_inverted = np.logical_not(line_summed)
        if debug:
            viewimg = np.tile(line_summed.astype(np.uint8) * 255, (25, 1))
            cv.imshow('character analysis: ' + str(i), viewimg)
        first = 0
        charpos.append([])
        while True:
            next = np.argmax(line_summed[first:])
            if next == 0:
                break
            first += next
            end = np.argmax(line_inverted[first:]) + first
            chars.append(l[:, first:end])
            charpos[current_row].append((first, end))
            first = end
        current_row += 1

    # add padding to the chars (if uneven amount of padding add one more column to the right)
    for i in range(len(chars)):
        c = chars[i]
        padding_size = Settings.char_shape[1] - c.shape[1]
        half_size = int(np.floor(padding_size / 2))
        padding = np.zeros((Settings.char_shape[0], half_size))
        chars[i] = np.hstack((padding, c, padding))
        # check for uneven padding
        if padding_size % 2:
            chars[i] = np.hstack((chars[i], np.zeros((Settings.char_shape[0], 1))))

    if debug:
        cv.imshow("all_chars", np.vstack(chars))
        cv.waitKey(0)
        cv.destroyAllWindows()

    # flatten it into a single row of 35 * 25 pixels and combine to one array of shape n x 35 * 25
    result = np.vstack([c.reshape(Settings.char_shape[0] * Settings.char_shape[1]) for c in chars])
    print("converted " + str(result.shape[0]) + " characters")

    return result, img, lines, charpos


def prep_img(img, debug=False):
    # this will later return multiple images for the answers / question for now it calls the split function with the question image
    return split_to_chars(img[530:650, 600:1140], debug)


if __name__ == "__main__":
    images = prep_img(cv.imread(path, cv.IMREAD_GRAYSCALE))
