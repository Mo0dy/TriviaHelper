# the settings for the image recognition and training collection algorithms

# the shape of the single char images
char_shape = (35, 35)

# the initial threshold for making the text image black and white
threshold = 120

# the areas of the question and answers after passed through the screen capture and clip function
youtube_areas = [[0, 0, 200, 520], [220, 30, 278, 473], [318, 32, 374, 481], [415, 32, 471, 476]]

# the begginning end end coordinates around the square with the important information in the screenshot clip function
beginning = 702, 220
end = 1222, 700

# the amount of knearest that will be searched
k_nearest = 1