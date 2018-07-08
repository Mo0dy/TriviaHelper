from TriviaHelper.Question import Question
from TriviaHelper.ImageRec import ImageRec
from TriviaHelper.SearchAlg import SearchAlg

# the main starting point of the program

# create the image
img = None

quest = ImageRec.image_rec(img)
answer = SearchAlg.search_alg(quest)
print(answer)

