from TriviaHelper.Question import Question
import urllib.request
import requests
import re
from bs4 import BeautifulSoup


# gets a Question object and returns the string of the correct answer
def search_alg(question):
    text = question.question
    print("question: " + text)
    for a in question.answers:
        text += " " + a
    text = urllib.parse.quote_plus(text)

    url = 'https://google.com/search?q=' + text
    response = requests.get(url)

    soup = BeautifulSoup(response.text, 'lxml')
    paragraphs = []
    for g in soup.find_all(class_='g'):
        paragraphs.append(g.text)
        print(g.text)
        print('-----')

    ans = question.answers
    answer_count = [0] * len(ans)

    for p in paragraphs:
        for i in range(len(ans)):
            if ans[i] in p:
                answer_count[i] += 1

    return ans[answer_count.index(max(answer_count))]


if __name__ == "__main__":
    search_alg(None)
