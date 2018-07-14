from googlesearch import search
import requests
import itertools
Value='Israel'
query = Value
def search_each_answear(answear):
    # searches in a number of websites to check where one of the answears appeared the most
    for j in search(query, tld="co.in", num=1, stop=1, pause=2):
        response = requests.get(j)
        print(response.content)
