import json
import requests
from newspaper import Article, Config

def scrapNews(topicName):
    api_key = "AIzaSyAOVoIz59KfO726SCDfccLnBw7BOq-ogWs"
    cse_id = "d07f9f6bb24b64497"
    query = topicName
    num_results = 10
    base_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": num_results,
    }
    article_list = []
    response = requests.get(base_url, params=params)
    results = response.json().get("items", [])

    for result in results:
        link = result.get("link")
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
        config = Config()
        config.browser_user_agent = user_agent

        article = Article(link, config=config)
        article.download()
        article.parse()
        article.nlp()
        dict = {}
        if article.summary[:2] != "Ad":
            dict['Summary'] = article.summary
            article_list.append(dict)
    with open('sentimental_analysis/realworld/news.json', 'w') as json_file:
        json.dump(article_list, json_file)

    print("Articles saved to news.json")
