import wikipediaapi

wiki_wiki = wikipediaapi.Wikipedia('en')

def get_wikipedia_articles(topics):
    articles = {}
    for topic in topics:
        page = wiki_wiki.page(topic)
        if page.exists():
            articles[topic] = page.text
    return articles

topics = ["Treaty of Versailles", "World War I", "Industrial Revolution"]
articles = get_wikipedia_articles(topics)

# Save articles to disk
import json
with open('wikipedia_articles.json', 'w') as f:
    json.dump(articles, f)
