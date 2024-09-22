import wikipediaapi
import json

def get_wikipedia_articles(topics):
    # Add a proper user agent string here
    user_agent = "llm_teacher/1.0 (bdavidriggins@domain.com)"
    wiki_wiki = wikipediaapi.Wikipedia('en', user_agent=user_agent)
    articles = {}
    for topic in topics:
        page = wiki_wiki.page(topic)
        if page.exists():
            articles[topic] = page.text
        else:
            print(f"Article for '{topic}' does not exist.")
    return articles

if __name__ == "__main__":
    topics = [
        "Treaty of Versailles",
        "World War I",
        "Industrial Revolution",
        # Add more topics as needed
    ]
    articles = get_wikipedia_articles(topics)
    with open('../data/wikipedia_articles.json', 'w') as f:
        json.dump(articles, f, indent=2)
    print("Wikipedia articles saved to data/wikipedia_articles.json")
