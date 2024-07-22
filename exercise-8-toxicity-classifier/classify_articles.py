from typing import Dict
from fundus import PublisherCollection, Crawler
from predict import Classifier


def crawl_headlines(num_headlines: int = 10) -> Dict[str, list[str]]:
    DailyStar_crw = Crawler(PublisherCollection.uk.DailyStar)
    TheSun_crw = Crawler(PublisherCollection.uk.TheSun)
    return {f"DailyStar": [article.title for article in DailyStar_crw.crawl(max_articles=num_headlines)],
            f"TheSun": [article.title for article in TheSun_crw.crawl(max_articles=num_headlines)]}


def predict_labels(publisher_headlines: Dict[str, list[str]]) -> Dict[str, list[list[str]]]:
    sentiments_per_publisher = {}
    model = Classifier(model_path="results/test_2")
    for publisher, headlines in publisher_headlines.items():
        sentiments_per_publisher[publisher] = []
        for headline in headlines:
            label = model.predict(headline)
            sentiments_per_publisher[publisher].append([headline, label])
    return sentiments_per_publisher


headlines = crawl_headlines()

sentiments_per_publisher = predict_labels(headlines)

for publisher, headlines in sentiments_per_publisher.items():
    print(f"Publisher: {publisher}")
    for headline, sentiment in headlines:
        print(f"{headline}: {sentiment}")
    print()
