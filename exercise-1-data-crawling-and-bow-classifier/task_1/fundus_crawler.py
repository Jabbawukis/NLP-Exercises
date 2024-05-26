from typing import Dict, List
from fundus import PublisherCollection, Crawler


def crawl_headlines(num_headlines: int = 10) -> Dict[str, list[str]]:
    # You can return a dictionary with the following structure, but you can adjust it as needed.:
    # {
    #     "publisher_1": ["headline_1", "headline_2", ..., "headline_10"],
    #     "publisher_2": ["headline_1", "headline_2", ..., "headline_10"]
    # }
    # headlines = {}

    latimes_crw = Crawler(PublisherCollection.us.LATimes)
    fox_crw = Crawler(PublisherCollection.us.FoxNews)

    headlines = {f"LATimes": [article.title for article in latimes_crw.crawl(max_articles=num_headlines)],
                 f"FoxNews": [article.title for article in fox_crw.crawl(max_articles=num_headlines)]}

    # List of publishers: https://github.com/flairNLP/fundus/blob/master/docs/supported_publishers.md

    return headlines
