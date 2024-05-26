from collections import Counter

from task_1.fundus_crawler import crawl_headlines
from task_1.flair_model import predict_labels


def print_statistics(sentiments_per_publisher):
    for publisher, headlines in sentiments_per_publisher.items():
        print(f"Publisher: {publisher}")
        sentiment_counter = Counter([headline[1] for headline in headlines])
        for sentiment, count in sentiment_counter.items():
            print(f"{sentiment}: {count} headlines")
        print()
    return


def main():
    headlines = crawl_headlines()

    sentiments_per_publisher = predict_labels(headlines)

    print_statistics(sentiments_per_publisher)


if __name__ == "__main__":
    main()
