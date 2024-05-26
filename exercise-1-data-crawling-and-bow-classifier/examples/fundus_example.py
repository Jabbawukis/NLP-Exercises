from fundus import PublisherCollection, Crawler


def main():
    # initialize the crawler for Washington Times
    crawler = Crawler(PublisherCollection.us.WashingtonTimes)

    # crawl 2 articles and print
    for article in crawler.crawl(max_articles=2):

        # print article overview
        print(article)

        # print only the title
        print(article.title)


if __name__ == "__main__":
    main()
