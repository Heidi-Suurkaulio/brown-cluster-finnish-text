from brown_clustering import BigramCorpus, BrownClustering

def main():
    # use some tokenized and preprocessed data
    sentences = [
        ["This", "is", "an", "example"],
        ["This", "is", "another", "example"]
    ]

    # create a corpus
    corpus = BigramCorpus(sentences, alpha=0.5, min_count=0)

    # (optional) print corpus statistics:
    corpus.print_stats()

    # create a clustering
    clustering = BrownClustering(corpus, m=4)

    # train the clustering
    clusters = clustering.train()


if __name__ == "__main__":
    main()
