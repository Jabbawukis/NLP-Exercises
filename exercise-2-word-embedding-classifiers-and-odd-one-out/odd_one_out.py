from itertools import combinations

from scipy.spatial import distance
from gensim.models.keyedvectors import Word2VecKeyedVectors

from data import load_odd_one_out


def main():
    # Load the pretrained embeddings
    pretrained_word_embeddings_file = "pretrained_embeddings/glove-wiki-gigaword-50.txt"
    data = load_odd_one_out()
    pretrained_embeddings = Word2VecKeyedVectors.load_word2vec_format(
        pretrained_word_embeddings_file
    )
    # Iterate over the examples
    for example in data:
        word_dict = {}
        out_of_vocab = False
        for combination in list(combinations(example, len(example) - 1)):  # Get the most similar word for each word
            # in the example
            try:
                sim = pretrained_embeddings.similarity(combination[0], combination[1])
            except KeyError:
                out_of_vocab = True
                break
            try:
                if word_dict[combination[0]][1] < sim:
                    word_dict[combination[0]] = [combination[1], sim]
            except KeyError:
                word_dict[combination[0]] = [combination[1], sim]
            try:
                if word_dict[combination[1]][1] < sim:
                    word_dict[combination[1]] = [combination[0], sim]
            except KeyError:
                word_dict[combination[1]] = [combination[0], sim]
        if out_of_vocab:
            print(f"Out of vocabulary word found in {example}.")  # Skip the example if out of vocabulary word found
            print(10 * "-")
            continue
        odd_one_out = set()
        votes = dict.fromkeys(word_dict, 0)  # Count the votes for each word
        for _, most_sim_word in word_dict.items():
            votes[most_sim_word[0]] += 1
        odd_one_out.add(min(votes, key=votes.get))  # Get the word with the least votes
        print(f"Odd one out in {example} is {odd_one_out.pop()}.")
        print(10 * "-")


if __name__ == "__main__":
    main()
