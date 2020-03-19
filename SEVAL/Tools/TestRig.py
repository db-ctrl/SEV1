from scipy.stats import entropy

words_in_clus = ["hi"]

word_count = 26

ent = entropy([len(words_in_clus) / word_count,  (word_count - len(words_in_clus)) / word_count], base=2)

print(ent)
