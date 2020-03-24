from scipy.stats import entropy

words = "of his bed. harry scrambled into them. the blank of his is together the didn\'t at the"

word_count = 26

ent = entropy([1/2, 1/2, 1/3, 1/4, 1/4], base=2)
print(ent)
