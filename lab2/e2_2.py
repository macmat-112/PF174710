from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

porter = PorterStemmer()
lancaster = LancasterStemmer()
words = ['generous', 'generation', 'generalize', 'generally', 'generic']
print(f"|{"Oryginał:":<15} | {"Porter:":<15} | {"Lancaster:":<15}|")
print(f"|{'-' * 51}|")
for i in range(len(words)):
    print(f"|{words[i]:<15} | {porter.stem(words[i]):<15} | {lancaster.stem(words[i]):<15}|")

# obydwa ucinają wszystkie wyrazy do jednego korzenia, ale Lancaster ucina o dwie litery więcej, niż Porter.
