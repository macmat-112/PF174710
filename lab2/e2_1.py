from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

porter = PorterStemmer()
text = "The researchers were studying the effectiveness of different approaches to natural language processing."
tokens = word_tokenize(text)
print(f"|{"Oryginał:":<15} → {"Stem:":<15}|")
print(f"|{'-' * 33}|")
for i in range(len(tokens)):
    print(f"|{tokens[i]:<15} → {porter.stem(tokens[i]):<15}|")
