import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Sørg for, at nødvendige data er downloadet
nltk.download('punkt')
nltk.download('stopwords')

# Eksempeltekst
text = """Natural Language Processing (NLP) is a field of artificial intelligence 
that focuses on the interaction between computers and humans using natural language. 
The goal of NLP is to enable computers to understand, interpret, and generate human language in a meaningful way."""

tokens = word_tokenize(text)
original_vocab = set(tokens)
print(f"Original tokens: {tokens}\n")

stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print(filtered_tokens)