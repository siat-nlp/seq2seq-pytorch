from nltk import word_tokenize
import spacy
import re

SPACE_NORMALIZER = re.compile(r"\s+")

def fair_tokenizer(text):
    text = SPACE_NORMALIZER.sub(" ", text)
    text = text.strip().split()
    return text

def nltk_tokenizer(text):
    return word_tokenize(text)

url = re.compile('(<url>.*</url>)')
spacy_en = spacy.load('en')
spacy_de = spacy.load('de')

def check(x):
    return len(x) >= 1 and not x.isspace()

def spacy_en_tokenizer(text):
    tokens = [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]
    return list(filter(check, tokens))

def spacy_de_tokenizer(text):
    tokens = [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]
    return list(filter(check, tokens))