import string
punct = set(string.punctuation);
def getStringWithPunctuationsRemoved(originalString):
    return ''.join(ch for ch in originalString if ch not in punct)
    
def getNumberOfCapitalLettersFromWord(inputWord):
    return sum(1 for c in inputWord if c.isupper())
