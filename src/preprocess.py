import re
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

stop_words = set(stopwords.words('english'))
negation_words = {'not', 'no', 'nor', 'never', "didn't", "don't", "won't", "isn't", "wasn't", "aren't"}
stop_words = stop_words - negation_words
stop_words.add('reuters')
stop_words.add('via')
stop_words.add('getty')
stop_words.add('screenshot')

lemmatizer = WordNetLemmatizer()

def remove_reuters_dateline(text):
    match = re.match(r'^.{0,100}?\(Reuters\)\s*-\s*', text)
    if match:
        return text[match.end():]
    return text

def remove_media_credits(text):
    text = re.sub(r'featured image[:\s](by|from)?.{0,60}', '', text, flags=re.IGNORECASE)
    return text

def repair_broken_contractions(text):
    text = re.sub(r"\b(couldn|didn|wasn|isn|wouldn|shouldn|aren|weren|hasn|haven|hadn|doesn|don) t\b", r"\1't", text, flags=re.IGNORECASE)
    return text

def clean_text(text):
    text = remove_reuters_dateline(text)
    text = remove_media_credits(text)
    text = repair_broken_contractions(text)
    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)