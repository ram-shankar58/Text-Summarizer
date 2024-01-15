import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK corpora if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess(text):
    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return words
def prepare_data(texts):
    # Tokenize the texts
    tokens = set(token for text in texts for token in preprocess(text))

    # Create the token-index mapping
    target_token_index = {token: i for i, token in enumerate(tokens)}

    # Create the reverse mapping
    reverse_target_char_index = {i: token for token, i in target_token_index.items()}

    # Compute the maximum sequence length
    max_decoder_seq_length = max(len(preprocess(text)) for text in texts)

    return target_token_index, reverse_target_char_index, max_decoder_seq_length
