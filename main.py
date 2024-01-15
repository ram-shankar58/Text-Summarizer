from models.model import TextSummarizer
from src.utils.preprocessing import *
from src.utils.postprocessing import *
import os
import json
import numpy as np
def main():
    # Load your data
    with open(os.path.join('data', 'data.json'), 'r') as f:
        data = json.load(f)
    documents = data['documents']
    summaries = data['summaries']
    # Combine all documents and summaries
    texts = documents + summaries

    # Prepare the data
    target_token_index, reverse_target_char_index, max_decoder_seq_length = prepare_data(texts)

    # Preprocess the text
    preprocessed_documents = [preprocess(doc) for doc in documents]
    preprocessed_summaries = [preprocess(summary) for summary in summaries]

    # Initialize the model
    num_encoder_tokens = 1000  # replace with your value
    num_decoder_tokens = 1000  # replace with your value
    latent_dim = 256  # replace with your value
    input_texts=documents
    target_texts=summaries
    # Tokenize the source texts
    tokens = set(token for text in input_texts for token in preprocess(text))

# Compute the number of unique tokens
    num_unique_tokens = len(tokens)

# Initialize the model
    model = TextSummarizer(num_unique_tokens, num_unique_tokens, latent_dim, target_token_index, reverse_target_char_index, max_decoder_seq_length)

    
        # Define the variables
    num_pairs = len(documents)  # or len(summaries)
    max_english_sentence_length = max(len(doc) for doc in preprocessed_documents + preprocessed_summaries)
    num_english_characters = len(target_token_index)
    input_texts = documents
    target_texts = summaries

    # Initialize the arrays with zeros
    encoder_input_data = np.zeros((num_pairs, max_english_sentence_length, num_english_characters))
    decoder_input_data = np.zeros((num_pairs, max_english_sentence_length, num_english_characters))
    decoder_target_data = np.zeros((num_pairs, max_english_sentence_length, num_english_characters))
    
# Create the token-index mapping
    input_token_index = {token: i for i, token in enumerate(tokens)}
    
    # Fill the arrays
    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        input_tokens = preprocess(input_text)
        target_tokens = preprocess(target_text)
        for t, token in enumerate(input_tokens):
            encoder_input_data[i, t, input_token_index[token]] = 1.
        for t, token in enumerate(target_tokens):
            decoder_input_data[i, t, target_token_index[token]] = 1.
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[token]] = 1.


    batch_size = 64  # replace with your value
    epochs = 100  # replace with your value
    validation_split = 0.2  # replace with your value
    model.train(encoder_input_data, decoder_input_data, decoder_target_data, batch_size, epochs, validation_split)

    # Generate the summary
    input_text = input('Enter your input sequence')  # replace with your value
    input_tokens = preprocess(input_text)
    input_seq = np.zeros((1, max_english_sentence_length, num_english_characters))
    for t, token in enumerate(input_tokens):
        input_seq[0, t, input_token_index[token]] = 1.
    summary = model.summarize(input_seq)

    # Postprocess the summary
    postprocessed_summary = postprocess(summary)

    # Print the summary
    print(postprocessed_summary)

if __name__ == "__main__":
    main()
