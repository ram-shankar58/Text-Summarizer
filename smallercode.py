import torch
import nltk
from nltk.corpus import brown
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from summarizer import Summarizer
from concurrent.futures import ThreadPoolExecutor

nltk.download('brown')
# Check if GPU is available and if not, use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

corpus=' '.join(brown.words())

# Initialize outside the functions
global_vectorizer = TfidfVectorizer().fit(corpus.split('.'))
summarizer_model = pipeline("summarization")

tokenizer_t5 = GPT2Tokenizer.from_pretrained('gpt2')
model_t5 = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
bert_model = Summarizer()

# Fit the vectorizer once on a large corpus of text


# Extractive Summarization Function using BERT
def bert_extractive_summarization(text, ratio=0.5):
    return bert_model(text, ratio=ratio)

# Abstractive Summarization Function using GPT-2
# Abstractive Summarization Function using GPT-2
def gpt2_abstractive_summarization(text):
    input_ids = tokenizer_t5.encode(text, return_tensors='pt', max_length=512, truncation=True)
    
    # Adjust max_length to match the length of the input_ids tensor
    max_length = input_ids.size(1)
    
    outputs = model_t5.generate(input_ids, max_length=max_length, num_beams=2, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    
    return tokenizer_t5.decode(outputs[0], skip_special_tokens=True)


# Paraphrase Function using GPT-2
def paraphrase_text_with_gpt2(text):
    input_ids = tokenizer_t5.encode(text, return_tensors='pt', max_length=512, truncation=True)
    max_length = input_ids.size(1)  # Get the length of the input tensor
    outputs = model_t5.generate(input_ids, max_length=max_length, num_beams=2, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    return tokenizer_t5.decode(outputs[0], skip_special_tokens=True)


# Query-Based Summarization Function
def query_based_summarization(query, text, num_sentences=3):
    combined_text = query + ' ' + text
    sentences = combined_text.split('.')
    vectors = global_vectorizer.transform(sentences)
    query_vector = vectors[0]
    cosine_similarities = cosine_similarity(query_vector, vectors).flatten()
    ranked_sentences = [(sentence, score) for sentence, score in zip(sentences, cosine_similarities)]
    ranked_sentences = sorted(ranked_sentences, key=lambda x: x[1], reverse=True)
    summary_sentences = [sentence for sentence, _ in ranked_sentences[:num_sentences]]
    return ' '.join(summary_sentences)




def combined_summarization_with_paraphrasing(text, query=None):
    with ThreadPoolExecutor() as executor:
        bert_extractive_future = executor.submit(bert_extractive_summarization, text)
        gpt2_abstractive_future = executor.submit(gpt2_abstractive_summarization, text)
        paraphrase_future = executor.submit(paraphrase_text_with_gpt2, text)
        if query:
            query_based_future = executor.submit(query_based_summarization, query, text)

    print("BERT Extractive Summary:")
    print(bert_extractive_future.result())
    print("\nGPT-2 Abstractive Summary:")
    print(gpt2_abstractive_future.result())
    print("\nParaphrased Text:")
    print(paraphrase_future.result())
    if query:
        print("\nQuery-Based Summary:")
        print(query_based_future.result())


# Example Usage
text = input('Enter your text')
yesorno=input('Do you have any query with respect to which summarisation has to be done? Yes or No')
if(yesorno.lower()=='yes'):
    query=input('Enter your query')

combined_summarization_with_paraphrasing(text, query)


