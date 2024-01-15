'''from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import Dataset, DataLoader, random_split
import torch

class SummaryDataset(Dataset):
    def __init__(self, documents, summaries):
        self.documents = documents
        self.summaries = summaries

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        return self.documents[idx], self.summaries[idx]

# Load your preprocessed data
documents = ["...", "...", "..."]  # Replace with your preprocessed documents
summaries = ["...", "...", "..."]  # Replace with your preprocessed summaries

# Initialize the T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Create a PyTorch Dataset
dataset = SummaryDataset(documents, summaries)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create PyTorch DataLoaders for training and validation
train_dataloader = DataLoader(train_dataset, batch_size=32)
val_dataloader = DataLoader(val_dataset, batch_size=32)

# Initialize the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Train the model
for epoch in range(10):  # 10 epochs
    model.train()
    for document, summary in train_dataloader:
        # Encode the document and summary
        input_ids = tokenizer(document, return_tensors='pt', truncation=True, padding=True, max_length=512)['input_ids']
        labels = tokenizer(summary, return_tensors='pt', truncation=True, padding=True, max_length=150)['input_ids']

        # Compute the loss
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # Backpropagate the loss
        loss.backward()

        # Clip the gradient norms (optional)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update the weights
        optimizer.step()
        optimizer.zero_grad()

    # Evaluate the model on the validation set
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for document, summary in val_dataloader:
            # Encode the document and summary
            input_ids = tokenizer(document, return_tensors='pt', truncation=True, padding=True, max_length=512)['input_ids']
            labels = tokenizer(summary, return_tensors='pt', truncation=True, padding=True, max_length=150)['input_ids']

            # Compute the loss
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            val_loss += loss.item()

    print(f'Validation loss after epoch {epoch}: {val_loss / len(val_dataloader)}')

# Save the trained model
model.save_pretrained('/model.p')'''

from transformers import T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer
import torch

def evaluate(model_path, test_documents, test_summaries):
    # Load the trained model
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1 = 0
    rouge2 = 0
    rougeL = 0

    # Iterate over the test data
    for document, reference_summary in zip(test_documents, test_summaries):
        # Encode the document
        input_ids = tokenizer(document, return_tensors='pt', truncation=True, padding=True, max_length=512)['input_ids']

        # Generate the summary
        output_ids = model.generate(input_ids)
        generated_summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Calculate the ROUGE scores
        scores = scorer.score(reference_summary, generated_summary)
        rouge1 += scores['rouge1'].fmeasure
        rouge2 += scores['rouge2'].fmeasure
        rougeL += scores['rougeL'].fmeasure

    # Calculate the average ROUGE scores
    rouge1 /= len(test_documents)
    rouge2 /= len(test_documents)
    rougeL /= len(test_documents)

    print(f'ROUGE-1: {rouge1}, ROUGE-2: {rouge2}, ROUGE-L: {rougeL}')

if __name__ == "__main__":
    model_path = '/model.p'  # Replace with the path to your trained model
    test_documents = ["...", "...", "..."]  # Replace with your test documents
    test_summaries = ["...", "...", "..."]  # Replace with your test summaries
    evaluate(model_path, test_documents, test_summaries)

