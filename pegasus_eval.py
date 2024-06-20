import tensorflow as tf
from transformers import PegasusTokenizer, TFPegasusForConditionalGeneration
from data_loader import load_and_preprocess_data
from datasets import load_metric

# Load and preprocess data
dataset = load_and_preprocess_data()

# Initialize tokenizer and model
tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
model = TFPegasusForConditionalGeneration.from_pretrained('./pegasus_model')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['article'], truncation=True, padding='max_length', max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Evaluate the model
metric = load_metric('rouge')
def evaluate_model(model, tokenizer, dataset):
    for batch in dataset['test']:
        inputs = tokenizer(batch['article'], truncation=True, padding='max_length', max_length=512, return_tensors="tf")
        summary_ids = model.generate(inputs['input_ids'])
        decoded_preds = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
        metric.add_batch(predictions=decoded_preds, references=batch['highlights'])
    return metric.compute()

# Get the evaluation results
results = evaluate_model(model, tokenizer, tokenized_datasets)
print(results)
