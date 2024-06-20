import torch
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from data_loader import load_and_preprocess_data
from transformers import DataCollatorForSeq2Seq

# Load and preprocess data
dataset = load_and_preprocess_data()

# Initialize tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['article'], truncation=True, padding='max_length', max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
    output_dir='./results',
    overwrite_output_dir=True,
    logging_steps=500,
    save_steps=1000,
    evaluation_strategy='epoch'
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./bart_model")
