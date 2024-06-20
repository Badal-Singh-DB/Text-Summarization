import tensorflow as tf
from transformers import PegasusTokenizer, TFPegasusForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from data_loader import load_and_preprocess_data
from transformers import DataCollatorForSeq2Seq

# Load and preprocess data
dataset = load_and_preprocess_data()

# Initialize tokenizer and model
tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
model = TFPegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['article'], truncation=True, padding='max_length', max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
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
trainer = Seq2SeqTrainer(
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
model.save_pretrained("./pegasus_model")
