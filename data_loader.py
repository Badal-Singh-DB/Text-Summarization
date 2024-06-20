import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset

def load_and_preprocess_data(filepath="data/news_summary.csv"):
    # Load data from CSV
    data = pd.read_csv(filepath)
    
    # Basic preprocessing
    data = data[['text', 'ctext']]
    data.columns = ['summary', 'article']
    
    # Split the data into train, validation, and test sets
    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
    
    # Convert to Hugging Face Dataset format
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)
    test_dataset = Dataset.from_pandas(test_data)
    
    return train_dataset, val_dataset, test_dataset
