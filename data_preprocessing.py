import pandas as pd
import numpy as np
import ast

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Clean and preprocess the data
def preprocess_data(df):
    # Remove rows with missing values in the target column (Price)
    df = df.dropna(subset=['Price'])
    
    # Handle any other missing values or issues as needed
    df['SquareFeet'] = df['SquareFeet'].replace(0, np.nan)
    df.dropna(subset=['SquareFeet'], inplace=True)
    
    # Convert any string columns to lists or proper formats if needed
    # (example of how you would handle columns like genres, keywords, etc.)
    
    return df
