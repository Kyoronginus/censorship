import pandas as pd
from googletrans import Translator
import glob
import os
from tqdm import tqdm  # Import tqdm for the progress bar

# Function to read all CSV files from a given folder
def read_csv_folder(folder_path):
    # Use glob to get all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    # List to store DataFrames
    dataframes = []
    
    # Iterate through the CSV files and read them
    for file in csv_files:
        try:
            df = pd.read_csv(file, sep=';', on_bad_lines='skip')
            dataframes.append(df)
            print(f"Loaded: {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Concatenate all DataFrames
    return pd.concat(dataframes, ignore_index=True)

# Define the folder paths
indo_folder = r'C:\Users\tohru\Documents\programming\censorship\indo'

# Read the CSV files from the folder
indo_data = read_csv_folder(indo_folder)

data = indo_data

# Initialize the translator
translator = Translator()

# Function to translate text
def translate_text(text, src='id', dest='en'):
    try:
        translation = translator.translate(text, src=src, dest=dest)
        return translation.text
    except Exception as e:
        print(f"Error translating '{text}': {e}")
        return text  # Return the original if translation fails

# Apply translation to the 'text' column with a progress bar
tqdm.pandas()  # Enable tqdm for pandas

# Show progress bar while applying translation
data['translated_text'] = data['text'].progress_apply(lambda x: translate_text(x))

# Save the translated dataset to a new CSV file
output_file = 'translated_dataset.csv'  # Define the output file
data[['translated_text', 'label']].to_csv(output_file, sep=';', index=False)

print(f"Dataset has been translated and saved as '{output_file}'.")
