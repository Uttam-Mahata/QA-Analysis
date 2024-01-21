import os
import pandas as pd
import spacy
import numpy as np
from nltk.tokenize import sent_tokenize

# Load spaCy English language model with word embeddings
nlp = spacy.load("en_core_web_sm")

def get_sentence_vector(sentence):
    # Get the sentence vector using spaCy
    return nlp(sentence).vector

def prepare_dataset():
    # Create an empty DataFrame with columns for vectors
    df_columns = [f'v_w{i+1}' for i in range(96)]
    df = pd.DataFrame(columns=['Question', 'Student'] + df_columns)

    # Loop through each question and each student's answer
    for question_number in range(1, 11):
        for student_number in range(1, 65):  # Adjusted the range to 15 students
            # Read the average vector from the file
            avg_vector_file_path = f"data/processed_data/avg_vectors/question{question_number}/student{student_number}_avg_vector.txt"
            with open(avg_vector_file_path, 'r', encoding='utf-8') as avg_vector_file:
                avg_vector_line = avg_vector_file.readline().strip().split(': ')[1]

            # Convert the string representation of the vector to a list of floats
            avg_vector = [float(v.split('=')[1]) for v in avg_vector_line.split(', ')]

            # Add the question and student number to the vector list
            row = [question_number, student_number] + avg_vector

            # Concatenate the row to the DataFrame
            df = pd.concat([df, pd.DataFrame([row], columns=df.columns)], ignore_index=True)

    # Save the DataFrame to a CSV file
    df.to_csv('student_vectors_dataset.csv', index=False)

    print('Dataset preparation completed successfully.')

# Call the function to prepare the dataset
prepare_dataset()
