import os
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np

# Load Sentence Transformer model
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_sentence_vector(sentence):
    # Use Sentence Transformer for sentence embedding
    sentence_embedding = sbert_model.encode(sentence, convert_to_tensor=True)
    sentence_embedding = np.array(sentence_embedding).tolist()
    return sentence_embedding

def load_reference_vectors(question_number):
    processed_reference_answer_file_path = f"data/processed_ref_answers/question{question_number}_processed_ref_answer.txt"
    reference_vectors = []

    try:
        with open(processed_reference_answer_file_path, 'r', encoding='utf-8') as processed_reference_answer_file:
            processed_reference_answers = processed_reference_answer_file.read().split('\n')

        for sentence_number, sentence in enumerate(processed_reference_answers, start=1):
            reference_vector = get_sentence_vector(sentence)
            reference_vectors.append(reference_vector)

    except FileNotFoundError:
        print(f"Processed reference answer file not found for Question {question_number}.")
        return None

    if reference_vectors:
        avg_vector = np.mean(reference_vectors, axis=0)
        return avg_vector
    else:
        print(f"No sentences to calculate the average vector for Question {question_number}.")
        return None

def process_reference_answers():
    # Create an empty DataFrame to store vectors for each question
    reference_vectors_df = pd.DataFrame()

    # Loop through each question
    for question_number in range(1, 11):
        # Create a list to store vectors for each reference answer
        reference_vectors = []

        # Read the processed reference answer from the file
        processed_reference_answer_file_path = f"data/processed_ref_answers/question{question_number}_processed_ref_answer.txt"
        with open(processed_reference_answer_file_path, 'r', encoding='utf-8') as processed_reference_answer_file:
            processed_reference_answers = processed_reference_answer_file.read().split('\n')

        # Vectorize each sentence and convert to numeric values
        for sentence_number, sentence in enumerate(processed_reference_answers, start=1):
            # Use Sentence Transformer for sentence vector
            sentence_vector = get_sentence_vector(sentence)
            reference_vectors.append(sentence_vector)

        if reference_vectors:
            # Calculate the average vector for all sentences
            avg_vector = np.mean(reference_vectors, axis=0)

            # Create a DataFrame to store the current question's data
            question_df = pd.DataFrame({
                'Question': [question_number],
                **{f'v{i+1}': avg_vector[i] for i in range(len(avg_vector))}
            })

            # Concatenate the current question's DataFrame with the main DataFrame
            reference_vectors_df = pd.concat([reference_vectors_df, question_df], ignore_index=True)

            # Print the average vector to the terminal
            print(f"Question {question_number} - Average Vector: {avg_vector}")
        else:
            print(f"Question {question_number} - No sentences to calculate the average vector.")

    # Save the DataFrame to a CSV file
    output_csv_path = 'reference_answer_vectors_woner.csv'
    reference_vectors_df.to_csv(output_csv_path, index=False)
    print(f'Reference answer vectors saved to {output_csv_path}.')

# Call the function to process reference answers and calculate average vectors
process_reference_answers()
