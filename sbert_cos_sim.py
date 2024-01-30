import os
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load Sentence Transformer model
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_sentence_vector(sentence):
    # Use Sentence Transformer for sentence embedding
    sentence_embedding = sbert_model.encode(sentence, convert_to_tensor=True).tolist()
    return sentence_embedding

def calculate_cosine_similarity(vectors1, vectors2):
    # Calculate cosine similarity between two sets of vectors
    return cosine_similarity(vectors1, vectors2)

def process_student_answers():
    # Lists to store vectors for each question
    question_vectors = []

    # Loop through each question and each student's answer
    for question_number in range(1, 3):
        question_vectors.append([])  # Initialize vectors list for the current question
        for student_number in range(1, 3):  # Adjusted the range to 15 students
            # Read the processed answer from the file
            processed_answer_file_path = f"data/processed_data/processed_answers/question{question_number}/student{student_number}_processed_answer.txt"
            with open(processed_answer_file_path, 'r', encoding='utf-8') as processed_answer_file:
                processed_answers = processed_answer_file.read().split('\n')

            # Vectorize each sentence and convert to numeric values
            sentence_vectors = []
            for sentence in processed_answers:
                # Use Sentence Transformer for sentence vector
                sentence_vector = get_sentence_vector(sentence)
                sentence_vectors.append(sentence_vector)

            if sentence_vectors:
                # Calculate the average vector for all sentences
                avg_vector = np.mean(sentence_vectors, axis=0)
                question_vectors[question_number - 1].append(avg_vector)

    # Calculate cosine similarity between different questions
    for q1 in range(1, 3):
        for q2 in range(1, 3):
            if q1 != q2:
                similarity_matrix = calculate_cosine_similarity(question_vectors[q1-1], question_vectors[q2-1])
                average_similarity = np.mean(similarity_matrix)
                print(f"Cosine Similarity between Question {q1} and Question {q2}: {average_similarity}")

    print('Cosine similarity calculation completed successfully.')

# Call the function to process student answers, calculate average vectors, and find cosine similarity
process_student_answers()
