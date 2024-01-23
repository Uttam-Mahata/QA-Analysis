import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

# Load pre-trained GloVe embeddings
glove_model = KeyedVectors.load_word2vec_format('glove.6B.300d.txt', binary=False, no_header=True)

def get_named_entity_pairs(sentence):
    # Tokenize the sentence into words
    words = sentence.split()
    
    # Extract named entities and form pairs
    named_entities = [word for word in words if word in glove_model]
    entity_pairs = [(named_entities[i], named_entities[j]) for i in range(len(named_entities)) for j in range(i+1, len(named_entities))]
    
    return entity_pairs

def get_sentence_vector(sentence, entity_pairs):
    # Tokenize the sentence into words
    words = sentence.split()
    
    # Get vectors for words within the range of entity pairs using GloVe embeddings
    vectors = [glove_model[word] for word in words if any(word in pair for pair in entity_pairs)]
    
    if vectors:
        # Calculate the average vector
        avg_vector = np.mean(vectors, axis=0)
        return avg_vector
    else:
        return np.zeros_like(glove_model['a'])  # Return a zero vector if no words are found

def calculate_cosine_similarity(vectors1, vectors2):
    # Calculate cosine similarity between two sets of vectors
    similarity_matrix = cosine_similarity([vectors1], [vectors2])
    return similarity_matrix[0, 0]

def process_student_answers():
    question_vectors = {}

    # Loop through each question and each student's answer
    for question_number in range(1, 3):
        question_vectors[question_number] = []

        for student_number in range(1, 2):  # Adjusted the range to 15 students
            # Read the processed answer from the file
            processed_answer_file_path = f"data/processed_data/processed_answers/question{question_number}/student{student_number}_processed_answer.txt"
            with open(processed_answer_file_path, 'r', encoding='utf-8') as processed_answer_file:
                processed_answers = processed_answer_file.read().split('\n')

            # Vectorize each sentence and convert to numeric values
            sentence_vectors = []
            for sentence in processed_answers:
                named_entity_pairs = get_named_entity_pairs(sentence)
                vectors = [get_sentence_vector(sentence, named_entity_pairs).tolist()]
                
                if vectors:
                    sentence_vectors.extend(vectors)

            if sentence_vectors:
                # Calculate the average vector for all sentences
                avg_vector = np.mean(sentence_vectors, axis=0)
                question_vectors[question_number].append(avg_vector)

    # Calculate cosine similarities between question vectors
    for q1 in range(1, 3):
        for q2 in range(1, 3):
            if q1 != q2:
                similarity = calculate_cosine_similarity(question_vectors[q1][0], question_vectors[q2][0])
                print(f"Cosine Similarity between Question {q1} and Question {q2}: {similarity}")

    print('Cosine similarity calculation completed successfully.')

# Call the function to calculate cosine similarities
process_student_answers()
