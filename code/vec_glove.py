import os
import numpy as np
from nltk.tokenize import sent_tokenize
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

def process_student_answers():
    # Loop through each question and each student's answer
    for question_number in range(1, 3):
        for student_number in range(1, 3):  # Adjusted the range to 15 students
            # Create directory to store average vectors for each question
            avg_vector_folder = f"data/processed_data/avg_vectors/question{question_number}"
            os.makedirs(avg_vector_folder, exist_ok=True)

            # Create file to store average vectors for each student's answer
            avg_vector_file_path = os.path.join(avg_vector_folder, f"student{student_number}_avg_vector.txt")

            with open(avg_vector_file_path, 'w', encoding='utf-8') as avg_vector_file:
                # Read the processed answer from the file
                processed_answer_file_path = f"data/processed_data/processed_answers/question{question_number}/student{student_number}_processed_answer.txt"
                with open(processed_answer_file_path, 'r', encoding='utf-8') as processed_answer_file:
                    processed_answers = processed_answer_file.read().split('\n')

                # Vectorize each sentence and convert to numeric values
                sentence_vectors = []
                for sentence_number, sentence in enumerate(processed_answers, start=1):
                    # Get named entity pairs
                    named_entity_pairs = get_named_entity_pairs(sentence)
                    
                    # Use named entity pairs to form vectors using GloVe embeddings
                    sentence_vector = get_sentence_vector(sentence, named_entity_pairs).tolist()
                    sentence_vectors.append(sentence_vector)
                    
                    print(f"Question {question_number}, Student {student_number}, Sentence {sentence_number} Vector: {sentence_vector}")

                if sentence_vectors:
                    # Calculate the average vector for all sentences
                    avg_vector = np.mean(sentence_vectors, axis=0)

                    # Save the average vector to the provided file for each student
                    avg_vector_file.write(f"Student{student_number}_Avg_Vector: {', '.join([f'v_w{i+1}={v}' for i, v in enumerate(avg_vector)])}\n")
                else:
                    avg_vector_file.write("No sentences to calculate the average vector.\n")

    print('Average vector calculation completed successfully.')

# Call the function to process student answers and calculate average vectors
process_student_answers()
