import os
import numpy as np
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer

# Load Sentence Transformer model
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_sentence_vector(sentence):
    # Use Sentence Transformer for sentence embedding
    sentence_embedding = sbert_model.encode(sentence, convert_to_tensor=True).tolist()
    return sentence_embedding

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
                    # Use Sentence Transformer for sentence vector
                    sentence_vector = get_sentence_vector(sentence)
                    sentence_vectors.append(sentence_vector)

                if sentence_vectors:
                    # Calculate the average vector for all sentences
                    avg_vector = np.mean(sentence_vectors, axis=0)

                    # Save the average vector to the provided file for each student
                    avg_vector_file.write(f"Student{student_number}_Avg_Vector: {', '.join([f'v_w{i+1}={v}' for i, v in enumerate(avg_vector)])}\n")

                    # Print the average vector to the terminal
                    print(f"Question {question_number}, Student {student_number} - Average Vector: {avg_vector}")
                else:
                    avg_vector_file.write("No sentences to calculate the average vector.\n")

    print('Average vector calculation completed successfully.')

# Call the function to process student answers and calculate average vectors
process_student_answers()
