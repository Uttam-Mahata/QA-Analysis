import os
import numpy as np
from nltk.tokenize import sent_tokenize
from gensim.models import KeyedVectors
import spacy

# Load spaCy English language model
nlp = spacy.load("en_core_web_sm")

def select_named_entity_pairs(named_entities):
    # Logic to select pairs of named entities based on entity type and non-identity
    selected_pairs = [(ent1, ent2) for ent1 in named_entities for ent2 in named_entities if ent1[1] == ent2[1] and ent1 != ent2]

    return selected_pairs

def vectorize_with_glove(selected_pairs, glove_model):
    # Vectorize each selected pair and take the average for the entire answer
    answer_vector = np.zeros(glove_model.vector_size)
    count = 0

    for pair in selected_pairs:
        for word in pair:
            if word in glove_model:
                word_vector = glove_model[word]
                answer_vector += word_vector
                count += 1

    if count != 0:
        answer_vector /= count

    return answer_vector

def process_student_answers():
    # Set the path to the raw student answers
    raw_data_path = os.path.join(os.getcwd(), 'data', 'raw_data', 'answers')

    # Load pre-trained GloVe model using gensim
    glove_model_path = 'glove.6B/glove.6B.50d.txt'  # Replace with the actual path
    glove_model = KeyedVectors.load_word2vec_format(glove_model_path, binary=False)

    # Loop through each question and each student's answer
    for i in range(1, 11):
        question_answers_path = os.path.join(raw_data_path, f'question{i}')

        for j in range(1, 16):  # Adjusted the range to 15 students
            # Read named entities from file
            named_entities_file_path = os.path.join(question_answers_path, f'student{j}_named_entities.txt')
            with open(named_entities_file_path, 'r', encoding='utf-8') as named_entities_file:
                named_entities_str = named_entities_file.read()
                named_entities = eval(named_entities_str)

            # Select pairs of named entities based on logic
            selected_pairs = select_named_entity_pairs(named_entities)

            # Read processed answer from file
            processed_answer_file_path = os.path.join(question_answers_path, f'student{j}_processed_answer.txt')
            with open(processed_answer_file_path, 'r', encoding='utf-8') as processed_answer_file:
                processed_answer = processed_answer_file.read()

            # Tokenize the answer into sentences
            sentences = sent_tokenize(processed_answer)

            # Vectorize each sentence and take the average for the entire answer
            answer_vector = np.zeros(glove_model.vector_size)
            for sentence in sentences:
                selected_pairs_in_sentence = [pair for pair in selected_pairs if any(word in sentence for word in pair)]
                sentence_vector = vectorize_with_glove(selected_pairs_in_sentence, glove_model)
                answer_vector += sentence_vector

            answer_vector /= len(sentences)  # Average over sentences

            # Save the answer vector to a file
            answer_vector_file_path = os.path.join(question_answers_path, f'student{j}_answer_vector.npy')
            np.save(answer_vector_file_path, answer_vector)

    print('Named entity pair selection and vectorization using GloVe completed successfully.')

# Call the function to process student answers
process_student_answers()
