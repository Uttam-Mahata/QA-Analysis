import os
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from itertools import combinations

nltk.download('punkt')

# Load spaCy English language model
nlp = spacy.load("en_core_web_sm")

def pos_tagging(sentence, question_number, student_number, sentence_number, pos_file):
    # Process the sentence using spaCy
    doc = nlp(sentence)

    # Extract POS tags
    pos_tags = [(token.text, token.pos_) for token in doc]

    # Save POS tags to the provided file for each sentence
    pos_file.write(f"S{sentence_number}: {', '.join([f'{pos[0]}: {pos[1]}' for pos in pos_tags])}\n")

    # Print POS tags to the terminal
    print(f"Question {question_number}, Student {student_number}, Sentence {sentence_number} - POS Tags: {pos_tags}")

def named_entity_extraction(sentence, question_number, student_number, sentence_number, ner_folder):
    # Process the sentence using spaCy
    doc = nlp(sentence)

    # Extract named entities
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Save named entities to the provided file for each sentence
    ner_file_path = os.path.join(ner_folder, f"student{student_number}_named_entities.txt")
    with open(ner_file_path, 'a', encoding='utf-8') as ner_file:
        ner_file.write(f"S{sentence_number}: {', '.join([f'{ne[0]}: {ne[1]}' for ne in named_entities])}\n")

    # Extract pairs of named entities
    entity_pairs = list(combinations([ne[0] for ne in named_entities], 2))

    # Save entity pairs to the provided file for each sentence
    entity_pairs_file_path = os.path.join(ner_folder, f"student{student_number}_entity_pairs.txt")
    with open(entity_pairs_file_path, 'a', encoding='utf-8') as entity_pairs_file:
        entity_pairs_file.write(f"S{sentence_number} Entity Pairs: {', '.join([f'({pair[0]}, {pair[1]})' for pair in entity_pairs])}\n")

    # Print entity pairs to the terminal
    print(f"Question {question_number}, Student {student_number}, Sentence {sentence_number} - Entity Pairs: {entity_pairs}")

def process_student_answers():
    # Loop through each question and each student's answer
    for question_number in range(1, 3):
        for student_number in range(1, 3):  # Adjusted the range to 15 students
            # Create directories to store processed answers, POS tags, named entities, and entity pairs
            processed_folder = f"data/processed_data/processed_answers/question{question_number}"
            os.makedirs(processed_folder, exist_ok=True)

            pos_folder = f"data/processed_data/pos_tags/question{question_number}"
            os.makedirs(pos_folder, exist_ok=True)

            ner_folder = f"data/processed_data/named_entities/question{question_number}"
            os.makedirs(ner_folder, exist_ok=True)

            # Create files to store POS tags, named entities, and entity pairs for each student's answer
            pos_file_path = os.path.join(pos_folder, f"student{student_number}_pos_tags.txt")
            ner_file_path = os.path.join(ner_folder, f"student{student_number}_named_entities.txt")

            with open(pos_file_path, 'w', encoding='utf-8') as pos_file, \
                 open(ner_file_path, 'w', encoding='utf-8') as ner_file:
                # Read the raw answer from the file
                answer_file_path = f"data/raw_data/answers/question{question_number}/student{student_number}_answer.txt"
                with open(answer_file_path, 'r', encoding='utf-8') as answer_file:
                    raw_answer = answer_file.read()

                # Save the processed answer to a file
                processed_answer_file_path = os.path.join(processed_folder, f'student{student_number}_processed_answer.txt')
                with open(processed_answer_file_path, 'w', encoding='utf-8') as processed_answer_file:
                    # Tokenize the answer into sentences after full stops and start a new line
                    sentences = sent_tokenize(raw_answer)
                    processed_answer_file.write('\n'.join(sentences))

                    # Process each sentence separately
                    for sentence_number, sentence in enumerate(sentences, start=1):
                        # Perform POS tagging, named entity recognition, and extract entity pairs, then save to files
                        pos_tagging(sentence, question_number, student_number, sentence_number, pos_file)
                        named_entity_extraction(sentence, question_number, student_number, sentence_number, ner_folder)

# Call the function to process student answers
process_student_answers()
