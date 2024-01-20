import os
import spacy
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK resources
nltk.download('punkt')

def pos_tagging(answer):
    # Load spaCy English language model
    nlp = spacy.load("en_core_web_sm")

    # Process the answer using spaCy
    doc = nlp(answer)

    # Extract POS tags
    pos_tags = [(token.text, token.pos_) for token in doc]

    return pos_tags

def named_entity_extraction(answer):
    # Load spaCy English language model
    nlp = spacy.load("en_core_web_sm")

    # Process the answer using spaCy
    doc = nlp(answer)

    # Extract named entities
    named_entities = [(ent.text, ent.label_) for ent in doc.ents]

    return named_entities

def process_student_answers():
    # Set the path to the raw student answers
    raw_data_path = os.path.join(os.getcwd(), 'data', 'raw_data', 'answers')

    # Loop through each question and each student's answer
    for i in range(1, 11):
        question_answers_path = os.path.join(raw_data_path, f'question{i}')

        # Create folders for processed answers, POS tags, and named entities
        processed_folder = os.path.join(question_answers_path, 'processed_answers')
        pos_tags_folder = os.path.join(question_answers_path, 'pos_tags')
        named_entities_folder = os.path.join(question_answers_path, 'named_entities')

        os.makedirs(processed_folder, exist_ok=True)
        os.makedirs(pos_tags_folder, exist_ok=True)
        os.makedirs(named_entities_folder, exist_ok=True)

        for j in range(1, 65):  # Adjusted the range to 15 students
            answer_file_path = os.path.join(question_answers_path, f'student{j}_answer.txt')

            # Read the raw answer from the file
            with open(answer_file_path, 'r', encoding='utf-8') as answer_file:
                raw_answer = answer_file.read()

            # Starting a new line after each full stop
            processed_answer = '\n'.join(sent_tokenize(raw_answer))

            # Save the processed answer to a file
            processed_answer_file_path = os.path.join(processed_folder, f'student{j}_processed_answer.txt')
            with open(processed_answer_file_path, 'w', encoding='utf-8') as processed_answer_file:
                processed_answer_file.write(processed_answer)

            # Perform POS tagging
            pos_tags = pos_tagging(processed_answer)

            # # Save POS tags to a file
            # pos_tags_file_path = os.path.join(pos_tags_folder, f'student{j}_pos_tags.txt')
            # with open(pos_tags_file_path, 'w', encoding='utf-8') as pos_tags_file:
            #     pos_tags_file.write(str(pos_tags))

            # Perform named entity extraction
            named_entities = named_entity_extraction(processed_answer)

            # # Save named entities to a file
            # named_entities_file_path = os.path.join(named_entities_folder, f'student{j}_named_entities.txt')
            # with open(named_entities_file_path, 'w', encoding='utf-8') as named_entities_file:
            #     named_entities_file.write(str(named_entities))

    print('POS tagging and named entity extraction using spaCy completed successfully.')

# Call the function to process student answers
process_student_answers()
