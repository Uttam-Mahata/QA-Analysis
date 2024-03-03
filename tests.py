import os
import flair
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# Initialize Flair NER tagger and POS tagger
flair_ner_tagger = flair.models.SequenceTagger.load('ner')
flair_pos_tagger = flair.models.SequenceTagger.load('pos')

def process_answer(answer, question_number, student_number, file_path_pos, file_path_ner):
    # Process each sentence in the answer
    for sentence_number, sentence in enumerate(sent_tokenize(answer), start=1):
        # Use Flair for NER tagging
        sentence_flair = flair.data.Sentence(sentence)
        flair_ner_tagger.predict(sentence_flair)
        named_entities = [(entity.text, entity.tag) for entity in sentence_flair.get_spans('ner')]
        
        # Use Flair for POS tagging
        sentence_flair = flair.data.Sentence(sentence)
        flair_pos_tagger.predict(sentence_flair)
        pos_tags = [(token.text, token.tag) for token in sentence_flair]
        
        # Write POS tags and named entities to files
        with open(file_path_pos, 'a', encoding='utf-8') as pos_file, \
             open(file_path_ner, 'a', encoding='utf-8') as ner_file:
            pos_file.write(f"S{sentence_number}: {', '.join([f'{pos[0]}: {pos[1]}' for pos in pos_tags])}\n")
            ner_file.write(f"S{sentence_number}: {', '.join([f'{ne[0]}: {ne[1]}' for ne in named_entities])}\n")
            
        # Print POS tags and named entities to the terminal
        # print(f"Question {question_number}, Student {student_number}, Sentence {sentence_number} - POS Tags: {pos_tags}")
        # print(f"Question {question_number}, Student {student_number}, Sentence {sentence_number} - Named Entities: {named_entities}")

def process_student_answers():
    # Loop through each question and each student's answer
    for question_number in range(1, 11):
        for student_number in range(1, 11):  # Adjusted the range to 15 students
            # Create directories to store POS tags and named entities
            pos_folder = f"data/processed_data/pos_tags/question{question_number}"
            os.makedirs(pos_folder, exist_ok=True)

            ner_folder = f"data/processed_data/named_entities/question{question_number}"
            os.makedirs(ner_folder, exist_ok=True)

            # Define file paths for POS tags and named entities
            file_path_pos = os.path.join(pos_folder, f"student{student_number}_pos_tags.txt")
            file_path_ner = os.path.join(ner_folder, f"student{student_number}_named_entities.txt")

            # Read the processed answer from the file
            processed_answer_file_path = f"data/processed_data/processed_answers/question{question_number}/student{student_number}_processed_answer.txt"
            with open(processed_answer_file_path, 'r', encoding='utf-8') as processed_answer_file:
                processed_answer = processed_answer_file.read()

            # Process the answer
            process_answer(processed_answer, question_number, student_number, file_path_pos, file_path_ner)

    print('POS tagging and named entity recognition using Flair completed successfully.')

# Call the function to process student answers
process_student_answers()
