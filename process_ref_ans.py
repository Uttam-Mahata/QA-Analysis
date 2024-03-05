import os
from nltk.tokenize import sent_tokenize

import nltk
nltk.download('punkt')

def process_reference_answers():
    # Loop through each question
    for question_number in range(1, 11):
        # Create directories to store processed answers
        processed_folder = f"data/processed_ref_answers"
        os.makedirs(processed_folder, exist_ok=True)

        # Create a file to store processed reference answer
        processed_reference_answer_file_path = os.path.join(processed_folder, f'question{question_number}_processed_ref_answer.txt')

        with open(processed_reference_answer_file_path, 'w', encoding='utf-8') as processed_reference_answer_file:
            # Read the raw reference answer from the file
            reference_answer_file_path = f"data/raw_data/reference_answers/question{question_number}_reference.txt"
            with open(reference_answer_file_path, 'r', encoding='utf-8') as reference_answer_file:
                raw_reference_answer = reference_answer_file.read()

            # Tokenize the answer into sentences after full stops and start a new line
            sentences = sent_tokenize(raw_reference_answer)
            processed_reference_answer_file.write('\n'.join(sentences))

    print('Processing and saving reference answers completed successfully.')

# Call the function to process reference answers
process_reference_answers()
