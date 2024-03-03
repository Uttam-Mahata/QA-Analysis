import os
import nltk
from nltk.tokenize import sent_tokenize


nltk.download('punkt')

def process_student_answers():
    # Loop through each question and each student's answer
    for question_number in range(1, 11):
        for student_number in range(1, 65):  # Adjusted the range to 15 students
            # Create directories to store processed answers
            processed_folder = f"data/processed_answers/question{question_number}"
            os.makedirs(processed_folder, exist_ok=True)

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

                

# Call the function to process student answers
process_student_answers()
