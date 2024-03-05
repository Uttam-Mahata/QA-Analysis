import os
import numpy as np
import pandas as pd

def load_avg_vectors(question_number, student_number):
    avg_vector_file_path = f"data/Avg_Vectors/question{question_number}/student{student_number}_avg_vector.txt"

    try:
        with open(avg_vector_file_path, 'r', encoding='utf-8') as avg_vector_file:
            avg_vector_line = avg_vector_file.readline()
            avg_vector_str = avg_vector_line.split(": ")[1].strip()
            avg_vector = [float(val.split("=")[1]) for val in avg_vector_str.split(", ")]
            return avg_vector
    except FileNotFoundError:
        print(f"Average vector file not found for Question {question_number}, Student {student_number}.")
        return None

def create_dataset_for_question(question_number):
    data = []

    # Loop through each student's answer for the given question
    for student_number in range(1, 65):
        # Load the existing average vector for the student
        avg_vector = load_avg_vectors(question_number, student_number)

        if avg_vector:
            # Append data to the dataset
            data.append({
                'Question': question_number,
                'Student': student_number,
                **{f'v{i+1}': avg_vector[i] for i in range(len(avg_vector))}
            })

    print(f'Dataset creation completed successfully for Question {question_number}.')

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv(f'complete_dataset_question_{question_number}.csv', index=False)
    print(f'Dataset saved to complete_dataset_question_{question_number}.csv')

# Call the function to create a dataset for each question using existing average vectors
for question_number in range(1, 11):
    create_dataset_for_question(question_number)
