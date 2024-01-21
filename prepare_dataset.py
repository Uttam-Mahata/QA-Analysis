import os
import numpy as np
import pandas as pd

def prepare_dataset():
    # Create an empty DataFrame to store the data
    columns = [f'v_w{i+1}' for i in range(300)]  # Assuming spaCy vectors are of size 300
    df = pd.DataFrame(columns=['Student'] + columns)

    # Loop through each student
    for student_number in range(1, 16):
        # Create a row for the student
        row = {'Student': f'Student{student_number}_Avg_Vector'}

        # Loop through each question and read the average vector
        for question_number in range(1, 11):
            avg_vector_file_path = f"data/processed_data/avg_vectors/question{question_number}/student{student_number}_avg_vector.txt"

            # Read the average vector from the file
            with open(avg_vector_file_path, 'r', encoding='utf-8') as avg_vector_file:
                avg_vector_str = avg_vector_file.read().split(': ')[1]

            # Convert the string representation to a numpy array
            avg_vector = np.array([float(value.split('=')[1]) for value in avg_vector_str.split(', ')])

            # Add the average vector values to the row
            row.update(zip(columns, avg_vector))

        # Append the row to the DataFrame
        df = df.append(row, ignore_index=True)

    # Save the DataFrame to a CSV file
    csv_file_path = "data/processed_data/average_vectors_dataset.csv"
    df.to_csv(csv_file_path, index=False)

    print(f'Dataset saved to {csv_file_path}.')

# Call the function to prepare the dataset
prepare_dataset()
