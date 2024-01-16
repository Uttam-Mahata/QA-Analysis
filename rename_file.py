import os

# Set the path to the parent directory containing the question folders
parent_directory = "Sakshi_Dataset"

# Iterate through the folders (questions) in the parent directory
for question_folder in os.listdir(parent_directory):
    question_folder_path = os.path.join(parent_directory, question_folder)

    # Check if the item in the directory is a folder
    if os.path.isdir(question_folder_path):
        # Set the starting count for each question
        start_count = 31

        # Lines to remove for each question
        lines_to_remove = []

        # Populate lines_to_remove based on the question
        if question_folder == "question1":
            lines_to_remove = ["Explain the concept of continental drift and provide evidence supporting this theory.", "Ans-"]
        elif question_folder == "question2":
            lines_to_remove = ["Elaborate on the concept of natural selection and its role in the theory of evolution. Provide examples of how natural selection can lead to the adaptation of species over time.", "Ans-"]
        elif question_folder == "question3":
            lines_to_remove = ["Explain the concept of social stratification and its impact on societies.", "Ans-"]
        elif question_folder == "question4":
            lines_to_remove = ["Discuss the impact of the French Revolution on European societies and global politics.", "Ans-"]
        elif question_folder == "question5":
            lines_to_remove = ["Discuss the significance of data science in extracting valuable insights from large datasets. Provide an example of a real-world application of data science.", "Ans-"]
        elif question_folder == "question6":
            lines_to_remove = ["What are the fundamental principles of machine learning, and how do they contribute to the development of intelligent systems?", "Ans-"]
        elif question_folder == "question7":
            lines_to_remove = ["What are the main components of a relational database management system (RDBMS), and how do they facilitate efficient data organization and retrieval?", "Ans-"]
        elif question_folder == "question8":
            lines_to_remove = ["How industrial revolution has impacted the modern world ?", "Ans-"]
        elif question_folder == "question9":
            lines_to_remove = ["What possible factors/tensions in the modern world may lead to World War 3?", "Ans-"]
        elif question_folder == "question10":
            lines_to_remove = ["Electric vehicles are said to be greener, but electricity production itself causes a lot of pollution. What do you think about this?", "Ans-"]

        # Iterate through the files in the question folder
        for count, filename in enumerate(os.listdir(question_folder_path), start=start_count):
            # Check if the file is a valid file (not a directory)
            if os.path.isfile(os.path.join(question_folder_path, filename)):
                # Generate the new filename based on the count and add .txt extension
                new_filename = f"student{count}_answer.txt"

                # Construct the full paths for the old and new filenames
                old_filepath = os.path.join(question_folder_path, filename)
                new_filepath = os.path.join(question_folder_path, new_filename)

                # Read the content of the file with UTF-8 encoding
                with open(old_filepath, 'r', encoding='utf-8') as file:
                    lines = file.readlines()

                # Remove specified lines from the content
                updated_lines = [line for line in lines if all(keyword not in line for keyword in lines_to_remove)]

                # Write the updated content back to the file with .txt extension
                with open(new_filepath, 'w', encoding='utf-8') as file:
                    file.writelines(updated_lines)

print("Lines have been removed, and files renamed with .txt extension successfully.")
