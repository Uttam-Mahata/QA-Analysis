import os

# Set the parent directory containing the question folders
parent_directory = "Deep_Dataset"

# Iterate through the folders (questions) in the parent directory
for question_folder in os.listdir(parent_directory):
    question_folder_path = os.path.join(parent_directory, question_folder)
    
    # Check if the item in the directory is a folder
    if os.path.isdir(question_folder_path):
        # Set the starting count for each question
        start_count = 51
        
        # Iterate through the files in the question folder
        for count, filename in enumerate(os.listdir(question_folder_path), start=start_count):
            # Check if the file is a .txt file
            if filename.endswith(".txt"):
                # Generate the new filename based on the count
                new_filename = f"student{count}_answer.txt"
                
                # Construct the full paths for the old and new filenames
                old_filepath = os.path.join(question_folder_path, filename)
                new_filepath = os.path.join(question_folder_path, new_filename)
                
                # Rename the file
                os.rename(old_filepath, new_filepath)

print("Files have been renamed successfully.")
