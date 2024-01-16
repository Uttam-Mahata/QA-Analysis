import os

# Set the path to the directory containing the files
directory_path = "Debosmita_Dataset/question4"

# Set the starting count
start_count = 16

# Iterate through the files in the directory
for count, filename in enumerate(os.listdir(directory_path), start=start_count):
    # Check if the file is a .txt file and starts with "answer"
    if filename.endswith(".txt"):
        # Generate the new filename based on the count
        new_filename = f"student{count}_answer.txt"
        
        # Construct the full paths for the old and new filenames
        old_filepath = os.path.join(directory_path, filename)
        new_filepath = os.path.join(directory_path, new_filename)
        
        # Rename the file
        os.rename(old_filepath, new_filepath)

print("Files have been renamed successfully.")
