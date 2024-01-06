import os

def create_project_directory_structure(num_subjects, num_students, num_questions):
    # Create data directory
    data_dir = os.path.join('data')
    os.makedirs(data_dir, exist_ok=True)

    # Create processed_data directory
    processed_data_dir = os.path.join('processed_data')
    os.makedirs(processed_data_dir, exist_ok=True)

    # Create embeddings directory
    embeddings_dir = os.path.join('embeddings')
    os.makedirs(embeddings_dir, exist_ok=True)

    # Create dataset directory
    dataset_dir = os.path.join('dataset')
    os.makedirs(dataset_dir, exist_ok=True)

    # Create clusters directory
    clusters_dir = os.path.join('clusters')
    os.makedirs(clusters_dir, exist_ok=True)

    # Create results directory
    results_dir = os.path.join('results')
    os.makedirs(results_dir, exist_ok=True)

    # Create reference folder
    reference_dir = os.path.join('reference')
    os.makedirs(reference_dir, exist_ok=True)

    # Create subject-wise directories and files
    for subject_num in range(1, num_subjects + 1):
        subject_dir = os.path.join(data_dir, f'Subject_{subject_num}')
        os.makedirs(subject_dir, exist_ok=True)

        # Create reference file for each question in the subject folder
        for question_id in range(1, num_questions + 1):
            reference_file = os.path.join(subject_dir, f'reference_question_{question_id}.txt')
            # Placeholder content for reference files
            with open(reference_file, 'w') as reference_file:
                reference_file.write(f"Reference answer for Subject {subject_num}, Question {question_id}")

        # Add specific Geography questions and answers for Subject_1
        if subject_num == 5:
            history_questions = [
                "Explain the causes and consequences of the Industrial Revolution in the 18th and 19th centuries.",
                "Discuss the impact of the French Revolution on European societies and global politics.",
                "Analyze the role of imperialism in shaping the development of colonies during the 19th and 20th centuries.",
                "Examine the factors that led to the outbreak of World War I. How did the war reshape global geopolitics?",
                "Discuss the events and consequences of the Great Depression in the 1930s.",
                "Explore the causes and outcomes of the Cold War between the United States and the Soviet Union.",
                "Analyze the social and political changes brought about by the Civil Rights Movement in the United States.",
                "Discuss the causes, events, and aftermath of the Cuban Missile Crisis during the Cold War.",
                "Examine the impact of decolonization on African and Asian nations in the mid-20th century.",
                "Discuss the factors that contributed to the fall of the Berlin Wall and the reunification of Germany.",
            ]
            for student_id in range(1, num_students + 1):
                student_dir = os.path.join(subject_dir, f'student_{student_id}')
                os.makedirs(student_dir, exist_ok=True)

                for question_id, geography_question in enumerate(history_questions, start=1):
                    question_filename = f'question_{question_id}.txt'
                    question_path = os.path.join(student_dir, question_filename)

                    # Write Geography question and answer content to each file
                    with open(question_path, 'w') as question_file:
                        question_file.write(f"History Question {question_id}\n{geography_question}\nAnswer for Question {question_id} by Student {student_id}")

    # Create processed_data subdirectories
    processed_data_subdirs = ['pos_tagged', 'named_entities']
    for subdir in processed_data_subdirs:
        subdir_path = os.path.join(processed_data_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)

        for student_id in range(1, num_students + 1):
            student_subdir = os.path.join(subdir_path, f'student_{student_id}')
            os.makedirs(student_subdir, exist_ok=True)

            for question_id in range(1, num_questions + 1):
                question_filename = f'question_{question_id}.txt'
                question_path = os.path.join(student_subdir, question_filename)

                # Placeholder content for processed_data files
                with open(question_path, 'w') as processed_data_file:
                    processed_data_file.write(f"Processed data for Question {question_id} by Student {student_id}")

    # Create embeddings subdirectories
    embeddings_subdirs = ['word2vec', 'glove', 'bert']
    for subdir in embeddings_subdirs:
        subdir_path = os.path.join(embeddings_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)

        for student_id in range(1, num_students + 1):
            embedding_file = os.path.join(subdir_path, f'student_{student_id}.npy')
            # Placeholder content for embeddings files
            with open(embedding_file, 'w') as embedding_file:
                embedding_file.write(f"Embeddings for Student {student_id}")

    # Create dataset files
    for student_id in range(1, num_students + 1):
        dataset_file = os.path.join(dataset_dir, f'student_{student_id}.npy')
        # Placeholder content for dataset files
        with open(dataset_file, 'w') as dataset_file:
            dataset_file.write(f"Dataset for Student {student_id}")

    # Create clusters subdirectories
    clusters_subdirs = [f'cluster_{i}' for i in range(1, num_subjects + 1)]
    for subdir in clusters_subdirs:
        subdir_path = os.path.join(clusters_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)

        for student_id in range(1, num_students + 1):
            cluster_file = os.path.join(subdir_path, f'student_{student_id}.txt')
            # Placeholder content for clusters files
            with open(cluster_file, 'w') as cluster_file:
                cluster_file.write(f"Clusters for Student {student_id}")

    # Create results file
    results_file = os.path.join(results_dir, 'marks.csv')
    # Placeholder content for results file
    with open(results_file, 'w') as results_file:
        results_file.write("StudentID, Marks\n")

if __name__ == "__main__":
    num_subjects = 5
    num_students = 10
    num_questions = 10
    create_project_directory_structure(num_subjects, num_students, num_questions)
