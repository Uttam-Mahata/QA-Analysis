import os

def create_directory_structure():
    # Get the current working directory
    project_root = os.getcwd()

    # Create necessary directories
    data_path = os.path.join(project_root, 'data')
    raw_data_path = os.path.join(data_path, 'raw_data')
    processed_data_path = os.path.join(data_path, 'processed_data')
    reference_path = os.path.join(raw_data_path, 'reference_answers')
    results_path = os.path.join(project_root, 'results')
    clusters_path = os.path.join(results_path, 'clusters')
    evaluation_scores_path = os.path.join(results_path, 'evaluation_scores')
    models_path = os.path.join(project_root, 'models')
    code_path = os.path.join(project_root, 'code')

    directories = [data_path, raw_data_path, processed_data_path, reference_path, results_path, clusters_path, evaluation_scores_path, models_path, code_path]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    # Create questions directory and text files
    questions_path = os.path.join(raw_data_path, 'questions')
    os.makedirs(questions_path, exist_ok=True)

    # List of questions
    questions = [
        "Explain the concept of continental drift and provide evidence supporting this theory.",
        "Elaborate on the concept of natural selection and its role in the theory of evolution. Provide examples of how natural selection can lead to the adaptation of species over time.",
        "Explain the concept of social stratification and its impact on societies.",
        "Discuss the impact of the French Revolution on European societies and global politics.",
        "Discuss the significance of data science in extracting valuable insights from large datasets. Provide an example of a real-world application of data science.",
        "What are the fundamental principles of machine learning, and how do they contribute to the development of intelligent systems?",
        "What are the main components of a relational database management system (RDBMS), and how do they facilitate efficient data organization and retrieval?",
        "How has the industrial revolution impacted the modern world?",
        "What possible factors/tensions in the modern world may lead to World War 3?",
        "Electric vehicles are said to be greener, but electricity production itself causes a lot of pollution. What do you think about this?"
    ]

    for i, question in enumerate(questions, start=1):
        question_file_path = os.path.join(questions_path, f'question{i}.txt')
        with open(question_file_path, 'w', encoding='utf-8') as question_file:
            question_file.write(question)

    # Create answers directory and text files
    answers_path = os.path.join(raw_data_path, 'answers')
    os.makedirs(answers_path, exist_ok=True)

    for i, question in enumerate(questions, start=1):
        question_answers_path = os.path.join(answers_path, f'question{i}')
        os.makedirs(question_answers_path, exist_ok=True)
        
        for j in range(1, 101):
            answer_file_path = os.path.join(question_answers_path, f'student{j}_answer.txt')
            with open(answer_file_path, 'w', encoding='utf-8') as answer_file:
                answer_file.write(f"Answer {j} to Question {i}")

    # Create reference answers and text files
    reference_answers_path = os.path.join(reference_path)
    os.makedirs(reference_answers_path, exist_ok=True)

    for i, question in enumerate(questions, start=1):
        reference_answer_file_path = os.path.join(reference_answers_path, f'question{i}_reference.txt')
        with open(reference_answer_file_path, 'w', encoding='utf-8') as reference_answer_file:
            reference_answer_file.write(f"Reference Answer to Question {i}")

    # Create subdirectories for processed_data
    processed_data_subdirectories = ['pos_tagged', 'vectors/word2vec', 'vectors/glove', 'vectors/bert']
    for subdir in processed_data_subdirectories:
        subdir_path = os.path.join(processed_data_path, subdir)
        os.makedirs(subdir_path, exist_ok=True)

    # Create evaluation_data text files
    evaluation_data_path = os.path.join(data_path, 'evaluation_data')
    os.makedirs(evaluation_data_path, exist_ok=True)

    for i, question in enumerate(questions, start=1):
        evaluation_file_path = os.path.join(evaluation_data_path, f'question{i}_evaluation.txt')
        with open(evaluation_file_path, 'w', encoding='utf-8') as evaluation_file:
            evaluation_file.write(f"Evaluation data for Question {i}")

    print('Directory structure created successfully in the current working directory.')

# Call the function to create the directory structure in the current working directory
create_directory_structure()
