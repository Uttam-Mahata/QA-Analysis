# Automated Student Answer Evaluation System

## Workflow Steps

### 1. Dataset Creation
- The dataset consists of 10 questions, each answered by 100 students.
- Preprocessing is applied to the dataset, including part-of-speech tagging for each sentence in the answers.

### 2. Named Entity Recognition
- Named entities are identified through the part-of-speech tagging process.

### 3. Related Entity Pairs Extraction
- Pairs of named entities are extracted from the identified named entities.

### 4. Vectorization using Word Embeddings
- Word embeddings such as Word2Vec, GloVe, and BERT are applied to generate vectors for each sentence.
- The focus is on creating vectors for individual sentences in an answer, considering pairs of named entities.

### 5. Sentence Vectorization
- Average vectors are calculated for each sentence, representing the overall content of the sentence.

### 6. Aggregation for Each Question
- The process of sentence vectorization is repeated for all 100 answers corresponding to a specific question.
- The final dataset is formed by calculating the average vector for each set of 100 answers. This average vector represents the overall understanding of each question by the students.

## Grade Assignment

After creating the dataset, the system aims to assign grades to all students based on their performance in answering the given questions. This involves cluster creation to analyze and categorize student responses.

## Automation

The entire process is automated to facilitate efficient and consistent evaluation of student answers.

