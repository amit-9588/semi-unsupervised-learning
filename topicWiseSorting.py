# ********* APPROACH 1 **************


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping
#
# # Sample dataset
# data = {
#     'question': [
#         'What is the capital of France?',
#         'How does a neural network work?',
#         'What is the Pythagorean theorem?',
#         'What is the capital of Germany?',
#         'How do you train a deep learning model?',
#         'What is the quadratic formula?',
#         'What is the capital of Spain?',
#         'What is backpropagation in neural networks?',
#         'What is the formula for the area of a circle?',
#         'What is a convolutional neural network?',
#         'What is the formula for the volume of a sphere?',
#         'How do you optimize a machine learning model?',
#         'What is the derivative of a function?',
#         'What is a decision tree in machine learning?',
#         'What is a matrix in mathematics?',
#         'What is the capital of Italy?',
#         'What is gradient descent?',
#         'What is the difference between supervised and unsupervised learning?',
#         'What is a neural network’s activation function?',
#         'What is the capital of Canada?',
#         'What is the difference between regression and classification?',
#         'What is the value of x in 2x + 3 = 5?',
#         'What is the integral of sin(x)?',
#         'What is the derivative of cos(x)?',
#         'What is the quadratic equation?',
#         'What is the area of a triangle?',
#         'What is the formula for the volume of a cone?',
#         'What is the Pythagorean theorem used for?',
#         'What is the difference between differentiation and integration?',
#         'What is the derivative of x^2?',
#         'What is the formula for the surface area of a sphere?',
#         'What is the integral of cos(x)?'
#     ],
#     'topic': [
#         'Geography', 'Computer Science', 'Mathematics', 'Geography', 'Computer Science', 'Mathematics',
#         'Geography', 'Computer Science', 'Mathematics', 'Computer Science', 'Mathematics', 'Computer Science',
#         'Mathematics', 'Computer Science', 'Mathematics', 'Geography', 'Computer Science', 'Computer Science',
#         'Computer Science', 'Geography', 'Computer Science',
#         'Mathematics', 'Mathematics', 'Mathematics', 'Mathematics', 'Mathematics', 'Mathematics',
#         'Mathematics', 'Mathematics', 'Mathematics', 'Mathematics', 'Mathematics'
#     ],
#     'sub_topic': [
#         'Capital Cities', 'Neural Networks', 'Geometry', 'Capital Cities', 'Deep Learning', 'Algebra',
#         'Capital Cities', 'Neural Networks', 'Geometry', 'Deep Learning', 'Geometry', 'Machine Learning',
#         'Calculus', 'Machine Learning', 'Matrices', 'Capital Cities', 'Optimization', 'Machine Learning',
#         'Neural Networks', 'Capital Cities', 'Machine Learning', 'Algebra', 'Integration', 'Differentiation',
#         'Algebra', 'Geometry', 'Geometry', 'Geometry', 'Calculus', 'Calculus', 'Geometry', 'Geometry'
#     ]
# }
#
# df = pd.DataFrame(data)
#
# # Preprocessing
# tfidf = TfidfVectorizer(stop_words='english')
# X = tfidf.fit_transform(df['question']).toarray()
#
# label_encoder_topic = LabelEncoder()
# y_topic = label_encoder_topic.fit_transform(df['topic'])
# y_topic = to_categorical(y_topic)
#
# label_encoder_sub_topic = LabelEncoder()
# y_sub_topic = label_encoder_sub_topic.fit_transform(df['sub_topic'])
# y_sub_topic = to_categorical(y_sub_topic)
#
# # Split data
# X_train, X_test, y_topic_train, y_topic_test, y_sub_topic_train, y_sub_topic_test = train_test_split(
#     X, y_topic, y_sub_topic, test_size=0.2, random_state=42)
#
# # Model for topic classification
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#
# # Model for topic classification
# model_topic = Sequential([
#     Dense(128, input_shape=(X.shape[1],), activation='relu'),
#     Dropout(0.5),
#     Dense(64, activation='relu'),
#     Dropout(0.5),
#     Dense(y_topic.shape[1], activation='softmax')  # Ensure the output layer has correct size for topics
# ])
#
# model_topic.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model_topic.fit(X_train, y_topic_train, epochs=500, batch_size=32, validation_data=(X_test, y_topic_test),
#                 callbacks=[early_stopping])
#
# # Model for sub-topic classification
# model_sub_topic = Sequential([
#     Dense(128, input_shape=(X.shape[1],), activation='relu'),
#     Dropout(0.5),
#     Dense(64, activation='relu'),
#     Dropout(0.5),
#     Dense(y_sub_topic.shape[1], activation='softmax')  # Ensure the output layer has correct size for sub-topics
# ])
#
# model_sub_topic.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model_sub_topic.fit(X_train, y_sub_topic_train, epochs=500, batch_size=32, validation_data=(X_test, y_sub_topic_test),
#                     callbacks=[early_stopping])
#
# # Predicting topics and sub-topics for new questions
# new_questions = ['What is the capital of Spain?', 'What is backpropagation?', 'What is Natural Language Processing?',
#                  'What is the value of x in 2x = 30 + x?', 'What is the value of sin^2(x) + cos^2(x) = 1?']
# new_X = tfidf.transform(new_questions).toarray()
#
# # Get predictions
# predicted_topics = model_topic.predict(new_X)
# predicted_sub_topics = model_sub_topic.predict(new_X)
#
# # Get the class labels with the highest probability
# predicted_topics_labels = label_encoder_topic.inverse_transform(np.argmax(predicted_topics, axis=1))
# predicted_sub_topics_labels = label_encoder_sub_topic.inverse_transform(np.argmax(predicted_sub_topics, axis=1))
#
# # Calculate the prediction percentages (confidence level)
# topic_confidences = np.max(predicted_topics, axis=1) * 100  # Maximum probability for topics
# sub_topic_confidences = np.max(predicted_sub_topics, axis=1) * 100  # Maximum probability for sub-topics
#
# # Set to "Unknown" if confidence is less than 10%
# threshold = 10
# predicted_topics_labels = [
#     "Unknown" if confidence < threshold else topic
#     for topic, confidence in zip(predicted_topics_labels, topic_confidences)
# ]
# predicted_sub_topics_labels = [
#     "Unknown" if confidence < threshold else sub_topic
#     for sub_topic, confidence in zip(predicted_sub_topics_labels, sub_topic_confidences)
# ]
#
# # Print results with matching percentages
# for question, topic, sub_topic, topic_confidence, sub_topic_confidence in zip(
#         new_questions, predicted_topics_labels, predicted_sub_topics_labels, topic_confidences, sub_topic_confidences):
#     print(
#         f'Question: {question} | Topic: {topic} ({topic_confidence:.2f}%) | Sub-Topic: {sub_topic} ({sub_topic_confidence:.2f}%)')


# ********* APPROACH 2 **************


# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import KMeans
# from sklearn.manifold import TSNE
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Step 1: Load unstructured text data (example: questions list)
# questions = [
#     'What is the capital of France?',
#     'How does a neural network work?',
#     'What is the Pythagorean theorem?',
#     'What is the capital of Germany?',
#     'How do you train a deep learning model?',
#     'What is the quadratic formula?',
#     'What is the capital of Spain?',
#     'What is backpropagation in neural networks?',
#     'What is the formula for the area of a circle?',
#     'What is a convolutional neural network?',
#     'What is the formula for the volume of a sphere?',
#     'How do you optimize a machine learning model?',
#     'What is the derivative of a function?',
#     'What is a decision tree in machine learning?',
#     'What is a matrix in mathematics?',
#     'What is the capital of Italy?',
#     'What is gradient descent?',
#     'What is the difference between supervised and unsupervised learning?',
#     'What is a neural network’s activation function?',
#     'What is the capital of Canada?',
#     'What is the difference between regression and classification?',
#     'What is the value of x in 2x + 3 = 5?',
#     'What is the integral of sin(x)?',
#     'What is the derivative of cos(x)?',
#     'What is the quadratic equation?',
#     'What is the area of a triangle?',
#     'What is the formula for the volume of a cone?',
#     'What is the Pythagorean theorem used for?',
#     'What is the difference between differentiation and integration?',
#     'What is the derivative of x^2?',
#     'What is the formula for the surface area of a sphere?',
#     'What is the integral of cos(x)?'
# ]
#
# # Step 2: Load the pre-trained Sentence-Transformer model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#
# # Step 3: Get embeddings for the text data (questions)
# embeddings = embedding_model.encode(questions, show_progress_bar=True)
#
# # Step 4: Elbow Method to find the optimal number of clusters
# distortions = []
# for i in range(1, 11):  # Try different cluster sizes from 1 to 10
#     kmeans = KMeans(n_clusters=i, random_state=42)
#     kmeans.fit(embeddings)
#     distortions.append(kmeans.inertia_)
#
# # Plot the Elbow curve
# plt.figure(figsize=(8, 6))
# plt.plot(range(1, 11), distortions, marker='o')
# plt.title('Elbow Method for Optimal Clusters')
# plt.xlabel('Number of clusters')
# plt.ylabel('Distortion')
# plt.show()
#
# # From the elbow method, pick the optimal number of clusters (let's assume it's 5 for now)
# num_clusters = 5
# kmeans = KMeans(n_clusters=num_clusters, random_state=42)
# kmeans.fit(embeddings)
#
# # Step 5: Get the cluster labels for each question
# labels = kmeans.labels_
#
# # Step 6: Create a DataFrame to view the topics and cluster assignments
# df = pd.DataFrame({
#     "Question": questions,
#     "Cluster": labels
# })
#
# # Display the clusters
# print("Questions with their respective clusters:")
# print(df)



# ********* APPROACH 3 **************



# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import KMeans
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Step 1: Load unstructured text data (example: questions list)
# questions = [
#     'What is the capital of France?',
#     'How does a neural network work?',
#     'What is the Pythagorean theorem?',
#     'What is the capital of Germany?',
#     'How do you train a deep learning model?',
#     'What is the quadratic formula?',
#     'What is the capital of Spain?',
#     'What is backpropagation in neural networks?',
#     'What is the formula for the area of a circle?',
#     'What is a convolutional neural network?',
#     'What is the formula for the volume of a sphere?',
#     'How do you optimize a machine learning model?',
#     'What is the derivative of a function?',
#     'What is a decision tree in machine learning?',
#     'What is a matrix in mathematics?',
#     'What is the capital of Italy?',
#     'What is gradient descent?',
#     'What is the difference between supervised and unsupervised learning?',
#     'What is a neural network’s activation function?',
#     'What is the capital of Canada?',
#     'What is the difference between regression and classification?',
#     'What is the value of x in 2x + 3 = 5?',
#     'What is the integral of sin(x)?',
#     'What is the derivative of cos(x)?',
#     'What is the quadratic equation?',
#     'What is the area of a triangle?',
#     'What is the formula for the volume of a cone?',
#     'What is the Pythagorean theorem used for?',
#     'What is the difference between differentiation and integration?',
#     'What is the derivative of x^2?',
#     'What is the formula for the surface area of a sphere?',
#     'What is the integral of cos(x)?'
# ]
#
# # Step 2: Load the pre-trained Sentence-Transformer model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#
# # Step 3: Get embeddings for the text data (questions)
# embeddings = embedding_model.encode(questions, show_progress_bar=True)
#
# # Step 4: Elbow Method to find the optimal number of clusters
# distortions = []
# for i in range(1, 11):  # Try different cluster sizes from 1 to 10
#     kmeans = KMeans(n_clusters=i, random_state=42)
#     kmeans.fit(embeddings)
#     distortions.append(kmeans.inertia_)
#
# # Plot the Elbow curve
# plt.figure(figsize=(8, 6))
# plt.plot(range(1, 11), distortions, marker='o')
# plt.title('Elbow Method for Optimal Clusters')
# plt.xlabel('Number of clusters')
# plt.ylabel('Distortion')
# plt.show()
#
# # From the elbow method, pick the optimal number of clusters (let's assume it's 5 for now)
# num_clusters = 5
# kmeans = KMeans(n_clusters=num_clusters, random_state=42)
# kmeans.fit(embeddings)
#
# # Step 5: Get the cluster labels for each question
# labels = kmeans.labels_
#
# # Step 6: Assign topics and sub-topics to clusters (manually after inspecting the clusters)
# cluster_labels = {
#     0: {'topic': 'Computer Science', 'sub_topic': 'Machine Learning'},
#     1: {'topic': 'Mathematics', 'sub_topic': 'Geometry'},
#     2: {'topic': 'Computer Science', 'sub_topic': 'Neural Networks'},
#     3: {'topic': 'Mathematics', 'sub_topic': 'Algebra'},
#     4: {'topic': 'Geography', 'sub_topic': 'Capital Cities'}
# }
#
# # Step 7: Create a DataFrame to view the questions with cluster assignments and labels
# df = pd.DataFrame({
#     "Question": questions,
#     "Cluster": labels
# })
#
# # Add the assigned topic and sub-topic labels based on clusters
# df['Topic'] = df['Cluster'].map(lambda x: cluster_labels[x]['topic'])
# df['Sub-Topic'] = df['Cluster'].map(lambda x: cluster_labels[x]['sub_topic'])
#
# # Display the clusters with topics and sub-topics
# print("Questions with their respective clusters and topics:")
# print(df)
#
# # Step 8: Make predictions for new questions
# new_questions = [
#     'What is the capital of Spain?',
#     'What is backpropagation?',
#     'What is Natural Language Processing?',
#     'What is the value of x in 2x = 30 + x?',
#     'What is the value of sin^2(x) + cos^2(x) = 1?'
# ]
#
# # Step 9: Get embeddings for the new questions
# new_embeddings = embedding_model.encode(new_questions, show_progress_bar=True)
#
# # Predict the cluster for the new questions
# new_labels = kmeans.predict(new_embeddings)
#
# # Map the predicted clusters to topics and sub-topics
# predicted_topics = [cluster_labels[label]['topic'] for label in new_labels]
# predicted_sub_topics = [cluster_labels[label]['sub_topic'] for label in new_labels]
#
# # Step 10: Display the results
# for question, topic, sub_topic in zip(new_questions, predicted_topics, predicted_sub_topics):
#     print(f"Question: {question} | Topic: {topic} | Sub-Topic: {sub_topic}")



# ********* APPROACH 4 **************



# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt
#
# # Step 1: Questions to cluster
# questions = [
#     'What is the capital of France?',
#     'How does a neural network work?',
#     'What is the Pythagorean theorem?',
#     'What is the capital of Germany?',
#     'How do you train a deep learning model?',
#     'What is the quadratic formula?',
#     'What is the capital of Spain?',
#     'What is backpropagation in neural networks?',
#     'What is the formula for the area of a circle?',
#     'What is a convolutional neural network?',
#     'What is the formula for the volume of a sphere?',
#     'How do you optimize a machine learning model?',
#     'What is the derivative of a function?',
#     'What is a decision tree in machine learning?',
#     'What is a matrix in mathematics?',
#     'What is the capital of Italy?',
#     'What is gradient descent?',
#     'What is the difference between supervised and unsupervised learning?',
#     'What is a neural network’s activation function?',
#     'What is the capital of Canada?',
#     'What is the difference between regression and classification?',
#     'What is the value of x in 2x + 3 = 5?',
#     'What is the integral of sin(x)?',
#     'What is the derivative of cos(x)?',
#     'What is the quadratic equation?',
#     'What is the area of a triangle?',
#     'What is the formula for the volume of a cone?',
#     'What is the Pythagorean theorem used for?',
#     'What is the difference between differentiation and integration?',
#     'What is the derivative of x^2?',
#     'What is the formula for the surface area of a sphere?',
#     'What is the integral of cos(x)?'
# ]
#
# # Step 2: Load labeled CSV data with 'Question', 'Topic', 'Sub-Topic'
# labeled_df = pd.read_csv("labeled_questions.csv")
#
# # Step 3: Load SentenceTransformer model
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
#
# # Step 4: Encode all questions and labeled questions
# question_embeddings = embedding_model.encode(questions, show_progress_bar=True)
# labeled_embeddings = embedding_model.encode(labeled_df['Question'].tolist(), show_progress_bar=True)
#
# # Step 5: Elbow Method to find optimal clusters
# distortions = []
# silhouette_scores = []
# for i in range(2, 25):  # Try cluster sizes from 2 to 25
#     kmeans = KMeans(n_clusters=i, random_state=42)
#     kmeans.fit(question_embeddings)
#     distortions.append(kmeans.inertia_)
#
#     # Calculate Silhouette Score
#     silhouette_avg = silhouette_score(question_embeddings, kmeans.labels_)
#     silhouette_scores.append(silhouette_avg)
#
# # Step 6: Plot Elbow and Silhouette scores
# fig, ax1 = plt.subplots(figsize=(8, 6))
#
# ax1.set_xlabel('Number of clusters')
# ax1.set_ylabel('Distortion', color='tab:red')
# ax1.plot(range(2, 25), distortions, marker='o', color='tab:red', label="Elbow Method (Distortion)")
#
# ax2 = ax1.twinx()  # Create another y-axis for silhouette score
# ax2.set_ylabel('Silhouette Score', color='tab:blue')
# ax2.plot(range(2, 25), silhouette_scores, marker='o', color='tab:blue', label="Silhouette Score")
#
# fig.tight_layout()
# plt.title('Elbow Method and Silhouette Score for Optimal Clusters')
# plt.show()
#
# # Step 7: Choose optimal number of clusters based on silhouette score (higher is better)
# optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2  # Adding 2 because range starts from 2
#
# print("Optimal number of clusters are: ", optimal_clusters)
#
# # Step 8: Perform KMeans clustering with the optimal number of clusters
# kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
# kmeans.fit(question_embeddings)
#
# # Step 9: Assign clusters to labeled questions
# labeled_cluster_ids = kmeans.predict(labeled_embeddings)
# labeled_df['Cluster'] = labeled_cluster_ids
#
# # Step 10: Build cluster to topic mapping based on labeled questions
# cluster_labels = {}
# for _, row in labeled_df.iterrows():
#     cluster = row['Cluster']
#     topic = row['Topic']
#     sub_topic = row['Sub-Topic']
#     cluster_labels[cluster] = {'topic': topic, 'sub_topic': sub_topic}
#
# # Step 11: Assign clusters to original questions
# question_clusters = kmeans.labels_
# df = pd.DataFrame({'Question': questions, 'Cluster': question_clusters})
#
# # Step 12: Map topic and sub-topic based on predicted cluster
# df['Topic'] = df['Cluster'].map(lambda x: cluster_labels.get(x, {"topic": "Unknown", "sub_topic": "Unknown"})['topic'])
# df['Sub-Topic'] = df['Cluster'].map(
#     lambda x: cluster_labels.get(x, {"topic": "Unknown", "sub_topic": "Unknown"})['sub_topic'])
#
# # Step 13: Display the results
# print(df)
#
# # Step 14: Make predictions for new questions
# new_questions = [
#     'What is the capital of Spain?',
#     'What is backpropagation?',
#     'What is Natural Language Processing?',
#     'What is the value of x in 2x = 30 + x?',
#     'What is the value of sin^2(x) + cos^2(x) = 1?'
# ]
#
# # Step 15: Get embeddings for the new questions
# new_embeddings = embedding_model.encode(new_questions, show_progress_bar=True)
#
# # Predict the cluster for the new questions
# new_labels = kmeans.predict(new_embeddings)
#
# # Map predicted clusters to topics and sub-topics
# predicted_topics = []
# predicted_sub_topics = []
#
# for label in new_labels:
#     topic_info = cluster_labels.get(label, {"topic": "Unknown", "sub_topic": "Unknown"})
#     predicted_topics.append(topic_info['topic'])
#     predicted_sub_topics.append(topic_info['sub_topic'])
#
# # Step 16: Display the results
# print("\n Questions Prediction:")
# for question, topic, sub_topic in zip(new_questions, predicted_topics, predicted_sub_topics):
#     print(f"Question: {question}\n → Topic: {topic} | Sub-Topic: {sub_topic}\n")




# ********* APPROACH 5 **************




import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift

# Disable parallelism warning from HuggingFace tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Avoid OpenMP conflicts by setting the number of threads
os.environ["OMP_NUM_THREADS"] = "1"

# Step 1: Load structured (labeled) and unstructured questions from CSV
df = pd.read_csv("labeled_questions2.csv")
labeled_df = df.dropna(subset=['Topic', 'Sub-Topic'])
unstructured_df = df[df['Topic'].isna()]

# Step 2: Store the original index for tracking
unstructured_questions = unstructured_df['Question'].tolist()
unstructured_indexes = unstructured_df.index.tolist()  # Retain the original index

# Step 3: Load SentenceTransformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Encode all questions (unstructured and labeled)
question_embeddings = embedding_model.encode(unstructured_questions, show_progress_bar=True).astype('float32')
labeled_embeddings = embedding_model.encode(labeled_df['Question'].tolist(), show_progress_bar=True).astype('float32')

# Step 5: Combine unstructured and labeled questions
all_questions = unstructured_questions + labeled_df['Question'].tolist()
all_embeddings = list(question_embeddings) + list(labeled_embeddings)

# Step 6: Define clustering models to try with dynamic parameter ranges
clustering_models = {
    "KMeans": {
        "model": KMeans(n_init=10, random_state=42),
        "params": {'n_clusters': range(2, 25)}  # Test with n_clusters from 2 to 25
    },
    "DBSCAN": {
        "model": DBSCAN(eps=0.5, min_samples=5),
        "params": {}  # No params, just use default
    },
    "HDBSCAN": {
        "model": hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True),
        "params": {}  # No params, just use default
    },
    "AgglomerativeClustering": {
        "model": AgglomerativeClustering(),
        "params": {'n_clusters': range(2, 25)}  # Test with n_clusters from 2 to 25
    },
    "MeanShift": {
        "model": MeanShift(),
        "params": {}  # No params, just use default
    }
}

# Step 7: Evaluate clustering models using silhouette score
best_model = None
best_score = -1
best_clusters = None
best_model_name = ""
optimal_num_clusters = 0

for model_name, config in clustering_models.items():
    model = config['model']
    param_grid = config['params']

    # Loop through each parameter combination
    if param_grid:  # If there are parameters to test
        for param, values in param_grid.items():
            for value in values:
                model.set_params(**{param: value})  # Set the current parameter value
                model.fit(all_embeddings)

                # Get the labels assigned by the model
                if hasattr(model, 'labels_'):
                    labels = model.labels_

                    # For DBSCAN and HDBSCAN, exclude noise points labeled as -1
                    if model_name in ['DBSCAN', 'HDBSCAN']:
                        valid_labels = labels[labels != -1]
                    else:
                        valid_labels = labels

                    # Ensure there are at least two clusters
                    if len(set(valid_labels)) >= 2:
                        try:
                            score = silhouette_score(all_embeddings, valid_labels)
                            print(f"{model_name} ({param}={value}) Silhouette Score: {score:.4f}")
                            if score > best_score:
                                best_score = score
                                best_model = model
                                best_clusters = model.labels_
                                best_model_name = model_name
                                optimal_num_clusters = len(set(best_clusters)) - (1 if -1 in best_clusters else 0)
                        except ValueError as e:
                            print(f"Skipping {model_name} ({param}={value}) due to invalid number of clusters: {e}")
    else:  # If no parameters to test, just run the model
        model.fit(all_embeddings)

        # Get the labels assigned by the model
        if hasattr(model, 'labels_'):
            labels = model.labels_

            # For DBSCAN and HDBSCAN, exclude noise points labeled as -1
            if model_name in ['DBSCAN', 'HDBSCAN']:
                valid_labels = labels[labels != -1]
            else:
                valid_labels = labels

            # Ensure there are at least two clusters
            if len(set(valid_labels)) >= 2:
                try:
                    score = silhouette_score(all_embeddings, valid_labels)
                    print(f"{model_name} Silhouette Score: {score:.4f}")
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_clusters = model.labels_
                        best_model_name = model_name
                        optimal_num_clusters = len(set(best_clusters)) - (1 if -1 in best_clusters else 0)
                except ValueError as e:
                    print(f"Skipping {model_name} due to invalid number of clusters: {e}")

# Step 8: Print the best model and optimal number of clusters
print(f"\nBest Clustering Model: {best_model_name} with {optimal_num_clusters} clusters")

# Step 9: Use the best model to assign clusters to labeled questions
labeled_df['Cluster'] = best_clusters[-len(labeled_df):]  # Assign clusters to labeled questions

# Step 10: Build cluster to topic mapping based on labeled questions
cluster_labels = {}
for _, row in labeled_df.iterrows():
    cluster = row['Cluster']
    topic = row['Topic']
    sub_topic = row['Sub-Topic']
    cluster_labels[cluster] = {'topic': topic, 'sub_topic': sub_topic}

# Step 11: Assign clusters to original unstructured questions
question_clusters = best_clusters[:len(unstructured_questions)]  # Only the unstructured questions
df_unstructured = pd.DataFrame({
    'Question': unstructured_questions,
    'Cluster': question_clusters,
    'Index': unstructured_indexes  # Add the original index column for tracking
})

# Step 12: Add Topic and Sub-Topic for unstructured questions (based on their predicted cluster)
df_unstructured['Topic'] = df_unstructured['Cluster'].map(
    lambda x: cluster_labels.get(x, {"topic": "Unknown", "sub_topic": "Unknown"})['topic'])
df_unstructured['Sub-Topic'] = df_unstructured['Cluster'].map(
    lambda x: cluster_labels.get(x, {"topic": "Unknown", "sub_topic": "Unknown"})['sub_topic'])

# Step 13: Combine labeled and unstructured questions in one final DataFrame
labeled_df['Cluster'] = best_clusters[-len(labeled_df):]  # Assign clusters to labeled questions
labeled_df['Topic'] = labeled_df['Topic']  # Keep the original topics for labeled questions
labeled_df['Sub-Topic'] = labeled_df['Sub-Topic']  # Keep the original sub-topics for labeled questions

# Step 14: Add the unstructured questions to the final DataFrame
final_df = pd.concat([df_unstructured[['Question', 'Cluster', 'Index', 'Topic', 'Sub-Topic']],
                       labeled_df[['Question', 'Cluster', 'Index', 'Topic', 'Sub-Topic']]],
                      ignore_index=False)

# Step 15: Re-sort the final DataFrame based on the original index
final_df = final_df.sort_values('Index')  # Re-sort by original index

# Step 16: Save the results to a CSV file
final_df.to_csv("unstructured_questions_with_clusters.csv", index=False)  # Don't include the index when saving

# Add a serial-wise 'Index' column
final_df['Index'] = range(1, len(final_df) + 1)

# Reorder the columns to make 'Index' the first column
final_df = final_df[['Index'] + [col for col in final_df.columns if col != 'Index']]

# Re-save the CSV file with the new 'Index' column
final_df.to_csv("unstructured_questions_with_clusters.csv", index=False)

# Display the final DataFrame with the added 'Index' column
print(final_df)

# Step 17: Make predictions for new questions using nearest neighbors
new_questions = [
    'What is the capital of Spain?',
    'What is backpropagation?',
    'What is Natural Language Processing?',
    'What is the value of x in 2x = 30 + x?',
    'What is the value of sin^2(x) + cos^2(x) = 1?',
    'Can you provide examples of deep learning applications?'
]

# Step 18: Get embeddings for the new questions
new_embeddings = embedding_model.encode(new_questions, show_progress_bar=True).astype('float32')

# Step 19: Use NearestNeighbors to assign new questions to the nearest labeled clusters
nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
nn.fit(labeled_embeddings)  # Fit the nearest neighbors on labeled question embeddings
distances, indices = nn.kneighbors(new_embeddings)  # Find the nearest labeled questions for new questions

# Step 20: Get the predicted clusters based on the nearest labeled questions
predicted_clusters = labeled_df['Cluster'].iloc[indices.flatten()]

# Step 21: Map predicted clusters to topics and sub-topics
predicted_topics = []
predicted_sub_topics = []

for label in predicted_clusters:
    topic_info = cluster_labels.get(label, {"topic": "Unknown", "sub_topic": "Unknown"})
    predicted_topics.append(topic_info['topic'])
    predicted_sub_topics.append(topic_info['sub_topic'])

# Step 22: Display the results for new questions
print("\nQuestions Prediction:")
for question, topic, sub_topic in zip(new_questions, predicted_topics, predicted_sub_topics):
    print(f"Question: {question}\n → Topic: {topic} | Sub-Topic: {sub_topic}\n")