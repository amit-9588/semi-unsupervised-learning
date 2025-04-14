import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import hdbscan
import faiss
import json

# Stability settings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# --------------- Load & Preprocess Data ---------------
def load_data(path):
    with open(path, 'r') as file:
        data = json.load(file)
    df = pd.DataFrame(data)
    labeled = df.dropna(subset=['Topic', 'Sub-Topic']).copy()
    unlabeled = df[df['Topic'].isna()].copy()
    return df, labeled, unlabeled


df, labeled_df, unstructured_df = load_data("labeled_questions2.json")
unstructured_qs = unstructured_df['Question'].tolist()
labeled_qs = labeled_df['Question'].tolist()

# --------------- Embeddings ---------------
model = SentenceTransformer("all-MiniLM-L6-v2")
emb_unstructured = model.encode(unstructured_qs, show_progress_bar=True).astype('float32')
emb_labeled = model.encode(labeled_qs, show_progress_bar=True).astype('float32')
all_embeddings = np.vstack([emb_unstructured, emb_labeled])

# --------------- Clustering Models Setup ---------------
clustering_models = {
    "KMeans": lambda k: KMeans(n_clusters=k, n_init=10, random_state=42),
    "Agglomerative": lambda k: AgglomerativeClustering(n_clusters=k),
    "DBSCAN": lambda _: DBSCAN(eps=0.5, min_samples=5),
    "HDBSCAN": lambda _: hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True),
    "MeanShift": lambda _: MeanShift()
}

# Function to tune DBSCAN and HDBSCAN hyperparameters
def tune_clustering_models(X):
    param_grid_dbscan = {'eps': [0.3, 0.5, 0.7], 'min_samples': [3, 5, 7]}
    param_grid_hdbscan = {'min_cluster_size': [5, 10, 15]}
    grid_search_dbscan = GridSearchCV(DBSCAN(), param_grid_dbscan, cv=3, n_jobs=-1)
    grid_search_hdbscan = GridSearchCV(hdbscan.HDBSCAN(), param_grid_hdbscan, cv=3, n_jobs=-1)

    # Fit DBSCAN GridSearch
    grid_search_dbscan.fit(X)
    print(f"Best DBSCAN Params: {grid_search_dbscan.best_params_}")

    # Fit HDBSCAN GridSearch
    grid_search_hdbscan.fit(X)
    print(f"Best HDBSCAN Params: {grid_search_hdbscan.best_params_}")

    return grid_search_dbscan.best_params_, grid_search_hdbscan.best_params_

# --------------- Evaluate Model ---------------
def evaluate_model(name, model, X):
    try:
        model.fit(X)
        labels = model.labels_
        valid = labels[labels != -1] if name in ['DBSCAN', 'HDBSCAN'] else labels
        if len(set(valid)) < 2:
            return -1, None
        return silhouette_score(X, valid), labels
    except Exception as e:
        print(f"⚠️ {name} failed: {e}")
        return -1, None

# --------------- Find Best Clustering ---------------
def find_best_model(X):
    best_score, best_labels, best_name, best_k = -1, None, "", 0
    for name, constructor in clustering_models.items():
        ks = range(2, 25) if name in ['KMeans', 'Agglomerative'] else [None]
        for k in ks:
            model = constructor(k)
            score, labels = evaluate_model(name, model, X)
            if score > best_score:
                best_score = score
                best_labels = labels
                best_name = name
                best_k = k if k else len(set(labels)) - (1 if -1 in labels else 0)
    return best_name, best_score, best_k, best_labels


best_model_name, best_score, optimal_clusters, best_labels = find_best_model(all_embeddings)
print(f"\n✅ Best Clustering: {best_model_name} with {optimal_clusters} clusters (Score: {best_score:.4f})")

# --------------- Assign Clusters & Topics ---------------
unstructured_labels = best_labels[:len(emb_unstructured)]
labeled_labels = best_labels[len(emb_unstructured):]
labeled_df['Cluster'] = labeled_labels

# Cluster to Topic Mapping using Majority Vote
cluster_labels = {}
cluster_groups = labeled_df.groupby("Cluster")
for cluster, group in cluster_groups:
    most_common_topic = group["Topic"].value_counts().idxmax()
    most_common_subtopic = group["Sub-Topic"].value_counts().idxmax()
    cluster_labels[cluster] = {"Topic": most_common_topic, "Sub-Topic": most_common_subtopic}

df_unstructured = pd.DataFrame({
    'Question': unstructured_qs,
    'Cluster': unstructured_labels,
    'Index': unstructured_df.index
})
df_unstructured['Topic'] = df_unstructured['Cluster'].map(lambda c: cluster_labels.get(c, {}).get("Topic", "Unknown"))
df_unstructured['Sub-Topic'] = df_unstructured['Cluster'].map(
    lambda c: cluster_labels.get(c, {}).get("Sub-Topic", "Unknown"))

labeled_df['Index'] = labeled_df.index
final_df = pd.concat([
    df_unstructured,
    labeled_df[['Question', 'Cluster', 'Index', 'Topic', 'Sub-Topic']]
]).sort_values('Index')

# Drop the existing 'Index' column if it exists
if 'Index' in final_df.columns:
    final_df.drop('Index', axis=1, inplace=True)

# Now insert the new 'Index' column
final_df.insert(0, 'Index', range(1, len(final_df) + 1))

# Save final results to JSON
result_data = final_df.to_dict(orient='records')
with open("unstructured_questions_with_clusters.json", 'w') as outfile:
    json.dump(result_data, outfile, indent=4)

print("📦 Saved to 'unstructured_questions_with_clusters.json'")

# --------------- Train Classifier ---------------
X_train, X_test, y_train, y_test = train_test_split(emb_labeled, labeled_labels, test_size=0.2, random_state=42)
classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train, y_train)
print("🤖 Classifier trained.")

# --------------- Setup FAISS ---------------
faiss_index = faiss.IndexFlatL2(emb_labeled.shape[1])
faiss_index.add(emb_labeled)
print(f"⚡ FAISS Index ready: {len(emb_labeled)} vectors.")

# --------------- Prediction Function ---------------
def predict_question(question, threshold=0.7, min_confidence=0.25):
    emb = model.encode([question], show_progress_bar=False).astype('float32')
    probs = classifier.predict_proba(emb)
    confidence = np.max(probs)
    pred_cluster = classifier.predict(emb)[0]

    if confidence >= threshold:
        method = "Classifier ✅"
    else:
        D, I = faiss_index.search(emb, 1)
        pred_cluster = labeled_df.iloc[I[0][0]]['Cluster']
        method = "FAISS 🔍"
        distance = D[0][0]
        confidence = max(0.0, 1.0 - distance / 1.5)  # normalize distance → confidence

    # If confidence is too low, mark as Unknown
    if confidence < min_confidence:
        topic, subtopic = "Unknown", "Unknown"
    else:
        topic_info = cluster_labels.get(pred_cluster, {"Topic": "Unknown", "Sub-Topic": "Unknown"})
        topic = topic_info["Topic"]
        subtopic = topic_info["Sub-Topic"]

    return method, topic, subtopic, confidence

# --------------- Demo Predictions ---------------
sample_questions = [
    'What is the capital of Spain?',
    'What is backpropagation?',
    'What is Natural Language Processing?',
    'What is the value of x in 2x = 30 + x?',
    'What is the value of sin^2(x) + cos^2(x) = 1?',
    'Can you provide examples of deep learning applications?',
    'What is the integral of cot(x)?',
    'What is the formula for the volume of a circle?'
]

for q in sample_questions:
    method, topic, subtopic, confidence = predict_question(q)
    print(f"\n🧠 Question: {q}")
    print(f"→ {method} | Topic: {topic} | Sub-Topic: {subtopic}")
    if confidence is not None:
        print(f"🎯 Confidence: {confidence * 100:.2f}%")
