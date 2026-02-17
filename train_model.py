import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle

# Load the dataset
music = pd.read_csv('spotify_millsongdata.csv')

# Create a combined 'tags' column
music['tags'] = music['song'] + ' ' + music['artist']

# Vectorize the 'tags' using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
vector_data = vectorizer.fit_transform(music['tags'])

# Train KNN model using Euclidean distance
knn_model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
knn_model.fit(vector_data)

# Save the model and vectorizer
pickle.dump(knn_model, open('knn_model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))