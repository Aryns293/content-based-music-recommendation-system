import streamlit as st
import pickle
import pandas as pd

knn_model = pickle.load(open('knn_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

music = pd.read_csv('spotify_millsongdata.csv')
music['tags'] = music['song'] + ' ' + music['artist']

st.title("🎵 Music Recommender System")

selected_song = st.selectbox("Select a song", music['song'].unique())

def recommend(song):
    song_tags = music[music['song'] == song]['tags'].values[0]
    song_vector = vectorizer.transform([song_tags])

    distances, indices = knn_model.kneighbors(song_vector, n_neighbors=6)

    recommended_music_names = []

    for i in indices[0][1:]:
        recommended_music_names.append(music.iloc[i].song)

    return recommended_music_names

if selected_song:
    recommended_music = recommend(selected_song)
    st.subheader("Recommended Songs:")
    for song in recommended_music:
        st.write(song)