# 🎵 Content-Based Music Recommendation System

A Machine Learning-powered music recommender system that suggests similar songs using **TF-IDF vectorization** and **K-Nearest Neighbors (KNN)**.

Built with Python, Scikit-learn, and Streamlit.

---

## 📌 Project Overview

This project implements a **content-based filtering recommendation system** that recommends songs based on textual similarity between song titles and artists.

The system:
- Extracts features using TF-IDF
- Computes similarity using KNN (Cosine Similarity)
- Displays recommendations via a Streamlit web app

---

## 🧠 How It Works

1. Combine song name + artist into a feature string  
2. Apply TF-IDF Vectorization  
3. Train KNN model  
4. Retrieve top similar songs  
5. Display recommendations in web UI  

---

## 🛠 Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- TF-IDF
- KNN (Cosine Similarity)

---

## 📂 Project Structure

```
music-recommender/
│
├── app.py
├── train_model.py
├── spotify_millsongdata.csv
├── knn_model.pkl
├── vectorizer.pkl
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/music-recommender.git
cd music-recommender
```

### 2️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install streamlit pandas scikit-learn numpy
```

### 4️⃣ Train Model (Only Once)

```bash
python train_model.py
```

### 5️⃣ Run App

```bash
streamlit run app.py
```

---

## 🎯 Features

✔ Content-Based Filtering  
✔ Cosine Similarity  
✔ Fast Recommendations  
✔ Clean Streamlit UI  
✔ Modular Code Structure  

---

## 📈 Future Improvements

- Add collaborative filtering
- Deploy online
- Add Spotify API integration
- Display album artwork
- Add similarity score

---

## 🧑‍💻 Author

Aryan Sharma<br>
Software Engineering<br>
DTU
---

## ⭐ If You Like This Project

Give it a star ⭐ on GitHub!
