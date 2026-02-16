import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity

# ======================================
# CONFIG
# ======================================
st.set_page_config(
    page_title="AI Movie Recommender",
    layout="wide"
)

TMDB_API_KEY = "f3846f894aea59d0f4f23ce80b0f9a16"

# ======================================
# SESSION STATE INIT
# ======================================
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# ======================================
# LOAD DATA
# ======================================
@st.cache_data
def load_data():
    df = pickle.load(open("df.pkl", "rb"))
    tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))

    df["title_clean"] = df["title"].str.lower().str.strip()
    return df, tfidf_matrix


df, tfidf_matrix = load_data()

# ======================================
# TMDB MOVIE DETAILS FETCH
# ======================================
def fetch_movie_data(title):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={title}"
        data = requests.get(url, timeout=5).json()

        if not data["results"]:
            return None

        movie = data["results"][0]
        movie_id = movie["id"]

        # Trailer Fetch
        trailer_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={TMDB_API_KEY}"
        trailer_data = requests.get(trailer_url).json()

        trailer_key = None
        if trailer_data["results"]:
            trailer_key = trailer_data["results"][0]["key"]

        return {
            "title": movie["title"],
            "overview": movie["overview"],
            "rating": movie["vote_average"],
            "release_date": movie["release_date"],
            "poster": "https://image.tmdb.org/t/p/w500" + movie["poster_path"]
            if movie.get("poster_path") else None,
            "trailer": trailer_key
        }

    except:
        return None

# ======================================
# RECOMMENDATION ENGINE
# ======================================
def recommend_movies(movie_title, top_n=6):

    idx = df[df["title_clean"] == movie_title.lower().strip()].index[0]

    sim_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    similar_indices = sim_scores.argsort()[::-1][1:top_n + 1]

    results = []
    for i in similar_indices:
        results.append(df.iloc[i]["title"])

    return results

# ======================================
# SIDEBAR WATCHLIST
# ======================================
# ======================================
# SIDEBAR: DEVELOPER DETAILS
# ======================================
st.sidebar.markdown("---")
st.sidebar.title("👨‍💻 Developer Details")

st.sidebar.markdown("""
**Name:** Gurudas Sunil Kadam  
🎓 MCA Student  
🏫 Shivaji University  
📍 Islampur, India  

---

📌 **Project:**  AI Movie Recommender  
⚙️ Built with Streamlit + TMDB API + Machine Learning  

---
🌐 Connect With Me
                    
📧 Email: kadamguru23@gmail.com  
📱 Contact: 9021292215 



🔗 LinkedIn Profile (https://www.linkedin.com/in/gurudas-kadam-714996322/)


💻 GitHub Repository (https://github.com/GURU2379108)

                     
                    
""")


# ======================================
# MAIN UI
# ======================================
st.title("🎬 Netflix AI Movie Recommender")
st.write("Powered by TF-IDF Similarity + TMDB API")

# ======================================
# ✅ FIX: Uniform Poster Size Styling
# ======================================
st.markdown(
    """
    <style>
    img {
        height: 320px !important;
        object-fit: cover !important;
        border-radius: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ======================================
# MOVIE SELECTION
# ======================================
movie_list = df["title"].values
selected_movie = st.selectbox("Select a Movie:", movie_list)

# ======================================
# RECOMMEND BUTTON
# ======================================
if st.button("🚀 Recommend Movies"):

    rec_movies = recommend_movies(selected_movie)

    st.subheader("✅ Recommended Movies")

    cols = st.columns(3)

    for i, movie in enumerate(rec_movies):

        details = fetch_movie_data(movie)

        with cols[i % 3]:

            if details:

                # ✅ Same Size Poster
                st.image(details["poster"], width=220)

                st.markdown(f"### {movie}")

                # ✅ Slide Down Details Section
                with st.expander("📌 View Details"):

                    st.write(f"⭐ Rating: {details['rating']}")
                    st.write(f"📅 Release Date: {details['release_date']}")

                    st.subheader("Overview")
                    st.info(details["overview"])

                    if details["trailer"]:
                        st.subheader("🎬 Trailer")
                        st.video(
                            f"https://www.youtube.com/watch?v={details['trailer']}"
                        )

                    # ✅ Watchlist Button
                    if st.button(
                        f"➕ Add {movie} to Watchlist",
                        key=f"watch_{movie}"
                    ):

                        if movie not in st.session_state.watchlist:
                            st.session_state.watchlist.append(movie)
                            st.success("Added to Watchlist!")
                        else:
                            st.warning("Already in Watchlist!")
