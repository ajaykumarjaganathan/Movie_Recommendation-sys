import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import time as t
#st.set_option('wideMode' , True)
st.set_page_config(layout="wide",page_title="Movie Recommendation", page_icon=":movie_camera:")

h1, h2, h3 = st.columns([2, 4, 1])
with h2:
    st.header("Movie Recommendation System")
#st.markdown("Rana Karmakar")
df = pd.read_csv("movies.csv")
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    df[feature] = df[feature].fillna('')
combined_features = df['genres'] + ' ' + df['keywords'] + ' ' + df['tagline'] + ' ' + df['cast'] + ' ' + df['director']
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

@st.cache_data
def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(
        movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


# movie_name = input(' Enter your favourite movie name : ')
list_of_all_titles = df['title'].tolist()
ip1, ip2,ip3 = st.columns([1.8,4,2.2])
with ip2:
    movie_name = st.selectbox(" ",list_of_all_titles,index=16)
    if not movie_name:
        st.warning("Please Select any Movie")
        st.stop()
    if movie_name:
        # st.write("Submitted Successfully")
        with st.spinner('Wait for it...'):
            t.sleep(2)
        # st.success('Recommendation Generated!')
with ip1:
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.markdown("Select Your Favorite Movie : ")
with ip3:
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.markdown("       Cinema is the most beautiful fraud in the world")

# movie_name = st.text_input("Enter Any Movie Name : ", value="Avatar")
find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

close_match = find_close_match[0]

index_of_the_movie = df[df.title == close_match]['index'].values[0]

similarity_score = list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

st.subheader(' Movies Suggested for you : ')

i = 1
rec_movies = []
rec_ids = []
image_url = []
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = df[df.index == index]['title'].values[0]
    # fetch_poster(index)
    if i < 11:
        rec_movies.append(title_from_index)
        i += 1
        # condition with df.values property
        mask = df['original_title'].values == title_from_index
        # new dataframe
        df_new = df[mask]
        ids = df_new["id"].to_numpy()
        # st.write(fetch_poster(ids))
        # st.image(fetch_poster(ids))
        rec_ids.append(ids)
try:
    for j in rec_ids:
        image_url.append(fetch_poster(j[0]))
except IndexError:
    st.write("Not Found")

col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    st.image(image_url[0])
with col2:
    st.image(image_url[1])
with col3:
    st.image(image_url[2])
with col4:
    st.image(image_url[3])
with col5:
    st.image(image_url[4])
try :
    with col6:
        st.image(image_url[5])
    n7, n8, n9, n10, n11, n12 = st.columns(6)
    with n7:
        st.image(image_url[6])
    with n8:
        st.image(image_url[7])
    with n9:
        st.image(image_url[6])
    with n10:
        st.image(image_url[7])
    with n11:
        st.image(image_url[8])
    with n12:
        st.image(image_url[9])
except IndexError:
    st.write("Not Found")
# Conclusion
ex1, ex2, ex3 = st.columns(3)
with ex1:
    with st.expander("About The Project"):
        st.markdown("Content-Based Recommendation System")
        st.write(
            "Recommender systems are a powerful new technology for extracting additional value for a business from its "
            "user databases.Recommender systems are a powerful new technology for extracting additional value for a "
            "business from its user databases. These systems help users find items they want to buy from a business. "
            "Recommender systems benefit users by enabling them to find items they like. Conversely, they help the "
            "business by generating more sales. Recommender systems are rapidly becoming a crucial tool in E-commerce "
            "on "
            "the Web. Recommender systems are being stressed by the huge volume of user data in existing corporate "
            "databases, and will be stressed even more by the increasing volume of user data available on the Web. New "
            "technologies are needed that can dramatically improve the scalability of recommender systems.")
with ex3:
    with st.expander("About Developer"):
        st.markdown("Rana Karmakar")
        st.write("I have a deep interest in Artificial Intelligence and Machine Learning ever since I got to know "
                 "about it; the sci-fi films, comics and stories have always fascinated me. I love to learn new "
                 "skills to keep myself up-to-date with the corporate world. I believe in maintaining a work-life "
                 "balance while learning and working upon different fields of interest.I have had a taste of many "
                 "different technologies: creating Websites, Software Development, Data Analysis, "
                 "creating Machine Learning models, Cloud Computing and Developing complex programs and more. "
                 "Feel free to give Feedback at ranakarmakar027@gmail.com")
with ex2:
    with st.expander("Contact"):
        st.markdown("ranakarmakar027@gmail.com")
        st.write("[Website](https://rana-reflective-porcupine-pf.eu-gb.mybluemix.net/)")
        st.write("[LinkedIn](https://www.linkedin.com/in/rana-karmakar-0972641a6)")
        st.write("Other Apps[Movie Recommendation System]("
                 "https://share.streamlit.io/ranakarmakar/streamlit_movie_recommendation/main/app.py)")
        st.write("Other Apps[Brain Tumor Detection]("
                 "https://share.streamlit.io/ranakarmakar/brain_tumor_classification/main/tumor.py)")

