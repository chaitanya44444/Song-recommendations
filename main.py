import os
import streamlit as st
import pandas as pd
import json
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from streamlit_mic_recorder import mic_recorder, speech_to_text

st.set_page_config(page_title="Song Recommender", page_icon="🎵", layout="centered", initial_sidebar_state="expanded")
st.markdown('<style>body {background-color: #0000000; color: black;}</style>', unsafe_allow_html=True)

SPOTIPY_CLIENT_ID = '36699e378ba8401aa2bc8b72a494b107'
SPOTIPY_CLIENT_SECRET = '56a4675e30ac4f7d9d06c1dcfdfa2d82'
SPOTIPY_REDIRECT_URI = 'http://localhost:8888/callback'

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,client_secret=SPOTIPY_CLIENT_SECRET,redirect_uri=SPOTIPY_REDIRECT_URI, scope="user-library-read playlist-modify-public"))

@st.cache_data
def load_data():
    tracks_data_parts = []
    for i in range(1, 16):
        try:
            part_data = pd.read_csv(f'data/tracks_part_{i}.csv')
            tracks_data_parts.append(part_data)
        except FileNotFoundError:
            st.warning(f'File data/tracks_part_{i}.csv not found.')

    tracks_data = pd.concat(tracks_data_parts, ignore_index=True) if tracks_data_parts else pd.DataFrame()

    artists_data_parts = []
    for i in range(1, 16):
        try:
            part_data = pd.read_csv(f'data/artists_part_{i}.csv')
            artists_data_parts.append(part_data)
        except FileNotFoundError:
            st.warning(f'File data/artists_part_{i}.csv not found.')

    artists_data = pd.concat(artists_data_parts, ignore_index=True) if artists_data_parts else pd.DataFrame()

    dict_artists_data = {}
    for i in range(1, 16):
        try:
            with open(f'data/dict_artists_part_{i}.json', 'r') as f:
                dict_artists_data.update(json.load(f))
        except FileNotFoundError:
            st.warning(f'File data/dict_artists_part_{i}.json not found.')
    
    selected_columns = ['name', 'artists', 'danceability', 'energy', 'popularity', 'acousticness', 'valence']
    if not tracks_data.empty:
        tracks_data = tracks_data[selected_columns]
        tracks_data.dropna(inplace=True)
        scaler = StandardScaler()
        tracks_data[['danceability', 'energy', 'popularity', 'acousticness', 'valence']] = scaler.fit_transform(
            tracks_data[['danceability', 'energy', 'popularity', 'acousticness', 'valence']]
        )
    else:
        scaler = None

    return tracks_data, artists_data, dict_artists_data, scaler

tracks_data, artists_data, dict_artists_data, scaler = load_data()

def find_similar_songs(song_or_artist, data, energy_level=None, top_n=10):
    song_or_artist = song_or_artist.lower()
    is_artist = data['artists'].str.lower().str.contains(song_or_artist)
    is_song = data['name'].str.lower().str.contains(song_or_artist)

    if is_artist.any():
        reference_song = data[is_artist].iloc[0]
    elif is_song.any():
        reference_song = data[is_song].iloc[0]
    else:
        return pd.DataFrame()

    reference_vector = reference_song[['danceability', 'energy', 'popularity', 'acousticness', 'valence']].values.reshape(1, -1)
    similarity_scores = cosine_similarity(data[['danceability', 'energy', 'popularity', 'acousticness', 'valence']], reference_vector)
    data['similarity'] = similarity_scores

    if energy_level is not None:
        energy_scaled = scaler.transform([[0, energy_level, 0, 0, 0]])[0][1]
        data = data[(data['energy'] >= energy_scaled - 0.2) & (data['energy'] <= energy_scaled + 0.2)]

    similar_songs = data.sort_values(by='similarity', ascending=False).head(top_n)
    similar_songs = similar_songs[similar_songs['name'] != reference_song['name']]

    return similar_songs[['artists', 'name', 'danceability', 'energy', 'popularity', 'acousticness', 'valence', 'similarity']]

def search_on_spotify(query):
    results = sp.search(q=query, type='track', limit=5)
    tracks = results['tracks']['items']
    return [{
        'name': track['name'],
        'artists': [artist['name'] for artist in track['artists']],
        'url': track['external_urls']['spotify'],
        'cover': track['album']['images'][0]['url'],
        'genre': track['album']['artists'][0].get('genres', [])  
    } for track in tracks]

def visualize_attributes(similar_songs):
    if not similar_songs.empty:
        attributes = ['danceability', 'energy', 'popularity', 'acousticness', 'valence']
        avg_values = similar_songs[attributes].mean().clip(lower=0)  
        
        avg_values = avg_values[avg_values > 0]

        plt.figure(figsize=(8, 4))
        wedges, texts, autotexts = plt.pie(avg_values, labels=avg_values.index, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Average Attributes of Recommended Songs')

        for text in texts:
            text.set_fontsize(12)
            text.set_color('white')
        for autotext in autotexts:
            autotext.set_color('black')  

        plt.legend(wedges, avg_values.index, title="Attributes", loc="upper right", bbox_to_anchor=(1, 1))

        st.pyplot(plt)

st.title("Song Recommender 🎵")
st.sidebar.title("Why u here???????????????????????????????????????????????????????????????")

user_input_method = st.radio("Choose input method:", ("Text Input", "Voice Input (Artist)", "Voice Input (Song)", "Voice Input (Both)"))

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

if user_input_method == "Text Input":
    st.session_state.user_input = st.text_input("Enter artist or song name:")

def handle_voice_input():
    if user_input_method == "Voice Input (Artist)":
        text = speech_to_text(language='en', start_prompt="Start recording artist", stop_prompt="Stop recording", key='artist_stt')
        if text:
            st.session_state.user_input = text
            st.success(f"Artist recognized: {text}")

    elif user_input_method == "Voice Input (Song)":
        text = speech_to_text(language='en', start_prompt="Start recording song", stop_prompt="Stop recording", key='song_stt')
        if text:
            st.session_state.user_input = text
            st.success(f"Song recognized: {text}")

    elif user_input_method == "Voice Input (Both)":
        text = speech_to_text(language='en', start_prompt="Start recording both", stop_prompt="Stop recording", key='both_stt')
        if text:
            st.session_state.user_input = text
            st.success(f"Input recognized: {text}")

if user_input_method != "Text Input":
    handle_voice_input()

energy_input = st.slider("Energy Level (0-1):", 0.0, 1.0, 0.8)
danceability_input = st.slider("Danceability (0-1):", 0.0, 1.0, 0.5)
acousticness_input = st.slider("Acousticness (0-1):", 0.0, 1.0, 0.5)
popularity_input = st.slider("Popularity (0-100):", 0, 100, 50)
valence_input = st.slider("Valence (0-1):", 0.0, 1.0, 0.5)
similarity_input = st.slider("Similarity (0-1):", 0.0, 1.0, 0.5)

recommend_btn = st.button("Get Recommendations")

if recommend_btn and st.session_state.user_input:
    similar_songs = find_similar_songs(st.session_state.user_input, tracks_data, energy_level=energy_input)
    if not similar_songs.empty:
        st.write("Top Similar Songs:")
        st.dataframe(similar_songs)
        visualize_attributes(similar_songs)
        st.write("Searching Spotify...")
        spotify_results = search_on_spotify(st.session_state.user_input)
        cols = st.columns(5)  
        for idx, song in enumerate(spotify_results):
            with cols[idx % 5]:  
                st.image(song['cover'], width=100) 
                st.write(f"**{song['name']}** by {', '.join(song['artists'])}")
                st.write(f"Genre: {', '.join(song['genre']) if song['genre'] else 'N/A'}")
                st.write(f"[Listen on Spotify]({song['url']})")
    else:
        st.write("No local data found. Searching Spotify...")
        spotify_results = search_on_spotify(st.session_state.user_input)

        cols = st.columns(5)  
        for idx, song in enumerate(spotify_results):
            with cols[idx % 5]:  
                st.image(song['cover'], width=100) 
                st.write(f"**{song['name']}** by {', '.join(song['artists'])}")
                st.write(f"Genre: {', '.join(song['genre']) if song['genre'] else 'N/A'}")
                st.write(f"[Listen on Spotify]({song['url']})")

st.markdown("Made with ❤️ by Chaitanya")

