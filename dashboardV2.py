import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import torch
import joblib
import re
import emoji
import plotly.graph_objects as go
from wordcloud import WordCloud
from collections import Counter
from langdetect import detect, DetectorFactory
from nltk.corpus import stopwords
import pickle
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from autocorrect import Speller
import nltk
from nltk.stem import WordNetLemmatizer

# ==========================
# 💼 Initialize API and Models
# ==========================
nltk.download("stopwords")
nltk.download('wordnet')

DetectorFactory.seed = 0  
stop_words = set(stopwords.words("english"))

# Initialize spell checker and lemmatizer
spell = Speller(lang='en')
lemmatizer = WordNetLemmatizer()

# Load Models
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

with open("fine_tuned_model.pkl", 'rb') as model_file:
    category_model = pickle.load(model_file)

with open("tokenizer.pkl", 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

with open("label_encoder.pkl", 'rb') as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
category_model.to(device)

vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_features=2)

# Initialize YouTube API
YOUTUBE_API_KEY = "AIzaSyDiKzS0goNfIPWMkmbzA9W3oSHLWZd0eWA"
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# ==========================
# 💼 Data Processing Functions
# ==========================
def extract_video_id(youtube_url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", youtube_url)
    return match.group(1) if match else None

def fetch_video_details(video_id):
    request = youtube.videos().list(part="snippet", id=video_id)
    response = request.execute()
    if response["items"]:
        snippet = response["items"][0]["snippet"]
        return snippet["title"], snippet["channelTitle"], snippet.get("description", "No description available")
    return "Unknown Title", "Unknown Author", "No description available"

def fetch_youtube_comments(video_id):
    comments, next_page_token = [], None
    try:
        while True:
            response = youtube.commentThreads().list(
                part="snippet", videoId=video_id, pageToken=next_page_token
            ).execute()
            
            if "items" not in response:
                return []
            
            for item in response["items"]:
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "Comment_Id": item["id"],
                    "Comment_Text": snippet["textDisplay"],
                    "Time_Stamp": snippet["publishedAt"],
                    "Comment_by": snippet["authorDisplayName"],
                    "Likes": snippet.get("likeCount", 0),
                    "Replies": item["snippet"].get("totalReplyCount", 0)
                })

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

        return comments
    except Exception:
        return []

def clean_text(text, sentiment_label=None):
    """Preprocess text to match the original training pipeline."""
    
    # Lowercasing
    text = text.lower()
    
    # Removing URLs
    text = re.sub(r"http\S+|www.\S+", "", text)
    
    # Removing HTML tags (e.g., <br>, <a href="...">)
    text = re.sub(r"<.*?>", " ", text)
    
    # Removing new lines
    text = re.sub(r"\n", " ", text)
    
    # Removing all punctuations
    text = re.sub(r"[^\w\s]", " ", text)
    
    # Removing numbers
    text = re.sub(r"\d+", "", text)
    
    # Removing emojis
    text = re.sub(r"[^\w\s#@/:%.,_-]", " ", text, flags=re.UNICODE)
    
    # Spell correction
    text = spell(text)
    
    # Lemmatization
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    
    # Stopword removal (with category-based exceptions)
    stop = set(stopwords.words("english"))
    negative_words = {'no', 'not'}
    interrogative_words = {'how', 'what', 'which', 'who', 'whom', 'why', 'do', 'does', 'is', 'are'}
    imperative_words = {'could', 'would', 'should', 'can'}

    if sentiment_label == "negative":
        filtered_words = [word for word in text.split() if word not in stop or word in negative_words]
    elif sentiment_label == "interrogative":
        filtered_words = [word for word in text.split() if word not in stop or word in interrogative_words]
    elif sentiment_label == "imperative":
        filtered_words = [word for word in text.split() if word not in stop or word in imperative_words]
    else:  # For positive, correction, and other categories
        filtered_words = [word for word in text.split() if word not in stop]
    
    return " ".join(filtered_words)


def preprocess_comments(comments):
    df = pd.DataFrame(comments)
    df["cleaned_text"] = df["Comment_Text"].apply(lambda x: clean_text(x) if x else "")
    return df[df["cleaned_text"] != ""]

def analyze_sentiment(text):
    try:
        text_vectorized = vectorizer.transform([text])
        return model.predict(text_vectorized)[0]
    except:
        return "Neutral"

def apply_sentiment_analysis(cleaned_comments):
    cleaned_comments["Sentiments"] = cleaned_comments["cleaned_text"].apply(analyze_sentiment)
    return cleaned_comments

def apply_category_classification(cleaned_comments):
    batch_size = 64
    category_model.eval()
    all_predictions = []

    for i in range(0, len(cleaned_comments), batch_size):
        batch_texts = cleaned_comments["cleaned_text"].iloc[i:i+batch_size].tolist()
        tokens = tokenizer(batch_texts, padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)

        with torch.no_grad():
            outputs = category_model(**tokens)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_predictions.extend(label_encoder.inverse_transform(predictions))

    cleaned_comments["Category"] = all_predictions
    return cleaned_comments

# Extract keywords using TF-IDF
def extract_keywords(text):
    try:
        tfidf_matrix = vectorizer_tfidf.fit_transform([text])
        feature_array = vectorizer_tfidf.get_feature_names_out()
        sorted_indices = tfidf_matrix.toarray().argsort()[0][-2:]
        return " ".join([feature_array[i] for i in sorted_indices])
    except ValueError:
        return ""

def apply_keyword_extraction(cleaned_comments):
    cleaned_comments["Keywords"] = cleaned_comments["cleaned_text"].apply(extract_keywords)
    return cleaned_comments

def fetch_process_analyze(youtube_url):
    video_id = extract_video_id(youtube_url)
    if video_id:
        comments = fetch_youtube_comments(video_id)
        cleaned_comments = preprocess_comments(comments)

        if not cleaned_comments.empty:
            cleaned_comments["Time_Stamp"] = pd.to_datetime(cleaned_comments["Time_Stamp"], errors='coerce')
            cleaned_comments = apply_sentiment_analysis(cleaned_comments)
            cleaned_comments = apply_category_classification(cleaned_comments)
            cleaned_comments = apply_keyword_extraction(cleaned_comments)
            return cleaned_comments
            
    return pd.DataFrame()

# ==========================
# 📊 Streamlit Dashboard
# ==========================
st.set_page_config(layout="wide")
st.title("📊 Real-Time YouTube Comments Analyzer")

st.sidebar.subheader("🔗 Enter YouTube Video URL")
youtube_url = st.sidebar.text_input("Paste YouTube URL and press Enter")
if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()


# Progress bar placeholders
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
time_taken = {}

df_final = pd.DataFrame()

if youtube_url:
    start_time = time.time()
    status_text.text("🔄 Fetching comments...")
    video_id = extract_video_id(youtube_url)
    title, author, _ = fetch_video_details(video_id)
    df_final = fetch_process_analyze(youtube_url)
    
    time_taken['fetching'] = time.time() - start_time
    progress_bar.progress(25)
    status_text.text("🔄 Processing comments...")
    time_taken['processing'] = time.time() - start_time
    progress_bar.progress(50)
    status_text.text("🔄 Applying Sentiment Analysis...")
    time_taken['sentiment'] = time.time() - start_time
    progress_bar.progress(75)
    status_text.text("✅ Processing Complete!")
    time_taken['total'] = time.time() - start_time
    progress_bar.progress(100)
    st.sidebar.success("✅ Data Fetched & Processed!")

    st.markdown(f"## 🎥 {title}")
    st.markdown(f"**Channel:** {author}")
    
    category_counts = df_final["Category"].value_counts()
    sentiment_counts = df_final.groupby("Category")["Sentiments"].value_counts().unstack(fill_value=0)
    pivot_df = df_final.pivot_table(index='Sentiments', columns='Category', aggfunc='size', fill_value=0)
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Sentiments")
        sentiment_counts = df_final["Sentiments"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, wedgeprops={'width': 0.3})
        ax.axis('equal')  
        st.pyplot(fig)

    with col2:
        st.subheader("📌 Categories")
        category_counts = df_final["Category"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90, wedgeprops={'width': 0.3})
        ax.axis('equal')  
        st.pyplot(fig)
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔹 Sentiments Over Time")
        sentiment_counts = df_final.groupby(df_final['Time_Stamp'].dt.date)['Sentiments'].value_counts().unstack().fillna(0)
        st.line_chart(sentiment_counts)

    with col2:
        st.subheader("📊 #Comments per Category Distribution of Sentiments")
        pivot_df = df_final.pivot_table(index='Category', columns='Sentiments', values='Comment_Id', aggfunc='count', fill_value=0)
        pivot_df_percentage = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
        fig, ax = plt.subplots(figsize=(8, 5))
        pivot_df_percentage.plot(kind='barh', stacked=True, ax=ax, colormap="Set2")
        ax.set_xlabel("Percentage")
        ax.set_title("#Comments per Category Distribution of Sentiments")
        ax.legend(title="Sentiments", bbox_to_anchor=(1.05, 1), loc="upper left")
        st.pyplot(fig)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔹 Sentiment-Category Table")
        sentiment_category_pivot = df_final.pivot_table(index='Sentiments', columns='Category', aggfunc='size', fill_value=0)
        st.dataframe(sentiment_category_pivot)

    with col2:
        st.subheader("🔹 Top 5 Keywords per Category & Sentiment")
        def get_top_keywords(df, sentiment, category):
            keywords_series = df[(df['Sentiments'] == sentiment) & (df['Category'] == category)]['Keywords'].dropna()
            all_keywords = ' '.join(keywords_series).split()
            top_keywords = [word for word, count in Counter(all_keywords).most_common(5)]
            return ', '.join(top_keywords) if top_keywords else "N/A"
        unique_sentiments = df_final["Sentiments"].unique()
        unique_categories = df_final["Category"].unique()
        keywords_data = []
        for sentiment in unique_sentiments:
            row = {}
            for category in unique_categories:
                row[category] = get_top_keywords(df_final, sentiment, category)
            keywords_data.append([sentiment] + list(row.values()))
        keywords_pivot_df = pd.DataFrame(keywords_data, columns=["Sentiments"] + list(unique_categories))
        st.dataframe(keywords_pivot_df)
    
    with st.container():
        st.subheader("🔹 Keywords Word Cloud")
        text = " ".join(df_final["Keywords"].dropna())
        if text.strip():
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)
        else:
            st.write("⚠ No keywords available to generate a word cloud.")

    with st.container():
        st.subheader("🔹 Keywords Word Cloud by Category")

        categories = df_final["Category"].unique()  # Get unique categories

        if len(categories) > 0:
            cols = st.columns(2)  # Create two columns for better visualization
            for i, category in enumerate(categories):
                with cols[i % 2]:  # Distribute charts across columns
                    st.subheader(f"📌 {category}")
                    category_text = " ".join(df_final[df_final["Category"] == category]["Keywords"].dropna())
                    if category_text.strip():
                        wordcloud = WordCloud(width=600, height=300, background_color='white').generate(category_text)
                        plt.figure(figsize=(8, 4))
                        plt.imshow(wordcloud, interpolation="bilinear")
                        plt.axis("off")
                        st.pyplot(plt)
                    else:
                        st.write(f"⚠ No keywords available for {category}.")
        else:
            st.write("⚠ No categories available to generate word clouds.")

    
    st.subheader("🆕 All Processed Comments")
    st.dataframe(df_final, height=600)
    
    save_filename = st.sidebar.text_input("Enter filename (without extension)", "processed_comments")
    if st.sidebar.button("Save CSV"):
        df_final.to_csv(f"{save_filename}.csv", index=False)
        st.sidebar.success(f"✅ Data saved as {save_filename}.csv")

