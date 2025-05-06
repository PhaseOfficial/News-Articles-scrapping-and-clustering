import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import nltk

# Download all required NLTK resources with error handling
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Set page config
st.set_page_config(page_title="News Story Clustering", layout="wide")

@st.cache_data
def load_data():
    """Load sample data or user-uploaded data"""
    sample_data = {
        'title': ['Market Trends 2023', 'Election Update', 'Art Exhibition Review', 
                 'Sports Championship', 'Tech Innovation', 'Political Debate'],
        'content': ['The stock market shows upward trends this quarter...', 
                   'The election results are coming in from all districts...',
                   'New art exhibition opens downtown featuring modern artists...',
                   'Local team wins championship after intense finals...',
                   'New tech product announced with revolutionary features...',
                   'Candidates debate key issues in televised session...'],
        'url': ['https://example.com/market', 'https://example.com/election',
               'https://example.com/art', 'https://example.com/sports',
               'https://example.com/tech', 'https://example.com/debate'],
        'category': ['business', 'politics', 'arts', 'sports', 'business', 'politics']
    }
    return pd.DataFrame(sample_data)

@st.cache_data
def preprocess_text(text):
    """Clean and preprocess text content with robust error handling"""
    try:
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Basic cleaning
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenization with fallback
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        # Lemmatization and stopword removal
        processed_tokens = []
        for word in tokens:
            if word.isalpha() and word not in stop_words:
                try:
                    lemma = lemmatizer.lemmatize(word)
                    processed_tokens.append(lemma)
                except:
                    processed_tokens.append(word)
        
        return ' '.join(processed_tokens)
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")
        return ""

def perform_clustering(df, n_clusters=4):
    """Cluster news stories based on content with error handling"""
    try:
        # Preprocess content
        df['processed_content'] = df['content'].apply(preprocess_text)
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X = vectorizer.fit_transform(df['processed_content'])
        
        # Cluster using K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(X)
        
        df['cluster'] = clusters
        return df, vectorizer, X
    
    except Exception as e:
        st.error(f"Clustering failed: {str(e)}")
        return df, None, None

def main():
    st.title("📰 News Story Clustering Platform")
    st.markdown("Group similar news stories using AI clustering")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        n_clusters = st.slider("Number of clusters", 2, 10, 4)
        uploaded_file = st.file_uploader("Upload your news data (CSV)", type="csv")
    
    # Load data
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            required_columns = {'title', 'content', 'url'}
            if not required_columns.issubset(df.columns):
                st.error(f"CSV must contain these columns: {required_columns}")
                st.stop()
            st.success("Uploaded data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.stop()
    else:
        df = load_data()
        st.info("Using sample data. Upload a CSV file to use your own data.")
    
    # Display raw data
    with st.expander("View Raw Data"):
        st.dataframe(df)
    
    # Cluster the stories
    clustered_df, vectorizer, X = perform_clustering(df, n_clusters)
    
    if vectorizer is None or X is None:
        st.error("Clustering failed. Please check your data and try again.")
        st.stop()
    
    # Show cluster distribution
    st.subheader("Cluster Distribution")
    cluster_counts = clustered_df['cluster'].value_counts().sort_index()
    st.bar_chart(cluster_counts)
    
    # Display clusters
    st.subheader("Explore Clusters")
    
    # Cluster selector
    selected_cluster = st.selectbox(
        "Select a cluster to explore:",
        sorted(clustered_df['cluster'].unique()))
    
    # Get stories in selected cluster
    cluster_data = clustered_df[clustered_df['cluster'] == selected_cluster]
    
    # Display cluster summary
    st.markdown(f"### Cluster {selected_cluster} Summary")
    st.markdown(f"**Number of stories:** {len(cluster_data)}")
    
    # Show top terms
    cluster_indices = cluster_data.index
    cluster_features = X[cluster_indices].sum(axis=0).A1
    top_indices = cluster_features.argsort()[-5:][::-1]
    top_terms = [vectorizer.get_feature_names_out()[i] for i in top_indices]
    st.markdown(f"**Key terms:** {', '.join(top_terms)}")
    
    # Display category distribution if available
    if 'category' in cluster_data.columns:
        st.markdown("**Category distribution:**")
        category_counts = cluster_data['category'].value_counts()
        st.bar_chart(category_counts)
    
    # Display stories in the cluster
    st.markdown("### Stories in this Cluster")
    for _, row in cluster_data.iterrows():
        with st.container(border=True):
            st.markdown(f"#### {row['title']}")
            if 'category' in row:
                st.markdown(f"*Category: {row['category']}*")
            st.markdown(f"{row['content'][:200]}...")
            st.markdown(f"[Read more]({row['url']})", unsafe_allow_html=True)

if __name__ == "__main__":
    main()