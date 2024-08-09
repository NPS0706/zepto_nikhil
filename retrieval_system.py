from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load the cleaned dataset
df = pd.read_csv('cleaned_data.csv')

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

# Fit and transform the combined text data
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

# Function to retrieve and rank products based on a search term
def retrieve_products(query, top_n=10):
    # Transform the search query using the same TF-IDF vectorizer
    query_vec = vectorizer.transform([query])
    
    # Calculate cosine similarity between the query and all products
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get the top n most similar products
    top_indices = cosine_similarities.argsort()[-top_n:][::-1]
    
    # Return the top n most relevant products
    return df.iloc[top_indices][['product_name', 'brand', 'retail_price', 'discounted_price', 'product_url']]

# Example usage
if __name__ == "__main__":
    example_results = retrieve_products("sofa", top_n=5)
    print(example_results)
