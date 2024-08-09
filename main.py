from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator

app = FastAPI()

df = pd.read_csv('flipkart_com-ecommerce_sample.csv')
df.fillna('', inplace=True)
df['combined'] = df['product_name'] + ' ' + df['description']

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined'])

class SearchRequest(BaseModel):
    query: str
    top_n: int = 10

translator = Translator()

@app.post("/search/tfidf/")
def search_tfidf(request: SearchRequest):
    translated_query = translator.translate(request.query, dest='en').text
    query_vec = vectorizer.transform([translated_query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[-request.top_n:][::-1]
    results = df.iloc[top_indices]
    return results[['product_name', 'product_url']].to_dict(orient='records')

@app.post("/search/keyword/")
def search_keyword(request: SearchRequest):
    pattern = request.query.lower()
    matches = df[df['combined'].str.lower().str.contains(pattern)]
    return matches.head(request.top_n)[['product_name', 'product_url']].to_dict(orient='records')


# Run the application
# To run: `uvicorn main:app --reload` via FastAPI 
    # Link: http://127.0.0.1:8000/search/?query=search_item_name
    # Link: http://127.0.0.1:8000/docs  for to test different search terms and view the results in a user-friendly interface.
# To run: 'streamlit run app_streamlit.py' VIA SteamLit
    # Link: http://localhost:8501/
