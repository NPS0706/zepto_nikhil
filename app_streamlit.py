import streamlit as st
import requests

st.title('Product Search Comparison')
search_method = st.radio("Choose a search method:", ["TF-IDF", "Keyword"])
query = st.text_input("Enter your search query:")
top_n = st.slider("Number of results:", 1, 20, 10)

if st.button('Search'):
    endpoint = f"http://localhost:8000/search/{search_method.lower()}/"
    response = requests.post(endpoint, json={'query': query, 'top_n': top_n})
    
    results = response.json()
    if results:
        for product in results:
            st.subheader(product['product_name'])
            st.write(f"URL: {product['product_url']}")
            st.write("---")
    else:
        st.error("No products found.")
