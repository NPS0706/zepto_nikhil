import pandas as pd

# Loading the dataset
df = pd.read_csv('flipkart_com-ecommerce_sample.csv')

# Filling missing 'brand' with 'Unknown'
df['brand'].fillna('Unknown', inplace=True)

# Dropping rows with missing 'retail_price' or 'discounted_price'
df.dropna(subset=['retail_price', 'discounted_price'], inplace=True)

# Dropping rows with missing 'description'
df.dropna(subset=['description'], inplace=True)

# Combining product name and description for better search results
df['combined_text'] = df['product_name'] + " " + df['description']

# Saving the cleaned dataset
df.to_csv('cleaned_data.csv', index=False)
