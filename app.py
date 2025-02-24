import gdown
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download dataset from Google Drive
# csv_file_id = "1Yw0BQRKypCh0d-DKaX5uQLgH4qb9Owt2"  # Replace with actual ID
# csv_url = f"https://drive.google.com/uc?id={csv_file_id}"
# gdown.download(csv_url, "full_dataset.csv", quiet=False)

# Load dataset
df = pd.read_csv("small_dataset.csv")

# Preprocess ingredients
def preprocess_ingredients(ingredient_list):
    if isinstance(ingredient_list, str):
        return ingredient_list
    elif isinstance(ingredient_list, list):
        return ' '.join(ingredient_list)
    return ""

df['NER'] = df['NER'].apply(preprocess_ingredients)

# Vectorize ingredients
vectorizer = CountVectorizer()
ingredient_matrix = vectorizer.fit_transform(df['NER'])

def recommend_recipes(user_ingredients, top_n=5):
    user_input = ' '.join(user_ingredients)
    user_vector = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vector, ingredient_matrix)
    top_indices = similarities.argsort()[0][-top_n:][::-1]
    return df.iloc[top_indices][['title', 'ingredients', 'directions']].to_dict(orient='records')

app = Flask(__name__)

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()
    if not data or "ingredients" not in data:
        return jsonify({"error": "Invalid input"}), 400

    user_ingredients = data["ingredients"]
    recommendations = recommend_recipes(user_ingredients)

    return jsonify({"recommended_recipes": recommendations})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
