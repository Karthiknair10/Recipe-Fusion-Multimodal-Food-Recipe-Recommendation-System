import json
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
model = load_model('image_features.json')
target_img = os.path.join(os.getcwd(), 'static/images')


@app.route('/')
def index_view():
    return render_template('index.html')


# Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg', 'jpeg', 'png'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


# Function to load and prepare the image in right shape
def read_image(filename):
    img = load_img(filename, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


@app.route('/predict', methods=['GET', 'POST'])
def recommend_recipes_by_image_similarity(recipe_id, num_recommendations=5):
    # Get the similarity scores for the target recipe
    target_similarity_scores = model[recipe_id]

    # Get the indices of recipes sorted by image similarity (excluding the target recipe itself)
    similar_recipe_indices = sorted(range(len(target_similarity_scores)), key=lambda i: target_similarity_scores[i], reverse=True)[1:]

    # Get the top N recommended recipe indices
    recommended_indices = similar_recipe_indices[:num_recommendations]

    # Return the recommended recipes
    recommended_recipes = data.iloc[recommended_indices]
    return recommended_recipes[['name', 'ingredients', 'instructions']]

# Example: Recommend recipes similar to recipe at index 0
recommended_recipes = recommend_recipes_by_image_similarity(0, num_recommendations=5)
print("Recommended Recipes based on Image Similarity:")
print(recommended_recipes)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8000)