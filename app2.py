import keras
import numpy as np
import pandas as pd
from flask import Flask, request, render_template


import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


nltk.download('punkt')
nltk.download('stopwords')

import mysql.connector
import json
import pandas as pd
from PIL import Image
import requests
from IPython.display import display
import keras
model = keras.models.load_model('fusionrecipe_rnn.h5')
data = pd.read_csv('fusionrecipe.csv')
data = data.rename_axis('recipe_id').reset_index()

image_path = "C:/Users/sanjo/Desktop/Project/recfus_img/"

with open('image_feat.json', 'r') as json_file:
    loaded_data = json.load(json_file)

def recommend_recipes_by_image_similarity(recipe_id, num_recommendations=5):
    target_similarity_scores = loaded_data[recipe_id]

    similar_recipe_indices = sorted(range(len(target_similarity_scores)), key=lambda i: target_similarity_scores[i], reverse=True)[1:]

    recommended_indices = similar_recipe_indices[:num_recommendations]

    recommended_recipes = data.iloc[recommended_indices]
    return recommended_recipes[['name', 'calories & servings', 'ingredients', 'instruction']]


connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Sanjog02",
    database="Register"
    )

try:
    cursor = connection.cursor()
except:
    print("connection failed")


# model = load_model('fusionrecipe_rnn.keras')
#model = keras.models.load_model('fusionrecipe_rnn.keras')
data = pd.read_csv('fusionrecipe.csv')
# Convert any non-string data to string
data['instruction'] = data['instruction'].astype(str)

# Preprocess the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['instruction'])
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in data['instruction']:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre'))

X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = y.reshape(y.shape[0],1)
# tokenizer=Tokenizer()
# text = data['instructions']
# tokenizer.fit_on_texts(text)
# total_words = len(tokenizer.word_index) + 1
# input_sequences = []
#
# for line in text:
#     token_list = tokenizer.texts_to_sequences([line])[0]
#     for i in range(1, len(token_list)):
#         n_gram_sequence = token_list[:i+1]
#         input_sequences.append(n_gram_sequence)
#
# max_sequence_length = max([len(x) for x in input_sequences])



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/tags')
def tags():
    return render_template('tags.html')
@app.route('/login_index')
def login_index():
    return render_template('login_index.html')
@app.route('/Login',methods=['GET', 'POST'])
def Login():
    if request.method == "POST":
        email= request.form.get("email")
        password = request.form.get("password")
        insert_query = "SELECT * FROM SignUp where email_id = %s and password = %s"
        print(email, password)
        values = (email, password)

        # Execute the query
        cursor.execute(insert_query, values)
        rows = cursor.fetchall()

        if not rows:
            print("Not login")
        else:
            return render_template('index.html')

    return render_template('login_index.html')

@app.route('/signup',methods=['GET', 'POST'])
def signup():
    if request.method == "POST":
        user_name = request.form.get("user_name")
        email= request.form.get("email")
        password = request.form.get("password")
        insert_query = "INSERT INTO SignUp (name, email_id, password) VALUES (%s, %s, %s)"
        print(user_name, email, password)
        values = (user_name, email, password)

        cursor.execute(insert_query, values)
        print("1234")
        cursor.execute("SELECT * FROM SignUp")
        rows = cursor.fetchall()

        for row in rows:
            print(row)


        return render_template('login_index.html')
@app.route('/recipes')
def recipes():
    return render_template('recipes.html')
@app.route('/recommend_recipe', methods=['POST', 'GET'])
def recommend_recipe():

    if request.method == 'POST':
        form_data = request.form
        recipe_name = form_data['recipe_name']
        print("recipe_name")

        print(recipe_name)
        recipe_model = form_data['Model']

        recommended_recipe = predict_recipe(recipe_name, model, max_sequence_length)
        recipe_name_split = list(recipe_name.split(" "))
        id = 20
        if recipe_model == "CNN":

            for recipe in data["name"]:
                if recipe_name_split[0] in list(recipe.split(" ")):
                    print(recipe_name_split)
                    id = data[data["name"]== recipe]["image_id"]
                    break
            recipe = recommend_recipes_by_image_similarity(int(id))
            print(recipe)
            name = recipe["name"]
            calories_servings = recipe["calories & servings"]
            ingredients = recipe["ingredients"]
            instruction = recipe["instruction"]
            my_dict = dict(zip(name, instruction))

            # combined_instructions = []
            # for i in range(len(instruction)):
            #     combined_instructions.append(f"{name[i]}: - {instruction[i]}")

            return render_template('predict.html', recommended_recipe=my_dict,instruction=instruction, recipe_name=recipe_name, model = recipe_model)
        else:
            print("Recommended Recipe Instructions and Ingredients:")
            print(recommended_recipe)
            data[data["instruction"]== recommended_recipe]
            split_data=recipe_name.split()
            for dat1 in split_data:
                for dat in data:
                    if dat1 in dat["ingrident"]:
                        print(dat["name"])

            modified_recipe_name = recommended_recipe.replace(recipe_name, 'recommended_recipe')
            rec = data[data["instruction"].str.startswith(modified_recipe_name)]
            print("Modified Recipe Name:", modified_recipe_name)
            print(rec)
            recommended_recipe = recommend_recipes_by_image_similarity(20, num_recommendations=5)

            return render_template('predict.html', recommended_recipe=recommended_recipe, recipe_name=recipe_name, model = recipe_model)

    else:
        recipe_name = request.args.get('recipe_name')
        recommended_recipes = predict_recipe(recipe_name, model, max_sequence_length)
        print("Recommended Recipe Instructions and Ingredients:")
        print(recommended_recipes)
        data[data["instruction"] == recommended_recipes]
        modified_recipe_name = recommended_recipes.replace(recipe_name, 'recommended_recipe')
        rec = data[data["instruction"].str.startswith(modified_recipe_name)]
        print("Modified Recipe Name:", modified_recipe_name)
        print(rec)
        recommended_recipes = recommend_recipes_by_image_similarity(20, num_recommendations=5)

        return render_template('predict.html', recommended_recipe=recommended_recipes, recipe_name=recipe_name)
def predict_recipe(recipe_name, model, max_sequence_length):
    recommended_recipe = generate_text(recipe_name, next_words=100, model=model, max_sequence_length=max_sequence_length)
    return recommended_recipe
def generate_text(seed_text, next_words, model, max_sequence_length):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted = np.argmax(predicted_probs)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text




if __name__ == '__main__':
    app.run(debug=True)
