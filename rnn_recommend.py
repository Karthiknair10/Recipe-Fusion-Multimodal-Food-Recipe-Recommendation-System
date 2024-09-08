# import keras
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow import keras

model = keras.models.load_model('model_recommend.h5')

print(model.summary())
data = pd.read_csv('cuisine_updated.csv')

# Preprocess the data
text = data['instructions']
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
total_words = len(tokenizer.word_index) + 1
input_sequences = []
for line in text:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre'))

X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = y.reshape(y.shape[0], 1)




# Function to generate text given a seed text
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



# Function to predict recipe instructions and ingredients based on the recipe name
def predict_recipe(recipe_name, model, max_sequence_length):
    recommended_recipe = generate_text(recipe_name, next_words=100, model=model, max_sequence_length=max_sequence_length)
    return recommended_recipe

# User input for the recipe name
# user_recipe_name = input("Enter a recipe name: ")

# Generate recommendations based on the user's input
# recommended_recipe = predict_recipe(user_recipe_name, model, max_sequence_length)
# print("Recommended Recipe Instructions and Ingredients:")
# print(recommended_recipe)


