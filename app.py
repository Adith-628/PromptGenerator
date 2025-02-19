# from flask import Flask,render_template,request,jsonify
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

# app = Flask(__name__)

# # Load the GPT-2 model and tokenizer
# tokenizer  = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2")

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/expand', methods=['POST'])
# def expand_prompt():
#     data = request.json
#     short_prompt = data.get("prompt")
    
#     if not short_prompt:
#         return jsonify({"error": "No prompt provided"}), 400
    
#     # encode input and generated text
#     input_ids = tokenizer.encode(short_prompt, return_tensors='pt')
#     outputs = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)
    
#     # decode the generated text
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
#     return jsonify({"detailed_prompt": generated_text})


# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load the GPT-2 model and tokenizer locally
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/expand", methods=["POST"])
def expand_prompt():
    data = request.json
    short_prompt = data.get("prompt")

    if not short_prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # Encode input and generate text
    input_ids = tokenizer.encode(short_prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)

    # Decode the generated text
    expanded_prompt = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({"detailed_prompt": expanded_prompt})

if __name__ == "__main__":
    app.run(debug=True)
