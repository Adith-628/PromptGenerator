from flask import Flask, request, jsonify, render_template
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import random
from transformers import pipeline

app = Flask(__name__)

# Load Transformers Model (Using GPT-2 for text generation)
generator = pipeline("text-generation", model="gpt2")

# Default sentence expansion templates
default_expansion_templates = [
    " This scene is filled with intricate details, enhancing the overall experience.",
    " The environment is carefully designed to create an immersive and realistic atmosphere.",
    " With its dynamic elements and rich textures, this setting feels alive and engaging.",
    " The visual composition is enhanced with subtle lighting and natural movements."
]

# Function to expand a prompt using NLTK
def expand_prompt_nltk(prompt):
    words = word_tokenize(prompt)  # Tokenize input text
    expanded_sentence = []

    for word in words:
        synonyms = wordnet.synsets(word)  # Get synonyms
        if synonyms:
            synonym = synonyms[0].lemmas()[0].name().replace("_", " ")  # Pick first synonym
            expanded_sentence.append(synonym)
        else:
            expanded_sentence.append(word)

    expanded_prompt = " ".join(expanded_sentence)
    additional_context = random.choice(default_expansion_templates)
    return expanded_prompt + additional_context

# Function to expand a prompt using Transformers
def expand_prompt_transformers(prompt):
    output = generator(
        f"Rewrite this prompt to be more detailed and descriptive, while keeping its original meaning but more explained:\n\n{prompt}",
        max_length=100, 
        num_return_sequences=1,
        do_sample=True,  # Enable sampling for variation
        temperature=0.8,  # Controls randomness (higher = more diverse)
        top_p=0.9  # Nucleus sampling for controlled creativity
    )
    return output[0]['generated_text'].strip()


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/expand", methods=["POST"])
def expand():
    data = request.json
    short_prompt = data.get("prompt")

    if not short_prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # Generate detailed prompts using both methods
    # nltk_expansion = expand_prompt_nltk(short_prompt)
    transformers_expansion = expand_prompt_transformers(short_prompt)

    return jsonify({
        # "nltk_expansion": nltk_expansion,
        "transformers_expansion": transformers_expansion
    })

if __name__ == "__main__":
    app.run(debug=True)
