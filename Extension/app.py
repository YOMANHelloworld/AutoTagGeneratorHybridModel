from flask import Flask, request, jsonify
import pickle
import torch
from transformers import BertTokenizer, BertModel
from flask_cors import CORS
import nltk

nltk.download('punkt')
from nltk.corpus import stopwords

nltk.download('stopwords')
stopwords_set = set(stopwords.words('english'))

with open('SavedModel/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('SavedModel/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load BERT model
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Function to predict keywords for a given title
def predict_keywords(titles):

    title = titles[0]

    filtered_text = title.lower()
    tokens = nltk.word_tokenize(filtered_text)

    filtered_tokens = [token for token in tokens if token not in stopwords_set]

    filtered_text = ' '.join(filtered_tokens)

    # Tokenize and pad the input title for BERT
    title_tokens = tokenizer.encode(filtered_text, max_length=128, truncation=True, padding='max_length', return_tensors='pt')

    # Get BERT embeddings
    with torch.no_grad():
        title_embedding = bert_model(title_tokens).last_hidden_state.mean(dim=1)

    # TF-IDF vectorization
    title_tfidf = vectorizer.transform([title])

    # Concatenate BERT embeddings with TF-IDF features
    title_combined = torch.cat([torch.tensor(title_tfidf.toarray(), dtype=torch.float32), title_embedding], dim=1)

    # Make prediction using the trained model
    predicted_keywords = model.predict(title_combined).tolist()  # Convert to list
    return predicted_keywords

def predict_tags(titles):
    
    X = predict_keywords(titles)
    
    return X

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.json
        input_titles = json_data.get('titles', [])

        predicted_tags = predict_keywords(input_titles)
        response_data = {'predicted_tags': predicted_tags}
        
        response = jsonify(response_data)
        response.headers.add('Access-Control-Allow-Origin', '*')  # Allow requests from any origin
        return response
    except Exception as e:
        print(f"Error processing request: {e}")
        error_response = {'error': 'An error occurred'}
        response = jsonify(error_response)
        response.headers.add('Access-Control-Allow-Origin', '*')  # Allow requests from any origin
        return response

if __name__ == '__main__':
    app.run()