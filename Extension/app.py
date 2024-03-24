from flask import Flask, request, jsonify
import pickle
import torch
from transformers import BertTokenizer, BertModel
from flask_cors import CORS

with open('SavedModel/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('SavedModel/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load BERT model
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Function to preprocess input titles
def preprocess_titles(titles):
    # Tokenize and pad the input data for BERT
    tokens = [tokenizer.encode(title, max_length=128, truncation=True, padding='max_length', return_tensors='pt')[0] for title in titles]
    tokens = torch.stack(tokens)
    
    # Get BERT embeddings
    with torch.no_grad():
        embeddings = bert_model(tokens).last_hidden_state.mean(dim=1)
    
    # Convert titles to TF-IDF features
    tfidf_features = vectorizer.transform(titles)
    
    # Concatenate BERT embeddings with TF-IDF features
    combined_features = torch.cat([torch.tensor(tfidf_features.toarray(), dtype=torch.float32), embeddings], dim=1)
    
    return combined_features

def predict_tags(titles):
    # Preprocess titles
    X = preprocess_titles(titles)
    
    # Make predictions using the loaded model
    predictions = model.predict(X)
    
    return predictions.tolist()

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.json
        input_titles = json_data.get('titles', [])

        predicted_tags = predict_tags(input_titles)
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