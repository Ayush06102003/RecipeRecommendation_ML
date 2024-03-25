import pickle, json, _json
from flask import Flask, render_template, jsonify, request
from recipe import get_similar_top5, get_details_top5, seg4

app = Flask(__name__)

seg4_vectorizer = pickle.load(open('seg4_vectorizer.pkl', 'rb'))
seg4_tfidf_matrix = pickle.load(open('seg4_tfidf_matrix.pkl', 'rb'))



@app.route('/predict', methods=['POST'])
def predict():
    # Receive the input data (tasks) from the frontend
    tasks_data = request.data.decode('utf-8')  # Decode bytes to string
    tasks_list = tasks_data.split(',')  # Assuming tasks are comma-separated
    
    # Process the tasks list as needed
    print("Received tasks:", tasks_list)
    
    # Use the loaded vectorizer and TF-IDF matrix to make recommendations
    similar_top5_indices = get_similar_top5(seg4_vectorizer, seg4_tfidf_matrix, tasks_list)
    recommended_details = get_details_top5(seg4, similar_top5_indices)
    
    # Return the top 5 recommended recipes as JSON to the frontend
    recommended_list = recommended_details.to_dict(orient='records')
    
    # Print the recommended list
    print("Recommended list:", recommended_list)
    
    return render_template('recipe.html', recommended_list=recommended_list)

    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    pass

if __name__ == "__main__":
    app.run(debug=True)
