import pickle
from flask import Flask, render_template, request, jsonify, session
from recipe import get_similar_top5, get_details_top5, seg4

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for sessions

seg4_vectorizer = pickle.load(open('seg4_vectorizer.pkl', 'rb'))
seg4_tfidf_matrix = pickle.load(open('seg4_tfidf_matrix.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    tasks_data = request.json.get('tasks', [])  
    tasks_list = [task.strip() for task in tasks_data if task.strip()] 
    
    similar_top5_indices = get_similar_top5(seg4_vectorizer, seg4_tfidf_matrix, tasks_list)
    recommended_details = get_details_top5(seg4, similar_top5_indices)
    
    recommended_list = recommended_details.to_dict(orient='records')
    
    # Store the recommended_list data in the session
    session['recommended_list'] = recommended_list
    
    return render_template('recipe.html', recommended_list=recommended_list)

@app.route('/display', methods=['GET'])
def display():
    # Retrieve the recommended_list data from the session
    recommended_list = session.get('recommended_list', [])
    return render_template('recipe.html', recommended_list=recommended_list)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    pass
    pass

if __name__ == "__main__":
    app.run(debug=True)
