import pickle
from flask import Flask, render_template, redirect, url_for, request, jsonify, session
from recipe import get_similar_top5, get_details_top5,search_top5,filtered_recipes_df,seg1,seg2,seg3, seg4

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for sessions

seg4_vectorizer = pickle.load(open('seg4_vectorizer.pkl', 'rb'))
seg4_tfidf_matrix = pickle.load(open('seg4_tfidf_matrix.pkl', 'rb'))

seg3_vectorizer = pickle.load(open('seg3_vectorizer.pkl', 'rb'))
seg3_tfidf_matrix = pickle.load(open('seg3_tfidf_matrix.pkl', 'rb'))

seg2_vectorizer = pickle.load(open('seg2_vectorizer.pkl', 'rb'))
seg2_tfidf_matrix = pickle.load(open('seg2_tfidf_matrix.pkl', 'rb'))

seg1_vectorizer = pickle.load(open('seg1_vectorizer.pkl', 'rb'))
seg1_tfidf_matrix = pickle.load(open('seg1_tfidf_matrix.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    category = request.json.get('category')
    print(category )
    
    tasks_data = request.json.get('tasks', [])  
    tasks_list = [task.strip() for task in tasks_data if task.strip()] 
    print(tasks_list)
    if(category == "Less than 40 minutes & High Calory" ):
        svc=seg2_vectorizer
        stm=seg2_tfidf_matrix
        ss=seg2
    
    elif(category == "Less than 40 minutes & Low Calory" ):
        svc=seg1_vectorizer
        stm=seg1_tfidf_matrix
        ss=seg1
    
    elif(category == "More than 40 minutes & High Calory" ):
        svc=seg4_vectorizer
        stm=seg4_tfidf_matrix
        ss=seg4
    
    else:
        svc=seg3_vectorizer
        stm=seg3_tfidf_matrix
        ss=seg3
            
    
    similar_top5_indices = get_similar_top5(svc, stm, tasks_list, ss)
    recommended_details = get_details_top5(ss, similar_top5_indices)
    
    recommended_list = recommended_details.to_dict(orient='records')
    
    # Store the recommended_list data in the session
    session['recommended_list'] = recommended_list
    
    return redirect(url_for('display'))

@app.route('/display', methods=['GET'])
def display():
    # Retrieve the recommended_list data from the session
    recommended_list = session.get('recommended_list', [])
    return render_template('recipe.html', recommended_list=recommended_list)


@app.route('/search', methods=['POST'])
def search():
    user_input = request.form['search_input']
    
    recommended_details=search_top5(user_input,filtered_recipes_df)
    
    recommended_list = recommended_details.to_dict(orient='records')
    
    # Store the recommended_list data in the session
    session['recommended_list'] = recommended_list
    
    return redirect(url_for('display'))


@app.route('/')
def home():
    return render_template('index.html')
    

if __name__ == "__main__":
    app.run(debug=True)
