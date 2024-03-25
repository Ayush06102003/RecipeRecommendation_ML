from flask import Flask, render_template, redirect, url_for, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if request.method == 'POST':
        # Here you can handle the search functionality if needed
        # For now, let's simply redirect to the recipe page
        return render_template('recipe.html')
        

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Here you can handle the search functionality if needed
        # For now, let's simply redirect to the recipe page
        return render_template('recipe.html')

if __name__ == "__main__":
    app.run(debug=True)
