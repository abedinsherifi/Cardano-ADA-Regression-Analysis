#import model # Import the python file containing the ML model
import cardano_model
from flask import Flask, request, render_template,jsonify # Import flask libraries

# Initialize the flask class and specify the templates directory
app = Flask(__name__,template_folder="templates")

# Default route set as 'home'
@app.route('/dino')
def home():
    return render_template('home.html') # Render home.html

# Route 'classify' accepts GET request
@app.route('/classify',methods=['POST','GET'])
def classify_type():
    try:
        a = request.args.get('Date') # Get parameters for sepal length
        b = request.args.get('High') # Get parameters for sepal width
        c = request.args.get('Low') # Get parameters for petal length
        d = request.args.get('Open') # Get parameters for petal width
        e = request.args.get('Volume') # Get parameters for petal width
        f = request.args.get('Marketcap') # Get parameters for petal width
        print(a)
        # Get the output from the classification model
        variety = cardano_model.classify(a, b, c, d, e, f)
        print(variety)

        # Render the output in new HTML page
        return render_template('output.html', variety=variety)
    except:
        return 'Error'

# Run the Flask server
if(__name__=='__main__'):
    app.run(debug=False) 
