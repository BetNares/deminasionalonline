from flask import Flask, jsonify
import os
from flask import Flask, render_template
from flask import Flask, render_template, request
import classifier

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")  

@app.route('/classify', methods=['POST'])
def classify():
    if (request.files['image']): 
        file = request.files['image']

        result = classifier.classifyImage(file)
        print('Model classification: ' + result)        
        return result 

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
