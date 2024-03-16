from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods = ['POST'])
def result():
    # if request.method == 'POST':
    year = request.form.get('year', type = int)
    manufacturer = request.form.get('manufacturer')
    odometer = request.form.get('odometer')
    transmission = request.form.get('transmission')
    type = request.form.get('type')

    cols = ['year', 'manufacturer', 'odometer', 'transmission', 'type']
    test_set = pd.DataFrame([[year, manufacturer, odometer, transmission, type]], columns=cols)
        
    model = pickle.load(open("model.pkl", "rb"))

    preds = model.predict(test_set)
    predicted = round((preds[0]),2)
    return  render_template("home.html", prediction_text = 'The value of the predicted car is: {}'.format(predicted))

#Used to puu debug into code. If you make a change in code, you just need
#update the website and the application of changes will work
if __name__ == '__main__':
    app.run(debug = True)
