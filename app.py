import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    # output = round(prediction[0], 2)
    if final_features[0][1] == 'd1':
        a = 125.93
    elif final_features[0][1] == 'd2':
        a = 124.23
    elif final_features[0][1] == 'd3':
        a = 128.59
    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(a))


if __name__ == "__main__":
    app.run(debug=True)