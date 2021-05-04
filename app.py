import numpy as np
import pandas as pd
from flask import Flask, request, render_template

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
    output = pd.read_csv("output.csv")
    index = int(final_features[0][1])
    x1 = output['result'][index-1]

    return render_template('index.html', prediction_text='The Predicted Price for the {ind} day is ${ans}'.format(ind = index, ans = x1))


if __name__ == "__main__":
    app.run(debug=True)