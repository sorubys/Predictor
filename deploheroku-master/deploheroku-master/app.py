#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2) 
    
    if output == 1:
        result_message = 'Your Team will win the match. Keep it up!'
    else:
        result_message = 'Your Team has less chances to win the match. Try to improve your performance!'
    return render_template('index.html', prediction_text='Result is  :{}'.format(result_message))

if __name__ == "__main__":
    app.run(debug=True)