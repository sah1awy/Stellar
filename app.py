import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
classes = {0:"GALAXY",1:"QSO",2:"STAR"}

app=Flask(__name__)
## Load the model
model=pickle.load(open('model.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=model.predict(new_data)
    return classes[output[0]]

@app.route('/predict',methods=['POST'])
# def predict():
#     data = [float(x) for x in request.form.values()]
#     print(data)
#     final_input = scalar.transform(np.array(data).reshape(1,-1))
#     print(final_input)
#     output = model.predict(final_input)[0]
#     return render_template('html.home',prediction_text="The Predicted Stellar Type is: {}".format(classes[output]))

def predict():
    # Get input data from request
    data = [float(x) for x in request.form.values()]
    print("Input data:", data)
    
    # Transform input data using the scalar
    final_input = scalar.transform(np.array(data).reshape(1, -1))
    print("Transformed input:", final_input)
    
    # Make prediction using the model
    output = model.predict(final_input)[0]
    print("Output prediction:", output)
    
    # Get the predicted class label from the classes array
    predicted_class = classes[output]
    print("Predicted class:", predicted_class)
    
    # Render the prediction in an HTML template
    return render_template('home.html', prediction_text=f'The Predicted Stellar Type is: {predicted_class}')

if __name__=="__main__":
    app.debug = True
    app.run()
