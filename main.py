#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    print(prediction[0])

    #output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="AQI for Jaipur {}".format(prediction[0]))


# In[7]:


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




