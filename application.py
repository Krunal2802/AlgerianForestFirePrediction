from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle

application = Flask(__name__)
app = application

## load linear regressor and standardscaler
lrg = pickle.load(open('models/Project_2_linear_regression.pkl','rb'))
sc = pickle.load(open('models/Project_2_Standard_scaler.pkl','rb'))


# @app.route("/")
# def index():
#     return render_template('index.html')

@app.route("/",methods=['GET','POST'])
def predict_data():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_scaled_data = sc.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])

        result = lrg.predict(new_scaled_data)

        return render_template('home.html',result=result[0])
    else:
        return render_template("home.html")

if __name__=="__main__":
    app.run(host="127.0.0.1", port=5000)