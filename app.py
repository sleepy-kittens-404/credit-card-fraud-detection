from flask import Flask,request,render_template
from numpy import np 
from pandas import pd
from srd.pipeline.predict_pipeline import CustomData,P
application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predictdata',methods=['GET','POST'])
def precict_datapoint():
    if request.method == 'GET':
        return render_template('prediction.html')
    else:
        data = CustomData()