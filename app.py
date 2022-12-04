from flask import Flask ,render_template,request
import pickle
from sklearn.linear_model import LogisticRegression
app=Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])

def predict():
    if request.method =='POST':
        exp1=request.form['exp']
        data=[[float(exp1)]]
        lr=pickle.load(open('salary.pkl','rb'))
        prediction=lr.predict(data)[0]

    return render_template('index.html',prediction=prediction)





if __name__ == '__main__':
    app.run(debug=True)