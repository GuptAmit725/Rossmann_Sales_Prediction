from flask import Flask, render_template, request
from flask_cors import CORS,cross_origin
from statsmodels.tsa.statespace.sarimax import SARIMAX
import datetime
import pandas as pd

app = Flask(__name__) # initializing a flask app
MODEL_PATH = 'sarima.pkl'
DATA = pd.read_csv('sales.csv')

@app.route('/',methods=['POST','GET']) # route to show the predictions in a web UI
@cross_origin()
def index():
    if request.method == 'POST':
        #  reading the inputs given by the user
        print('Getting input...')
        year = request.form['year']
        month = request.form['month']
        day = request.form['day']
        print('input recieved')
        if str(month)[0]=='0':
            month = month[1:]
        if str(day)[0]=='0':
            day = day[1:]
        print(year,month,day)
        d1 = f'{year}-{month}-{day}'
        d2 = '2013-01-01'
        d1 = datetime.datetime.strptime(d1, "%Y-%m-%d")
        d2 = datetime.datetime.strptime(d2, "%Y-%m-%d")

        n_days = (d1-d2).days
        print(n_days)

        (P, D, G, s) = (1,1,3,12)
        s_mod = SARIMAX(
                        DATA,
                        order=(P,D,G),
                        seasonal_order=(P,D,G,s),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                        )
        s_fit = s_mod.fit(disp=0)

        print('model trained')
        forecast = round(s_fit.predict(start=n_days, end=n_days, exog=None, dynamic=False),0)
        print(forecast)
        # showing the prediction results in a UI
        return render_template('results.html',prediction=forecast.values[0])
    else:
        print('Something is still wrong')
        return render_template('results.html')



if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app
