import datetime
import json
from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
import os
import pickle
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from django.conf import settings
from rest_framework import views
from rest_framework import status
from rest_framework.response import Response
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from .models import ExchangeRate
from django.shortcuts import render
from statsmodels.tsa.arima.model import ARIMA


class Train(views.APIView):
    def post(self, request):
        iris = datasets.load_iris()
        mapping = dict(zip(np.unique(iris.target), iris.target_names))

        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.DataFrame(iris.target).replace(mapping)
        model_name = request.data.pop('model_name')

        try:
            clf = RandomForestClassifier(**request.data)
            clf.fit(X, y)
        except Exception as err:
            return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

        path = os.path.join(settings.MODEL_ROOT, model_name)
        with open(path, 'wb') as file:
            pickle.dump(clf, file)
        return Response(status=status.HTTP_200_OK)


class Predict(views.APIView):
    def post(self, request):
        predictions = []
        for entry in request.data:
            model_name = entry.pop('model_name')
            path = os.path.join(settings.MODEL_ROOT, model_name)
            with open(path, 'rb') as file:
                model = pickle.load(file)
            try:
                result = model.predict(pd.DataFrame([entry]))
                predictions.append(result[0])

            except Exception as err:
                return Response(str(err), status=status.HTTP_400_BAD_REQUEST)

        return Response(predictions, status=status.HTTP_200_OK)


def load_data_from_excel(request):
    data = pd.read_excel('exchange_rates.xlsx', parse_dates=['Date'], date_parser=lambda x: pd.to_datetime(x, format='%d.%m.%Y'))
    for index, row in data.iterrows():
        ExchangeRate.objects.create(
            date=row['Date'],
            aud=row['AUD'],
            usd=row['USD'],
            # Add other fields here
            eur = row['EUR'],
            cny = row['CNY'],
            kgs = row['KGS'],
            nok = row['NOK'],
            uah = row['UAH'],
            krw = row['KRW'],
            jpy = row['JPY'],
            mxn = row['MXN']
        )
    return HttpResponse("Data loaded successfully")

def display_exchange_rates(request):
    exchange_rates = ExchangeRate.objects.all()
    dates = [exchange.date.strftime('%Y-%m-%d') for exchange in exchange_rates]
    aud = [exchange.aud for exchange in exchange_rates]
    usd = [exchange.usd for exchange in exchange_rates]
    eur = [exchange.eur for exchange in exchange_rates]
    cny = [exchange.cny for exchange in exchange_rates]
    kgs = [exchange.kgs for exchange in exchange_rates]
    nok = [exchange.nok for exchange in exchange_rates]
    uah = [exchange.uah for exchange in exchange_rates]
    krw = [exchange.krw for exchange in exchange_rates]
    jpy = [exchange.jpy for exchange in exchange_rates]
    mxn = [exchange.mxn for exchange in exchange_rates]
    return render(request, 'template.html', {
        'exchange_rates': exchange_rates, 
        'dates': json.dumps(dates),
        'aud': json.dumps(aud),
        'usd': json.dumps(usd),
        'eur': json.dumps(eur),
        'cny': json.dumps(cny),
        'kgs': json.dumps(kgs),
        'nok': json.dumps(nok),
        'uah': json.dumps(uah),
        'krw': json.dumps(krw),
        'jpy': json.dumps(jpy),
        'mxn': json.dumps(mxn),
    })

def index(request):
    return render(request, 'index.html')

def predict_linear_regression(request):
    data = ExchangeRate.objects.all()

    # Convert dates to timestamps
    dates = [(entry.date - datetime.date(1970, 1, 1)).total_seconds() for entry in data]
    usd_values = [entry.usd for entry in data]
    aud_values = [entry.aud for entry in data]
    eur_values = [entry.eur for entry in data]

    X = np.array(dates).reshape(-1, 1)

    model_usd = LinearRegression()
    model_usd.fit(X, usd_values)

    model_aud = LinearRegression()
    model_aud.fit(X, aud_values)

    model_eur = LinearRegression()
    model_eur.fit(X, eur_values)

    # Generating future dates
    future_dates = np.linspace(np.min(X), np.max(X) + 86400 * 7, 100)

    # Predicting values for each currency
    predicted_usd = model_usd.predict(future_dates.reshape(-1, 1))
    predicted_aud = model_aud.predict(future_dates.reshape(-1, 1))
    predicted_eur = model_eur.predict(future_dates.reshape(-1, 1))

    # Combining predictions for all currencies into a single list
    predicted_data = [{
        'date': datetime.datetime.utcfromtimestamp(date).strftime('%Y-%m-%d'),
        'usd': usd,
        'aud': aud,
        'eur': eur
    } for date, usd, aud, eur in zip(future_dates, predicted_usd, predicted_aud, predicted_eur)]


    return render(request, 'predict_data.html', {'predicted_data': json.dumps(predicted_data)})

def arima_forecast(request):
    # Retrieve data from ExchangeRate model
    data = ExchangeRate.objects.all()

    # Prepare data for modeling
    dates = [entry.date for entry in data]
    usd_values = [entry.usd for entry in data]
    aud_values = [entry.aud for entry in data]
    eur_values = [entry.eur for entry in data]

    # Create a DataFrame with date and currency values
    df = pd.DataFrame({'Date': dates, 'USD': usd_values, 'AUD': aud_values, 'EUR': eur_values})

    # Set 'Date' column as index
    df.set_index('Date', inplace=True)

    # Fit ARIMA model for each currency
    arima_models = {}
    for currency in ['USD', 'AUD', 'EUR']:
        model = ARIMA(df[currency], order=(5,1,0))
        arima_models[currency] = model.fit()

    # Forecast future values
    forecast_period = 30  # Adjust as needed
    forecast_dates = pd.date_range(start=df.index[-1], periods=forecast_period+1, freq='D')
    forecast_dates_str = forecast_dates.strftime('%Y-%m-%d')  # Convert DatetimeIndex to list of strings
    forecast_results = {}
    for currency, model in arima_models.items():
        forecast = model.forecast(steps=forecast_period)
        forecast_results[currency] = list(forecast)

    context = {
        'forecast_dates': json.dumps(list(forecast_dates_str)[1:]),  # Convert to list and exclude first element
        'forecast_results': json.dumps(forecast_results),
    }

    return render(request, 'arima_forecast.html', context)