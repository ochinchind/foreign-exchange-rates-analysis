import json
from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
import os
import pickle
import numpy as np
import pandas as pd
from sklearn import datasets
from django.conf import settings
from rest_framework import views
from rest_framework import status
from rest_framework.response import Response
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from .models import ExchangeRate
from django.shortcuts import render


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