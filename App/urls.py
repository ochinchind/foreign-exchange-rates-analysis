from App.views import Train, Predict
from django.urls import path
from .views import arima_forecast, display_exchange_rates, load_data_from_excel, index, lstm_forecast, predict_linear_regression, prophet_forecast

app_name = 'App'

urlpatterns = [
    path('train/', Train.as_view(), name="train"),
    path('predict/', Predict.as_view(), name="predict"),
    path('load-data/', load_data_from_excel, name='load_data'),
    path('exchange-rates/', display_exchange_rates, name='exchange_rates'),
    path('index', index, name="index"),
    path('predict_linear_regression', predict_linear_regression, name='predict_linear_regression'),
    path('arima_forecast', arima_forecast, name='arima_forecast'),
    path('lstm_forecast', lstm_forecast, name='lstm_forecast'),
    path('prophet_forecast', prophet_forecast, name='prophet_forecast')
]