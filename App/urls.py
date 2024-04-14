from App.views import Train, Predict
from django.urls import path

app_name = 'App'

urlpatterns = [
    path('train/', Train.as_view(), name="train"),
    path('predict/', Predict.as_view(), name="predict"),
]