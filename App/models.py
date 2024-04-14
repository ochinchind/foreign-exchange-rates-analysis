from django.db import models

# Create your models here.
class ExchangeRate(models.Model):
    date = models.DateField()
    aud = models.FloatField()
    usd = models.FloatField()
    eur = models.FloatField()
    cny = models.FloatField()
    kgs = models.FloatField()
    nok = models.FloatField()
    uah = models.FloatField()
    krw = models.FloatField()
    jpy = models.FloatField()
    mxn = models.FloatField()