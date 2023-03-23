import requests
import json

API_URL = "https://www.alphavantage.co/query"

data = {
    "function": "TIME_SERIES_INTRADAY_EXTENDED",
    "symbol": "IBM",
    "interval":"1min",
    "outputsize": "compact",
    "datatype": "csv",
    "apikey": "EHSP3TEMSVZG4J9J"
    }
response = requests.get(API_URL, params=data)

print(response.text)

with open('IBM_last30days_1min.csv', 'w') as out:
    out.write(response.text)

