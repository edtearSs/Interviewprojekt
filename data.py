import numpy as np
import pandas as pd

def get_dataframe():
    data = pd.read_csv('autoscout24.csv')
    data = data.rename(
        columns={'mileage': 'Kilometerstand', 'make': 'Hersteller', 'model': 'Modell', 'fuel': 'Kraftstoff',
                 'gear': 'Schaltung', 'offerType': 'Zustand', 'price': 'Preis', 'hp': 'PS', 'year': 'Jahr'})
    data = data.dropna()
    data = data[data['Jahr'] >= 2011]

    data['Kilometerstand'] = data['Kilometerstand'].round(0).astype(int)
    data['Preis'] = data['Preis'].round(0).astype(int)
    data['PS'] = data['PS'].round(0).astype(int)
    data['Jahr'] = data['Jahr'].round(0).astype(int)

    data_num = data.drop(columns=['Hersteller', 'Modell', 'Kraftstoff', 'Schaltung', 'Zustand'])

    hersteller_count = data['Hersteller'].value_counts()
    data_five = data[data['Hersteller'].isin(hersteller_count.index[:5])]

    return data, data_num, data_five


