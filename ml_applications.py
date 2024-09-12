# ml_applications.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import numpy as np

def forecast_enrollment(data, state):
    print(f"Debug: forecast_enrollment called with data shape: {data.shape}")
    print(f"Debug: forecast_enrollment data head: {data.head()}")

    # Convert 'Year' to a numeric format for forecasting
    data['Year_Numeric'] = data['Year'].apply(lambda x: int(x.split('-')[0]))
    
    X = data['Year_Numeric'].values.reshape(-1, 1)
    y = data['Enrollment'].values

    model = LinearRegression()
    model.fit(X, y)

    last_year = X.max()
    future_years = np.array(range(last_year + 1, last_year + 4)).reshape(-1, 1)
    forecast = model.predict(future_years)

    forecast_data = pd.DataFrame({
        'ds': [f"{year}-{year+1}" for year in future_years.flatten()],
        'yhat': forecast
    })

    print(f"Debug: forecast_enrollment output shape: {forecast_data.shape}")
    print(f"Debug: forecast_enrollment output head: {forecast_data.head()}")

    return forecast_data

# Clustering using KMeans
def cluster_demographics(data, n_clusters=3):
    clustering_data = data.groupby('Demographic_Gender').sum().reset_index()
    kmeans = KMeans(n_clusters=n_clusters)
    clustering_data['Cluster'] = kmeans.fit_predict(clustering_data[['Enrollment']])
    return clustering_data

# Anomaly Detection using Isolation Forest
def detect_anomalies(data):
    anomaly_data = data.groupby('Year').sum().reset_index()
    model = IsolationForest(contamination=0.1)
    anomaly_data['Anomaly'] = model.fit_predict(anomaly_data[['Enrollment']])
    return anomaly_data
