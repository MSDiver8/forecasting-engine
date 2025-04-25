from typing import List, Dict, Any
import models
import pandas as pd

class Auto_real_forecast:
    def __init__(self):
        self.available_models = {           # Словарь название модели : название ее функции
            'RW': models.RW_real_forecast,
            'RWS': models.RWS_real_forecast,
            'RWD': models.RWD_real_forecast,
            'RWDS': models.RWDS_real_forecast,
            'TS': models.TS_real_forecast,
            'ARIMA': models.ARIMA_real_forecast
        }

        self.model_args = {         # Словарь название модели : входные параметры ее функции
            'RW': ['df', 'Forecast_horizon'],
            'RWS': ['df', 'Forecast_horizon', 'Seasonality'],
            'RWD': ['df', 'Forecast_horizon', 'Frequency'],
            'RWDS': ['df', 'Forecast_horizon', 'Seasonality'],
            'TS': ['df', 'Forecast_horizon'],
            'ARIMA': ['df', 'Forecast_horizon', 'Frequency']
        }


    
    def RW_real_forecast(Self,
                         Data: pd.DataFrame,
                         Forecast_horizon: int) -> pd.DataFrame:
        '''Запускаем модель RW реального прогноза'''
        return models.RW_real_forecast(Data, Forecast_horizon)