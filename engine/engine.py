from typing import List, Dict, Any
import models
import pandas as pd

class Forecasting:
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
   
    def RW_real_forecast(Data: pd.DataFrame,
                         Forecast_horizon: int) -> pd.DataFrame:
        '''Запускаем модель RW реального прогноза'''
        return models.RW_real_forecast(Data, Forecast_horizon)
    
    def RWD_real_forecast(Data: pd.DataFrame,
                          Forecast_horizon: int,
                          Frequency: str) -> pd.DataFrame:
        '''Запускаем модель RWD реального прогноза'''
        return models.RWD_real_forecast(Data, Forecast_horizon, Frequency)
    
    def RWS_real_forecast(Data: pd.DataFrame,
                          Forecast_horizon: int,
                          Seasonality: int) -> pd.DataFrame:
        '''Запускаем модель RWS реального прогноза'''
        return models.RWS_real_forecast(Data, Forecast_horizon, Seasonality)
    
    def RWDS_real_forecast(Data: pd.DataFrame,
                           Forecast_horizon: int,
                           Seasonality: int) -> pd.DataFrame:
        '''Запускаем модель RWDS реального прогноза'''
        return models.RWDS_real_forecast(Data, Forecast_horizon, Seasonality)
    
    def TS_real_forecast(Data: pd.DataFrame,
                         Forecast_horizon: int) -> pd.DataFrame:
        '''Запускаем модель TS реального прогноза'''
        return models.TS_real_forecast(Data, Forecast_horizon)
    
    def ARIMA_real_forecast(Data: pd.DataFrame,
                            Forecast_horizon: int,
                            Frequency: str) -> pd.DataFrame:
        '''Запускаем модель ARIMA реального прогноза'''
        return models.ARIMA_real_forecast(Data, Forecast_horizon, Frequency)