from typing import List, Dict, Any
import models
import utils
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
    
    # Функция для построения реального прогноза
    def RW_real_forecast(Data: pd.DataFrame,
                         Forecast_horizon: int) -> pd.DataFrame:
        '''Запускаем модель RW реального прогноза'''
        return models.RW_real_forecast(Data, Forecast_horizon)
    # дабвить вывод в виде дф с датой
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
    
    # Функции для построения псевдовневыборочного прогноза
    def ps_RW_forecast(Data: pd.DataFrame,
                       Deep_forecast_period: int,
                       Forecast_horizon: int) -> pd.DataFrame:
        '''Запускаем модель RW псевдовневыборочного прогноза'''
        return models.ps_RW_forecast(Data, Deep_forecast_period, Forecast_horizon)
    
    def ps_RWD_forecast(Data: pd.DataFrame,
                        Deep_forecast_period: int,
                        Forecast_horizon: int,
                        Frequency: str) -> pd.DataFrame:
        '''Запускаем модель RWD псевдовневыборочного прогноза'''
        return models.ps_RWD_forecast(Data, Deep_forecast_period, Forecast_horizon, Frequency)
    
    def ps_RWS_forecast(Data: pd.DataFrame,
                        Deep_forecast_period: int,
                        Forecast_horizon: int,
                        Seasonality: int) -> pd.DataFrame:
        '''Запускаем модель RWS псевдовневыборочного прогноза'''
        return models.ps_RWS_forecast(Data, Deep_forecast_period, Forecast_horizon, Seasonality)
    
    def ps_RWDS_forecast(Data: pd.DataFrame,
                         Deep_forecast_period: int,
                         Forecast_horizon: int,
                         Seasonality : int) -> pd.DataFrame:
        '''Запускаем модель RWDS псевдовневыборочного прогноза'''
        return models.ps_RWDS_forecast(Data, Deep_forecast_period, Forecast_horizon, Seasonality)
    
    def ps_TS_forecast(Data: pd.DataFrame,
                       Deep_forecast_period: int,
                       Forecast_horizon: int,
                       ) -> pd.DataFrame:
        '''Запускаем модель TS псевдовневыборочного прогноза'''
        return models.ps_TS_forecast(Data, Deep_forecast_period, Forecast_horizon)
    
    def ps_ARIMA_forecast(Data: pd.DataFrame,
                          Deep_forecast_period: int,
                          Forecast_horizon: int,
                          Frequency: str) -> pd.DataFrame:
        '''Запускаем модель ARIMA псевдовневыборочного прогноза'''
        return models.ps_ARIMA_forecast(Data, Deep_forecast_period, Forecast_horizon, Frequency)
    
    # Функция Автоматического прогноза - Конструктор моделей
    def Auto_forecast(Data: pd.DataFrame,
                      Frequency: str) -> pd.DataFrame:
        '''Запускаем автоматический прогноз "Конструктор моделей"'''
        return utils.Auto_forecast(Data, Frequency)