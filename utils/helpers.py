import pandas as pd
import math
import sklearn.metrics as sm

from exceptions.exceptions import Auto_params_selection_Error

from models.models import ps_RW_forecast, ps_RWD_forecast, ps_RWS_forecast, ps_RWDS_forecast, ps_TS_forecast, ps_ARIMA_forecast

# Фукция автоматического подбора параметров для базовых моделей
def auto_params_selection(data: pd.DataFrame,
                          freq: str):
    df = data.copy()
    dic_auto_params = {'Forecast horizon': [], # Горизонт псевдовневыборочного прогноза
                       'Deep forecast period':[], # Длина псевдовневыборочного прогноза
                       'Seasonality': [], # параметр сезонности, учавствующий в уравнении модели (дополнительный снос)
                       'Window size': []} # ширина скользящего окна
    
    # горизонт прогнозирования от длины дф
    # дописать что 'Forecast horizon' не может быть больше 'Deep forecast period'
    '''Экспертами устанавливаются фиксированные значения для Forecast horizon'''
    '''Для коротких рядов допускается взятие 20% от длины ряда'''
    horizon_values = {'horizon_m': 12,
                      'horizon_q': 4,
                      'horizon_y': 3}
    ratio_horizon_to_len = math.floor(0.2 * len(df)) # Отношение горизонта псевдовневыборочного прогноза к длине всего ряда. актуально для коротких рядов.
    if freq == 'M':
        if ratio_horizon_to_len > 12:
            dic_auto_params['Forecast horizon'] = horizon_values['horizon_m']   # ряд стандартной длины
        else: 
            dic_auto_params['Forecast horizon'] = ratio_horizon_to_len     # для коротких рядов
    elif freq == 'Q':
        if ratio_horizon_to_len > 4:
            dic_auto_params['Forecast horizon'] = horizon_values['horizon_q']
        else: 
            dic_auto_params['Forecast horizon'] = ratio_horizon_to_len
    elif freq == 'Y':
        if ratio_horizon_to_len > 3:
            dic_auto_params['Forecast horizon'] = horizon_values['horizon_y']
        else: 
            dic_auto_params['Forecast horizon'] = ratio_horizon_to_len
    else:
        raise Auto_params_selection_Error("Невозможно определить горизонт псевдовневыборочного прогноза")
    
    '''Экспертами устанавливаются фиксированные значения для Deep forecast period'''
    '''Для коротких рядов допускается взятие 40% от длины ряда'''
    
    deep_forecast_values = {'deep_forecast_period_m': 24,
                            'deep_forecast_period_q': 8,
                            'deep_forecast_period_y': 6}
    ratio_deep_forcast_period_to_len = math.floor(0.4 * len(df)) # Отношение длины псевдовневыборочного прогноза к длине всего ряда. актуально для коротких рядов.
    if freq == 'M':
        if ratio_deep_forcast_period_to_len > 12:
            dic_auto_params['Deep forecast period'] = deep_forecast_values['deep_forecast_period_m']
        else: 
            dic_auto_params['Deep forecast period'] = ratio_deep_forcast_period_to_len
    elif freq == 'Q':
        if ratio_deep_forcast_period_to_len > 4:
            dic_auto_params['Deep forecast period'] = deep_forecast_values['deep_forecast_period_q']
        else: 
            dic_auto_params['Deep forecast period'] = ratio_deep_forcast_period_to_len
    elif freq == 'Y':
        if ratio_deep_forcast_period_to_len > 3:
            dic_auto_params['Deep forecast period'] = deep_forecast_values['deep_forecast_period_y']
        else: 
            dic_auto_params['Deep forecast period'] = ratio_deep_forcast_period_to_len
    else:
        raise Auto_params_selection_Error("Невозможно определить длину псевдовневыборочного прогноза")
    
    '''Экспертами устанавливаются фиксированные значения для Seasonality'''
    seasonality_values = {'seasonality_m': 12,  # для годовых рядов отсутствует сезонности
                          'seasonality_q': 4,
                          'seasonality_y' : None}
    if freq == 'M':
        dic_auto_params['Seasonality'] = seasonality_values['seasonality_m']
    elif freq == 'Q':
        dic_auto_params['Seasonality'] = seasonality_values['seasonality_q']
    elif freq == 'Y':
        dic_auto_params['Seasonality'] = seasonality_values['seasonality_y']

    '''Экспертами устанавливаются фиксированные значения для Windowsize'''      # редактировать параметры
    windowsize_values = {'windowsize_m': 36,
                         'windowsize_q': 12,
                         'windowsize_y': 3}
    ratio_windowsize_to_len = math.floor(0.2 * len(df)) # Отношение ширины скользящего окна псевдовневыборочного прогноза к длине всего ряда. актуально для коротких рядов.
    if freq == 'M':
        if ratio_windowsize_to_len > 12:
            dic_auto_params['Window size'] = windowsize_values['windowsize_m']
        else: 
            dic_auto_params['Window size'] = ratio_windowsize_to_len
    elif freq == 'Q':
        if ratio_windowsize_to_len > 4:
            dic_auto_params['Window size'] = windowsize_values['windowsize_q']
        else: 
            dic_auto_params['Window size'] = ratio_windowsize_to_len
    elif freq == 'Y':
        if ratio_windowsize_to_len > 3:
            dic_auto_params['Window size'] = windowsize_values['windowsize_y']
        else: 
            dic_auto_params['Window size'] = ratio_windowsize_to_len
    else:
        raise Auto_params_selection_Error("Невозможно определить ширину скользящего окна")
    return dic_auto_params

# Функция рассчета усредненных  MAPE для каждого горизонта прогнозирования в отдельности
def MAPE_step_by_step(Data: pd.DataFrame,
                      Dataframe_model: pd.DataFrame,
                      Deep_forecast_period: int,
                      Forecast_horizon: int):
    
    df_real = Data.copy()
    df_real.obs = df_real.obs.astype(float) 
    
    df_model = Dataframe_model.copy()
    
    train_period =  len(df_real.obs) - Deep_forecast_period
    quantity_pseudo_foracasts = Deep_forecast_period - Forecast_horizon + 1
    
    Real_Data_list = [df_real.obs[train_period + i:train_period+quantity_pseudo_foracasts + i] for i in range(Forecast_horizon)]
    
    Model_Data_list = [df_model.iloc[:, i] for i in range(Forecast_horizon)]
    
    
    Errors = []
    for i in range(Forecast_horizon):       
        Errors.append(round(math.sqrt(sm.mean_absolute_percentage_error(Real_Data_list[i], Model_Data_list[i])), 2))
    
    return Errors

# Функция определения минимального MAPE на каждом шаге и формирование списка с "лучшими" моделями для каждого шага прогнозирования
def Psevdo_forecast_test_MAPE(Data: pd.DataFrame,
                        Deep_forecast_period: int,
                        Forecast_horizon: int,
                        Seasonality : int,
                        Frequency: str):
    
    if Frequency == 'Y':
        Forecast_dict_Y = {'RW' : ps_RW_forecast(Data, Deep_forecast_period, Forecast_horizon),            # строим псевдо прогноз для всех моделей
                           'RWD' : ps_RWD_forecast(Data, Deep_forecast_period, Forecast_horizon, Frequency),
                           'TS' : ps_TS_forecast(Data, Deep_forecast_period, Forecast_horizon),
                           'ARIMA' : ps_ARIMA_forecast(Data, Deep_forecast_period, Forecast_horizon, Frequency)
                        }
    
        Erorr_dict_Y = {k : MAPE_step_by_step(Data,
                                        Forecast_dict_Y[k],
                                        Deep_forecast_period,
                                        Forecast_horizon)
                        for k in Forecast_dict_Y.keys()
                        }
    
        List_of_model_number_Y = [min(Erorr_dict_Y, key=lambda key: Erorr_dict_Y[key][i]) for i in range(Forecast_horizon)] # функция min() примененная к словарю обращается к его ключам. С помощью lambda функции выбирается сначала только первые элементы, затем только вторые и тд. затем функция min() выбирает минимальный элемент среди первых, затем среди вторых и тд, когда выбирается миниму, выражение возвращает ключ в котором он содержится
    
        return List_of_model_number_Y
    
    else:
        Forecast_dict = {'RW' : ps_RW_forecast(Data, Deep_forecast_period, Forecast_horizon),            # строим псевдо прогноз для всех моделей
                         'RWS' : ps_RWS_forecast(Data, Deep_forecast_period, Forecast_horizon, Seasonality),
                         'RWD' : ps_RWD_forecast(Data, Deep_forecast_period, Forecast_horizon, Frequency),
                         'RWDS' : ps_RWDS_forecast(Data, Deep_forecast_period, Forecast_horizon, Seasonality),
                         'TS' : ps_TS_forecast(Data, Deep_forecast_period, Forecast_horizon),
                         'ARIMA' : ps_ARIMA_forecast(Data, Deep_forecast_period, Forecast_horizon, Frequency)
                        }
    
        Erorr_dict = {k : MAPE_step_by_step(Data,
                                        Forecast_dict[k],
                                        Deep_forecast_period,
                                        Forecast_horizon)
                        for k in Forecast_dict.keys()
                        }
    
        List_of_model_number = [min(Erorr_dict, key=lambda key: Erorr_dict[key][i]) for i in range(Forecast_horizon)] # функция min() примененная к словарю обращается к его ключам. С помощью lambda функции выбирается сначала только первые элементы, затем только вторые и тд. затем функция min() выбирает минимальный элемент среди первых, затем среди вторых и тд, когда выбирается миниму, выражение возвращает ключ в котором он содержится
    
        return List_of_model_number
