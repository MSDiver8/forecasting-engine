import pandas as pd
import math
import sklearn.metrics as sm

from exceptions.exceptions import Auto_params_selection_Error

from models.models import ps_RW_forecast, ps_RWD_forecast, ps_RWS_forecast, ps_RWDS_forecast, ps_TS_forecast, ps_ARIMA_forecast

from models.models import RW_real_forecast, RWD_real_forecast, RWS_real_forecast, RWDS_real_forecast, TS_real_forecast, ARIMA_real_forecast

# Фукция автоматического подбора параметров для базовых моделей
def auto_params_selection(data: pd.DataFrame,
                          freq: str):
    df = data.copy()
    dic_auto_params = {'Forecast_horizon': [], # Горизонт псевдовневыборочного прогноза
                       'Deep_forecast_period':[], # Длина псевдовневыборочного прогноза
                       'Seasonality': [], # параметр сезонности, учавствующий в уравнении модели (дополнительный снос)
                       'Window_size': []} # ширина скользящего окна
    
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
            dic_auto_params['Forecast_horizon'] = horizon_values['horizon_m']   # ряд стандартной длины
        else: 
            dic_auto_params['Forecast_horizon'] = ratio_horizon_to_len     # для коротких рядов
    elif freq == 'Q':
        if ratio_horizon_to_len > 4:
            dic_auto_params['Forecast_horizon'] = horizon_values['horizon_q']
        else: 
            dic_auto_params['Forecast_horizon'] = ratio_horizon_to_len
    elif freq == 'Y':
        if ratio_horizon_to_len > 3:
            dic_auto_params['Forecast_horizon'] = horizon_values['horizon_y']
        else: 
            dic_auto_params['Forecast_horizon'] = ratio_horizon_to_len
    else:
        raise Auto_params_selection_Error("Невозможно определить горизонт псевдовневыборочного прогноза")
    
    '''Экспертами устанавливаются фиксированные значения для Deep forecast period'''
    '''Для коротких рядов допускается взятие 40% от длины ряда'''
    
    deep_forecast_values = {'deep_forecast_period_m': 24,
                            'deep_forecast_period_q': 8,
                            'deep_forecast_period_y': 6}
    ratio_deep_forcast_period_to_len = math.floor(0.4 * len(df)) # Отношение длины псевдовневыборочного прогноза к длине всего ряда. актуально для коротких рядов.
    if freq == 'M':
        if ratio_deep_forcast_period_to_len > 24:
            dic_auto_params['Deep_forecast_period'] = deep_forecast_values['deep_forecast_period_m']
        else: 
            dic_auto_params['Deep_forecast_period'] = ratio_deep_forcast_period_to_len
    elif freq == 'Q':
        if ratio_deep_forcast_period_to_len > 8:
            dic_auto_params['Deep_forecast_period'] = deep_forecast_values['deep_forecast_period_q']
        else: 
            dic_auto_params['Deep_forecast_period'] = ratio_deep_forcast_period_to_len
    elif freq == 'Y':
        if ratio_deep_forcast_period_to_len > 6:
            dic_auto_params['Deep_forecast_period'] = deep_forecast_values['deep_forecast_period_y']
        else: 
            dic_auto_params['Deep_forecast_period'] = ratio_deep_forcast_period_to_len
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
            dic_auto_params['Window_size'] = windowsize_values['windowsize_m']
        else: 
            dic_auto_params['Window_size'] = ratio_windowsize_to_len
    elif freq == 'Q':
        if ratio_windowsize_to_len > 4:
            dic_auto_params['Window_size'] = windowsize_values['windowsize_q']
        else: 
            dic_auto_params['Window_size'] = ratio_windowsize_to_len
    elif freq == 'Y':
        if ratio_windowsize_to_len > 3:
            dic_auto_params['Window_size'] = windowsize_values['windowsize_y']
        else: 
            dic_auto_params['Window_size'] = ratio_windowsize_to_len
    else:
        raise Auto_params_selection_Error("Невозможно определить ширину скользящего окна")
    return dic_auto_params

# Функция рассчета усредненных  MAPE для каждого горизонта прогнозирования в отдельности
def MAPE_step_by_step(Data: pd.DataFrame,
                      Dataframe_model: pd.DataFrame,
                      Deep_forecast_period: int,
                      Forecast_horizon: int):
    '''Функция рассчета усредненных  MAPE для каждого горизонта прогнозирования в отдельности'''
    df_real = Data.copy()
    df_real.obs = df_real.obs.astype(float) 
    
    df_model = Dataframe_model.copy()
    
    train_period =  len(df_real.obs) - Deep_forecast_period
    quantity_pseudo_foracasts = Deep_forecast_period - Forecast_horizon + 1
    
    Real_Data_list = [df_real.obs[train_period + i:train_period+quantity_pseudo_foracasts + i] for i in range(Forecast_horizon)]
    
    Model_Data_list = [df_model.iloc[:, i] for i in range(Forecast_horizon)]
    
    
    Errors = []
    for i in range(Forecast_horizon):        
        Errors.append(round(sm.mean_absolute_percentage_error(Real_Data_list[i], Model_Data_list[i])), 2)
    
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

# Функция Автоматического прогноза - Конструктор моделей
def Auto_forecast(Data : pd.DataFrame,
                  Frequency : str):

    # Создается 2 словоря available_models и model_args, это делается для оптимизации работы функции.
    # Для построени комбинированного по шагам прогноза будут задействованы только необходимые функции,
    # которые получаются в результате работы функции Psevdo_forecast_test_MAPE. 
    
    # Функции для построения реального прогноза
    available_models = {
        'RW': RW_real_forecast,
        'RWS': RWS_real_forecast,
        'RWD': RWD_real_forecast,
        'RWDS': RWDS_real_forecast,
        'TS': TS_real_forecast,
        'ARIMA': ARIMA_real_forecast
    }
    # Параметры функций для построения реального прогноза
    model_args = {
        'RW': ['Data', 'Forecast_horizon'],
        'RWS': ['Data', 'Forecast_horizon', 'Seasonality'],
        'RWD': ['Data', 'Forecast_horizon', 'Frequency'],
        'RWDS': ['Data', 'Forecast_horizon', 'Seasonality'],
        'TS': ['Data', 'Forecast_horizon'],
        'ARIMA': ['Data', 'Forecast_horizon', 'Frequency']
    }
    # dic_auto_params - Словарь с автоматически продобранными значениями базовых прогнозных функций. 
    dic_auto_params = auto_params_selection(Data, Frequency)

    # List_of_model_number - Список с результатами по псевдовневыборочному тесту 
    # то есть, список моделей, которые участвуют в построение реального прогноза.
    List_of_model_number = Psevdo_forecast_test_MAPE(Data,
                                                     dic_auto_params['Deep_forecast_period'],
                                                     dic_auto_params['Forecast_horizon'],
                                                     dic_auto_params['Seasonality'],
                                                     Frequency)
    
    # Конструирование реального прогноза по шагам прогнозирования. 
    Forecast, Model_name, Steps = [], [], []    # Для записи результатов
    for i in range(len(List_of_model_number)):
        # По списку List_of_model_number, определются только нужные для прогноза модели,
        # для нужных моделей вызываются соответсвующие им функции из available_models и
        # аргументы соответсвующие вызванным функциям из model_args.             
        model_func = available_models.get(List_of_model_number[i])
        required_args = model_args.get(List_of_model_number[i], [])
        # call_args - словарь, в который записывается название аргумента функции и
        # его значение. Это происходит только для той функции которая соответсвует итерации (List_of_model_number[i]).
        call_args = {
            arg: (Data if arg == 'Data' else Frequency if arg == 'Frequency' else dic_auto_params[arg])
            for arg in required_args
            if arg in ['Data', 'Frequency'] or arg in dic_auto_params
        }
        # Для записи результата прогноза вызывается нужная модель из списка List_of_model_number и
        # соотвесвующее номеру шага (горизонту прогнозирования) прогнозное значение.
        Forecast.append(model_func(**call_args)[i])
        Model_name.append(List_of_model_number[i])
        Steps.append(f'Горизонт {i+1}')
    
    # Создание временной даты соответсвующей прогнозным значениям
    Data["date"] = pd.to_datetime(Data["date"], dayfirst=True, format="%d.%m.%Y")
    Date = pd.date_range(Data.iloc[-1,0] + (Data.iloc[-1,0] - Data.iloc[-2,0]),
                         periods = dic_auto_params['Forecast_horizon'],
                         freq = Frequency
                         ).strftime('%d.%m.%Y').tolist()
    
    # Записываем итоговый результат в виде датафрейма
    results = pd.DataFrame({
    'Дата': Date,
    'Шаги': Steps,
    'Модель': Model_name,
    'Прогноз': Forecast
    })

    return results