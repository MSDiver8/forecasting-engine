import pandas as pd
import statsmodels.formula.api as smf
import pmdarima as pmd

from exceptions.exceptions import Real_Forecast_Error

# Модели для псевдовневыборочного прогноза
def ps_RW_forecast(Data: pd.DataFrame,
                   Deep_forecast_period: int,
                   Forecast_horizon: int):

        df = Data.copy()
        df.obs = df.obs.astype(float) # значения переводятся в формат float
        base_period =  len(df['obs']) - Deep_forecast_period  # базовый период, на котором оценивается модель
        quantity_pseudo_forecasts = Deep_forecast_period - Forecast_horizon + 1 # количество псевдовневыборочных прогнозов

        forecast_table = [] # двумерный массив прогнозов
        for i in range(quantity_pseudo_forecasts):
            forecast_table.append([df.iloc[base_period - 1 + i, 1] for _ in range(Forecast_horizon)]) # из дф берется последняя известная точка, прогноз - значения этой точки, итог массив длиной forecast_horizon с одним и тем же числом.

        ## групировка по шагам
        steps_table=[] # таблица прогнозов по шагам. каждому столбцу соответсвует номер шага
        for i in range(Forecast_horizon):                                                
                steps_table.append([forecast_table[j][i] for j in range(quantity_pseudo_forecasts)])
                
        dic = { f's{i+1}' : steps_table[i] for i in range(Forecast_horizon)} # записываем в словарь
        df = pd.DataFrame.from_dict(dic)        
        return(df)
def ps_RWS_forecast(Data: pd.DataFrame,
                    Deep_forecast_period: int,
                    Forecast_horizon: int,
                    Seasonality : int):
    
    df = Data.copy()
    df.obs = df.obs.astype(float) # значения переводятся в формат float
    base_period =  len(df['obs']) - Deep_forecast_period # длина базового периода
    quantity_pseudo_foracasts = Deep_forecast_period - Forecast_horizon + 1 # количество псевдовневыборочных прогнозов  
    
    forecast_table = [] # двумерный массив прогнозов
    for i in range(quantity_pseudo_foracasts):
        forecast_table.append(df.iloc[base_period - Seasonality + i : base_period - Seasonality + i + Forecast_horizon, 1].to_list()) # из дф берется масив с данными смещенными на один месяц, так для каждого момента прогнозирования
    
    # групировка по шагам
    steps_table=[]
    for i in range(Forecast_horizon):                                                
            steps_table.append([forecast_table[j][i] for j in range(quantity_pseudo_foracasts)])
    #print(s)
    dic = { f's{i+1}' : steps_table[i] for i in range(Forecast_horizon)}
    df = pd.DataFrame.from_dict(dic)
    
    return(df)
def ps_RWD_forecast(Data: pd.DataFrame,
                    Deep_forecast_period: int,
                    Forecast_horizon: int,
                    Frequency: str
                    ):
        '''Преобразовываем dataframe в формат, подходящий для библиотеки statsmodel'''
        df = Data.copy()
        df.obs = df.obs.astype(float) # значения переводятся в формат float
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="%d.%m.%Y")
        df.index = df['date'] # Индекс дата
        df = df.drop('date', axis = 1)
        df = df.asfreq(Frequency) # Установка частотности
        base_period =  len(df['obs']) - Deep_forecast_period  
        quantity_pseudo_foracasts = Deep_forecast_period - Forecast_horizon + 1                 # количество псевдовневыборочных прогнозов

        # составление двумерного массива прогнозов                                                                             
        forecast_table = []
        for i in range(quantity_pseudo_foracasts):
                #const_RWD_r = sma.ARIMA(df['obs'][:base_period + i], order=(0, 1, 0), enforce_stationarity=False).fit().params.const     # рассчитываем константу смещения, для каждого нового момента прогнозирования она переоценивается. 
                const_RWD_r = df['obs'].diff().mean().round(2)
                forecast_table.append([df.iloc[base_period - 1 + i, 0] + (j + 1) * const_RWD_r for j in range(Forecast_horizon)])                         # формула модели RWD, заполняется масив для каждого момента прогнозирования. Затем это записывается в общий масив прогнозов.
        
        
        # групировка по шагам
        steps_table=[]
        for i in range(Forecast_horizon):                                                
                steps_table.append([forecast_table[j][i] for j in range(quantity_pseudo_foracasts)])
        dic = { f's{i+1}' : steps_table[i] for i in range(Forecast_horizon)}
        df = pd.DataFrame.from_dict(dic)
        
        return(round(df,2))
def ps_RWDS_forecast(Data: pd.DataFrame,
                     Deep_forecast_period: int,
                     Forecast_horizon: int,
                     Seasonality : int):
    df = Data.copy()
    df.obs = df.obs.astype(float) # значения переводятся в формат float
    base_period =  len(df['obs']) - Deep_forecast_period     # длина базового периода
    quantity_pseudo_foracasts = Deep_forecast_period - Forecast_horizon + 1 # количество псевдовневыборочных прогнозов
                                                                                    
    forecast_table = []   # список прогнозов
    for i in range(quantity_pseudo_foracasts):
        forecast_table.append(
            [
            round(
                df.iloc[base_period + j - Seasonality - 1 + i, 1] +
                df.iloc[:base_period + j, 1].diff(Seasonality).mean(),
                2)
            for j in range(Forecast_horizon)
            ]
        )
            
    # групировка по шагам
    steps_table=[]
    for i in range(Forecast_horizon):                                                
            steps_table.append([forecast_table[j][i] for j in range(quantity_pseudo_foracasts)])
    #print(s)
    dic = { f's{i+1}' : steps_table[i] for i in range(Forecast_horizon)}
    df = pd.DataFrame.from_dict(dic)
    
    return df
def ps_TS_forecast(Data: pd.DataFrame,
                   Deep_forecast_period: int,
                   Forecast_horizon: int):
    df = Data.copy()
    df['T'] = [i + 1 for i in range(len(df['obs']))]
    df.obs = df.obs.astype(float)
    base_period =  len(df) - Deep_forecast_period  
    quantity_pseudo_foracasts = Deep_forecast_period - Forecast_horizon + 1                 # количество псевдовневыборочных прогнозов
                                                                                  
    forecast_table = []
    for i in range(quantity_pseudo_foracasts):
        alpha = smf.ols('obs ~ T', data=df.iloc[:base_period + i,[1, 2]]).fit().params['Intercept']     # рассчитываем константу 1
        betta = smf.ols('obs ~ T', data=df.iloc[:base_period + i,[1, 2]]).fit().params['T']     # рассчитываем константу 2
        forecast_table.append([alpha + df.iloc[base_period + i + j, 2] * betta for j in range(Forecast_horizon)]) #посмотреть
        #print(alpha)
        #print(betta)
    
    
    
    # групировка по шагам
    steps_table=[]
    for i in range(Forecast_horizon):                                                
            steps_table.append([forecast_table[j][i] for j in range(quantity_pseudo_foracasts)])
    #print(s)
    dic = { f's{i+1}' : steps_table[i] for i in range(Forecast_horizon)}
    df = pd.DataFrame.from_dict(dic)
    df = round(df,2)
    return(df)
def ps_ARIMA_forecast(Data: pd.DataFrame,
                      Deep_forecast_period: int,
                      Forecast_horizon: int,
                      Frequency: str):
    
    df = Data.copy()
    df.obs = df.obs.astype(float) # значения переводятся в формат float
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="%d.%m.%Y")
    df.index = df['date'] # Индекс дата
    df = df.drop('date', axis = 1)
    df = df.asfreq(Frequency) # Установка частотности
    base_period =  len(df['obs']) - Deep_forecast_period  
    quantity_pseudo_foracasts = Deep_forecast_period - Forecast_horizon + 1                 # количество псевдовневыборочных прогнозов
                                                                               
    forecast_table = []
    for i in range(quantity_pseudo_foracasts):
        forecast_table.append(pmd.auto_arima(df.obs[:base_period + i],
                                             information_criterion = 'bic',
                                             max_p = 5,
                                             max_d = 2,
                                             max_q = 5,
                                             error_action="ignore",
                                             stepwise=True).predict(Forecast_horizon).tolist())
    
    # групировка по шагам
    steps_table=[]
    for i in range(Forecast_horizon):                                                
            steps_table.append([forecast_table[j][i] for j in range(quantity_pseudo_foracasts)])
    #print(s)
    dic = { f's{i+1}' : steps_table[i] for i in range(Forecast_horizon)}
    df = pd.DataFrame.from_dict(dic)
    df = round(df,2)
    
    return df

# Модели для реального прогноза
def RW_real_forecast(Data: pd.DataFrame,
                     Forecast_horizon: int):

        df = Data.copy()
        df.obs = df.obs.astype(float) # значения переводятся в формат float

        RW_forecast_list = [df.iloc[-1,1] for _ in range(Forecast_horizon)] # из дф берется последняя известная точка, прогноз - значения этой точки, итог массив длиной forecast_horizon с одним и тем же числом.
     
        return(RW_forecast_list)
def RWS_real_forecast(Data: pd.DataFrame,       
                      Forecast_horizon: int,
                      Seasonality: int):
    
    if Seasonality < Forecast_horizon:
        raise Real_Forecast_Error('Seasonality < Forecast_horizon, Функция подходит только для горизонта прогноза не привышающего величину сезонного сдвига')
    
    df = Data.copy()
    df.obs = df.obs.astype(float) # значения переводятся в формат float 
    
    RWS_forecast_list = [df.iloc[- Seasonality + i ,1] for i in range(Forecast_horizon)] # из дф берется масив с данными смещенными на один месяц, так для каждого момента прогнозирования
    
    return(RWS_forecast_list)
def RWD_real_forecast(Data: pd.DataFrame,
                 Forecast_horizon: int,
                 Frequency: str):
    
        df = Data.copy()
        df.obs = df.obs.astype(float) # значения переводятся в формат float
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="%d.%m.%Y")
        df.index = df['date'] # Индекс дата
        df = df.drop('date', axis = 1)
        df = df.asfreq(Frequency) # Установка частотности                                                                       
        
        #const_RWD_r = sma.ARIMA(df['obs'], order=(0, 1, 0), enforce_stationarity=False).fit().params.const.round(2)     # рассчитываем константу на всей выборке
        const_RWD_r = df['obs'].diff().mean().round(2)
        RWD_forecast_list = [df.iloc[-1, 0] + (j + 1) * const_RWD_r for j in range(Forecast_horizon)]    # формула модели RWD
        
        return(RWD_forecast_list)
def RWDS_real_forecast(Data: pd.DataFrame,
                  Forecast_horizon: int,
                  Seasonality: int):
    
    if Seasonality < Forecast_horizon:
        raise Real_Forecast_Error('Seasonality < Forecast_horizon, Функция подходит только для горизонта прогноза не привышающего величину сезонного сдвига')
    
    df = Data.copy()
    df.obs = df.obs.astype(float) # значения переводятся в формат float
                                                                                    
    
    RWDS_forecast_list = [round(df.iloc[- Seasonality - 1 + i, 1] + df.obs.diff(Seasonality).mean(), 2) for i in range(Forecast_horizon)]
            
    
    return RWDS_forecast_list
def TS_real_forecast(Data: pd.DataFrame,            
                Forecast_horizon: int):
    
    df = Data.copy()
    df['T'] = [i + 1 for i in range(len(df['obs']))]
    df.obs = df.obs.astype(float)
                                                                                 
    
    alpha = smf.ols('obs ~ T', data=df.iloc[:,[1, 2]]).fit().params['Intercept']    # рассчитываем константу 1
    betta = smf.ols('obs ~ T', data=df.iloc[:,[1, 2]]).fit().params['T']     # рассчитываем константу 2
    TS_forecast_list = [round(alpha + (df[-1:].index.to_list()[0] + i) * betta, 2) for i in range(Forecast_horizon)] 
        
    return(TS_forecast_list)
def ARIMA_real_forecast(Data: pd.DataFrame,
                      Forecast_horizon : int, 
                      Frequency: str):
    
    df = Data.copy()
    df.obs = df.obs.astype(float) # значения переводятся в формат float
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, format="%d.%m.%Y")
    df.index = df['date'] # Индекс дата
    df = df.drop('date', axis = 1)
    df = df.asfreq(Frequency) # Установка частотности  
    
    ARIMA_forecast_list = pmd.auto_arima(df.obs, information_criterion = 'bic', d = 1, stepwise=True).predict(Forecast_horizon).to_list()
    ARIMA_forecast_list = [round(num ,2) for num in ARIMA_forecast_list]
    
    return ARIMA_forecast_list