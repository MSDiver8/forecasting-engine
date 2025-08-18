class Psevdo_Forecast_Error(Exception):
    """Ошибка, возникающая при посторение псевдо прогноза моделями"""
    pass

class Real_Forecast_Error(Exception):
    """Ошибка, возникающая при посторение реального прогноза моделями"""
    pass

class Auto_params_selection_Error(Exception):
    """Ошибка, возникающая при автоматическом выборе параметров для базовых функций прогнозирования"""
    pass