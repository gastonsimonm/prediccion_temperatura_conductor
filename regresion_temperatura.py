import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


    # 1. Cargamos los datos del archivo
    df = pd.read_csv('datos_temp.csv')

    # 2. Definimos variables dependientes (y) e independientes (x)
    x = df[['Temp_Ambiente_C', 'Velocidad_Viento_ms']]
    y = df['Temp_Conductor_C']

    # 3. Entrenamos el modelo de Regresión Lineal
    model = LinearRegression()
    model.fit(x, y)

    # 4. Calculamos métricas y coeficientes
    y_pred = model.predict(x)
    r2 = r2_score(y, y_pred)        # El coeficiente indica que tan bien se ajusta el modelo al problema (debe ser cercano a 1)
    ordenada = model.intercept_
    coef_temp = model.coef_[0]
    coef_viento = model.coef_[1]

    # Mostramos los valores en pantalla
    print("\n--- RESULTADOS DEL MODELO ---")
    print(f'Coeficiente de determinación r^2 = {r2}')             
    print(f'Ordenada al origen = {ordenada}')
    print(f'Coef. temperatura ambiente = {coef_temp}')
    print(f'Coef. velocidad del viento = {coef_viento}')
    
    print("ECUACIÓN")
    print(f'T_conductor = {ordenada} + ({coef_temp} * T_ambiente) + ({coef_viento:} * V_viento)')

    # 5. Ingresamos valor de la Velocidad del Viento y Temperatura Ambiente
    print("\n--- PREDICCIÓN TEMPERATURA DEL CONDUCTOR ---")
    try:
        v_viento = float(input('Ingrese la velocidad del viento (m/s): '))
        t_ambiente = float(input('Ingrese la temperatura ambiente (°C): '))
        t_conductor = ordenada + (coef_temp * t_ambiente) + (coef_viento * v_viento)
        print(f'Temperatura del conductor estimada = {t_conductor} °C')
    except ValueError:
        print("Error: Por favor, ingrese únicamente valores numéricos.")


