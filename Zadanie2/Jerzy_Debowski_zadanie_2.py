import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


def fahrenheit_to_celsius(temperature):
    return round((temperature - 32) * 5 / 9, 2)


temperatures_df = pd.read_csv('data.csv', sep=',', header=4)
temperatures_df.columns = [
    'Data',
    'Temperatura',
    'Odchyłka',
]
temperatures_df['Data'] = temperatures_df['Data'].apply(
    lambda x: pd.to_datetime(str(x), format='%Y%m').year
)
temperatures_df['Temperatura'] = temperatures_df['Temperatura'].apply(
    lambda x: fahrenheit_to_celsius(x)
)
temperatures_df['Odchyłka'] = temperatures_df['Odchyłka'].apply(
    lambda x: fahrenheit_to_celsius(x)
)

temperatures_df_description = temperatures_df.describe()
print(f'Opis ramki danych:\n{temperatures_df_description}')

slope, intercept, r_value, p_value, std_err = stats.linregress(
    x=temperatures_df['Data'],
    y=temperatures_df['Temperatura']
)

probably_january_2022_temperature = round(slope * 2022 + intercept, 2)
january_2022_temperature = fahrenheit_to_celsius(31.17)
print(
    f'Prognozowana średnia temperatura w styczniu 2022: {probably_january_2022_temperature}\n'
    f'Faktyczna średnia temperatura w styczniu 2022: {january_2022_temperature}'
)

y_pred = slope * temperatures_df['Data'] + intercept
plt.scatter(temperatures_df['Data'], temperatures_df['Temperatura'], label='Dane rzeczywiste')
plt.plot(temperatures_df['Data'], y_pred, color='red', label='Regresja liniowa')
plt.xlabel('Rok')
plt.ylabel('Temperatura')
plt.legend()
plt.title('Regresja liniowa w szeregu czasowym')
plt.show()

