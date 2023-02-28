import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Чтобы определить движение цены фьючерса ETHUSDT,
# исключая движения, вызванные влиянием цены BTCUSDT,
# был выбран метод линейной регрессии.
# При помощи него можно смоделировать взаимосвязь между
# зависимой (цена фьючерса ETHUSDT) и независимой
# переменной (цена BTCUSDT).
# Анализируя коэффициенты независимой переменной,
# можно определить влияние цены BTCUSDT на цену
# ETHUSDT и исключить его.

# данные из датасета cryptocurrency historical prices из kaggle
btc_df = pd.read_csv('coin_Bitcoin.csv')
eth_df = pd.read_csv('coin_Ethereum.csv')

# конвертация дат в datetime объекты для выполнения временных операций
btc_df['Date'] = pd.to_datetime(btc_df['Date'])
eth_df['Date'] = pd.to_datetime(eth_df['Date'])

# фильтрация нужных колонок
btc_df = btc_df.loc[:, ['Date', 'Close']]
eth_df = eth_df.loc[:, ['Date', 'Close']]

# объединение датафреймов по Date
merged_df = pd.merge(btc_df, eth_df, on='Date', suffixes=('_btc', '_eth'))

# подгонка модели линейной регрессии к ценам BTC и ETH
reg = LinearRegression().fit(merged_df[['Close_btc']], merged_df[['Close_eth']])

# прогнозирование цены ETH на основе цены BTC
predicted_eth_price = reg.predict(merged_df[['Close_btc']])

# вычитание прогнозируемого движения цены ETH из фактического движения цены ETH
eth_price_movements_excl_btc = merged_df['Close_eth'] - predicted_eth_price.flatten()

fig, ax = plt.subplots()
ax.plot(merged_df['Date'], eth_price_movements_excl_btc, label='Результат')
ax.plot(merged_df['Date'], merged_df['Close_eth'], label='Фактические даные')
ax.set_xlabel('Дата')
ax.set_ylabel('ETH')
ax.legend()
plt.show()
