import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings


def adfuller_test(sales):
    result = adfuller(sales)
    labels = ['ADF Test Statistics', 'P-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):
        print(label+' : '+str(value))
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis(Ho), "
              "reject the null hypothesis, Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis, Time series has a unit root, indicating it is non-stationary")


if __name__ == '__main__':
    df = pd.read_csv('Perrin-Freres-monthly-champagne-sales-millions.csv')
    # print(df.head())
    # print(df.tail())
    df.columns = ['Month', 'Sales']
    # print(df.head())
    df.drop(106, axis=0, inplace=True)
    df.drop(105, axis=0, inplace=True)
    # print(df.tail())
    df['Month'] = pd.to_datetime(df['Month'])
    # print(df.head())
    df.set_index('Month', inplace=True)
    # print(df.head())
    print(df.describe())
    # df.plot()
    plt.show()

    from statsmodels.tsa.stattools import adfuller
    test_result = adfuller(df['Sales'])
    print(test_result)
    print(adfuller_test(df['Sales']))
    df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(1)
    df['Seasonal First Difference'] = df['Sales'] - df['Sales'].shift(12)
    print(df.head(14))
    print(adfuller_test(df['Seasonal First Difference'].dropna()))
    df['Seasonal First Difference'].plot()
    # plt.show()

    from pandas.plotting import autocorrelation_plot
    autocorrelation_plot(df['Sales'])
    # plt.show()

    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import statsmodels as sm

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsaplots.plot_acf(df['Seasonal First Difference'].iloc[13:], lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsaplots.plot_pacf(df['Seasonal First Difference'].iloc[13:], lags=40, ax=ax2)
    print("#########################################")
    print(df['Sales'])
    print(df['Sales First Difference'])

    # from statsmodels.tsa.arima_model import ARIMA
    # model = ARIMA(df['Sales'], order=(1, 1, 1))
    # model_fit = model.fit()

    import statsmodels.api as sm
    model = sm.tsa.arima.ARIMA(df['Sales'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    df['forecast'] = results.predict(start=90, end=103, dynamic=True)
    df[['Sales', 'forecast']].plot(figsize=(12, 8))
    plt.show()










