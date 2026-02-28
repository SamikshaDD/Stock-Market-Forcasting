import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import EMAIndicator, SMAIndicator
import datetime
from datetime import date
import plotly.graph_objects as go
from plotly.offline import iplot
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
from tensorflow import keras
from sklearn.metrics import mean_squared_error

st.title('Stock X')
st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your Stocks below')


def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize','Analyze', 'Predict'])
    
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Analyze':
        dfframe()
    else:
        LSTM_ALGO(df)



@st.cache_resource
def download_df(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df



Stock = st.sidebar.text_input('Enter a Stock Symbol', value='MSFT')
Stock = Stock.upper()
today = datetime.date.today()
duration = st.sidebar.number_input('Enter the duration', value=1500)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)
if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
        download_df(Stock, start_date, end_date)
    else:
        st.sidebar.error('Error: End date must fall after start date')



df = download_df(Stock, start_date, end_date)


def tech_indicators():
    st.header('Technical Indicators')
    Stock = st.radio('Choose a Technical Indicator to Visualize', ['Chart', 'BB', 'SMA', 'EMA'])

    # Bollinger bands
    bb_indicator = BollingerBands(df.Close)
    bb = df.copy()
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    # Creating a new dfframe
    bb = bb[['Close', 'bb_h', 'bb_l']]
    # SMA
    sma = SMAIndicator(df.Close, window=14).sma_indicator()
    # EMA
    ema = EMAIndicator(df.Close).ema_indicator()

    if Stock == 'Chart':
        st.write('Candlestick chart')
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlesticks',
            increasing_line_color='green',  # Color for bullish days
            decreasing_line_color='red',    # Color for bearish days
            hoverinfo='x+y',           # Additional info on hover
            hoverlabel=dict(font=dict(size=12)),
            text=df.index.strftime('%Y-%m-%d'),  # Date as text on hover
            whiskerwidth=0.2,               # Width of whiskers
            line=dict(width=1),             # Width of candlestick lines
        )])
        fig.update_layout(height=800, width=800)
        st.plotly_chart(fig)
    
    elif Stock == 'BB':
        st.write('BollingerBands')
        fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True, subplot_titles = ('chart', 'Slider'), vertical_spacing = 0.1, row_width = [0.2, 0.7])

        # ----------------
        # Candlestick Plot
        fig.add_trace(go.Candlestick(
                             x=df.index,
                             open=df['Open'],
                             high=df['High'],
                             low=df['Low'],
                             close=df['Close'],
                             name='Candlesticks',
                             increasing_line_color='green',  # Color for bullish days
                             decreasing_line_color='red',    # Color for bearish days
                             hoverinfo='x+y+name',           # Additional info on hover
                             hoverlabel=dict(font=dict(size=12)),
                             text=df.index.strftime('%Y-%m-%d'),  # Date as text on hover
                             whiskerwidth=0.2,               # Width of whiskers
                             line=dict(width=1),
                             ))
        # bb
        fig.add_trace(go.Scatter(x=df.index,
                         y = bb['bb_h'],
                         line_color = 'white',
                         line_width = 0.8,
                         name = 'higher band'),
              row = 1, col = 1)
        fig.add_trace(go.Scatter(x=df.index,
                         y = bb['bb_l'],
                         line_color = 'white',
                         line_width = 0.8,
                         name = 'lower band'),
              row = 1, col = 1)
        fig.update_layout(height=700, width=700)
        st.plotly_chart(fig)


    elif Stock == 'SMA':
        st.write('Simple Moving Average')
        fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True, subplot_titles = ('Chart', 'Slider'), vertical_spacing = 0.1, row_width = [0.2, 0.7])

        # ----------------
        # Candlestick Plot
        fig.add_trace(go.Candlestick(
                             x=df.index,
                             open=df['Open'],
                             high=df['High'],
                             low=df['Low'],
                             close=df['Close'],
                             name='Candlesticks',
                             increasing_line_color='green',  # Color for bullish days
                             decreasing_line_color='red',    # Color for bearish days
                             hoverinfo='x+y+name',           # Additional info on hover
                             hoverlabel=dict(font=dict(size=12)),
                             text=df.index.strftime('%Y-%m-%d'),  # Date as text on hover
                             whiskerwidth=0.2,               # Width of whiskers
                             line=dict(width=1),
                             ))
        # Moving Average
        fig.add_trace(go.Scatter(x=df.index,
                         y = sma,
                         line_color = 'white',
                         line_width = 0.8,
                         name = 'sma'),
              row = 1, col = 1)
        fig.update_layout(height=800, width=800)
        st.plotly_chart(fig)
    else:
        st.write('Expoenetial Moving Average')
        fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True, subplot_titles = ('Chart', 'Slider'), vertical_spacing = 0.1, row_width = [0.2, 0.7])

        # ----------------
        # Candlestick Plot
        fig.add_trace(go.Candlestick(
                             x=df.index,
                             open=df['Open'],
                             high=df['High'],
                             low=df['Low'],
                             close=df['Close'],
                             name='Candlesticks',
                             increasing_line_color='green',  # Color for bullish days
                             decreasing_line_color='red',    # Color for bearish days
                             hoverinfo='x+y+name',           # Additional info on hover
                             hoverlabel=dict(font=dict(size=12)),
                             text=df.index.strftime('%Y-%m-%d'),  # Date as text on hover
                             whiskerwidth=0.2,               # Width of whiskers
                             line=dict(width=1),
                             ))
        # exponential Moving Average
        fig.add_trace(go.Scatter(x=df.index,
                         y = ema,
                         line_color = 'white',
                         line_width = 0.8,
                         name = 'ema'),
              row = 1, col = 1)
        fig.update_layout(height=800, width=800)
        st.plotly_chart(fig)


def dfframe():
    st.header('Analysis of ' + str(Stock))
    st.dataframe(df.tail(10))
    fundamental_data, news , technical_analysis = st.tabs(["Fundamental Data", "Top 10 News", "Technical Analysis"])
    
    from alpha_vantage.fundamentaldata import FundamentalData
    key = ' 4BLVRHXAUWVRIZTR'

    with fundamental_data:
        fd = FundamentalData(key,output_format = 'pandas')
        st.subheader('Balance Sheet')
        balance_sheet = fd.get_balance_sheet_annual(Stock)[0]
        bs = balance_sheet.T[2:]
        bs.columns = list(balance_sheet.T.iloc[0])
        st.write(bs)
        
        st.subheader('Income Statement')
        income_statement = fd.get_income_statement_annual(Stock)[0]
        is1 = income_statement.T[2:]
        is1.columns = list(income_statement.T.iloc[0])
        st.write(is1)
        
        st.subheader('Cash Flow Statement')
        cash_flow = fd.get_cash_flow_annual(Stock)[0]
        cf = cash_flow.T[2:]
        cf.columns = list(cash_flow.T.iloc[0])
        st.write(cf)


    
    from finvizfinance.quote import finvizfinance
    stocknews = finvizfinance(Stock)

    with news:
     st.header('News of '+ str(Stock))
     news_df = stocknews.ticker_news()
     for i in range(10):
        st.subheader(f'News {i+1}')
        st.write(news_df['Date'][i])
        st.write(news_df['Title'][i])
        st.write(news_df['Link'][i])

    with technical_analysis:

        st.write("<br>", unsafe_allow_html=True)
                
        from tradingview_ta import TA_Handler, Interval
        handler = TA_Handler(
            symbol=Stock,
            screener="america",
            exchange="NASDAQ",
            interval=Interval.INTERVAL_1_DAY
        )
        
        T1=handler.get_analysis().summary
        
        
        T2=handler.get_analysis().oscillators
        
        T3=handler.get_analysis().moving_averages
        Summery, Oscillators_analysis, MovingAverage_analysis = st.tabs(["Overall Analysis", "Oscillators Analysis", "Moving Average Analysis"])
        with Summery:
            st.write("<br>", unsafe_allow_html=True)
            for key, value in T1.items():
                st.write(f"{key}: {value}")
        
        with Oscillators_analysis:
            st.write("<br>", unsafe_allow_html=True)
            for key, value in list(T2.items())[:4]:
                st.write(f"{key}: {value}")
            st.header("individual oscillators:")
            for key, value in T2['COMPUTE'].items():
                st.write(f"{key}: {value}")
          
        
        with MovingAverage_analysis: 
            st.write("<br>", unsafe_allow_html=True)    
            for key, value in list(T3.items())[:4]:
                st.write(f"{key}: {value}")
            st.header("individual moving averages:")
            for key, value in T3['COMPUTE'].items():
                st.write(f"{key}: {value}")   




def LSTM_ALGO(df):
    dfSet=df.copy()
    dfSet1=dfSet.copy()
    dfSet1['Difference'] = dfSet1['Close'].shift(-1) - dfSet1['Close']

    dfSet1 = dfSet1.drop(dfSet1.index[-1])
    dfSet1 = dfSet1[['Difference']]

    dfset_train = dfSet1.iloc[0:int(0.8 * len(dfSet1)), :]
    dfset_test = dfSet1.iloc[int(0.8 * len(dfSet1)):, :]

    # Feature Scaling
    from sklearn.preprocessing import MinMaxScaler

    training_set = dfSet1.iloc[:, 0:1].values  # Extracting the "Difference" column as numpy array

    # Scaling the data
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # Creating the data structure with 7 timesteps and 1 output
    X_train = []  # Memory with 7 days from day i
    y_train = []  # Day i
    for i in range(30, len(training_set_scaled)):
        X_train.append(training_set_scaled[i - 30:i, 0])
        y_train.append(training_set_scaled[i, 0])

    # Convert lists to numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping: Adding 3rd dimension
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Now, to integrate the forecasting part
    X_forecast = np.array(X_train[-1, 1:])  # Selecting the last 6 values from X_train
    X_forecast = np.append(X_forecast, y_train[-1])  # Appending the last actual y_train value
    X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))  # Reshaping for LSTM input

    # Now, X_train and y_train are prepared for training, and X_forecast is prepared for forecasting.
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import LSTM
    import os
    from keras.models import load_model
    import math

    load_existing_model = True  # Define the variable load_existing_model
    save_model = True  # Define the variable save_model

    if load_existing_model and os.path.exists(str (Stock) + "lstm_model.h5"):
        regressor = load_model( str (Stock) + "lstm_model.h5")
    else:
        regressor = Sequential()
        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units=200, return_sequences=True))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units=200, return_sequences=True))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.1))
        regressor.add(Dense(units=1))
        regressor.compile(optimizer='adam', loss='mean_squared_error')

        regressor.fit(X_train, y_train, epochs=512, batch_size=8)

        if save_model:
            regressor.save(str (Stock) + "lstm_model.h5")

        

    # Testing

    real_stock_price = dfset_test.iloc[:, 0:1].values

    # To predict, we need stock prices of 7 days before the test set
    # So combine train and test set to get the entire df set
    dfset_total = pd.concat((dfset_train, dfset_test), axis=0)
    testing_set = dfset_total[len(dfset_total) - len(dfset_test) - 30:].values
    testing_set = testing_set.reshape(-1, 1)
    # -1=till last row, (-1,1)=>(80,1). otherwise only (80,0)

    # Feature scaling
    testing_set = sc.transform(testing_set)

    # Create df structure
    X_test = []
    for i in range(30, len(testing_set)):
        X_test.append(testing_set[i - 30:i, 0])
    # Convert list to numpy arrays
    X_test = np.array(X_test)

    # Reshaping: Adding 3rd dimension
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Testing Prediction
    predicted_stock_price = regressor.predict(X_test)

    # Getting original prices back from scaled values
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    # Forecasting Prediction
    forecasted_stock_price = regressor.predict(X_forecast)

    # Getting original prices back from scaled values
    forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)

    lstm_pred = forecasted_stock_price[0, 0]

    latest_closed_price = dfSet.iloc[-1]['Close']
    print("Latest closed price:", latest_closed_price)

    final_forecast= latest_closed_price + lstm_pred

    
    error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
    final_forecast= round(final_forecast, 3)
    error_lstm = round(error_lstm, 2)

    print("##############################################################################")
  
     
    st.text("Tomorrow's final prediction is: ")
    st.header(final_forecast)
    st.text("RSME : " + str(error_lstm))
if __name__ == '__main__':
    main()
