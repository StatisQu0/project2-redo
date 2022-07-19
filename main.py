# Inspired by Python Engineer at https://www.youtube.com/watch?v=0E_31WqVzCY&t=530s

from sqlalchemy import column
import streamlit as st
from datetime import datetime
from sympy import fu
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from dateutil.relativedelta import relativedelta
import numpy as np

START = datetime.now() - relativedelta(years=3) 
TODAY = datetime.today()

st.title('Stock Prediction')
stocks = ('ALK', 'DAL', 'LUV', 'SAVE', 'ALGT', 'AAL', 'ULCC', 'HA', 'JBLU', 'SNCY', 'UAL')
# stocks = ('AMZN', 'SAVE', 'SYY')

selected_stock = st.selectbox('Select Dataset', stocks)

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace = True)
    return data

data_load_state = st.text('Load Data...')
data = load_data(selected_stock)
data_load_state.text('Loading Data...Done!')

st.subheader('Raw Data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text = 'Time Series Data', xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

plot_raw_data()

# Forcasting

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns = {'Date' : 'ds', 'Close' : 'y'})
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods = period)
# df_no_weekdays = df[df['ds'].dt.dayofweek < 5]
future = future[future['ds'].dt.dayofweek < 5]
forcast = m.predict(future)

st.subheader('Forcast Data')
st.write(forcast.tail())

st.write('forcast data')
fig1 = plot_plotly(m, forcast)
st.plotly_chart(fig1)

st.write('forcast components')
fig2 = m.plot_components(forcast)
st.write(fig2)