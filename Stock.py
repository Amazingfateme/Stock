import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image


st.write('''
# STOCK VISUALIZER 

         
**FATEME HASHEMI**
''')

img=Image.open("C:/Users/amazing/Desktop/exchangeproject/Deloitte BrandVoice_ Transforming The Balance Sheet_ Navigating New Lease Standards For Success.png")

st.image(img,width=500)

st.sidebar.header('Insert Data')

def data():
    n=st.sidebar.text_input('How many days you wanna predict?',5)
    symbol = st.sidebar.selectbox('select the symbol:',['FOOLAD','AMZN','AAPL','GOLD'])
    return n,symbol

def get_data():
    if symbol == 'FOOLAD':
        df = pd.read_csv('C:/Users/amazing/Desktop/exchangeproject/foolad.csv')
    if symbol == 'AMZN':
        df = pd.read_csv('C:/Users/amazing/Desktop/exchangeproject/AMZN.csv')
    if symbol == 'AAPL':
        df = pd.read_csv('C:/Users/amazing/Desktop/exchangeproject/AAPL.csv')
    if symbol == 'GOLD':
        df = pd.read_csv('C:/Users/amazing/Desktop/exchangeproject/GOLD.csv')
    df = df.set_index(pd.DatetimeIndex(df['Date'].values))
    return df 


def get_company_name(symbol):
    if symbol == 'FOOLAD':
        return 'FOOLAD'
    elif symbol == 'AMZN':
        return 'AMAZON'
    elif symbol == 'AAPL':
        return 'APPLE'
    elif symbol == 'GOLD':
        return 'GOLD'
    else:
        return 'NONE'

n , symbol = data()
df=get_data()
company = get_company_name(symbol)
st.header(company +' '+ 'Close Price\n')
st.line_chart(df['Close'])
st.header(company +' '+ 'Volume\n')
st.line_chart(df['Volume'])
st.header('Stock Datas')
st.write(df.describe())


df = df[['Close']]
forecast = int(n)
df['Prediction']=df[['Close']].shift(-forecast)

x=np.array(df.drop(labels='Prediction',axis=1))
x=x[:-forecast]
y=np.array(df['Prediction'])
y=y[:-forecast]


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)
mysvr = SVR(kernel='rbf',C=1000,gamma=0.1)
mysvr.fit(xtrain,ytrain)
svmconf = mysvr.score(xtest,ytest)
st.header('SVM accuracy')
st.success(svmconf)


x_forecast = np.array(df.drop(labels='Prediction',axis=1))[-forecast:]
svmpred = mysvr.predict(x_forecast)
st.header('SVM Prediction')
st.success(svmpred)

lr = LinearRegression()
lr.fit(xtrain,ytrain)
lrconf= lr.score(xtest,ytest)
st.header('LR accuracy')
st.success(lrconf)


lrpred = lr.predict(x_forecast)
st.header('LR Prediction')
st.success(lrpred)




