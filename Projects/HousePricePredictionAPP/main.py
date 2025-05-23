import streamlit as st

import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from pandas import DataFrame

def generate_house_data (n_samples=1000):
    np.random.seed(50)
    size = np.random.normal(1400, 50, n_samples)
    price = size * 50 + np.random.normal(0, 50, n_samples)
    return pd.DataFrame({'size': size, 'price': price})

def train_model():
    df = generate_house_data(n_samples=100)
    X = df[['size']]
    Y = df['price']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    return model

def main():
    st.title('HOUSE PRICE PREDICTION APP')
    
    st.write('PUT IN YOUR HOUSE SIZE TO KNOW ITS PRICE')
    
    model = train_model()
    
    size = st.number_input ('HOUSE SIZE', min_value=100, max_value=10000, value=1500)
    
    if st.button('PREDICT PRICE'):
        predicted_price = model.predict([[size]])
        st.success(f'ESTIMATED PRICE: ${predicted_price[0]:,.2f}')
        
        df = generate_house_data()
        
        fig = px.scatter(df, x='size', y='price', title= 'SIZE vs HOUSE PRICE')
        fig.add_scatter(x=[size], y=[predicted_price[0]],
                mode='markers',
                marker=dict(size=15, color='green'),
                name= 'PREDICTION')
        
        st.plotly_chart(fig)
        
        
if __name__ == '__main__':
    main()
    
    
    
    
