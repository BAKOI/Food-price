# Save the Streamlit app code to a file named 'app.py'

app_code = """
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

st.title("Commodity Food Price Analysis and Prediction")

# File upload
st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)
    
    # Display the dataframe
    st.header("Data Overview")
    st.write(data)

    # Visualize the data
    st.header("Price Trend")
    plt.figure(figsize=(10, 5))
    for column in data.columns[1:]:
        plt.plot(data['Date'], data[column], label=column)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

    # Simple price prediction
    st.header("Price Prediction")
    commodity = st.selectbox("Select Commodity", data.columns[1:])
    
    # Prepare data for prediction
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    df = data[[commodity]].dropna()

    # Feature engineering
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Year'] = df.index.year

    # Train-test split
    X = df[['Day', 'Month', 'Year']]
    y = df[commodity]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Display the results
    st.write("Mean Squared Error:", mean_squared_error(y_test, predictions))
    st.write("Predicted Prices:", predictions)

    # Predict future prices
    st.header("Predict Future Prices")
    future_dates = pd.date_range(start=data.index[-1], periods=30, freq='D')
    future_df = pd.DataFrame({
        'Day': future_dates.day,
        'Month': future_dates.month,
        'Year': future_dates.year
    })
    future_predictions = model.predict(future_df)
    
    future_df['Predicted Price'] = future_predictions
    future_df.set_index(future_dates, inplace=True)
    st.write(future_df[['Predicted Price']])
    
    # Plot future prices
    plt.figure(figsize=(10, 5))
    plt.plot(future_df.index, future_df['Predicted Price'], label='Predicted Price', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

else:
    st.info("Please upload a CSV file.")
"""

# Save the code to a file
file_path = '/mnt/data/app.py'
with open(file_path, 'w') as file:
    file.write(app_code)

file_path
