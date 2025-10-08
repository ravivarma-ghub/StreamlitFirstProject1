import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Title and description
st.title('📈 Linear Regression Model')
st.write('Upload your CSV file to train the model and make predictions.')

# File uploader
uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])

if uploaded_file is not None:
    # ✅ Correct way to read CSV into a DataFrame
    df = pd.read_csv(uploaded_file)

    st.subheader('📊 Uploaded Data')
    st.write(df.head())

    # Splitting data
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Predictions
    y_pred = model.predict(x_test)

    # Evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader('📉 Model Performance')
    st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")

    # Plot Actual vs Predicted
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, color='blue', alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted')
    st.pyplot(fig)

    # Prediction input
    st.subheader('🔮 Predict with New Data')
    input_data = {}
    for col in x.columns:
        input_data[col] = st.number_input(f'Enter value for **{col}**', value=0.0)
    
    if st.button('Predict'):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        st.success(f'Predicted Value: {prediction[0]:.4f}')