from sklearn.calibration import LabelEncoder
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import datetime as dt

# Set page title
st.set_page_config(page_title="Gold Price Regression Analysis")

# App title
st.title("Gold Price Regression Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        df.drop(['Vol.', 'Change %'], axis=1, inplace=True)
        #df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date']=df['Date'].map(dt.datetime.toordinal)
        df.sort_values(by='Date', ascending=True, inplace=True)
        df.reset_index(drop=True, inplace=True)
        NumCols = df.columns.drop(['Date'])
        df[NumCols] = df[NumCols].replace({',': ''}, regex=True)
        df[NumCols] = df[NumCols].astype('float64')
        
        st.write("Data Preview:")
        st.write(df.head())


        # Select features and target
        st.subheader("Select Features and Target")
        features = st.multiselect("Select features", df.columns.tolist())
        target = st.selectbox("Select target variable", df.columns.tolist())

        if features and target:
            # Prepare the data
            X = df[features]
            y = df[target]
            # Check for non-numeric columns
            non_numeric = X.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric) > 0:
                st.warning(f"Non-numeric columns detected: {', '.join(non_numeric)}. These will be dropped.")
                X = X.select_dtypes(include=[np.number])

            # Handle missing values
            imputer = SimpleImputer(strategy='mean')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

            # Normalize features
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Display results
            st.subheader("Regression Results")
            st.write(f"Mean Squared Error: {mse:.2f}")
            st.write(f"R-squared Score: {r2:.2f}")

            # Plot actual vs predicted values
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            ax.set_title("Actual vs Predicted Values")
            st.pyplot(fig)

            # Feature importance
            st.subheader("Feature Importance")
            importance = pd.DataFrame({'feature': X.columns, 'importance': model.coef_})
            importance = importance.sort_values('importance', ascending=False)
            st.bar_chart(importance.set_index('feature'))

            # Prediction
            st.subheader("Make a Prediction")
            user_input = {}
            for feature in X.columns:
                user_input[feature] = st.number_input(f"Enter value for {feature}")

            if st.button("Predict"):
                user_df = pd.DataFrame([user_input])
                user_df = pd.DataFrame(scaler.transform(user_df), columns=user_df.columns)
                prediction = model.predict(user_df)
                st.write(f"Predicted {target}: {prediction[0]:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your data and ensure it's in the correct format.")

else:
    st.write("Please upload a CSV file to begin the analysis.")