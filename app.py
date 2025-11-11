import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Retail Sales Forecasting", layout="wide")

st.title("📈 Retail Sales Forecasting Dashboard")
st.markdown("This app uses a **Random Forest Regressor** to predict future sales using your dataset.")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("model/superstore_sales_rf.pkl")
st.sidebar.success("✅ Model loaded successfully!")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload your sales CSV (with a 'date' and 'sales' column):", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower()
    
    if 'date' not in df.columns or 'sales' not in df.columns:
        st.error("❌ CSV must contain 'date' and 'sales' columns.")
    else:
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)

        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head())

        # Plot uploaded sales data
        st.subheader("Sales Over Time")
        st.line_chart(df.set_index('date')['sales'])

        # -----------------------------
        # FORECAST SECTION
        # -----------------------------
        st.subheader("🔮 Forecast Future Sales")

        # Slider to select forecast period
        periods = st.slider("Select number of days to forecast:", min_value=7, max_value=90, value=30, step=7)

        if st.button("⚡ Forecast Sales"):
            # Prepare features
            last_date = df['date'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods)

            future_df = pd.DataFrame({
                'date': future_dates,
                'day': future_dates.day,
                'month': future_dates.month,
                'year': future_dates.year,
                'dayofweek': future_dates.dayofweek
            })

            # Predict with your Random Forest model
            future_df['predicted_sales'] = model.predict(future_df[['day','month','year','dayofweek']])

            # Combine actual + forecast data
            past_df = df[['date','sales']].copy()
            past_df['type'] = 'Actual'
            future_df['sales'] = future_df['predicted_sales']
            future_df['type'] = 'Forecast'

            combined = pd.concat([past_df, future_df[['date','sales','type']]])

            # Plot
            st.subheader(f"📊 Forecast for next {periods} days")
            fig, ax = plt.subplots(figsize=(10,5))
            for label, grp in combined.groupby('type'):
                ax.plot(grp['date'], grp['sales'], label=label)
            plt.legend()
            plt.title("Actual vs Forecasted Sales")
            plt.xlabel("Date")
            plt.ylabel("Sales")
            st.pyplot(fig)

            # Show forecast data
            st.subheader("📄 Forecasted Data")
            st.dataframe(future_df[['date','predicted_sales']])

            # Download button
            csv = future_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Forecast CSV", csv, "sales_forecast.csv", "text/csv")
