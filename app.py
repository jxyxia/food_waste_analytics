import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

# Set page layout
st.set_page_config(page_title="🌍 Food Waste Analytics", layout="wide")

# --- Load data from SQLite ---
def get_data():
    conn = sqlite3.connect("food_waste.db")
    df = pd.read_sql_query("SELECT * FROM food_waste", conn)
    conn.close()
    return df

# Load and clean
df = get_data()
df.columns = df.columns.str.strip()

def get_column(df, name):
    return next((col for col in df.columns if col.lower() == name.lower()), None)

food_col = get_column(df, "Total Waste (Tons)")
loss_col = get_column(df, "Economic Loss (Million $)")
country_col = get_column(df, "country")
year_col = get_column(df, "Year")
food_category_col = get_column(df, "Food Category")

st.title("🌍 Food Waste Analytics & Prediction System")
st.markdown("Analyze, visualize, and predict global food waste trends over time.")

if not (food_col and loss_col and country_col and year_col):
    st.error("❌ Required columns are missing in the database.")
    st.stop()

# Summary Metrics
total_waste = df[food_col].sum()
total_loss = df[loss_col].sum()
col1, col2 = st.columns(2)
col1.metric("🌾 Total Food Waste (Tonnes)", f"{total_waste:,.2f}")
col2.metric("💸 Total Economic Loss ($)", f"${total_loss:,.2f}")
st.markdown("---")

# Sidebar Filters
st.sidebar.header("🔍 Filter Data")
selected_country = st.sidebar.selectbox("🌎 Select Country", df[country_col].unique())
selected_food_category = st.sidebar.selectbox("🍱 Select Food Category", ["Overall"] + sorted(df[food_category_col].unique()))

# Apply Filters
filtered_data = df[df[country_col] == selected_country]
if selected_food_category != "Overall":
    filtered_data = filtered_data[filtered_data[food_category_col] == selected_food_category]

if filtered_data.empty:
    st.warning("⚠️ No data available for the selected filters!")
    st.stop()

# Linear Regression Model
def train_model(df, year_col, target_col):
    df = df.dropna(subset=[year_col, target_col])
    X = df[[year_col]].values.reshape(-1, 1)
    y = df[target_col].values
    model = LinearRegression()
    model.fit(X, y)
    return model

def predict_future(model, start_year, end_year):
    future_years = np.arange(start_year, end_year + 1).reshape(-1, 1)
    predictions = model.predict(future_years)
    return future_years.flatten(), predictions

# Plot: Food Waste
st.subheader(f"📈 Food Waste Trends in {selected_country}")
plot_df = filtered_data.groupby(year_col, as_index=False)[food_col].mean()
fig = px.line(plot_df, x=year_col, y=food_col, markers=True, title="Actual Food Waste")
fig.update_layout(title_font=dict(size=24))

model = train_model(plot_df, year_col, food_col)
start_year = max(plot_df[year_col])
future_years, future_preds = predict_future(model, start_year, start_year + 5)

future_df = pd.DataFrame({year_col: future_years, food_col: future_preds})
fig.add_scatter(x=future_df[year_col], y=future_df[food_col], mode='lines+markers',
                name='Predicted', line=dict(dash='dash', color='green'))
fig.update_layout(xaxis_title="Year", yaxis_title="Food Waste (Tonnes)", xaxis_title_font=dict(size=18))
st.plotly_chart(fig, use_container_width=True)

# Plot: Economic Loss
st.subheader(f"💰 Economic Loss in {selected_country}")
plot_df = filtered_data.groupby(year_col, as_index=False)[loss_col].mean()
fig = px.line(plot_df, x=year_col, y=loss_col, markers=True, title="Actual Economic Loss")
fig.update_layout(title_font=dict(size=24))

model = train_model(plot_df, year_col, loss_col)
future_years, future_preds = predict_future(model, start_year, start_year + 5)

future_df = pd.DataFrame({year_col: future_years, loss_col: future_preds})
fig.add_scatter(x=future_df[year_col], y=future_df[loss_col], mode='lines+markers',
                name='Predicted', line=dict(dash='dash', color='orange'))
fig.update_layout(xaxis_title="Year", yaxis_title="Economic Loss (Million $)", xaxis_title_font=dict(size=18))
st.plotly_chart(fig, use_container_width=True)

# Download Filtered Data
st.markdown("### 📤 Download Filtered Data")
csv = filtered_data.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", data=csv, file_name="filtered_food_waste.csv", mime="text/csv")
