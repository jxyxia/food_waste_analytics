import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st

# Load data
conn = sqlite3.connect("food_waste.db")
df = pd.read_sql_query("SELECT year, food_waste_tonnes FROM food_waste", conn)
conn.close()

# Prepare data
X = df[["year"]]
y = df["food_waste_tonnes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict future waste
st.subheader("🔮 Food Waste Prediction for Next 5 Years")
future_years = pd.DataFrame({"year": [2026, 2027, 2028, 2029, 2030]})
predictions = model.predict(future_years)

# Display results
for year, waste in zip(future_years["year"], predictions):
    st.write(f"**{year}:** {waste:.2f} tonnes")
