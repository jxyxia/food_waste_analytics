import sqlite3
import pandas as pd

# Load dataset
df = pd.read_csv("dataset/global_food_wastage_dataset.csv")

# Connect to SQLite
conn = sqlite3.connect("food_waste.db")
cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE IF NOT EXISTS food_waste (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    country TEXT,
    year INTEGER,
    food_waste_tonnes REAL,
    economic_loss REAL
)
""")

# Insert data
df.to_sql("food_waste", conn, if_exists="replace", index=False)

print("Database setup complete!")
conn.commit()
conn.close()
