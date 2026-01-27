import pandas as pd
from sqlalchemy import create_engine, text

# ----------------------------
# 1️⃣ Connect to AWS RDS MySQL
# ----------------------------
db_url = "mysql+mysqlconnector://admin:Datascience25@database-1.c89agc4261l8.us-east-1.rds.amazonaws.com:3306/testdb"
engine = create_engine(db_url)
print("✅ CONNECTED TO THE DATABASE")

# ----------------------------
# 2️⃣ Create `users` table (if not exists)
# ----------------------------
create_users_table = text("""
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    age INT
)
""")

with engine.connect() as conn:
    conn.execute(create_users_table)
    conn.commit()
print("✅ 'users' table is ready")

# ----------------------------
# 3️⃣ Load CSV
# ----------------------------
csv_path = r"C:\Users\CISLAB\Downloads\StudentsPerformance_Dataset.csv"
df = pd.read_csv(csv_path)
print("✅ CSV loaded successfully")
print("Columns in CSV:", df.columns.tolist())

# ----------------------------
# 4️⃣ Save CSV to a separate table
# ----------------------------
df.to_sql(
    name="students_performance",
    con=engine,
    if_exists="replace",  # replace table if it already exists
    index=False
)
print("✅ CSV uploaded successfully into 'students_performance' table")

# ----------------------------
# 5️⃣ Verify tables and data
# ----------------------------
print("\nTables in database:")
print(pd.read_sql("SHOW TABLES", engine))

print("\nSample data from 'students_performance':")
print(pd.read_sql("SELECT * FROM students_performance LIMIT 5", engine))

# ----------------------------
# 6️⃣ Test connection
# ----------------------------
with engine.connect() as conn:
    result = conn.execute(text("SELECT 1"))
    print("\n✅ AWS RDS connection test successful, returned:", result.scalar())
