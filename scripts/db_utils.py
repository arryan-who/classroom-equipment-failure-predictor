import sqlite3
import pandas as pd

DB_PATH = "data/equipment.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def create_table():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS equipment_data (
        equipment_type TEXT,
        block TEXT,
        room_number TEXT,
        equipment_id TEXT,

        age_years INTEGER,
        daily_usage_hours REAL,
        days_since_last_maintenance INTEGER,
        last_maintenance_type INTEGER,

        avg_temperature_week REAL,
        max_temperature_week REAL,
        filter_cleaning_gap_days INTEGER,

        touch_responsiveness INTEGER,
        ghost_touch_issue INTEGER,
        software_updated_recently INTEGER,

        switch_cycles_per_day INTEGER,
        frequent_flickering INTEGER,

        desired_temperature REAL,
        occupancy_level INTEGER,

        failure INTEGER,
        dataset_version TEXT
    )
    """)

    conn.commit()
    conn.close()

def insert_data(df, version):
    conn = get_connection()
    df["dataset_version"] = version
    df.to_sql("equipment_data", conn, if_exists="append", index=False)
    conn.close()

def fetch_data(version=None, equipment=None):
    conn = get_connection()

    query = "SELECT * FROM equipment_data WHERE 1=1"

    params = []

    if version is not None:
        query += " AND dataset_version = ?"
        params.append(version)

    if equipment is not None:
        query += " AND equipment_type = ?"
        params.append(equipment)

    df = pd.read_sql(query, conn, params=params)
    conn.close()
    return df