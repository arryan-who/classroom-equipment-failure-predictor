import numpy as np
import pandas as pd
import os
from db_utils import create_table, insert_data

np.random.seed(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def random_location():
    blocks = ["A", "B", "C"]
    return np.random.choice(blocks), str(np.random.randint(101, 410))


def generate_data(equipment, drift=1.0, n=500):

    rows = []

    for i in range(n):

        block, room = random_location()

        age = np.random.randint(1, 10)
        usage = np.random.uniform(2, 10) * drift
        maintenance_gap = np.random.randint(5, 60)
        maintenance_type = np.random.choice([0, 1])

        base = (
            0.3 * age +
            0.4 * usage +
            0.3 * maintenance_gap -
            0.5 * (1 - maintenance_type)
        )

        row = {
            "equipment_type": equipment,
            "block": block,
            "room_number": room,
            "equipment_id": f"{equipment}_{i}",

            "age_years": age,
            "daily_usage_hours": usage,
            "days_since_last_maintenance": maintenance_gap,
            "last_maintenance_type": maintenance_type,

            "avg_temperature_week": 0,
            "max_temperature_week": 0,
            "filter_cleaning_gap_days": 0,

            "touch_responsiveness": 0,
            "ghost_touch_issue": 0,
            "software_updated_recently": 0,

            "switch_cycles_per_day": 0,
            "frequent_flickering": 0,

            "desired_temperature": 0,
            "occupancy_level": 0,
        }

        # PROJECTOR
        if equipment == "projector":
            row["avg_temperature_week"] = np.random.uniform(25, 35) * drift
            row["max_temperature_week"] = row["avg_temperature_week"] + np.random.uniform(2, 8)
            row["filter_cleaning_gap_days"] = np.random.randint(10, 90)

            base += (
                0.3 * row["avg_temperature_week"] +
                0.4 * row["max_temperature_week"] +
                0.3 * row["filter_cleaning_gap_days"]
            )

        # SMARTBOARD
        elif equipment == "smartboard":
            row["touch_responsiveness"] = np.random.choice([0, 1, 2])
            row["ghost_touch_issue"] = np.random.choice([0, 1])
            row["software_updated_recently"] = np.random.choice([0, 1])

            base += (
                0.5 * row["touch_responsiveness"] +
                0.4 * row["ghost_touch_issue"] -
                0.3 * row["software_updated_recently"]
            )

        # LIGHTING
        elif equipment == "lighting":
            row["switch_cycles_per_day"] = np.random.randint(5, 80)
            row["frequent_flickering"] = np.random.choice([0, 1])

            base += (
                0.4 * row["switch_cycles_per_day"] +
                0.5 * row["frequent_flickering"]
            )

        # AC
        elif equipment == "ac":
            row["avg_temperature_week"] = np.random.uniform(26, 40) * drift
            row["max_temperature_week"] = row["avg_temperature_week"] + np.random.uniform(3, 10)
            row["desired_temperature"] = np.random.uniform(18, 24)
            row["occupancy_level"] = np.random.randint(5, 50)
            row["filter_cleaning_gap_days"] = np.random.randint(10, 90)

            cooling_load = row["avg_temperature_week"] - row["desired_temperature"]

            base += (
                0.3 * cooling_load +
                0.3 * row["max_temperature_week"] +
                0.2 * row["occupancy_level"]
            )

        prob = sigmoid(base / 10)
        row["failure"] = np.random.choice([0, 1], p=[1 - prob, prob])

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    os.makedirs("data", exist_ok=True)
    create_table()

    versions = {"v1": 1.0, "v2": 1.3, "v3": 1.6}
    equipment_types = ["projector", "smartboard", "lighting", "ac"]

    for v, drift in versions.items():
        full = pd.DataFrame()

        for eq in equipment_types:
            df = generate_data(eq, drift)
            full = pd.concat([full, df])

        insert_data(full, v)
        print(f"Inserted dataset {v} into database")


if __name__ == "__main__":
    main()