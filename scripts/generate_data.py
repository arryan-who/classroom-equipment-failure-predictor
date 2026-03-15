import pandas as pd
import numpy as np
import random

rows = 6000

equipment_types = ["projector", "smartboard", "lighting", "ac"]

data = []

for i in range(rows):

    equipment = random.choice(equipment_types)

    age = np.random.randint(1, 12)
    daily_usage = np.random.uniform(1, 10)
    maintenance_gap = np.random.randint(0, 180)
    room_temperature = np.random.uniform(20, 38)
    power_fluctuation = np.random.randint(0, 6)

    lamp_usage = np.random.randint(100, 4000)
    touch_errors = np.random.randint(0, 20)
    switch_cycles = np.random.randint(5, 60)
    room_occupancy = np.random.randint(10, 80)
    temperature_difference = np.random.uniform(3, 15)

    if equipment == "projector":

        risk = (
            0.35 * age +
            0.3 * (lamp_usage / 4000) +
            0.2 * daily_usage +
            0.1 * room_temperature +
            0.05 * (maintenance_gap / 180)
        )

    elif equipment == "smartboard":

        risk = (
            0.3 * age +
            0.25 * touch_errors +
            0.2 * power_fluctuation +
            0.15 * daily_usage +
            0.1 * (maintenance_gap / 180)
        )

    elif equipment == "lighting":

        risk = (
            0.3 * age +
            0.25 * switch_cycles +
            0.25 * power_fluctuation +
            0.1 * daily_usage +
            0.1 * (maintenance_gap / 180)
        )

    elif equipment == "ac":

        risk = (
            0.3 * age +
            0.25 * temperature_difference +
            0.2 * daily_usage +
            0.15 * room_occupancy +
            0.1 * (maintenance_gap / 180)
        )

    failure = 1 if risk > 4 else 0

    data.append([
        equipment,
        age,
        daily_usage,
        maintenance_gap,
        room_temperature,
        power_fluctuation,
        lamp_usage,
        touch_errors,
        switch_cycles,
        room_occupancy,
        temperature_difference,
        failure
    ])

columns = [
    "equipment_type",
    "equipment_age_years",
    "daily_usage_hours",
    "maintenance_gap_days",
    "room_temperature",
    "power_fluctuation_events",
    "lamp_usage_hours",
    "touch_error_rate",
    "switch_cycles_per_day",
    "room_occupancy",
    "temperature_difference",
    "failure_occurred"
]

df = pd.DataFrame(data, columns=columns)

df.to_csv("data/equipment_data.csv", index=False)

print("Dataset generated successfully")