import pandas as pd
import numpy as np
import random

rows = 10000

equipment_types = ["projector","smartboard","lighting","ac"]

data = []

for i in range(rows):

    equipment = random.choice(equipment_types)

    equipment_age = np.random.randint(1,12)

    daily_usage_hours = np.random.uniform(1,10)

    maintenance_gap_days = np.random.randint(0,180)

    last_maintenance_type = np.random.choice(
        ["preventive","corrective"],
        p=[0.75,0.25]
    )

    room_temperature = np.random.uniform(20,35)

    power_fluctuations = np.random.randint(0,5)

    # equipment specific measurable factors

    projector_operating_hours = np.random.randint(200,5000)

    touch_error_rate = np.random.randint(0,12)

    switch_cycles_per_day = np.random.randint(5,60)

    room_occupancy = np.random.randint(10,80)

    temperature_difference = np.random.uniform(3,15)

    filter_cleaning_gap_days = np.random.randint(0,150)

    firmware_update_gap_days = np.random.randint(0,300)

    voltage_variation_events = np.random.randint(0,5)

    # risk score initialization
    risk = 0

    # maintenance factor

    if last_maintenance_type == "corrective":
        risk += 0.8

    # equipment specific risk logic

    if equipment == "projector":

        risk += (
            0.25*equipment_age +
            0.25*(projector_operating_hours/5000) +
            0.2*daily_usage_hours +
            0.15*(filter_cleaning_gap_days/150) +
            0.15*(maintenance_gap_days/180)
        )

    elif equipment == "smartboard":

        risk += (
            0.25*equipment_age +
            0.2*touch_error_rate +
            0.2*(firmware_update_gap_days/300) +
            0.2*power_fluctuations +
            0.15*daily_usage_hours
        )

    elif equipment == "lighting":

        risk += (
            0.25*equipment_age +
            0.25*switch_cycles_per_day/60 +
            0.2*voltage_variation_events +
            0.15*daily_usage_hours +
            0.15*(maintenance_gap_days/180)
        )

    elif equipment == "ac":

        risk += (
            0.25*equipment_age +
            0.25*temperature_difference/15 +
            0.2*daily_usage_hours +
            0.15*room_occupancy/80 +
            0.15*(filter_cleaning_gap_days/150)
        )

    # convert risk score into probability

    probability = 1 / (1 + np.exp(-risk))

    # simulate failure

    failure = np.random.choice(
        [0,1],
        p=[1-probability, probability]
    )

    data.append([
        equipment,
        equipment_age,
        daily_usage_hours,
        maintenance_gap_days,
        last_maintenance_type,
        room_temperature,
        power_fluctuations,
        projector_operating_hours,
        touch_error_rate,
        switch_cycles_per_day,
        room_occupancy,
        temperature_difference,
        filter_cleaning_gap_days,
        firmware_update_gap_days,
        voltage_variation_events,
        failure
    ])

columns = [
    "equipment_type",
    "equipment_age_years",
    "daily_usage_hours",
    "maintenance_gap_days",
    "last_maintenance_type",
    "room_temperature",
    "power_fluctuation_events",
    "projector_operating_hours",
    "touch_error_rate",
    "switch_cycles_per_day",
    "room_occupancy",
    "temperature_difference",
    "filter_cleaning_gap_days",
    "firmware_update_gap_days",
    "voltage_variation_events",
    "failure_within_30_days"
]

df = pd.DataFrame(data,columns=columns)

df.to_csv("data/equipment_data.csv",index=False)

print("Dataset generated successfully")