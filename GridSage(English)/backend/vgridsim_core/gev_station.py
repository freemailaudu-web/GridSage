import numpy as np
import random
import os
from dataclasses import dataclass, field
import pandas as pd
@dataclass
class EVParameters:
    """Define the general physical parameters of electric vehicles"""
    capacity_kwh: float = 70.0
    max_charge_kw: float = 60.0
    max_discharge_kw: float = 25.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.9


@dataclass
class EVChargeSession:
    """Define scene information for a single EV charging session"""
    ev_id: int
    spot_id: int
    arrival_hour: int
    departure_hour: int
    initial_soc: float
    final_soc: float = field(init=False, default=0.0)


def hour_multiplier_value(values, hour: int, default: float = 1.0) -> float:
    if not values:
        return default
    if isinstance(values, list):
        return float(values[hour]) if hour < len(values) else default
    if isinstance(values, dict):
        for candidate in (hour, str(hour), f"{hour:02d}"):
            if candidate in values:
                return float(values[candidate])
        for range_key, value in values.items():
            text = str(range_key)
            if "-" not in text:
                continue
            try:
                start, end = text.split("-", 1)
                start_hour = int(float(start))
                end_hour = int(float(end))
            except (TypeError, ValueError):
                continue
            if start_hour <= int(hour) <= end_hour:
                return float(value)
    return default


def average_hour_multiplier(values, default: float = 1.0) -> float:
    if not values:
        return default
    multipliers = [hour_multiplier_value(values, hour, default) for hour in range(24)]
    return sum(multipliers) / len(multipliers)


# --- G-EVs charging station core category ---
class GEVStation:
    """
    G-EVs charging station module.
    """
    def __init__(self, station_id: str, num_spots: int = 30, ev_params: EVParameters = None):
        self.station_id = station_id
        self.num_spots = num_spots
        self.ev_params = ev_params if ev_params else EVParameters()
        self.daily_sessions: list[EVChargeSession] = []
        # Add ID to log
        print(f"G-EVs charging station (ID: {self.station_id}) has been created and contains {self.num_spots} charging piles.")
    def _is_timeslot_free(self, schedule: list, new_arrival: int, new_departure: int) -> bool:
        """
        Internal helper function: Checks if the new time period overlaps with the scheduled time period.
        """
        for arrival, departure in schedule:
            # Check overlapping conditions: (StartA < EndB) and (EndA > StartB)
            if new_arrival < departure and new_departure > arrival:
                return False # If there is overlap, it is not idle
        return True # If there is no overlap at the end of the loop, it is idle

    def load_scenarios_from_csv(self, all_scenarios_df: pd.DataFrame):
        """
        Load the charging session belonging to this station from a DataFrame containing all charging station scenes.

        Parameters:
            all_scenarios_df (pd.DataFrame): Complete DataFrame containing all charging events.
        """
        self.daily_sessions = []

        # Filter out scene data that only belongs to the current charging station instance
        # Assume that the GEVStation instance has a station_id attribute
        station_scenarios_df = all_scenarios_df[all_scenarios_df['station_id'] == self.station_id]

        if station_scenarios_df.empty:
            print(f"[Message]: No charging events belonging to charging station '{self.station_id}' were found in the CSV data.")
            return

        # Verify whether the charging pile ID is outside the scope of this site
        if 'spot_id_in_station' in station_scenarios_df.columns:
            max_spot_id = station_scenarios_df['spot_id_in_station'].max()
            if max_spot_id >= self.num_spots:
                print(f"[Error]: There is a charging pile ID ({max_spot_id}) assigned to the charging station '{self.station_id}' in the data file")
                print(f" Exceeded the actual number of charging piles ({self.num_spots}) owned by this station. Please check your data!")
                raise ValueError(f"The charging pile ID of charging station {self.station_id} is out of bounds.")

        # Convert each row in the DataFrame to an EVChargeSession object
        for index, row in station_scenarios_df.iterrows():
            session = EVChargeSession(
                ev_id=index,
                spot_id=row['spot_id_in_station'],
                arrival_hour=int(row['arrival_hour']),
                departure_hour=int(row['departure_hour']),
                initial_soc=float(row['initial_soc'])
            )
            self.daily_sessions.append(session)

        print(f"Successfully loaded {len(self.daily_sessions)} charging sessions from CSV for charging station '{self.station_id}'.")



    def generate_daily_scenarios(self, num_evs_to_generate: int = 120, arrival_hour_multipliers=None):
        """
        Generate random EV charging session scenarios throughout the day.
        The arrival time is generated using a three-peak (morning, noon, and evening) mixed normal distribution model.
        There are the most vehicles during the evening peak hours.
        """
        self.daily_sessions = []
        spot_schedules = [[] for _ in range(self.num_spots)]

        # --- 1. Define the three peak period parameters ---
        # The format is: (center time, time standard deviation, vehicle number weight)
        # NOTE: All weights must add up to 1.0
        PEAK_MORNING = (8, 1.5, 0.30) # Morning peak: 8 o'clock is the center, standard deviation is 1.5 hours, accounting for 30% of the total traffic flow
        PEAK_NOON = (12, 1.0, 0.20) # Noon peak: 12 o'clock is the center, standard deviation is 1 hour, accounting for 20% of the total traffic flow
        PEAK_EVENING = (18, 2.0, 0.50) # Evening peak: 18 o'clock is the center, standard deviation is 2 hours, accounting for 50% of the total traffic flow (highest weight)

        peaks = [PEAK_MORNING, PEAK_NOON, PEAK_EVENING]
        peak_weights = [p[2] for p in peaks]

        hour_weights = None
        if arrival_hour_multipliers:
            hour_weights = []
            for hour in range(24):
                base_weight = sum(
                    peak_weight * np.exp(-0.5 * ((hour - center) / std) ** 2)
                    for center, std, peak_weight in peaks
                )
                hour_weights.append(max(0.0, base_weight * hour_multiplier_value(arrival_hour_multipliers, hour)))
            if not any(weight > 0 for weight in hour_weights):
                num_evs_to_generate = 0
                hour_weights = None


        generated_count = 0
        attempts = 0
        while generated_count < num_evs_to_generate and attempts < num_evs_to_generate * 5:
            attempts += 1

            # --- 2. Generate random arrival time ---
            if hour_weights:
                arrival = random.choices(range(24), weights=hour_weights, k=1)[0]
            else:
                # 2.1. First randomly select a peak based on weight
                chosen_peak = random.choices(peaks, weights=peak_weights, k=1)[0]

                # 2.2. Generate a specific arrival time from the normal distribution corresponding to the selected peak
                arrival_float = np.random.normal(loc=chosen_peak[0], scale=chosen_peak[1])

                # 2.3. Convert floating point time to integer hours and make sure it is in the range of 0-23 points
                arrival = int(round(arrival_float))
                arrival = max(0, min(23, arrival))

            # --- 3. Generate integer residence time (fast charging mode) ---
            # Randomly generate an integer stay time of 1 or 2 hours
            stay_duration = random.randint(1, 3)

            departure = arrival + stay_duration
            departure = min(24, departure)

            if arrival >= departure:
                continue

            # --- 4. Allocate charging piles (logic remains unchanged) ---
            available_spots = list(range(self.num_spots))
            random.shuffle(available_spots)

            assigned_spot = -1
            for spot_id in available_spots:
                if self._is_timeslot_free(spot_schedules[spot_id], arrival, departure):
                    assigned_spot = spot_id
                    spot_schedules[spot_id].append((arrival, departure))
                    break

            if assigned_spot != -1:
                initial_soc = round(random.uniform(0.1, 0.5), 2)
                session = EVChargeSession(
                    ev_id=generated_count,
                    spot_id=assigned_spot,
                    arrival_hour=arrival,
                    departure_hour=departure,
                    initial_soc=initial_soc
                )
                self.daily_sessions.append(session)
                generated_count += 1

        print(f"Attempted to generate {num_evs_to_generate} EV scenes, successfully created {len(self.daily_sessions)}.")

        if generated_count < num_evs_to_generate:
            print(f"[Warning]: Only {generated_count} / {num_evs_to_generate} EV scenes were successfully generated. It may be that the charging pile is full or the time slot conflicts.")
        else:
            print(f"{len(self.daily_sessions)} EV scenes were successfully generated.")

    def get_scenario_for_baseline(self):
        """
        Convert the generated scene data into the format required by the baseline optimization model.
        Corrected the BOC (battery state of charge) transfer logic to ensure that the initial SOC is filled correctly.
        """
        arrival_times = [[] for _ in range(self.num_spots)]
        departure_times = [[] for _ in range(self.num_spots)]
        present_cars = np.zeros((self.num_spots, 24), dtype=int)

        # Initialize a BOC array of 25 time points (t=0 to t=24)
        boc_initial = np.zeros((self.num_spots, 25))

        # Step 1: Mark the vehicle presence period and record the initial SOC at the arrival time
        for session in self.daily_sessions:
            spot, arr, dep = session.spot_id, session.arrival_hour, session.departure_hour

            # Ensure that no index goes out of bounds
            arr = min(arr, 23)
            dep = min(dep, 24)

            if dep > arr:
                arrival_times[spot].append(arr)
                departure_times[spot].append(dep)
                present_cars[spot, arr:dep] = 1
                boc_initial[spot, arr] = session.initial_soc


        # Step 2: Fill BOC forward. If a vehicle is present at a certain hour but has a BOC of 0,
        # Fill with the BOC value of the previous hour to ensure that the SOC data is continuous.
        for spot in range(self.num_spots):
            for hour in range(1, 25): # Start checking from t=1
                # If there are cars present in the current hour (present_cars only reaches 23, so use hour-1)
                # And the current BOC is empty (0), but the BOC was not empty one hour ago
                if hour < 24 and present_cars[spot, hour - 1] == 1 and boc_initial[spot, hour] == 0:
                    boc_initial[spot, hour] = boc_initial[spot, hour - 1]

        # In order to allow external code to access parameters like the chargym environment, create a simple simulation object
        class MockOriginalEnv:
            def __init__(self, station):
                self.number_of_cars = station.num_spots
                self.EV_Param = {
                    'EV_capacity': station.ev_params.capacity_kwh,
                    'charging_rate': station.ev_params.max_charge_kw,
                    'discharging_rate': station.ev_params.max_discharge_kw,
                    'charging_effic': station.ev_params.charge_efficiency,
                    'discharging_effic': station.ev_params.discharge_efficiency,
                }
                self.Invalues = {
                    'present_cars': present_cars,
                    'BOC': boc_initial, # Use the modified BOC array
                    'ArrivalT': arrival_times,
                    'DepartureT': departure_times,
                }

        return MockOriginalEnv(self)
