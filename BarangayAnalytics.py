import pandas as pd
import logging
from datetime import datetime, timedelta
import pytz
from collections import Counter
import os
from models import lr_road, lr_fire
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_time_range(time_filter):
    now = datetime.now(pytz.timezone('Asia/Manila'))
    if time_filter == 'today':
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now
    elif time_filter == 'daily':
        start = now - timedelta(days=1)
        end = now
    elif time_filter == 'weekly':
        start = now - timedelta(days=7)
        end = now
    elif time_filter == 'monthly':
        start = now - timedelta(days=30)
        end = now
    elif time_filter == 'yearly':
        start = now - timedelta(days=365)
        end = now
    else:
        start = now - timedelta(days=7)
        end = now
    return start, end

def generate_mock_data(time_filter, incident_type='road'):
    start, end = get_time_range(time_filter)
    num_entries = 100 if time_filter == 'yearly' else 50 if time_filter == 'monthly' else 20 if time_filter == 'weekly' else 10
    timestamps = pd.date_range(start, end, periods=num_entries, tz='Asia/Manila')
    if incident_type == 'road':
        emergency_types = ['Collision', 'Rollover', 'Pedestrian']
        causes = ['Speeding', 'Drunk Driving', 'Distracted Driving']
        road_condition = ['Dry', 'Wet', 'Icy']
        vehicle_type = ['Car', 'Motorcycle', 'Truck']
        data = {
            'timestamp': timestamps,
            'emergency_type': np.random.choice(emergency_types, size=num_entries),
            'cause': np.random.choice(causes, size=num_entries),
            'responded': np.random.choice([True, False], size=num_entries),
            'weather': np.random.choice(['Sunny', 'Rainy', 'Foggy'], size=num_entries),
            'road_condition': np.random.choice(road_condition, size=num_entries),
            'vehicle_type': np.random.choice(vehicle_type, size=num_entries),
            'barangay': np.random.choice(['Barangay 1', 'Barangay 2', 'Barangay 3'], size=num_entries)
        }
    elif incident_type == 'fire':
        emergency_types = ['Electrical', 'Arson', 'Cooking']
        causes = ['Electrical Fault', 'Arson', 'Cooking Accident']
        property_type = ['Residential', 'Commercial', 'Industrial']
        data = {
            'timestamp': timestamps,
            'emergency_type': np.random.choice(emergency_types, size=num_entries),
            'cause': np.random.choice(causes, size=num_entries),
            'responded': np.random.choice([True, False], size=num_entries),
            'weather': np.random.choice(['Sunny', 'Rainy', 'Foggy'], size=num_entries),
            'property_type': np.random.choice(property_type, size=num_entries),
            'barangay': np.random.choice(['Barangay 1', 'Barangay 2', 'Barangay 3'], size=num_entries)
        }
    df = pd.DataFrame(data)
    if incident_type == 'road' and lr_road:
        features = pd.get_dummies(df[['weather', 'road_condition', 'vehicle_type']]).fillna(0)
        df['predicted_cause'] = lr_road.predict(features)
    elif incident_type == 'fire' and lr_fire:
        features = pd.get_dummies(df[['weather', 'property_type']]).fillna(0)
        df['predicted_cause'] = lr_fire.predict(features)
    return df

def load_csv_data_road(file_path, time_filter):
    try:
        file_path_full = os.path.join('dataset', file_path)
        if not os.path.exists(file_path_full):
            logger.info("CSV file not found: {}. Generating mock data.".format(file_path_full))
            return generate_mock_data(time_filter, 'road')
        
        df = pd.read_csv(file_path_full)
        if df.empty:
            logger.info("CSV file {} is empty. Generating mock data.".format(file_path))
            return generate_mock_data(time_filter, 'road')
        
        column_mapping = {
            'Date': 'Date', 'Time': 'Time', 'Barangay': 'Barangay', 'Weather': 'Weather',
            'Road_Condition': 'Road_Condition', 'Vehicle_Type': 'Vehicle_Type',
            'Accident_Type': 'Accident_Type', 'Latitude': 'Latitude', 'Longitude': 'Longitude',
            'Day_of_Week': 'Day_of_Week', 'Injuries': 'Injuries', 'Fatalities': 'Fatalities',
            'Driver_Age': 'Driver_Age', 'Driver_Gender': 'Driver_Gender'
        }
        
        missing_columns = [col for col in column_mapping.keys() if col not in df.columns]
        if missing_columns:
            logger.info("Missing columns in {}: {}. Generating mock data.".format(file_path, missing_columns))
            return generate_mock_data(time_filter, 'road')
        
        # Proceed with processing (timestamp creation, renaming, filtering, etc.)
        df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M', errors='coerce')
        df['timestamp'] = df['timestamp'].dt.tz_localize('Asia/Manila')
        # ... rest of the function ...
        
        return df
    except Exception as e:
        logger.info("Failed to load CSV {}: {}. Generating mock data.".format(file_path, e))
        return generate_mock_data(time_filter, 'road')

def load_csv_data_fire(file_path, time_filter):
    try:
        file_path_full = os.path.join('dataset', file_path)
        if not os.path.exists(file_path_full):
            logger.warning(f"CSV file not found: {file_path_full}. Generating mock data.")
            return generate_mock_data(time_filter, 'fire')
        
        df = pd.read_csv(file_path_full)
        
        # Define expected columns and map them
        column_mapping = {
            'Date': 'Date',
            'Time': 'Time',
            'Barangay': 'Barangay',
            'Weather': 'Weather',
            'Property_Type': 'Property_Type',
            'Fire_Cause': 'Fire_Cause',
            'Latitude': 'Latitude',
            'Longitude': 'Longitude',
            'Day_of_Week': 'Day_of_Week',
            'Fire_Severity': 'Fire_Severity',
            'Casualty_Count': 'Casualty_Count',
            'Response_Time': 'Response_Time',
            'Fire_Duration': 'Fire_Duration'
        }
        
        missing_columns = [col for col in column_mapping.keys() if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing columns in {file_path}: {missing_columns}. Generating mock data.")
            return generate_mock_data(time_filter, 'fire')
        
        # Create timezone-aware timestamp
        df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M', errors='coerce')
        df['timestamp'] = df['timestamp'].dt.tz_localize('Asia/Manila')
        
        # Rename columns to standardized names
        df.rename(columns={
            'Barangay': 'barangay',
            'Weather': 'weather',
            'Property_Type': 'property_type',
            'Fire_Cause': 'cause',
            'Latitude': 'latitude',
            'Longitude': 'longitude',
            'Day_of_Week': 'day_of_week',
            'Fire_Severity': 'fire_severity',
            'Casualty_Count': 'casualty_count',
            'Response_Time': 'response_time',
            'Fire_Duration': 'fire_duration'
        }, inplace=True)
        
        df['emergency_type'] = 'Fire'
        df['responded'] = True
        
        start, end = get_time_range(time_filter)
        df = df[(df['timestamp'].notna()) & (df['timestamp'] >= start) & (df['timestamp'] <= end)]
        
        if lr_fire:
            features = pd.get_dummies(df[['weather', 'property_type']]).fillna(0)
            df['predicted_cause'] = lr_fire.predict(features)
        
        return df
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {e}. Generating mock data.")
        return generate_mock_data(time_filter, 'fire')

def get_barangay_trends(time_filter):
    try:
        road_df = load_csv_data_road('road_accident.csv', time_filter)
        fire_df = load_csv_data_fire('fire_incident.csv', time_filter)
        start, end = get_time_range(time_filter)
        
        if time_filter == 'today':
            labels = [(start + timedelta(hours=i)).strftime('%H:%M') for i in range(24)]
            total = [len(road_df[road_df['timestamp'].dt.hour == i]) + len(fire_df[fire_df['timestamp'].dt.hour == i]) for i in range(24)]
            responded = [len(road_df[(road_df['timestamp'].dt.hour == i) & (road_df['responded'] == True)]) + 
                         len(fire_df[(fire_df['timestamp'].dt.hour == i) & (fire_df['responded'] == True)]) for i in range(24)]
        elif time_filter == 'daily':
            labels = [(start + timedelta(hours=i)).strftime('%H:%M') for i in range(24)]
            total = [len(road_df[road_df['timestamp'].dt.hour == i]) + len(fire_df[fire_df['timestamp'].dt.hour == i]) for i in range(24)]
            responded = [len(road_df[(road_df['timestamp'].dt.hour == i) & (road_df['responded'] == True)]) + 
                         len(fire_df[(fire_df['timestamp'].dt.hour == i) & (fire_df['responded'] == True)]) for i in range(24)]
        elif time_filter == 'weekly':
            labels = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
            total = [len(road_df[road_df['timestamp'].dt.date == (start + timedelta(days=i)).date()]) + 
                     len(fire_df[fire_df['timestamp'].dt.date == (start + timedelta(days=i)).date()]) for i in range(7)]
            responded = [len(road_df[(road_df['timestamp'].dt.date == (start + timedelta(days=i)).date()) & (road_df['responded'] == True)]) + 
                         len(fire_df[(fire_df['timestamp'].dt.date == (start + timedelta(days=i)).date()) & (fire_df['responded'] == True)]) for i in range(7)]
        elif time_filter == 'monthly':
            labels = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(0, 30, 2)]
            total = [len(road_df[(road_df['timestamp'].dt.date >= (start + timedelta(days=i)).date()) & 
                                 (road_df['timestamp'].dt.date < (start + timedelta(days=i+2)).date())]) + 
                     len(fire_df[(fire_df['timestamp'].dt.date >= (start + timedelta(days=i)).date()) & 
                                 (fire_df['timestamp'].dt.date < (start + timedelta(days=i+2)).date())]) for i in range(0, 30, 2)]
            responded = [len(road_df[(road_df['timestamp'].dt.date >= (start + timedelta(days=i)).date()) & 
                                     (road_df['timestamp'].dt.date < (start + timedelta(days=i+2)).date()) & 
                                     (road_df['responded'] == True)]) + 
                         len(fire_df[(fire_df['timestamp'].dt.date >= (start + timedelta(days=i)).date()) & 
                                     (fire_df['timestamp'].dt.date < (start + timedelta(days=i+2)).date()) & 
                                     (fire_df['responded'] == True)]) for i in range(0, 30, 2)]
        else:
            labels = [(start + timedelta(days=i*30)).strftime('%Y-%m') for i in range(12)]
            total = [len(road_df[(road_df['timestamp'].dt.year == (start + timedelta(days=i*30)).year) & 
                                 (road_df['timestamp'].dt.month == (start + timedelta(days=i*30)).month)]) + 
                     len(fire_df[(fire_df['timestamp'].dt.year == (start + timedelta(days=i*30)).year) & 
                                 (fire_df['timestamp'].dt.month == (start + timedelta(days=i*30)).month)]) for i in range(12)]
            responded = [len(road_df[(road_df['timestamp'].dt.year == (start + timedelta(days=i*30)).year) & 
                                     (road_df['timestamp'].dt.month == (start + timedelta(days=i*30)).month) & 
                                     (road_df['responded'] == True)]) + 
                         len(fire_df[(fire_df['timestamp'].dt.year == (start + timedelta(days=i*30)).year) & 
                                     (fire_df['timestamp'].dt.month == (start + timedelta(days=i*30)).month) & 
                                     (fire_df['responded'] == True)]) for i in range(12)]
        return {'labels': labels, 'total': total, 'responded': responded}
    except Exception as e:
        logger.error(f"Error in get_barangay_trends: {e}")
        return {'labels': [], 'total': [], 'responded': []}

def get_barangay_distribution(time_filter):
    try:
        road_df = load_csv_data_road('road_accident.csv', time_filter)
        fire_df = load_csv_data_fire('fire_incident.csv', time_filter)
        distribution = Counter(road_df['emergency_type']) + Counter(fire_df['emergency_type'])
        return {k: {'total': v, 'responded': len(road_df[(road_df['emergency_type'] == k) & (road_df['responded'] == True)]) + 
                    len(fire_df[(fire_df['emergency_type'] == k) & (fire_df['responded'] == True)])} for k, v in distribution.items()}
    except Exception as e:
        logger.error(f"Error in get_barangay_distribution: {e}")
        return {'Unknown': {'total': 0, 'responded': 0}}

def get_barangay_causes(time_filter, barangay=''):
    try:
        road_df = load_csv_data_road('road_accident.csv', time_filter)
        fire_df = load_csv_data_fire('fire_incident.csv', time_filter)
        if barangay and 'barangay' in road_df.columns and 'barangay' in fire_df.columns:
            road_df = road_df[road_df['barangay'] == barangay]
            fire_df = fire_df[fire_df['barangay'] == barangay]
        road_causes = Counter(road_df['predicted_cause'] if 'predicted_cause' in road_df.columns else road_df['cause'])
        fire_causes = Counter(fire_df['predicted_cause'] if 'predicted_cause' in fire_df.columns else fire_df['cause'])
        return {'road': dict(road_causes), 'fire': dict(fire_causes)}
    except Exception as e:
        logger.error(f"Error in get_barangay_causes: {e}")
        return {'road': {'Unknown': 0}, 'fire': {'Unknown': 0}}
