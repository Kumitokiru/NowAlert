import pandas as pd
import logging
from datetime import datetime, timedelta
import pytz
from collections import Counter
import os
from models import lr_fire
import numpy as np



logger = logging.getLogger(__name__)

# Assuming lr_fire is a pre-trained model available in the scope
lr_fire = None  # Placeholder; replace with actual model if available

def get_time_range(time_filter):
    # Placeholder function; replace with actual implementation
    from datetime import datetime, timedelta
    end = datetime.now()
    if time_filter == 'today':
        start = end.replace(hour=0, minute=0, second=0, microsecond=0)
    elif time_filter == 'daily':
        start = end - timedelta(days=1)
    elif time_filter == 'weekly':
        start = end - timedelta(days=7)
    elif time_filter == 'monthly':
        start = end - timedelta(days=30)
    elif time_filter == 'yearly':
        start = end - timedelta(days=365)
    else:
        start = end - timedelta(days=7)  # Default to weekly
    return start, end

def generate_mock_data(time_filter, incident_type='fire'):
    # Placeholder function; replace with actual implementation
    start, end = get_time_range(time_filter)
    num_entries = 100
    dates = pd.date_range(start, end, periods=num_entries)
    barangays = ['Barangay 1', 'Barangay 2', 'Barangay 3']
    weather_types = ['Sunny', 'Rainy', 'Cloudy']
    property_types = ['Residential', 'Commercial', 'Industrial']
    causes = ['Natural', 'Electrical', 'Cooking']
    
    data = {
        'timestamp': dates,
        'barangay': np.random.choice(barangays, size=num_entries),
        'weather': np.random.choice(weather_types, size=num_entries),
        'property_type': np.random.choice(property_types, size=num_entries),
        'cause': np.random.choice(causes, size=num_entries),
        'emergency_type': [incident_type] * num_entries,
        'responded': np.random.choice([True, False], size=num_entries)
    }
    return pd.DataFrame(data)

def load_csv_data(file_path, time_filter, incident_type='fire'):
    try:
        file_path_full = os.path.join('dataset', file_path)
        if not os.path.exists(file_path_full):
            logger.warning(f"CSV file not found: {file_path_full}. Generating mock data.")
            return generate_mock_data(time_filter, incident_type)
        
        df = pd.read_csv(file_path_full)
        
        # Define required columns for mapping
        required_columns = ['Date', 'Time', 'Barangay', 'Weather', 'Property_Type', 'Fire_Cause']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing columns in {file_path}: {missing_columns}. Generating mock data.")
            return generate_mock_data(time_filter, incident_type)
        
        # Combine Date and Time into timestamp
        df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M', errors='coerce')
        
        # Rename columns to match expected names
        df.rename(columns={
            'Barangay': 'barangay',
            'Weather': 'weather',
            'Property_Type': 'property_type',
            'Fire_Cause': 'cause'
        }, inplace=True)
        
        # Add emergency_type and responded columns
        df['emergency_type'] = 'Fire'
        df['responded'] = True
        
        # Filter by time range
        start, end = get_time_range(time_filter)
        df = df[(df['timestamp'].notna()) & (df['timestamp'] >= start) & (df['timestamp'] <= end)]
        
        # Apply predictions if model is available
        if lr_fire:
            features = pd.get_dummies(df[['weather', 'property_type']]).fillna(0)
            df['predicted_cause'] = lr_fire.predict(features)
        
        return df
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {e}. Generating mock data.")
        return generate_mock_data(time_filter, incident_type)

def get_bfp_trends(time_filter, barangay=''):
    try:
        fire_df = load_csv_data('fire_incident.csv', time_filter, incident_type='fire')
        if barangay and 'barangay' in fire_df.columns:
            fire_df = fire_df[fire_df['barangay'] == barangay]
        start, end = get_time_range(time_filter)
        if time_filter == 'today':
            labels = [(start + timedelta(hours=i)).strftime('%H:%M') for i in range(24)]
            total = [len(fire_df[fire_df['timestamp'].dt.hour == i]) for i in range(24)]
            responded = [len(fire_df[(fire_df['timestamp'].dt.hour == i) & (fire_df['responded'] == True)]) for i in range(24)]
        elif time_filter == 'daily':
            labels = [(start + timedelta(hours=i)).strftime('%H:%M') for i in range(24)]
            total = [len(fire_df[fire_df['timestamp'].dt.hour == i]) for i in range(24)]
            responded = [len(fire_df[(fire_df['timestamp'].dt.hour == i) & (fire_df['responded'] == True)]) for i in range(24)]
        elif time_filter == 'weekly':
            labels = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
            total = [len(fire_df[fire_df['timestamp'].dt.date == (start + timedelta(days=i)).date()]) for i in range(7)]
            responded = [len(fire_df[(fire_df['timestamp'].dt.date == (start + timedelta(days=i)).date()) & (fire_df['responded'] == True)]) for i in range(7)]
        elif time_filter == 'monthly':
            labels = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(0, 30, 2)]
            total = [len(fire_df[(fire_df['timestamp'].dt.date >= (start + timedelta(days=i)).date()) & 
                                 (fire_df['timestamp'].dt.date < (start + timedelta(days=i+2)).date())]) for i in range(0, 30, 2)]
            responded = [len(fire_df[(fire_df['timestamp'].dt.date >= (start + timedelta(days=i)).date()) & 
                                     (fire_df['timestamp'].dt.date < (start + timedelta(days=i+2)).date()) & 
                                     (fire_df['responded'] == True)]) for i in range(0, 30, 2)]
        else:
            labels = [(start + timedelta(days=i*30)).strftime('%Y-%m') for i in range(12)]
            total = [len(fire_df[(fire_df['timestamp'].dt.year == (start + timedelta(days=i*30)).year) & 
                                 (fire_df['timestamp'].dt.month == (start + timedelta(days=i*30)).month)]) for i in range(12)]
            responded = [len(fire_df[(fire_df['timestamp'].dt.year == (start + timedelta(days=i*30)).year) & 
                                     (fire_df['timestamp'].dt.month == (start + timedelta(days=i*30)).month) & 
                                     (fire_df['responded'] == True)]) for i in range(12)]
        return {'labels': labels, 'total': total, 'responded': responded}
    except Exception as e:
        logger.error(f"Error in get_bfp_trends: {e}")
        return {'labels': [], 'total': [], 'responded': []}

def get_bfp_distribution(time_filter, barangay=''):
    try:
        fire_df = load_csv_data('fire_incident.csv', time_filter, incident_type='fire')
        if barangay and 'barangay' in fire_df.columns:
            fire_df = fire_df[fire_df['barangay'] == barangay]
        distribution = Counter(fire_df['emergency_type'])
        return {k: {'total': v, 'responded': len(fire_df[(fire_df['emergency_type'] == k) & (fire_df['responded'] == True)])} 
                for k, v in distribution.items()}
    except Exception as e:
        logger.error(f"Error in get_bfp_distribution: {e}")
        return {'Unknown': {'total': 0, 'responded': 0}}

def get_bfp_causes(time_filter, barangay=''):
    try:
        fire_df = load_csv_data('fire_incident.csv', time_filter, incident_type='fire')
        if barangay and 'barangay' in fire_df.columns:
            fire_df = fire_df[fire_df['barangay'] == barangay]
        fire_causes = Counter(fire_df['predicted_cause'] if 'predicted_cause' in fire_df else fire_df['cause'])
        return {'fire': dict(fire_causes)}
    except Exception as e:
        logger.error(f"Error in get_bfp_causes: {e}")
        return {'fire': {'Unknown': 0}}
