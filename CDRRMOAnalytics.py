import pandas as pd
import logging
from datetime import datetime, timedelta
import pytz
from collections import Counter
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_time_range(time_filter):
    """Determine the start and end times based on the time filter."""
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
        start = now - timedelta(days=7)  # Default to weekly
        end = now
    return start, end

def load_csv_data(file_path, time_filter, barangay=None):
    """Load and filter CSV data based on time and barangay."""
    try:
        file_path_full = os.path.join('dataset', file_path)
        if not os.path.exists(file_path_full):
            logger.error(f"CSV file not found: {file_path_full}")
            return pd.DataFrame()
        
        # Read CSV without parse_dates to avoid errors
        df = pd.read_csv(file_path_full)
        
        # Check if 'timestamp' column exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            start, end = get_time_range(time_filter)
            df = df[(df['timestamp'].notna()) & (df['timestamp'] >= start) & (df['timestamp'] <= end)]
        else:
            logger.warning(f"'timestamp' column not found in {file_path}. Skipping time-based filtering.")
        
        if barangay and 'barangay' in df.columns:
            df = df[df['barangay'] == barangay]
        return df
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {e}")
        return pd.DataFrame()

def get_cdrrmo_trends(time_filter, barangay=None):
    """Get incident trends over time."""
    try:
        road_df = load_csv_data('road_accident.csv', time_filter, barangay)
        if road_df.empty or 'timestamp' not in road_df.columns:
            logger.warning("No data available for trends")
            return {'labels': [], 'total': [], 'responded': []}
        start, end = get_time_range(time_filter)
        if time_filter == 'today':
            labels = [(start + timedelta(hours=i)).strftime('%H:%M') for i in range(24)]
            total = [len(road_df[road_df['timestamp'].dt.hour == i]) for i in range(24)]
            responded = [len(road_df[(road_df['timestamp'].dt.hour == i) & (road_df['responded'] == True)]) for i in range(24)]
        elif time_filter == 'daily':
            labels = [(start + timedelta(hours=i)).strftime('%H:%M') for i in range(24)]
            total = [len(road_df[road_df['timestamp'].dt.hour == i]) for i in range(24)]
            responded = [len(road_df[(road_df['timestamp'].dt.hour == i) & (road_df['responded'] == True)]) for i in range(24)]
        elif time_filter == 'weekly':
            labels = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
            total = [len(road_df[road_df['timestamp'].dt.date == (start + timedelta(days=i)).date()]) for i in range(7)]
            responded = [len(road_df[(road_df['timestamp'].dt.date == (start + timedelta(days=i)).date()) & (road_df['responded'] == True)]) for i in range(7)]
        elif time_filter == 'monthly':
            labels = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(0, 30, 2)]
            total = [len(road_df[(road_df['timestamp'].dt.date >= (start + timedelta(days=i)).date()) & (road_df['timestamp'].dt.date < (start + timedelta(days=i+2)).date())]) for i in range(0, 30, 2)]
            responded = [len(road_df[(road_df['timestamp'].dt.date >= (start + timedelta(days=i)).date()) & (road_df['timestamp'].dt.date < (start + timedelta(days=i+2)).date()) & (road_df['responded'] == True)]) for i in range(0, 30, 2)]
        else:
            labels = [(start + timedelta(days=i*30)).strftime('%Y-%m') for i in range(12)]
            total = [len(road_df[(road_df['timestamp'].dt.year == (start + timedelta(days=i*30)).year) & (road_df['timestamp'].dt.month == (start + timedelta(days=i*30)).month)]) for i in range(12)]
            responded = [len(road_df[(road_df['timestamp'].dt.year == (start + timedelta(days=i*30)).year) & (road_df['timestamp'].dt.month == (start + timedelta(days=i*30)).month) & (road_df['responded'] == True)]) for i in range(12)]
        return {'labels': labels, 'total': total, 'responded': responded}
    except Exception as e:
        logger.error(f"Error in get_cdrrmo_trends: {e}")
        return {'labels': [], 'total': [], 'responded': []}

def get_cdrrmo_distribution(time_filter, barangay=None):
    """Get distribution of emergency types."""
    try:
        road_df = load_csv_data('road_accident.csv', time_filter, barangay)
        if road_df.empty or 'emergency_type' not in road_df.columns or 'responded' not in road_df.columns:
            logger.warning("No data available for distribution")
            return {'Road Accident': {'total': 0, 'responded': 0}}
        distribution = Counter(road_df['emergency_type'])
        return {k: {'total': v, 'responded': len(road_df[(road_df['emergency_type'] == k) & (road_df['responded'] == True)])} for k, v in distribution.items()}
    except Exception as e:
        logger.error(f"Error in get_cdrrmo_distribution: {e}")
        return {'Road Accident': {'total': 0, 'responded': 0}}

def get_cdrrmo_causes(time_filter, barangay=None):
    """Get distribution of accident causes."""
    try:
        road_df = load_csv_data('road_accident.csv', time_filter, barangay)
        if road_df.empty or 'cause' not in road_df.columns:
            logger.warning("No data available for causes")
            return {'Unknown': 0}
        causes = Counter(road_df['cause'])
        return dict(causes)
    except Exception as e:
        logger.error(f"Error in get_cdrrmo_causes: {e}")
        return {'Unknown': 0}

def get_cdrrmo_weather_impact(time_filter, barangay=None):
    """Get distribution of weather conditions impacting accidents."""
    try:
        road_df = load_csv_data('road_accident.csv', time_filter, barangay)
        if road_df.empty or 'weather' not in road_df.columns:
            logger.warning("No data available for weather impact")
            return {'Clear': 0, 'Rainy': 0, 'Foggy': 0}
        weather_impact = Counter(road_df['weather'])
        return dict(weather_impact)
    except Exception as e:
        logger.error(f"Error in get_cdrrmo_weather_impact: {e}")
        return {'Clear': 0, 'Rainy': 0, 'Foggy': 0}

def get_cdrrmo_road_conditions(time_filter, barangay=None):
    """Get distribution of road conditions."""
    try:
        road_df = load_csv_data('road_accident.csv', time_filter, barangay)
        if road_df.empty or 'road_condition' not in road_df.columns:
            logger.warning("No data available for road conditions")
            return {'Dry': 0, 'Wet': 0, 'Slippery': 0}
        road_conditions = Counter(road_df['road_condition'])
        return dict(road_conditions)
    except Exception as e:
        logger.error(f"Error in get_cdrrmo_road_conditions: {e}")
        return {'Dry': 0, 'Wet': 0, 'Slippery': 0}

def get_cdrrmo_vehicle_types(time_filter, barangay=None):
    """Get distribution of vehicle types involved in accidents."""
    try:
        road_df = load_csv_data('road_accident.csv', time_filter, barangay)
        if road_df.empty or 'vehicle_type' not in road_df.columns:
            logger.warning("No data available for vehicle types")
            return {'Car': 0, 'Motorcycle': 0, 'Truck': 0}
        vehicle_types = Counter(road_df['vehicle_type'])
        return dict(vehicle_types)
    except Exception as e:
        logger.error(f"Error in get_cdrrmo_vehicle_types: {e}")
        return {'Car': 0, 'Motorcycle': 0, 'Truck': 0}

def get_cdrrmo_driver_age(time_filter, barangay=None):
    """Get distribution of driver age groups."""
    try:
        road_df = load_csv_data('road_accident.csv', time_filter, barangay)
        if road_df.empty or 'driver_age' not in road_df.columns:
            logger.warning("No data available for driver age")
            return {'18-25': 0, '26-40': 0, '41-60': 0, '60+': 0}
        driver_age = Counter(road_df['driver_age'])
        return dict(driver_age)
    except Exception as e:
        logger.error(f"Error in get_cdrrmo_driver_age: {e}")
        return {'18-25': 0, '26-40': 0, '41-60': 0, '60+': 0}

def get_cdrrmo_driver_gender(time_filter, barangay=None):
    """Get distribution of driver genders."""
    try:
        road_df = load_csv_data('road_accident.csv', time_filter, barangay)
        if road_df.empty or 'driver_gender' not in road_df.columns:
            logger.warning("No data available for driver gender")
            return {'Male': 0, 'Female': 0}
        driver_gender = Counter(road_df['driver_gender'])
        return dict(driver_gender)
    except Exception as e:
        logger.error(f"Error in get_cdrrmo_driver_gender: {e}")
        return {'Male': 0, 'Female': 0}

def get_cdrrmo_accident_type(time_filter, barangay=None):
    """Get distribution of accident types."""
    try:
        road_df = load_csv_data('road_accident.csv', time_filter, barangay)
        if road_df.empty or 'accident_type' not in road_df.columns:
            logger.warning("No data available for accident type")
            return {'Collision': 0, 'Rollover': 0, 'Pedestrian': 0}
        accident_type = Counter(road_df['accident_type'])
        return dict(accident_type)
    except Exception as e:
        logger.error(f"Error in get_cdrrmo_accident_type: {e}")
        return {'Collision': 0, 'Rollover': 0, 'Pedestrian': 0}

def get_cdrrmo_injuries(time_filter, barangay=None):
    """Get total injuries over time."""
    try:
        road_df = load_csv_data('road_accident.csv', time_filter, barangay)
        if road_df.empty or 'timestamp' not in road_df.columns or 'injuries' not in road_df.columns:
            logger.warning("No data available for injuries")
            return [0] * (24 if time_filter in ['today', 'daily'] else 7 if time_filter == 'weekly' else 15 if time_filter == 'monthly' else 12)
        start, end = get_time_range(time_filter)
        if time_filter == 'today':
            injuries = [road_df[road_df['timestamp'].dt.hour == i]['injuries'].sum() for i in range(24)]
        elif time_filter == 'daily':
            injuries = [road_df[road_df['timestamp'].dt.hour == i]['injuries'].sum() for i in range(24)]
        elif time_filter == 'weekly':
            injuries = [road_df[road_df['timestamp'].dt.date == (start + timedelta(days=i)).date()]['injuries'].sum() for i in range(7)]
        elif time_filter == 'monthly':
            injuries = [road_df[(road_df['timestamp'].dt.date >= (start + timedelta(days=i)).date()) & (road_df['timestamp'].dt.date < (start + timedelta(days=i+2)).date())]['injuries'].sum() for i in range(0, 30, 2)]
        else:
            injuries = [road_df[(road_df['timestamp'].dt.year == (start + timedelta(days=i*30)).year) & (road_df['timestamp'].dt.month == (start + timedelta(days=i*30)).month)]['injuries'].sum() for i in range(12)]
        return injuries
    except Exception as e:
        logger.error(f"Error in get_cdrrmo_injuries: {e}")
        return [0] * (24 if time_filter in ['today', 'daily'] else 7 if time_filter == 'weekly' else 15 if time_filter == 'monthly' else 12)

def get_cdrrmo_fatalities(time_filter, barangay=None):
    """Get total fatalities over time."""
    try:
        road_df = load_csv_data('road_accident.csv', time_filter, barangay)
        if road_df.empty or 'timestamp' not in road_df.columns or 'fatalities' not in road_df.columns:
            logger.warning("No data available for fatalities")
            return [0] * (24 if time_filter in ['today', 'daily'] else 7 if time_filter == 'weekly' else 15 if time_filter == 'monthly' else 12)
        start, end = get_time_range(time_filter)
        if time_filter == 'today':
            fatalities = [road_df[road_df['timestamp'].dt.hour == i]['fatalities'].sum() for i in range(24)]
        elif time_filter == 'daily':
            fatalities = [road_df[road_df['timestamp'].dt.hour == i]['fatalities'].sum() for i in range(24)]
        elif time_filter == 'weekly':
            fatalities = [road_df[road_df['timestamp'].dt.date == (start + timedelta(days=i)).date()]['fatalities'].sum() for i in range(7)]
        elif time_filter == 'monthly':
            fatalities = [road_df[(road_df['timestamp'].dt.date >= (start + timedelta(days=i)).date()) & (road_df['timestamp'].dt.date < (start + timedelta(days=i+2)).date())]['fatalities'].sum() for i in range(0, 30, 2)]
        else:
            fatalities = [road_df[(road_df['timestamp'].dt.year == (start + timedelta(days=i*30)).year) & (road_df['timestamp'].dt.month == (start + timedelta(days=i*30)).month)]['fatalities'].sum() for i in range(12)]
        return fatalities
    except Exception as e:
        logger.error(f"Error in get_cdrrmo_fatalities: {e}")
        return [0] * (24 if time_filter in ['today', 'daily'] else 7 if time_filter == 'weekly' else 15 if time_filter == 'monthly' else 12)