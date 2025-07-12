import pandas as pd
import logging
from datetime import datetime, timedelta
import pytz
from collections import Counter
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load machine learning model
try:
    dt_classifier = joblib.load('decision_tree_model.pkl')
    logger.info("Machine learning model loaded successfully")
except Exception as e:
    dt_classifier = None
    logger.error(f"Failed to load machine learning model: {e}")

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

def load_csv_data(file_path, time_filter, barangay=None):
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        start, end = get_time_range(time_filter)
        df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
        if barangay:
            df = df[df['barangay'] == barangay]
        return df
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {e}")
        return pd.DataFrame()

def get_pnp_trends(time_filter, barangay=None):
    try:
        df = load_csv_data('road_accident.csv', time_filter, barangay)
        
        start, end = get_time_range(time_filter)
        if time_filter == 'today':
            labels = [(start + timedelta(hours=i)).strftime('%H:%M') for i in range(24)]
            total = [len(df[df['timestamp'].dt.hour == i]) for i in range(24)]
            responded = [len(df[(df['timestamp'].dt.hour == i) & (df['responded'] == True)]) for i in range(24)]
        elif time_filter == 'daily':
            labels = [(start + timedelta(hours=i)).strftime('%H:%M') for i in range(24)]
            total = [len(df[df['timestamp'].dt.hour == i]) for i in range(24)]
            responded = [len(df[(df['timestamp'].dt.hour == i) & (df['responded'] == True)]) for i in range(24)]
        elif time_filter == 'weekly':
            labels = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
            total = [len(df[df['timestamp'].dt.date == (start + timedelta(days=i)).date()]) for i in range(7)]
            responded = [len(df[(df['timestamp'].dt.date == (start + timedelta(days=i)).date()) & (df['responded'] == True)]) for i in range(7)]
        elif time_filter == 'monthly':
            labels = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(0, 30, 2)]
            total = [len(df[(df['timestamp'].dt.date >= (start + timedelta(days=i)).date()) & (df['timestamp'].dt.date < (start + timedelta(days=i+2)).date())]) for i in range(0, 30, 2)]
            responded = [len(df[(df['timestamp'].dt.date >= (start + timedelta(days=i)).date()) & (df['timestamp'].dt.date < (start + timedelta(days=i+2)).date()) & (df['responded'] == True)]) for i in range(0, 30, 2)]
        else:
            labels = [(start + timedelta(days=i*30)).strftime('%Y-%m') for i in range(12)]
            total = [len(df[(df['timestamp'].dt.year == (start + timedelta(days=i*30)).year) & (df['timestamp'].dt.month == (start + timedelta(days=i*30)).month)]) for i in range(12)]
            responded = [len(df[(df['timestamp'].dt.year == (start + timedelta(days=i*30)).year) & (df['timestamp'].dt.month == (start + timedelta(days=i*30)).month) & (df['responded'] == True)]) for i in range(12)]
        
        return {'labels': labels, 'total': total, 'responded': responded}
    except Exception as e:
        logger.error(f"Error in get_pnp_trends: {e}")
        return {'labels': [], 'total': [], 'responded': []}

def get_pnp_distribution(time_filter, barangay=None):
    try:
        df = load_csv_data('road_accident.csv', time_filter, barangay)
        distribution = Counter(df['emergency_type'])
        return {k: {'total': v, 'responded': len(df[(df['emergency_type'] == k) & (df['responded'] == True)])} for k, v in distribution.items()}
    except Exception as e:
        logger.error(f"Error in get_pnp_distribution: {e}")
        return {'Road Accident': {'total': 0, 'responded': 0}}

def get_pnp_causes(time_filter, barangay=None):
    try:
        df = load_csv_data('road_accident.csv', time_filter, barangay)
        causes = Counter(df['cause'])
        if dt_classifier:
            try:
                recent = df.tail(10)
                if not recent.empty and 'features' in recent.columns:
                    predictions = dt_classifier.predict(recent['features'].values.reshape(-1, 1))
                    predicted_causes = Counter(predictions)
                    causes.update(predicted_causes)
            except Exception as e:
                logger.error(f"Error in model prediction for causes: {e}")
        return dict(causes)
    except Exception as e:
        logger.error(f"Error in get_pnp_causes: {e}")
        return {'Unknown': 0}

def get_pnp_weather_impact(time_filter, barangay=None):
    try:
        df = load_csv_data('road_accident.csv', time_filter, barangay)
        return dict(Counter(df['weather']))
    except Exception as e:
        logger.error(f"Error in get_pnp_weather_impact: {e}")
        return {'Clear': 0, 'Rainy': 0, 'Foggy': 0}

def get_pnp_road_conditions(time_filter, barangay=None):
    try:
        df = load_csv_data('road_accident.csv', time_filter, barangay)
        return dict(Counter(df['road_condition']))
    except Exception as e:
        logger.error(f"Error in get_pnp_road_conditions: {e}")
        return {'Dry': 0, 'Wet': 0, 'Slippery': 0}

def get_pnp_vehicle_types(time_filter, barangay=None):
    try:
        df = load_csv_data('road_accident.csv', time_filter, barangay)
        return dict(Counter(df['vehicle_type']))
    except Exception as e:
        logger.error(f"Error in get_pnp_vehicle_types: {e}")
        return {'Car': 0, 'Motorcycle': 0, 'Truck': 0}

def get_pnp_driver_age(time_filter, barangay=None):
    try:
        df = load_csv_data('road_accident.csv', time_filter, barangay)
        return dict(Counter(df['driver_age']))
    except Exception as e:
        logger.error(f"Error in get_pnp_driver_age: {e}")
        return {'18-25': 0, '26-40': 0, '41-60': 0, '60+': 0}

def get_pnp_driver_gender(time_filter, barangay=None):
    try:
        df = load_csv_data('road_accident.csv', time_filter, barangay)
        return dict(Counter(df['driver_gender']))
    except Exception as e:
        logger.error(f"Error in get_pnp_driver_gender: {e}")
        return {'Male': 0, 'Female': 0}

def get_pnp_accident_type(time_filter, barangay=None):
    try:
        df = load_csv_data('road_accident.csv', time_filter, barangay)
        return dict(Counter(df['accident_type']))
    except Exception as e:
        logger.error(f"Error in get_pnp_accident_type: {e}")
        return {'Collision': 0, 'Rollover': 0, 'Pedestrian': 0}

def get_pnp_injuries(time_filter, barangay=None):
    try:
        df = load_csv_data('road_accident.csv', time_filter, barangay)
        start, end = get_time_range(time_filter)
        if time_filter == 'today':
            labels = [(start + timedelta(hours=i)).strftime('%H:%M') for i in range(24)]
            injuries = [df[df['timestamp'].dt.hour == i]['injuries'].sum() for i in range(24)]
        elif time_filter == 'daily':
            labels = [(start + timedelta(hours=i)).strftime('%H:%M') for i in range(24)]
            injuries = [df[df['timestamp'].dt.hour == i]['injuries'].sum() for i in range(24)]
        elif time_filter == 'weekly':
            labels = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
            injuries = [df[df['timestamp'].dt.date == (start + timedelta(days=i)).date()]['injuries'].sum() for i in range(7)]
        elif time_filter == 'monthly':
            labels = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(0, 30, 2)]
            injuries = [df[(df['timestamp'].dt.date >= (start + timedelta(days=i)).date()) & (df['timestamp'].dt.date < (start + timedelta(days=i+2)).date())]['injuries'].sum() for i in range(0, 30, 2)]
        else:
            labels = [(start + timedelta(days=i*30)).strftime('%Y-%m') for i in range(12)]
            injuries = [df[(df['timestamp'].dt.year == (start + timedelta(days=i*30)).year) & (df['timestamp'].dt.month == (start + timedelta(days=i*30)).month)]['injuries'].sum() for i in range(12)]
        return injuries
    except Exception as e:
        logger.error(f"Error in get_pnp_injuries: {e}")
        return [0] * (24 if time_filter in ['today', 'daily'] else 7 if time_filter == 'weekly' else 15 if time_filter == 'monthly' else 12)

def get_pnp_fatalities(time_filter, barangay=None):
    try:
        df = load_csv_data('road_accident.csv', time_filter, barangay)
        start, end = get_time_range(time_filter)
        if time_filter == 'today':
            labels = [(start + timedelta(hours=i)).strftime('%H:%M') for i in range(24)]
            fatalities = [df[df['timestamp'].dt.hour == i]['fatalities'].sum() for i in range(24)]
        elif time_filter == 'daily':
            labels = [(start + timedelta(hours=i)).strftime('%H:%M') for i in range(24)]
            fatalities = [df[df['timestamp'].dt.hour == i]['fatalities'].sum() for i in range(24)]
        elif time_filter == 'weekly':
            labels = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
            fatalities = [df[df['timestamp'].dt.date == (start + timedelta(days=i)).date()]['fatalities'].sum() for i in range(7)]
        elif time_filter == 'monthly':
            labels = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(0, 30, 2)]
            fatalities = [df[(df['timestamp'].dt.date >= (start + timedelta(days=i)).date()) & (df['timestamp'].dt.date < (start + timedelta(days=i+2)).date())]['fatalities'].sum() for i in range(0, 30, 2)]
        else:
            labels = [(start + timedelta(days=i*30)).strftime('%Y-%m') for i in range(12)]
            fatalities = [df[(df['timestamp'].dt.year == (start + timedelta(days=i*30)).year) & (df['timestamp'].dt.month == (start + timedelta(days=i*30)).month)]['fatalities'].sum() for i in range(12)]
        return fatalities
    except Exception as e:
        logger.error(f"Error in get_pnp_fatalities: {e}")
        return [0] * (24 if time_filter in ['today', 'daily'] else 7 if time_filter == 'weekly' else 15 if time_filter == 'monthly' else 12)
