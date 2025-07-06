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

def get_bfp_trends(time_filter, barangay=None):
    try:
        df = load_csv_data('fire_incident.csv', time_filter, barangay)
        
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
        logger.error(f"Error in get_bfp_trends: {e}")
        return {'labels': [], 'total': [], 'responded': []}

def get_bfp_distribution(time_filter, barangay=None):
    try:
        df = load_csv_data('fire_incident.csv', time_filter, barangay)
        distribution = Counter(df['emergency_type'])
        return {k: {'total': v, 'responded': len(df[(df['emergency_type'] == k) & (df['responded'] == True)])} for k, v in distribution.items()}
    except Exception as e:
        logger.error(f"Error in get_bfp_distribution: {e}")
        return {'Fire': {'total': 0, 'responded': 0}}

def get_bfp_causes(time_filter, barangay=None):
    try:
        df = load_csv_data('fire_incident.csv', time_filter, barangay)
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
        logger.error(f"Error in get_bfp_causes: {e}")
        return {'Unknown': 0}

def get_bfp_weather_impact(time_filter, barangay=None):
    try:
        df = load_csv_data('fire_incident.csv', time_filter, barangay)
        return dict(Counter(df['weather']))
    except Exception as e:
        logger.error(f"Error in get_bfp_weather_impact: {e}")
        return {'Clear': 0, 'Rainy': 0, 'Foggy': 0}

def get_bfp_property_types(time_filter, barangay=None):
    try:
        df = load_csv_data('fire_incident.csv', time_filter, barangay)
        return dict(Counter(df['property_type']))
    except Exception as e:
        logger.error(f"Error in get_bfp_property_types: {e}")
        return {'Residential': 0, 'Commercial': 0, 'Industrial': 0}

def get_bfp_fire_severity(time_filter, barangay=None):
    try:
        df = load_csv_data('fire_incident.csv', time_filter, barangay)
        return dict(Counter(df['severity']))
    except Exception as e:
        logger.error(f"Error in get_bfp_fire_severity: {e}")
        return {'Low': 0, 'Medium': 0, 'High': 0}

def get_bfp_casualty_count(time_filter, barangay=None):
    try:
        df = load_csv_data('fire_incident.csv', time_filter, barangay)
        start, end = get_time_range(time_filter)
        if time_filter == 'today':
            labels = [(start + timedelta(hours=i)).strftime('%H:%M') for i in range(24)]
            casualties = [df[df['timestamp'].dt.hour == i]['casualty_count'].sum() for i in range(24)]
        elif time_filter == 'daily':
            labels = [(start + timedelta(hours=i)).strftime('%H:%M') for i in range(24)]
            casualties = [df[df['timestamp'].dt.hour == i]['casualty_count'].sum() for i in range(24)]
        elif time_filter == 'weekly':
            labels = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
            casualties = [df[df['timestamp'].dt.date == (start + timedelta(days=i)).date()]['casualty_count'].sum() for i in range(7)]
        elif time_filter == 'monthly':
            labels = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(0, 30, 2)]
            casualties = [df[(df['timestamp'].dt.date >= (start + timedelta(days=i)).date()) & (df['timestamp'].dt.date < (start + timedelta(days=i+2)).date())]['casualty_count'].sum() for i in range(0, 30, 2)]
        else:
            labels = [(start + timedelta(days=i*30)).strftime('%Y-%m') for i in range(12)]
            casualties = [df[(df['timestamp'].dt.year == (start + timedelta(days=i*30)).year) & (df['timestamp'].dt.month == (start + timedelta(days=i*30)).month)]['casualty_count'].sum() for i in range(12)]
        return casualties
    except Exception as e:
        logger.error(f"Error in get_bfp_casualty_count: {e}")
        return [0] * (24 if time_filter in ['today', 'daily'] else 7 if time_filter == 'weekly' else 15 if time_filter == 'monthly' else 12)

def get_bfp_response_time(time_filter, barangay=None):
    try:
        df = load_csv_data('fire_incident.csv', time_filter, barangay)
        start, end = get_time_range(time_filter)
        if time_filter == 'today':
            labels = [(start + timedelta(hours=i)).strftime('%H:%M') for i in range(24)]
            response_times = [df[df['timestamp'].dt.hour == i]['response_time'].mean() for i in range(24)]
        elif time_filter == 'daily':
            labels = [(start + timedelta(hours=i)).strftime('%H:%M') for i in range(24)]
            response_times = [df[df['timestamp'].dt.hour == i]['response_time'].mean() for i in range(24)]
        elif time_filter == 'weekly':
            labels = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
            response_times = [df[df['timestamp'].dt.date == (start + timedelta(days=i)).date()]['response_time'].mean() for i in range(7)]
        elif time_filter == 'monthly':
            labels = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(0, 30, 2)]
            response_times = [df[(df['timestamp'].dt.date >= (start + timedelta(days=i)).date()) & (df['timestamp'].dt.date < (start + timedelta(days=i+2)).date())]['response_time'].mean() for i in range(0, 30, 2)]
        else:
            labels = [(start + timedelta(days=i*30)).strftime('%Y-%m') for i in range(12)]
            response_times = [df[(df['timestamp'].dt.year == (start + timedelta(days=i*30)).year) & (df['timestamp'].dt.month == (start + timedelta(days=i*30)).month)]['response_time'].mean() for i in range(12)]
        return [0 if pd.isna(x) else x for x in response_times]
    except Exception as e:
        logger.error(f"Error in get_bfp_response_time: {e}")
        return [0] * (24 if time_filter in ['today', 'daily'] else 7 if time_filter == 'weekly' else 15 if time_filter == 'monthly' else 12)

def get_bfp_fire_duration(time_filter, barangay=None):
    try:
        df = load_csv_data('fire_incident.csv', time_filter, barangay)
        start, end = get_time_range(time_filter)
        if time_filter == 'today':
            labels = [(start + timedelta(hours=i)).strftime('%H:%M') for i in range(24)]
            durations = [df[df['timestamp'].dt.hour == i]['fire_duration'].mean() for i in range(24)]
        elif time_filter == 'daily':
            labels = [(start + timedelta(hours=i)).strftime('%H:%M') for i in range(24)]
            durations = [df[df['timestamp'].dt.hour == i]['fire_duration'].mean() for i in range(24)]
        elif time_filter == 'weekly':
            labels = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(7)]
            durations = [df[df['timestamp'].dt.date == (start + timedelta(days=i)).date()]['fire_duration'].mean() for i in range(7)]
        elif time_filter == 'monthly':
            labels = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(0, 30, 2)]
            durations = [df[(df['timestamp'].dt.date >= (start + timedelta(days=i)).date()) & (df['timestamp'].dt.date < (start + timedelta(days=i+2)).date())]['fire_duration'].mean() for i in range(0, 30, 2)]
        else:
            labels = [(start + timedelta(days=i*30)).strftime('%Y-%m') for i in range(12)]
            durations = [df[(df['timestamp'].dt.year == (start + timedelta(days=i*30)).year) & (df['timestamp'].dt.month == (start + timedelta(days=i*30)).month)]['fire_duration'].mean() for i in range(12)]
        return [0 if pd.isna(x) else x for x in durations]
    except Exception as e:
        logger.error(f"Error in get_bfp_fire_duration: {e}")
        return [0] * (24 if time_filter in ['today', 'daily'] else 7 if time_filter == 'weekly' else 15 if time_filter == 'monthly' else 12)