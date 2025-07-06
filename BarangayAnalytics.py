import pandas as pd
import logging
from datetime import datetime, timedelta
import pytz
from collections import Counter
import os
import joblib
import numpy as np

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load trained models globally
MODEL_DIR = 'training/Road Models/'
models = {}
model_files = {
    
    'accident_type': 'lr_road_accident.pkl'
}
for key, filename in model_files.items():
    try:
        models[key] = joblib.load(os.path.join(MODEL_DIR, filename))
        logger.info(f"Loaded model: {filename}")
    except Exception as e:
        logger.error(f"Failed to load {filename}: {e}")
        models[key] = None

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
    try:
        file_path_full = os.path.join('dataset', file_path)
        if not os.path.exists(file_path_full):
            logger.error(f"CSV file not found: {file_path_full}")
            return pd.DataFrame()
        
        df = pd.read_csv(file_path_full)
        if 'Date' in df.columns:  # Replace 'timestamp' with the actual column name
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            start, end = get_time_range(time_filter)
            df = df[(df['Date'].notna()) & (df['Date'] >= start) & (df['Date'] <= end)]
        else:
            logger.warning(f"'Date' column not found in {file_path}. Skipping time-based filtering.")
        
        if barangay and 'barangay' in df.columns:
            df = df[df['barangay'] == barangay]
        return df
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {e}")
        return pd.DataFrame()

def get_baseline_probabilities(model, default_classes):
    """Compute baseline probabilities from a logistic regression model's intercepts."""
    if model is None or not hasattr(model, 'intercept_') or not hasattr(model, 'classes_'):
        logger.warning("Model not available or invalid")
        return dict.fromkeys(default_classes, 0)
    try:
        intercepts = model.intercept_
        classes = model.classes_
        exp_intercepts = np.exp(intercepts)
        probs = exp_intercepts / exp_intercepts.sum()
        return dict(zip(classes, probs))
    except Exception as e:
        logger.error(f"Error computing probabilities: {e}")
        return dict.fromkeys(default_classes, 0)

def get_barangay_trends(time_filter, barangay=None):
    """Get incident trends over time (uses CSV data)."""
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
        logger.error(f"Error in get_barangay_trends: {e}")
        return {'labels': [], 'total': [], 'responded': []}

def get_barangay_distribution(time_filter, barangay=None):
    """Get distribution of emergency types (simplified, using accident_type model)."""
    default = {'Road Accident': {'total': 0, 'responded': 0}}
    model = models.get('accident_type')
    if model:
        probs = get_baseline_probabilities(model, ['Road Accident'])
        return {'Road Accident': {'total': probs.get('Road Accident', 0), 'responded': 0}}
    return default

def get_barangay_causes(time_filter, barangay=None):
    """Get distribution of accident causes (no direct model, using default)."""
    return {'Unknown': 0}  # No specific model provided for causes

def get_barangay_weather_impact(time_filter, barangay=None):
    """Get distribution of weather conditions using model."""
    default = {'Clear': 0, 'Rainy': 0, 'Foggy': 0}
    model = models.get('weather')
    return get_baseline_probabilities(model, default.keys())

def get_barangay_road_conditions(time_filter, barangay=None):
    """Get distribution of road conditions using model."""
    default = {'Dry': 0, 'Wet': 0, 'Slippery': 0}
    model = models.get('road_condition')
    return get_baseline_probabilities(model, default.keys())

def get_barangay_vehicle_types(time_filter, barangay=None):
    """Get distribution of vehicle types using model."""
    default = {'Car': 0, 'Motorcycle': 0, 'Truck': 0}
    model = models.get('vehicle_type')
    return get_baseline_probabilities(model, default.keys())

def get_barangay_driver_age(time_filter, barangay=None):
    """Get distribution of driver age groups using model."""
    default = {'18-25': 0, '26-40': 0, '41-60': 0, '60+': 0}
    model = models.get('driver_age')
    return get_baseline_probabilities(model, default.keys())

def get_barangay_driver_gender(time_filter, barangay=None):
    """Get distribution of driver genders using model."""
    default = {'Male': 0, 'Female': 0}
    model = models.get('driver_gender')
    return get_baseline_probabilities(model, default.keys())

def get_barangay_accident_type(time_filter, barangay=None):
    """Get distribution of accident types using model."""
    default = {'Collision': 0, 'Rollover': 0, 'Pedestrian': 0}
    model = models.get('accident_type')
    return get_baseline_probabilities(model, default.keys())

def get_barangay_injuries(time_filter, barangay=None):
    """Get total injuries over time (uses CSV data)."""
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
        logger.error(f"Error in get_barangay_injuries: {e}")
        return [0] * (24 if time_filter in ['today', 'daily'] else 7 if time_filter == 'weekly' else 15 if time_filter == 'monthly' else 12)

def get_barangay_fatalities(time_filter, barangay=None):
    """Get total fatalities over time (uses CSV data)."""
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
        logger.error(f"Error in get_barangay_fatalities: {e}")
        return [0] * (24 if time_filter in ['today', 'daily'] else 7 if time_filter == 'weekly' else 15 if time_filter == 'monthly' else 12)