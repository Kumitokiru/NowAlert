import pandas as pd
import logging
from datetime import datetime, timedelta
import pytz
from collections import Counter
import os
from models import lr_road, lr_fire  # Updated import

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

def load_csv_data(file_path, time_filter):
    try:
        file_path_full = os.path.join('dataset', file_path)
        if not os.path.exists(file_path_full):
            logger.error(f"CSV file not found: {file_path_full}")
            return pd.DataFrame()
        df = pd.read_csv(file_path_full)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            start, end = get_time_range(time_filter)
            df = df[(df['timestamp'].notna()) & (df['timestamp'] >= start) & (df['timestamp'] <= end)]
        return df
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {e}")
        return pd.DataFrame()

def get_barangay_trends(time_filter):
    try:
        road_df = load_csv_data('road_accident.csv', time_filter)
        fire_df = load_csv_data('fire_incident.csv', time_filter)
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
        road_df = load_csv_data('road_accident.csv', time_filter)
        fire_df = load_csv_data('fire_incident.csv', time_filter)
        distribution = Counter(road_df['emergency_type']) + Counter(fire_df['emergency_type'])
        return {k: {'total': v, 'responded': len(road_df[(road_df['emergency_type'] == k) & (road_df['responded'] == True)]) + 
                    len(fire_df[(fire_df['emergency_type'] == k) & (fire_df['responded'] == True)])} for k, v in distribution.items()}
    except Exception as e:
        logger.error(f"Error in get_barangay_distribution: {e}")
        return {'Unknown': {'total': 0, 'responded': 0}}

def get_barangay_causes(time_filter):
    try:
        road_df = load_csv_data('road_accident.csv', time_filter)
        fire_df = load_csv_data('fire_incident.csv', time_filter)
        road_causes = Counter(road_df['cause'])
        fire_causes = Counter(fire_df['cause'])
        if lr_road and not road_df.empty:
            features = road_df[['weather', 'road_condition', 'vehicle_type']].fillna(0)
            predictions = lr_road.predict(features)
            road_causes.update(predictions)
        if lr_fire and not fire_df.empty:
            features = fire_df[['weather', 'property_type']].fillna(0)
            predictions = lr_fire.predict(features)
            fire_causes.update(predictions)
        return {'road': dict(road_causes), 'fire': dict(fire_causes)}
    except Exception as e:
        logger.error(f"Error in get_barangay_causes: {e}")
        return {'road': {'Unknown': 0}, 'fire': {'Unknown': 0}}

# Rename functions to reflect BFP context if needed
def get_bfp_trends(time_filter):
    return get_barangay_trends(time_filter)

def get_bfp_distribution(time_filter):
    return get_barangay_distribution(time_filter)

def get_bfp_causes(time_filter):
    return get_barangay_causes(time_filter)
