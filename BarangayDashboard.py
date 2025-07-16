from alert_data import alerts
from collections import Counter
import onnxruntime as ort
import numpy as np
import logging
import os

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def predict_emergency_type(image, fire_session, road_session):
    if fire_session is None or road_session is None:
        logger.warning("ONNX models not loaded, returning unknown prediction")
        return 'unknown', 0.0
    try:
        input_name_road = road_session.get_inputs()[0].name
        output_road = road_session.run(None, {input_name_road: image})
        prob_road = float(output_road[0][0])

        input_name_fire = fire_session.get_inputs()[0].name
        output_fire = fire_session.run(None, {input_name_fire: image})
        prob_fire = float(output_fire[0][0])

        if prob_road > prob_fire and prob_road > 0.5:
            return 'road_accident', prob_road
        elif prob_fire > prob_road and prob_fire > 0.5:
            return 'fire_incident', prob_fire
        else:
            return 'unknown', max(prob_road, prob_fire)
    except Exception as e:
        logger.error(f"Prediction failed in BarangayDashboard: {e}")
        return 'unknown', 0.0

def get_barangay_stats():
    types = [a.get('emergency_type', 'unknown') for a in alerts if a.get('role') == 'barangay' or a.get('barangay')]
    return Counter(types)

def get_latest_alert():
    if alerts:
        return list(alerts)[-1]
    return None
