from alert_data import alerts
from collections import Counter
import onnxruntime as ort
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load ONNX models
fire_model = os.path.join('training', 'Fire Models', 'fire_incident_model.onnx')
road_model = os.path.join('training', 'Road Models', 'road_accident_model.onnx')

try:
    fire_session = ort.InferenceSession(fire_model)
    road_session = ort.InferenceSession(road_model)
    logger.info("ONNX models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading ONNX models: {e}")
    fire_session = None

def predict_emergency_type(image):
    if road_model is None or fire_model is None:
        logger.warning("ONNX models not loaded, returning unknown prediction")
        return 'unknown', 0.0
    try:
        input_name_road = road_model.get_inputs()[0].name
        output_road = road_model.run(None, {input_name_road: image})
        prob_road = float(output_road[0][0])

        input_name_fire = fire_model.get_inputs()[0].name
        output_fire = fire_model.run(None, {input_name_fire: image})
        prob_fire = float(output_fire[0][0])

        if prob_road > prob_fire and prob_road > 0.5:
            return 'road_accident', prob_road
        elif prob_fire > prob_road and prob_fire > 0.5:
            return 'fire_incident', prob_fire
        else:
            return 'unknown', 0.0
    except Exception as e:
        logger.error(f"Prediction failed in CDRRMODashboard: {e}")
        return 'unknown', 0.0


def get_pnp_stats():
    types = [a.get('emergency_type', 'unknown') for a in alerts if a.get('role') == 'pnp' or a.get('municipality')]
    return Counter(types)

def get_latest_alert():
    if alerts:
        return list(alerts)[-1]
    return None
