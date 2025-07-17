import joblib
import os
import logging

logger = logging.getLogger(__name__)

# Load road accident model
road_models_path = os.path.join(os.path.dirname(__file__), 'training', 'Road Models')
try:
    lr_road = joblib.load(os.path.join(road_models_path, 'lr_road_accident.pkl'))
    logger.info("lr_road_accident.pkl loaded successfully.")
except FileNotFoundError:
    logger.error("lr_road_accident.pkl not found.")
    lr_road = None
except Exception as e:
    logger.error(f"Error loading lr_road_accident.pkl: {e}")
    lr_road = None

# Load fire incident model
fire_models_path = os.path.join(os.path.dirname(__file__), 'training', 'Fire Models')
try:
    lr_fire = joblib.load(os.path.join(fire_models_path, 'lr_fire_incident.pkl'))
    logger.info("lr_fire_incident.pkl loaded successfully.")
except FileNotFoundError:
    logger.error("lr_fire_incident.pkl not found.")
    lr_fire = None
except Exception as e:
    logger.error(f"Error loading lr_fire_incident.pkl: {e}")
    lr_fire = None