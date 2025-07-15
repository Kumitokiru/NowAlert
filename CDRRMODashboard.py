from alert_data import alerts
from collections import Counter
import logging

logger = logging.getLogger(__name__)

def predict_emergency(preprocessed_img, fire_session, road_session):
    if fire_session is None or road_session is None:
        return 'unknown', 0.0
    try:
        fire_input_name = fire_session.get_inputs()[0].name
        fire_output_name = fire_session.get_outputs()[0].name
        road_input_name = road_session.get_inputs()[0].name
        road_output_name = road_session.get_outputs()[0].name
        
        fire_pred = fire_session.run([fire_output_name], {fire_input_name: preprocessed_img})[0]
        road_pred = road_session.run([road_output_name], {road_input_name: preprocessed_img})[0]
        
        fire_prob = fire_pred[0][0]
        road_prob = road_pred[0][0]
        
        if fire_prob > road_prob:
            if fire_prob > 0.5:
                return 'fire_incident', fire_prob
            else:
                return 'unknown', 0.0
        else:
            if road_prob > 0.5:
                return 'road_accident', road_prob
            else:
                return 'unknown', 0.0
    except Exception as e:
        logger.error(f"Prediction failed in CDRRMODashboard: {e}")
        return 'unknown', 0.0


def get_cdrrmo_stats():
    types = [a.get('emergency_type', 'unknown') for a in alerts if a.get('role') == 'cdrrmo' or a.get('municipality')]
    return Counter(types)

def get_latest_alert():
    if alerts:
        return list(alerts)[-1]
    return None
