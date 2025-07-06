import pandas as pd
import logging
from datetime import datetime, timedelta
import pytz
import logging

def get_pnp_trends():
    try:
        pnp_alerts = [a for a in alerts if a.get('role') == 'pnp' or a.get('emergency_type') == 'road_accident']
        today = datetime.now(pytz.timezone('Asia/Manila')).date()
        labels = [(today - timedelta(days=i)).strftime('%b %d') for i in range(6, -1, -1)]
        total = [0] * 7
        responded = [0] * 7
        
        for alert in pnp_alerts:
            alert_date = datetime.fromisoformat(alert['timestamp']).date()
            days_ago = (today - alert_date).days
            if 0 <= days_ago < 7:
                total[6 - days_ago] += 1
                if alert.get('responded', False):
                    responded[6 - days_ago] += 1
        
        return {'labels': labels, 'total': total, 'responded': responded}
    except Exception as e:
        logger.error(f"Error in get_pnp_trends: {e}")
        return {'labels': [], 'total': [], 'responded': []}

def get_pnp_distribution():
    try:
        pnp_alerts = [a for a in alerts if a.get('role') == 'pnp' or a.get('emergency_type') == 'road_accident']
        distribution = defaultdict(lambda: {'total': 0, 'responded': 0})
        for alert in pnp_alerts:
            emergency_type = alert.get('emergency_type', 'unknown')
            distribution[emergency_type]['total'] += 1
            if alert.get('responded', False):
                distribution[emergency_type]['responded'] += 1
        return distribution
    except Exception as e:
        logging.error(f"Error in get_pnp_distribution: {e}", exc_info=True)
        return {}

def get_pnp_causes():
    try:
        causes = {'Speeding': 20, 'Drunk Driving': 10, 'Distracted Driving': 15, 'Weather': 5}
        return causes
    except Exception as e:
        logging.error(f"Error in get_pnp_causes: {e}", exc_info=True)
        return {}