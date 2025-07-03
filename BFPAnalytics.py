from alert_data import alerts
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import pytz
import logging

def get_bfp_trends():
    try:
        bfp_alerts = [a for a in alerts if a.get('role') == 'bfp' or a.get('emergency_type') == 'fire']
        today = datetime.now(pytz.timezone('Asia/Manila')).date()
        labels = [(today - timedelta(days=i)).strftime('%b %d') for i in range(6, -1, -1)]
        total = [0] * 7
        responded = [0] * 7
        
        for alert in bfp_alerts:
            alert_date = datetime.fromisoformat(alert['timestamp']).date()
            days_ago = (today - alert_date).days
            if 0 <= days_ago < 7:
                total[6 - days_ago] += 1
                if alert.get('responded', False):
                    responded[6 - days_ago] += 1
        
        return {'labels': labels, 'total': total, 'responded': responded}
    except Exception as e:
        logging.error(f"Error in get_bfp_trends: {e}", exc_info=True)
        return {'labels': [], 'total': [], 'responded': []}

def get_bfp_distribution():
    try:
        bfp_alerts = [a for a in alerts if a.get('role') == 'bfp' or a.get('emergency_type') == 'fire']
        distribution = defaultdict(lambda: {'total': 0, 'responded': 0})
        for alert in bfp_alerts:
            emergency_type = alert.get('emergency_type', 'unknown')
            distribution[emergency_type]['total'] += 1
            if alert.get('responded', False):
                distribution[emergency_type]['responded'] += 1
        return distribution
    except Exception as e:
        logging.error(f"Error in get_bfp_distribution: {e}", exc_info=True)
        return {}

def get_bfp_causes():
    try:
        causes = {'Electrical': 15, 'Arson': 5, 'Cooking': 10, 'Smoking': 2}
        return causes
    except Exception as e:
        logging.error(f"Error in get_bfp_causes: {e}", exc_info=True)
        return {}