from alert_data import alerts
from collections import Counter

def get_cdrrmo_stats():
    types = [a.get('emergency_type', 'unknown') for a in alerts if a.get('role') == 'cdrrmo' or a.get('municipality')]
    return Counter(types)

def get_latest_alert():
    if alerts:
        return list(alerts)[-1]
    return None
