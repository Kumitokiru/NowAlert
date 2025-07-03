from alert_data import alerts
from collections import Counter

def get_pnp_stats():
    types = [a.get('emergency_type', 'unknown') for a in alerts if a.get('role') == 'pnp' or a.get('municipality')]
    return Counter(types)