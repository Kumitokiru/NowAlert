from alert_data import alerts
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import pytz
import logging

def get_barangay_trends(time_filter='weekly'):
    try:
        barangay_alerts = [a for a in alerts if a.get('role') == 'barangay' or a.get('barangay')]
        today = datetime.now(pytz.timezone('Asia/Manila')).date()
        
        # Determine time range
        if time_filter == 'today':
            start_date = today
            labels = [today.strftime('%b %d')]
        elif time_filter == 'daily':
            start_date = today - timedelta(days=1)
            labels = [(today - timedelta(days=i)).strftime('%b %d') for i in range(1, -1, -1)]
        elif time_filter == 'weekly':
            start_date = today - timedelta(days=6)
            labels = [(today - timedelta(days=i)).strftime('%b %d') for i in range(6, -1, -1)]
        elif time_filter == 'monthly':
            start_date = today - timedelta(days=29)
            labels = [(today - timedelta(days=i)).strftime('%b %d') for i in range(29, -1, -1)]
        elif time_filter == 'yearly':
            start_date = today - timedelta(days=364)
            labels = [(today - timedelta(days=i)).strftime('%b %d') for i in range(364, -1, -1)]
        else:
            start_date = today - timedelta(days=6)
            labels = [(today - timedelta(days=i)).strftime('%b %d') for i in range(6, -1, -1)]
        
        total = [0] * len(labels)
        responded = [0] * len(labels)
        
        for alert in barangay_alerts:
            alert_date = datetime.fromisoformat(alert['timestamp']).date()
            days_ago = (today - alert_date).days
            if 0 <= days_ago < len(labels):
                total[len(labels) - 1 - days_ago] += 1
                if alert.get('responded', False):
                    responded[len(labels) - 1 - days_ago] += 1
        
        return {'labels': labels, 'total': total, 'responded': responded}
    except Exception as e:
        logging.error(f"Error in get_barangay_trends: {e}", exc_info=True)
        return {'labels': [], 'total': [], 'responded': []}

def get_barangay_distribution(time_filter='weekly'):
    try:
        barangay_alerts = [a for a in alerts if a.get('role') == 'barangay' or a.get('barangay')]
        today = datetime.now(pytz.timezone('Asia/Manila')).date()
        
        # Determine time range
        if time_filter == 'today':
            start_date = today
        elif time_filter == 'daily':
            start_date = today - timedelta(days=1)
        elif time_filter == 'weekly':
            start_date = today - timedelta(days=6)
        elif time_filter == 'monthly':
            start_date = today - timedelta(days=29)
        elif time_filter == 'yearly':
            start_date = today - timedelta(days=364)
        else:
            start_date = today - timedelta(days=6)
        
        distribution = defaultdict(lambda: {'total': 0, 'responded': 0})
        for alert in barangay_alerts:
            alert_date = datetime.fromisoformat(alert['timestamp']).date()
            if alert_date >= start_date:
                emergency_type = alert.get('emergency_type', 'unknown')
                distribution[emergency_type]['total'] += 1
                if alert.get('responded', False):
                    distribution[emergency_type]['responded'] += 1
        return distribution
    except Exception as e:
        logging.error(f"Error in get_barangay_distribution: {e}", exc_info=True)
        return {}

def get_barangay_causes(time_filter='weekly'):
    try:
        # Placeholder for cause analysis; replace with actual logic if data available
        causes = {'Fire': 15, 'Road Accident': 5, 'Others': 2}
        return causes
    except Exception as e:
        logging.error(f"Error in get_barangay_causes: {e}", exc_info=True)
        return {}
