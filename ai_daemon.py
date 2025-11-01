import psutil
import requests
import time
from threading import Thread
from flask import Flask, request, jsonify

# --- Configuration ---
CPU_THRESHOLD = 80.0  # percent
MEMORY_THRESHOLD = 80.0  # percent
CHECK_INTERVAL = 10  # seconds
N8N_WEBHOOK_URL = "http://localhost:5678/webhook/ai-daemon-alert" # Replace with your n8n webhook URL

# --- Flask App for receiving status updates ---
app = Flask(__name__)
daemon_status = {"status": "running", "last_alert": None}

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify(daemon_status)

@app.route('/update', methods=['POST'])
def update_status():
    data = request.json
    daemon_status.update(data)
    return jsonify({"message": "Status updated successfully"})

def run_flask_app():
    app.run(port=5002, debug=True, use_reloader=False)

# --- AI Daemon Logic ---
def monitor_system():
    """Monitors CPU and memory usage and sends alerts."""
    while True:
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        alert_payload = None

        if cpu_percent > CPU_THRESHOLD:
            alert_payload = {
                "type": "cpu_alert",
                "value": cpu_percent,
                "threshold": CPU_THRESHOLD,
                "message": f"High CPU usage detected: {cpu_percent}%"
            }

        if memory_percent > MEMORY_THRESHOLD:
            alert_payload = {
                "type": "memory_alert",
                "value": memory_percent,
                "threshold": MEMORY_THRESHOLD,
                "message": f"High Memory usage detected: {memory_percent}%"
            }

        if alert_payload:
            try:
                requests.post(N8N_WEBHOOK_URL, json=alert_payload)
                daemon_status["last_alert"] = alert_payload
            except requests.exceptions.RequestException as e:
                print(f"Error sending webhook: {e}")

        time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    # Start the Flask app in a background thread
    flask_thread = Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()

    # Start the system monitor
    monitor_system()
