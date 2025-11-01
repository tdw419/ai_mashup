import sys
import os
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from threading import Thread
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.workflow import BuilderFirstWorkflow
from engine.stores.lancedb_store import LanceHistoryStore
import yaml

app = Flask(__name__)
socketio = SocketIO(app)

# In-memory store for task status
tasks = {}

def run_vista_vloop(task_id, goal):
    """Function to run the VISTA V-Loop and emit updates."""
    try:
        def log(message):
            socketio.emit('log', {'task_id': task_id, 'log': message})
            tasks[task_id]['logs'] += message + '\n'

        tasks[task_id] = {"status": "running", "logs": ""}
        log("Starting VISTA V-Loop...")

        # Load config
        with open("../config/builder_mode.yaml") as f:
            cfg = yaml.safe_load(f)

        # Initialize components
        db = LanceHistoryStore(uri="../.lancedb")

        from engine.llm import litellm_completion
        llm = litellm_completion

        workflow = BuilderFirstWorkflow(llm, db, cfg)

        log("Workflow initialized. Running task...")

        final_result, report = workflow.run(goal)

        log("V-Loop completed.")
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["report"] = report
        socketio.emit('task_completed', {'task_id': task_id, 'report': report})

    except Exception as e:
        log(f"An error occurred: {e}")
        tasks[task_id]["status"] = "failed"
        socketio.emit('task_failed', {'task_id': task_id, 'error': str(e)})


@app.route('/vloop/run', methods=['POST'])
def run_task():
    goal = request.json.get('goal')
    if not goal:
        return jsonify({"error": "Goal not provided"}), 400

    task_id = f"task_{len(tasks) + 1}"

    socketio.start_background_task(run_vista_vloop, task_id, goal)

    return jsonify({"task_id": task_id})

@app.route('/vloop/status/<task_id>')
def task_status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    return jsonify(task)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001, allow_unsafe_werkzeug=True)
