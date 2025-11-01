# VISTA V-Loop

This project is a sophisticated, recursive AI system that avoids "analysis paralysis" by being "builder-first" (solution-oriented).

## AI Daemon

This project includes an AI Daemon that monitors system resources and triggers an n8n workflow to analyze and respond to alerts.

### Running the AI Daemon and n8n Workflow

To run the AI Daemon and the n8n workflow, follow these steps:

1.  **Start your n8n instance.** Make sure your n8n instance is running and accessible.
2.  **Import the AI Daemon Handler workflow.** In your n8n instance, import the `n8n_workflows/ai_daemon_handler.json` workflow.
3.  **Activate the workflow.** Once imported, activate the workflow in the n8n UI.
4.  **Run the AI Daemon.** In your terminal, run the following command from the root of the project:

    ```bash
    python ai_daemon.py
    ```

The daemon will now monitor your system's CPU and memory usage. If any of the thresholds are breached, it will send a webhook to your n8n workflow, which will then analyze the alert and send a status update back to the daemon. You can view the status of the daemon by visiting `http://localhost:5002/status` in your browser.
