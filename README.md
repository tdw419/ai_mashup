# AI Ping-Pong Orchestrator GUI

This project provides a self-contained graphical application to orchestrate conversations between two AI models running in LM Studio. It is designed to be a user-friendly, all-in-one tool for local multi-agent experimentation.

## Features

- **Self-Contained GUI:** A single script, `pingpong.py`, launches a comprehensive Tkinter GUI. No command-line flags are needed.
- **Advanced Orchestration:**
  - **Model Health Checks:** Validates that models are responsive before starting a conversation.
  - **Resilient Client:** Includes exponential backoff and retry logic for API calls.
  - **Live In-App Transcript:** Watch the conversation unfold in real-time directly within the GUI.
  - **Performance Monitoring:** A dedicated tab shows live metrics like tokens/second and response times for each model.
- **Flexible Configuration:**
  - **"Available Models" Browser:** Automatically fetches and displays models from your running LM Studio server, with hardware-aware recommendations.
  - **Save/Load Configurations:** Save and load your experimental setups using YAML or JSON files.
  - **Role-Based Interaction:** Easily define system prompts and conversation patterns to assign specific roles to each model.
- **Detailed Logging:** Automatically saves a complete log of each conversation in both JSON (with metadata) and Markdown formats.

## Setup

### 1. Prerequisites
- Python 3.7+
- LM Studio (with the API server enabled)

### 2. Install Dependencies
This project requires a few Python packages. Install them using pip:
```bash
pip install requests pyyaml
```
*Note: `tkinter` is usually included with Python, but if not, you may need to install it separately (`sudo apt-get install python3-tk` on Debian/Ubuntu).*

### 3. Download Models in LM Studio
1. Open LM Studio and download the models you wish to use (e.g., `qwen2.5-coder-1.5b` and `tinyllama-1.1b-chat-v1.0`).
2. Go to the "Local Server" tab and start the server (e.g., at `http://localhost:1234`).

## How to Use

Simply run the `pingpong.py` script to launch the application.

```bash
python pingpong.py
```

### Using the Application
1.  **Models Tab:** On startup, the application will try to connect to LM Studio. Use the "Available Models" tab to see what's running. You can select models from the list and assign them to "Model A" or "Model B".
2.  **Configuration Tab:** Fine-tune your experiment. Set the initial prompt, the number of rounds, system prompts for each model, and other generation parameters.
3.  **Start Conversation:** Once configured, click the "Start Conversation" button.
4.  **Monitor:** Watch the dialogue unfold in the "Live Conversation" tab and see performance data in the "Performance Metrics" tab.

## Files

- `pingpong.py`: The single, self-contained GUI application script.
- `config.yaml`: An example configuration file that can be loaded into the GUI.
- `transcripts/`: The directory where all conversation logs are saved.
