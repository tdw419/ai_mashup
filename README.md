# AI Ping-Pong Orchestrator

This project provides a tool to orchestrate conversations between two AI models running in LM Studio. It is a unified application that can be run in three different modes: a full graphical interface (GUI), a terminal-based interface (TUI), or a standard command-line interface (CLI).

## Features

- **Unified Application:** A single script, `pingpong.py`, provides all functionality.
- **Multiple Modes:**
  - **GUI Mode (`--mode gui`):** A comprehensive Tkinter GUI for easy configuration, live conversation viewing, and performance monitoring.
  - **TUI Mode (`--mode tui`):** A real-time, color-coded terminal interface using `rich` to watch the conversation unfold.
  - **CLI Mode (`--mode cli`):** A headless mode for scripting and batch processing.
- **Advanced Orchestration:**
  - **Health Checks:** Validates that models are responsive before starting a conversation.
  - **Resilient Client:** Includes exponential backoff and retry logic for API calls.
  - **Performance Metrics:** Tracks tokens/second and response times.
- **Flexible Configuration:** Supports configuration via a `config.yaml` file for reproducible experiments.
- **Role-Based Interaction:** Easily define system prompts and conversation patterns to assign specific roles to each model.
- **Transcript Logging:** Automatically saves a complete log of each conversation in both JSON and Markdown formats.

## Setup

### 1. Prerequisites
- Python 3.7+
- LM Studio (with the API server enabled)

### 2. Install Dependencies
This project requires a few Python packages. Install them using pip:
```bash
pip install requests pyyaml rich
```
*Note: `tkinter` is usually included with Python, but if not, you may need to install it separately (`sudo apt-get install python3-tk` on Debian/Ubuntu).*

### 3. Download Models in LM Studio
1. Open LM Studio and download the models you wish to use (e.g., `Phi-3-mini-4k-instruct-Q4_K_M.gguf`).
2. Go to the "Local Server" tab and start the server (e.g., at `http://localhost:1234`).

## How to Use

The application is run through the `pingpong.py` script, using the `--mode` flag to select the interface.

### GUI Mode (Recommended for Interactive Use)

This is the easiest way to get started. It provides a full interface for configuration and live viewing.

```bash
python pingpong.py --mode gui
```

### TUI Mode

This mode provides a rich, real-time view of the conversation directly in your terminal.

```bash
python pingpong.py --mode tui --config my_config.yaml
```

### CLI Mode

This mode is best for automated runs. It prints the conversation to the console without a special interface.

```bash
python pingpong.py --mode cli --config my_config.yaml
```

## Configuration

The conversation is controlled by a `config.yaml` file. You can create your own or modify the provided one. The script will look for `config.yaml` by default, but you can specify a different file with the `--config` argument.

See `config.yaml` for a detailed example of all available options.

## Files

- `pingpong.py`: The single, unified application script.
- `config.yaml`: An example configuration file.
- `transcripts/`: The directory where all conversation logs are saved.
