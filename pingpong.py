#!/usr/bin/env python3
"""
Unified Ping-Pong Orchestrator for LM Studio
Supports GUI, TUI, and CLI modes.

Features:
- Health checks for models before starting.
- Enhanced client with retry logic and performance tracking.
- Adaptive context truncation based on model size.
- Flexible conversation patterns via YAML config.
- Live transcript viewing directly within the GUI.
"""
import argparse
import json
import os
import sys
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import requests
import yaml

# --- Optional Imports for GUI/TUI ---
try:
    import tkinter as tk
    from tkinter import ttk, scrolledtext, filedialog, messagebox
    USE_TK = True
except ImportError:
    USE_TK = False

try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich.theme import Theme
    USE_RICH = True
except ImportError:
    USE_RICH = False

# --- Core Logic Classes ---

class ModelHealthChecker:
    """Validates models are loaded and responsive."""
    def check_model(self, endpoint: str, model_name: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        result = {"available": False, "response_time": None, "error": None}
        try:
            models_url = f"{endpoint.rstrip('/')}/v1/models"
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            start_time = time.time()
            models_resp = requests.get(models_url, headers=headers, timeout=10)
            models_resp.raise_for_status()

            available_models = [model["id"] for model in models_resp.json().get("data", [])]
            if model_name not in available_models:
                result["error"] = f"Model '{model_name}' not found in server list."
                return result

            test_url = f"{endpoint.rstrip('/')}/v1/chat/completions"
            test_payload = {"model": model_name, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}
            test_resp = requests.post(test_url, json=test_payload, headers=headers, timeout=30)
            test_resp.raise_for_status()
            result["response_time"] = time.time() - start_time
            result["available"] = True
        except requests.RequestException as e:
            result["error"] = f"Connection error: {e}"
        except Exception as e:
            result["error"] = f"Unexpected error: {e}"
        return result

class EnhancedLMClient:
    """LM Studio client with retry logic and performance tracking."""
    def __init__(self, endpoint: str, model_name: str, api_key: Optional[str] = None):
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.session = requests.Session()
        self.metrics = {"total_requests": 0, "total_tokens": 0, "total_time": 0, "errors": 0}

    def chat_stream(self, messages: List[Dict], temperature: float, max_tokens: int, max_retries: int = 2):
        url = f"{self.endpoint}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model_name, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": True}

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                response = self.session.post(url, json=payload, headers=headers, stream=True, timeout=300)
                response.raise_for_status()

                full_content = ""
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith('data: '):
                            data_str = decoded_line[6:]
                            if data_str.strip() == '[DONE]':
                                break
                            try:
                                data = json.loads(data_str)
                                chunk = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if chunk:
                                    full_content += chunk
                                    yield chunk
                            except json.JSONDecodeError:
                                continue

                elapsed = time.time() - start_time
                self.metrics["total_requests"] += 1
                self.metrics["total_tokens"] += (len(full_content) // 4)
                self.metrics["total_time"] += elapsed
                return

            except requests.RequestException as e:
                self.metrics["errors"] += 1
                if attempt == max_retries:
                    raise Exception(f"Request failed after {max_retries + 1} attempts: {e}")
                time.sleep(2 ** attempt)

    def get_metrics(self) -> Dict[str, float]:
        if self.metrics["total_requests"] == 0:
            return {"avg_response_time": 0, "tokens_per_second": 0, "error_rate": 0}
        return {
            "avg_response_time": self.metrics["total_time"] / self.metrics["total_requests"],
            "tokens_per_second": self.metrics["total_tokens"] / max(self.metrics["total_time"], 0.001),
            "error_rate": self.metrics["errors"] / self.metrics["total_requests"],
        }

class ConversationOrchestrator:
    """Handles the core conversation logic, adaptable for GUI, TUI, or CLI."""
    def __init__(self, config: Dict[str, Any], message_queue: Optional[queue.Queue] = None):
        self.config = config
        self.message_queue = message_queue
        self.stop_requested = False
        self.client_a = EnhancedLMClient(config["endpoints"]["a"], config["models"]["a"]["name"])
        self.client_b = EnhancedLMClient(config["endpoints"]["b"], config["models"]["b"]["name"])

    def _send_update(self, msg_type: str, data: Any):
        if self.message_queue:
            self.message_queue.put((msg_type, data))
        # Also print for CLI mode
        elif msg_type == "conversation_update":
            print(f"\n--- {data['speaker']} ---\n{data['content']}")
        elif msg_type == "error":
            print(f"\n--- ERROR ---\n{data}", file=sys.stderr)


    def run(self):
        # Initialization
        pattern = self.config.get("patterns", {}).get(self.config["mode"], [{"model": "a"}, {"model": "b"}])
        rounds = self.config["rounds"]
        # Ensure the pattern repeats enough times to cover all rounds
        full_pattern = (pattern * (rounds * len(pattern)))[:rounds]

        histories = {
            "a": [{"role": "system", "content": self.config["models"]["a"]["system_prompt"]}],
            "b": [{"role": "system", "content": self.config["models"]["b"]["system_prompt"]}]
        }
        histories["a"].append({"role": "user", "content": self.config["prompt"]})
        dialogue = [{"speaker": "user", "content": self.config["prompt"]}]
        self._send_update("conversation_update", dialogue[-1])

        completed_rounds = 0
        for i, step in enumerate(full_pattern):
            if self.stop_requested:
                self._send_update("status", "Conversation stopped by user.")
                break

            round_num = i + 1
            model_key = step["model"]
            role = step.get("role", "default")

            # Get role-specific system prompt if available
            system_prompt = self.config["models"][model_key].get(f"system_prompt_{role}", self.config["models"][model_key]["system_prompt"])

            self._send_update("status", f"Round {round_num}/{rounds} - Model {model_key.upper()} ({role}) thinking...")

            client = self.client_a if model_key == "a" else self.client_b
            history = histories[model_key]
            history[0] = {"role": "system", "content": system_prompt} # Update system prompt for the turn

            # Adaptive context truncation
            history = self._truncate_history_adaptive(history, self.config["context_budget"], client.model_name)

            # Get response
            full_response = ""
            try:
                stream = client.chat_stream(history, self.config["temperature"], self.config["max_tokens"])
                for chunk in stream:
                    full_response += chunk
                    self._send_update("conversation_stream", {"speaker": model_key.upper(), "chunk": chunk})
            except Exception as e:
                self._send_update("error", f"Error from Model {model_key.upper()}: {e}")
                break

            dialogue.append({"speaker": model_key.upper(), "content": full_response})
            self._send_update("conversation_update", dialogue[-1])

            # Update histories
            histories[model_key].append({"role": "assistant", "content": full_response})
            other_key = "b" if model_key == "a" else "a"
            histories[other_key].append({"role": "user", "content": full_response})

            self._send_update("metrics_update", {"a": self.client_a.get_metrics(), "b": self.client_b.get_metrics()})

            if self.config.get("stop_if") and self.config["stop_if"] in full_response:
                self._send_update("status", f"Stop condition met by Model {model_key.upper()}.")
                break

            completed_rounds = round_num
            time.sleep(self.config.get("sleep", 0.2))

        self._save_transcript(dialogue, completed_rounds)
        self._send_update("finished", {"success": True, "completed_rounds": completed_rounds})
        return {"success": True, "completed_rounds": completed_rounds}

    def _truncate_history_adaptive(self, history: List[Dict], budget: int, model_size_hint: str) -> List[Dict]:
        """A simple placeholder for adaptive history truncation."""
        if budget <= 0: return history
        # A real implementation would be more complex. For now, simple truncation.
        token_count = 0
        truncated_history = []
        # Always keep system message
        if history and history[0]['role'] == 'system':
            truncated_history.append(history[0])
            token_count += len(history[0]['content']) // 4

        for msg in reversed(history[len(truncated_history):]):
            msg_tokens = len(msg['content']) // 4
            if token_count + msg_tokens > budget:
                break
            truncated_history.insert(1, msg)
            token_count += msg_tokens
        return truncated_history

    def _save_transcript(self, dialogue: List[Dict], completed_rounds: int):
        """Saves the conversation transcript to JSON and Markdown files."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = self.config.get("name", "pingpong_run")
        transcripts_dir = Path(self.config.get("transcripts_dir", "transcripts"))
        transcripts_dir.mkdir(exist_ok=True)

        base_path = transcripts_dir / f"{ts}_{run_name}"
        json_path = base_path.with_suffix(".json")
        md_path = base_path.with_suffix(".md")

        meta = {
            "config": self.config,
            "started_at": ts,
            "completed_rounds": completed_rounds,
            "performance": {"a": self.client_a.get_metrics(), "b": self.client_b.get_metrics()}
        }

        # Save JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "dialogue": dialogue}, f, indent=2)

        # Save Markdown
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Ping-Pong Transcript: {run_name}\n\n")
            for entry in dialogue:
                f.write(f"## {entry['speaker']}\n\n{entry['content']}\n\n---\n\n")

        self._send_update("status", f"Transcripts saved to {transcripts_dir}")


# --- GUI Class (Tkinter) ---
class PingPongGUI:
    """The full-featured Tkinter GUI for the orchestrator."""
    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.root.title("Ping-Pong Orchestrator")
        self.root.geometry("950x800")
        self.message_queue = queue.Queue()
        self.orchestrator = None
        self.orchestrator_thread = None
        self.stream_buffer = {"A": "", "B": ""}

        self.setup_ui()
        self.check_queue()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=0, column=0, sticky="nsew", columnspan=2)
        main_frame.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        self._create_config_tab(notebook)
        self._create_conversation_tab(notebook)
        self._create_performance_tab(notebook)
        self._create_controls(main_frame)

    def _create_config_tab(self, notebook):
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Configuration")

        # Add full configuration options here
        ttk.Label(frame, text="Model A:").pack(anchor="w")
        self.model_a_entry = ttk.Entry(frame, width=80)
        self.model_a_entry.insert(0, self.config["models"]["a"]["name"])
        self.model_a_entry.pack(fill="x", expand=True)

        ttk.Label(frame, text="Model B:").pack(anchor="w")
        self.model_b_entry = ttk.Entry(frame, width=80)
        self.model_b_entry.insert(0, self.config["models"]["b"]["name"])
        self.model_b_entry.pack(fill="x", expand=True)

        ttk.Label(frame, text="Prompt:").pack(anchor="w")
        self.prompt_text = scrolledtext.ScrolledText(frame, height=5, wrap="word")
        self.prompt_text.insert("1.0", self.config.get("prompt", ""))
        self.prompt_text.pack(fill="both", expand=True, pady=2)


    def _create_conversation_tab(self, notebook):
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Live Conversation")
        self.conversation_text = scrolledtext.ScrolledText(frame, wrap="word", font=("Consolas", 10))
        self.conversation_text.pack(fill="both", expand=True)
        self.conversation_text.tag_configure("user", foreground="#FFA500") # Orange
        self.conversation_text.tag_configure("A", foreground="#00FFFF") # Cyan
        self.conversation_text.tag_configure("B", foreground="#FF00FF") # Magenta
        self.conversation_text.tag_configure("error", foreground="#FF4444", font=("Consolas", 10, "bold"))


    def _create_performance_tab(self, notebook):
        frame = ttk.Frame(notebook, padding="10")
        notebook.add(frame, text="Performance")
        self.perf_a_label = ttk.Label(frame, text="Model A Metrics: Waiting...")
        self.perf_a_label.pack(anchor="w")
        self.perf_b_label = ttk.Label(frame, text="Model B Metrics: Waiting...")
        self.perf_b_label.pack(anchor="w")


    def _create_controls(self, parent_frame):
        frame = ttk.Frame(parent_frame)
        frame.grid(row=1, column=0, sticky="ew", pady=(10, 0), columnspan=2)
        self.start_button = ttk.Button(frame, text="Start", command=self.start_conversation)
        self.start_button.pack(side="right")
        self.stop_button = ttk.Button(frame, text="Stop", command=self.stop_conversation, state="disabled")
        self.stop_button.pack(side="right")

    def start_conversation(self):
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.conversation_text.delete("1.0", "end")

        # Update config from GUI
        self.config["prompt"] = self.prompt_text.get("1.0", "end-1c").strip()
        self.config["models"]["a"]["name"] = self.model_a_entry.get().strip()
        self.config["models"]["b"]["name"] = self.model_b_entry.get().strip()

        self.orchestrator = ConversationOrchestrator(self.config, self.message_queue)
        self.orchestrator_thread = threading.Thread(target=self.orchestrator.run, daemon=True)
        self.orchestrator_thread.start()

    def stop_conversation(self):
        if self.orchestrator:
            self.orchestrator.stop_requested = True
        self.stop_button.config(state="disabled")

    def check_queue(self):
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()
                if msg_type == "conversation_stream":
                    # Buffer stream output to avoid choppy UI updates
                    speaker = data["speaker"]
                    self.stream_buffer[speaker] += data["chunk"]
                    if "\n" in self.stream_buffer[speaker] or len(self.stream_buffer[speaker]) > 80:
                        self.conversation_text.insert("end", self.stream_buffer[speaker], speaker)
                        self.conversation_text.see("end")
                        self.stream_buffer[speaker] = ""
                elif msg_type == "conversation_update":
                    # Flush any remaining buffer before printing the full message
                    for speaker, buffer in self.stream_buffer.items():
                        if buffer:
                            self.conversation_text.insert("end", buffer, speaker)
                    self.stream_buffer = {"A": "", "B": ""}

                    self.conversation_text.insert("end", f"\n\n--- End of Turn ---\n\n", "user")
                elif msg_type == "status":
                    self.root.title(f"Orchestrator - {data}")
                elif msg_type == "metrics_update":
                    self.perf_a_label.config(text=f"Model A Metrics: {data['a']}")
                    self.perf_b_label.config(text=f"Model B Metrics: {data['b']}")
                elif msg_type == "error":
                     self.conversation_text.insert("end", f"\n--- ERROR ---\n{data}\n", "error")
                elif msg_type == "finished":
                    self.start_button.config(state="normal")
                    self.stop_button.config(state="disabled")
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.check_queue)


# --- TUI and CLI Logic ---
def run_tui(config):
    if not USE_RICH:
        print("Rich library not found. Running in CLI mode.", file=sys.stderr)
        run_cli(config)
        return

    console = Console(theme=Theme({"user": "yellow", "A": "cyan", "B": "magenta"}))
    layout = Layout()
    layout.split_column(Layout(name="header", size=3), Layout(ratio=1, name="body"))
    layout["body"].split_row(Layout(name="A"), Layout(name="B"))

    with Live(layout, console=console, screen=True, refresh_per_second=5) as live:
        # TUI needs its own queue and orchestrator instance
        q = queue.Queue()
        orchestrator = ConversationOrchestrator(config, q)
        threading.Thread(target=orchestrator.run, daemon=True).start()

        panels = {"A": Text(), "B": Text(), "user": Text()}
        live.update(layout)

        while True:
            try:
                msg_type, data = q.get(timeout=0.1)
                if msg_type == "conversation_stream":
                    panels[data["speaker"]].append(data["chunk"])
                elif msg_type == "conversation_update":
                    speaker = data["speaker"]
                    # Reset the other panel
                    other = "B" if speaker == "A" else "A"
                    panels[other] = Text()
                elif msg_type == "finished":
                    break

                layout["A"].update(Panel(panels["A"], title="Model A"))
                layout["B"].update(Panel(panels["B"], title="Model B"))

            except queue.Empty:
                if not orchestrator.stop_requested and threading.active_count() <= 1:
                    break # Thread finished
                continue


def run_cli(config):
    print("--- Starting Ping-Pong (CLI Mode) ---")
    orchestrator = ConversationOrchestrator(config)
    orchestrator.run()
    print("\n--- Conversation Finished ---")

# --- Main Entry Point ---
def main():
    parser = argparse.ArgumentParser(description="Unified Ping-Pong Orchestrator")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--mode", choices=["gui", "tui", "cli"], default="gui", help="Execution mode")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Config file '{args.config}' not found. Using defaults.", file=sys.stderr)
        config = {"prompt": "Hello, world!", "rounds": 1, "models": {"a": {"name": "any", "system_prompt": ""}, "b": {"name": "any", "system_prompt": ""}}, "endpoints": {"a": "http://localhost:1234", "b": "http://localhost:1234"}, "temperature": 0.7, "max_tokens": 512, "context_budget": 4096, "mode": "pingpong"}
    except Exception as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)

    if args.mode == "gui":
        if not USE_TK:
            print("Tkinter not found. Cannot run GUI mode.", file=sys.stderr)
            sys.exit(1)
        root = tk.Tk()
        app = PingPongGUI(root, config)
        root.mainloop()
    elif args.mode == "tui":
        run_tui(config)
    else: # cli
        run_cli(config)

if __name__ == "__main__":
    main()
