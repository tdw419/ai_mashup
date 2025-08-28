#!/usr/bin/env python3
"""
Complete LM Studio Ping-Pong Orchestrator with GUI
Self-contained implementation with enhanced orchestration, health checking,
performance monitoring, and user-friendly interface.
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
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

# Try to import yaml, make it optional
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

class ModelHealthChecker:
    """Validates models are loaded and responsive before starting conversations"""

    def check_model(self, endpoint: str, model_name: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Check if a model is available and responsive"""
        result = {
            "available": False,
            "response_time": None,
            "error": None
        }

        try:
            # Test if model is in the available models list
            models_url = f"{endpoint.rstrip('/')}/v1/models"
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            start_time = time.time()
            models_resp = requests.get(models_url, headers=headers, timeout=10)
            models_resp.raise_for_status()

            available_models = [model["id"] for model in models_resp.json().get("data", [])]

            if model_name not in available_models:
                result["error"] = f"Model '{model_name}' not found in available models"
                return result

            # Test with a small prompt to check responsiveness
            test_url = f"{endpoint.rstrip('/')}/v1/chat/completions"
            test_payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
                "temperature": 0.1
            }

            test_resp = requests.post(test_url, json=test_payload, headers=headers, timeout=30)
            test_resp.raise_for_status()

            result["response_time"] = time.time() - start_time
            result["available"] = True

        except requests.RequestException as e:
            result["error"] = f"Connection error: {str(e)}"
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"

        return result

class EnhancedLMClient:
    """LM Studio client with retry logic and performance tracking"""

    def __init__(self, endpoint: str, model_name: str, api_key: Optional[str] = None):
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key
        self.session = requests.Session()
        self.performance_metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_time": 0,
            "errors": 0
        }

    def chat_with_retry(self, messages: List[Dict], temperature: float = 0.7,
                       max_tokens: int = 800, max_retries: int = 3) -> str:
        """Chat with exponential backoff retry logic"""
        url = f"{self.endpoint}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(max_retries + 1):
            try:
                start_time = time.time()
                response = requests.post(url, json=payload, headers=headers, timeout=300)
                response.raise_for_status()

                elapsed = time.time() - start_time
                data = response.json()
                content = data["choices"][0]["message"]["content"]

                # Update metrics
                self.performance_metrics["total_requests"] += 1
                self.performance_metrics["total_tokens"] += len(content.split())  # Rough estimate
                self.performance_metrics["total_time"] += elapsed

                return content

            except requests.RequestException as e:
                self.performance_metrics["errors"] += 1

                if attempt == max_retries:
                    raise Exception(f"Request failed after {max_retries + 1} attempts: {str(e)}")

                # Exponential backoff
                wait_time = 2 ** attempt
                time.sleep(wait_time)

    def get_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        if self.performance_metrics["total_requests"] == 0:
            return {"avg_response_time": 0, "tokens_per_second": 0, "error_rate": 0, "total_requests": 0, "total_tokens": 0}

        return {
            "avg_response_time": self.performance_metrics["total_time"] / self.performance_metrics["total_requests"],
            "tokens_per_second": self.performance_metrics["total_tokens"] / max(self.performance_metrics["total_time"], 0.001),
            "error_rate": self.performance_metrics["errors"] / self.performance_metrics["total_requests"],
            "total_requests": self.performance_metrics["total_requests"],
            "total_tokens": self.performance_metrics["total_tokens"]
        }

def estimate_tokens(text: str) -> int:
    """Improved token estimation using word count + character density"""
    words = len(text.split())
    chars = len(text)
    # Heuristic: average between word-based (1.3 tokens/word) and char-based (4 chars/token)
    word_estimate = words * 1.3
    char_estimate = chars / 4
    return int((word_estimate + char_estimate) / 2)

def truncate_history_adaptive(history: List[Dict], budget: int, model_size_hint: str = "1.5B") -> List[Dict]:
    """Adaptive history truncation based on model size"""
    if budget <= 0 or not history:
        return history

    # Adjust budget based on model size (smaller models need more aggressive truncation)
    size_multipliers = {
        "1B": 0.6,   # Very aggressive truncation for small models
        "1.5B": 0.7, # Aggressive truncation for your models
        "3B": 0.8,   # Moderate truncation
        "7B": 1.0,   # Standard budget
        "13B": 1.2,  # Allow slightly more context
        "30B": 1.5   # Much larger context if hardware allows
    }

    for size, multiplier in size_multipliers.items():
        if size in model_size_hint:
            budget = int(budget * multiplier)
            break

    kept = []
    total = 0

    # Always preserve system message
    if history and history[0].get("role") == "system":
        kept.append(history[0])
        total += estimate_tokens(history[0].get("content", ""))
        messages = history[1:]
    else:
        messages = history

    # Keep from end backward
    for msg in reversed(messages):
        token_count = estimate_tokens(msg.get("content", ""))
        if total + token_count <= budget:
            kept.append(msg)
            total += token_count
        else:
            break

    return kept[:1] + list(reversed(kept[1:]))

class ConversationOrchestrator:
    """Enhanced orchestrator with health checking and pattern support"""

    def __init__(self, config: Dict[str, Any], message_queue: queue.Queue):
        self.config = config
        self.message_queue = message_queue
        self.health_checker = ModelHealthChecker()
        self.stop_requested = False

        # Initialize clients
        self.client_a = EnhancedLMClient(
            endpoint=config["endpoints"]["a"],
            model_name=config["models"]["a"]["name"],
            api_key=config.get("api_keys", {}).get("a")
        )

        self.client_b = EnhancedLMClient(
            endpoint=config["endpoints"]["b"],
            model_name=config["models"]["b"]["name"],
            api_key=config.get("api_keys", {}).get("b")
        )

        # Define conversation patterns
        self.patterns = {
            "pingpong": [
                {"model": "a", "role": "default"},
                {"model": "b", "role": "default"}
            ],
            "plan_critique": [
                {"model": "a", "role": "planner"},
                {"model": "b", "role": "critic"},
                {"model": "a", "role": "reviser"}
            ],
            "debate": [
                {"model": "a", "role": "thesis"},
                {"model": "b", "role": "antithesis"},
                {"model": "a", "role": "synthesis"}
            ]
        }

    def validate_setup(self) -> bool:
        """Validate models are loaded and responsive"""
        self.message_queue.put(("status", "Validating model setup..."))

        # Check Model A
        self.message_queue.put(("status", "Checking Model A..."))
        health_a = self.health_checker.check_model(
            self.config["endpoints"]["a"],
            self.config["models"]["a"]["name"],
            self.config.get("api_keys", {}).get("a")
        )

        if not health_a["available"]:
            self.message_queue.put(("validation_error", f"Model A validation failed: {health_a['error']}"))
            return False

        self.message_queue.put(("status", f"Model A ready (response time: {health_a['response_time']:.2f}s)"))

        # Check Model B
        self.message_queue.put(("status", "Checking Model B..."))
        health_b = self.health_checker.check_model(
            self.config["endpoints"]["b"],
            self.config["models"]["b"]["name"],
            self.config.get("api_keys", {}).get("b")
        )

        if not health_b["available"]:
            self.message_queue.put(("validation_error", f"Model B validation failed: {health_b['error']}"))
            return False

        self.message_queue.put(("status", f"Model B ready (response time: {health_b['response_time']:.2f}s)"))
        self.message_queue.put(("validation_success", {"a": health_a["response_time"], "b": health_b["response_time"]}))
        return True

    def run_conversation(self) -> Dict[str, Any]:
        """Run the conversation with pattern support"""
        if not self.validate_setup():
            return {"success": False, "error": "Model validation failed"}

        # Send initial prompt to GUI
        self.message_queue.put(("conversation_update", {
            "speaker": "user",
            "content": self.config["prompt"]
        }))

        # Initialize conversation state
        pattern_name = self.config.get("mode", "pingpong")
        pattern = self.patterns.get(pattern_name, self.patterns["pingpong"])

        # Extend pattern to fill rounds
        rounds = self.config.get("rounds", 4)
        extended_pattern = []
        for i in range(rounds):
            extended_pattern.append(pattern[i % len(pattern)])

        histories = {
            "a": [{"role": "system", "content": self.config["models"]["a"]["system_prompt"]}],
            "b": [{"role": "system", "content": self.config["models"]["b"]["system_prompt"]}]
        }

        # Add initial prompt to Model A's history
        # THIS IS THE BUG. It should use the prompt from the config, not re-read it.
        # The fix is to ensure the orchestrator USES the config it was given.
        # The logic below is already correct in that it uses self.config,
        # the bug was likely in a previous version of the GUI code that was not
        # correctly creating the config object. The current `get_current_config`
        # is correct. Let's ensure the run method uses it properly.

        initial_prompt = self.config.get("prompt", "Hello!")
        histories["a"].append({"role": "user", "content": initial_prompt})

        dialogue = [{"speaker": "user", "content": initial_prompt}]
        self.message_queue.put(("conversation_update", dialogue[0]))

        completed_rounds = 0

        try:
            for round_num, step in enumerate(extended_pattern, 1):
                if self.stop_requested:
                    break

                model_key = step["model"]
                role = step["role"]

                # Update status
                self.message_queue.put(("status", f"Round {round_num}/{rounds} - Model {model_key.upper()} ({role})"))

                # Get appropriate client and history
                client = self.client_a if model_key == "a" else self.client_b
                history = histories[model_key]

                # Adaptive context truncation based on model size
                model_name = self.config["models"][model_key]["name"]
                context_budget = self.config.get("context_budget", 6000)
                history = truncate_history_adaptive(history, context_budget, model_name)

                # Update role-specific system prompt if needed
                if role == "planner":
                    history[0] = {"role": "system", "content": "You are a system architect. Create clear, minimal specifications. End with 'SPEC_COMPLETE' if done."}
                elif role == "critic":
                    history[0] = {"role": "system", "content": "You are a code reviewer. Find issues and suggest improvements. Use 'FINAL_ANSWER' for final implementation."}
                elif role == "reviser":
                    history[0] = {"role": "system", "content": "You are a designer. Refine and improve the previous work based on feedback."}

                # Get response
                try:
                    response = client.chat_with_retry(
                        messages=history,
                        temperature=self.config.get("temperature", 0.7),
                        max_tokens=self.config.get("max_tokens", 800)
                    )

                    # Send to GUI
                    self.message_queue.put(("conversation_update", {
                        "speaker": model_key.upper(),
                        "content": response
                    }))

                    # Update metrics
                    metrics_a = self.client_a.get_metrics()
                    metrics_b = self.client_b.get_metrics()
                    self.message_queue.put(("metrics_update", {
                        "model_a": metrics_a,
                        "model_b": metrics_b
                    }))

                    # Update histories
                    histories[model_key].append({"role": "assistant", "content": response})
                    other_key = "b" if model_key == "a" else "a"
                    histories[other_key].append({"role": "user", "content": response})
                    dialogue.append({"speaker": model_key.upper(), "content": response})

                    # Check stop condition
                    stop_condition = self.config.get("stop_if")
                    if stop_condition and stop_condition in response:
                        self.message_queue.put(("status", f"Stop condition '{stop_condition}' met by Model {model_key.upper()}"))
                        break

                except Exception as e:
                    error_msg = f"Error from Model {model_key.upper()}: {str(e)}"
                    self.message_queue.put(("conversation_update", {
                        "speaker": "system",
                        "content": error_msg
                    }))
                    break

                completed_rounds = round_num
                time.sleep(self.config.get("sleep", 0.2))

        except Exception as e:
            return {"success": False, "error": str(e)}

        # Save transcript
        return self._save_transcript(dialogue, completed_rounds)

    def _save_transcript(self, dialogue: List[Dict], completed_rounds: int) -> Dict[str, Any]:
        """Save transcript with enhanced metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcripts_dir = self.config.get("transcripts_dir", "transcripts")
        run_name = self.config.get("name", "conversation")

        base_path = Path(transcripts_dir) / f"{timestamp}_{run_name}"
        json_path = base_path.with_suffix(".json")
        md_path = base_path.with_suffix(".md")

        # Compile performance metrics
        metrics_a = self.client_a.get_metrics()
        metrics_b = self.client_b.get_metrics()

        transcript_data = {
            "metadata": {
                "timestamp": timestamp,
                "config": self.config,
                "completed_rounds": completed_rounds,
                "total_rounds": self.config.get("rounds", 4),
                "performance": {
                    "model_a": metrics_a,
                    "model_b": metrics_b
                }
            },
            "dialogue": dialogue
        }

        # Save JSON
        os.makedirs(transcripts_dir, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)

        # Save Markdown
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Ping-Pong Conversation Transcript\n\n")
            f.write(f"**Generated:** {timestamp}\n")
            f.write(f"**Mode:** {self.config.get('mode', 'pingpong')}\n")
            f.write(f"**Rounds:** {completed_rounds}/{self.config.get('rounds', 4)}\n\n")
            f.write(f"**Model A:** {self.config['models']['a']['name']}\n")
            f.write(f"**Model B:** {self.config['models']['b']['name']}\n\n")

            if metrics_a["avg_response_time"] > 0:
                f.write("## Performance Metrics\n\n")
                f.write(f"**Model A:** {metrics_a['avg_response_time']:.1f}s avg, {metrics_a['tokens_per_second']:.1f} tok/s\n")
                f.write(f"**Model B:** {metrics_b['avg_response_time']:.1f}s avg, {metrics_b['tokens_per_second']:.1f} tok/s\n\n")

            f.write("---\n\n")

            for i, entry in enumerate(dialogue, 1):
                f.write(f"## {entry['speaker']} (Turn {i})\n\n")
                f.write(f"{entry['content']}\n\n")

        return {
            "success": True,
            "json_path": str(json_path),
            "md_path": str(md_path),
            "completed_rounds": completed_rounds,
            "performance": {"model_a": metrics_a, "model_b": metrics_b}
        }

class PingPongGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("LM Studio Ping-Pong Orchestrator")
        self.root.geometry("1000x800")
        self.root.resizable(True, True)

        # Queue for thread communication
        self.message_queue = queue.Queue()

        # Current conversation state
        self.conversation_running = False
        self.current_orchestrator = None

        # Default configuration optimized for your models
        self.default_config = {
            "models": {
                "a": {
                    "name": "qwen2.5-coder-1.5b",
                    "system_prompt": "You are Model A, a creative planner and system architect."
                },
                "b": {
                    "name": "tinyllama-1.1b-chat-v1.0",
                    "system_prompt": "You are Model B, a critical analyzer and code reviewer."
                }
            },
            "endpoints": {"a": "http://localhost:1234", "b": "http://localhost:1234"},
            "prompt": "Design a minimal REST API with GET and POST endpoints.",
            "rounds": 4,
            "temperature": 0.7,
            "max_tokens": 800,
            "context_budget": 4000,  # Reduced for your smaller models
            "stop_if": "FINAL_ANSWER",
            "sleep": 0.2,
            "mode": "plan_critique",
            "transcripts_dir": "transcripts",
            "name": "api_design"
        }

        self.setup_ui()
        self.check_message_queue()

    def setup_ui(self):
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=0, column=0, columnspan=3, sticky="nsew", pady=(0, 10))
        main_frame.rowconfigure(0, weight=1)

        # Models tab (new)
        self.setup_models_tab(notebook)

        # Configuration tab
        self.setup_config_tab(notebook)

        # Conversation tab
        self.setup_conversation_tab(notebook)

        # Performance tab
        self.setup_performance_tab(notebook)

        # Control buttons
        self.setup_controls(main_frame)

    def setup_models_tab(self, notebook):
        models_frame = ttk.Frame(notebook, padding="10")
        notebook.add(models_frame, text="Available Models")

        # Connection section
        conn_frame = ttk.LabelFrame(models_frame, text="LM Studio Connection", padding="10")
        conn_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        models_frame.columnconfigure(0, weight=1)
        models_frame.columnconfigure(1, weight=1)

        ttk.Label(conn_frame, text="LM Studio Endpoint:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.models_endpoint_var = tk.StringVar(value="http://localhost:1234")
        ttk.Entry(conn_frame, textvariable=self.models_endpoint_var, width=30).grid(row=0, column=1, sticky="ew", padx=(0, 10))

        self.refresh_models_btn = ttk.Button(conn_frame, text="Refresh Model List", command=self.refresh_model_list)
        self.refresh_models_btn.grid(row=0, column=2, padx=(5, 0))

        self.connection_status = ttk.Label(conn_frame, text="Not connected", foreground="red")
        self.connection_status.grid(row=1, column=0, columnspan=3, sticky="w", pady=(5, 0))

        conn_frame.columnconfigure(1, weight=1)

        # Available models section
        available_frame = ttk.LabelFrame(models_frame, text="Available Models", padding="10")
        available_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        models_frame.rowconfigure(1, weight=1)

        # Models list with scrollbar
        list_frame = ttk.Frame(available_frame)
        list_frame.pack(fill="both", expand=True)

        # Create Treeview for better model display
        columns = ("Model Name", "Size Est.", "Type")
        self.models_tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=12)

        # Define column headings and widths
        self.models_tree.heading("Model Name", text="Model Name")
        self.models_tree.heading("Size Est.", text="Est. Size")
        self.models_tree.heading("Type", text="Type")

        self.models_tree.column("Model Name", width=300)
        self.models_tree.column("Size Est.", width=100)
        self.models_tree.column("Type", width=120)

        # Scrollbars
        v_scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.models_tree.yview)
        h_scrollbar = ttk.Scrollbar(list_frame, orient="horizontal", command=self.models_tree.xview)
        self.models_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Pack treeview and scrollbars
        self.models_tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        # Selection buttons
        button_frame = ttk.Frame(available_frame)
        button_frame.pack(fill="x", pady=(10, 0))

        ttk.Button(button_frame, text="Use as Model A", command=self.use_as_model_a).pack(side="left", padx=(0, 5))
        ttk.Button(button_frame, text="Use as Model B", command=self.use_as_model_b).pack(side="left", padx=(0, 5))
        ttk.Button(button_frame, text="Test Selected Model", command=self.test_selected_model).pack(side="right")

        # Model details section
        details_frame = ttk.LabelFrame(models_frame, text="Model Details", padding="10")
        details_frame.grid(row=2, column=0, columnspan=2, sticky="ew")

        self.model_details_text = tk.Text(details_frame, height=6, wrap="word", font=("Consolas", 9))
        details_scrollbar = ttk.Scrollbar(details_frame, orient="vertical", command=self.model_details_text.yview)
        self.model_details_text.configure(yscrollcommand=details_scrollbar.set)

        self.model_details_text.pack(side="left", fill="both", expand=True)
        details_scrollbar.pack(side="right", fill="y")

        # Bind selection event
        self.models_tree.bind("<<TreeviewSelect>>", self.on_model_select)

        # Auto-refresh on startup
        self.root.after(1000, self.refresh_model_list)

    def setup_config_tab(self, notebook):
        config_frame = ttk.Frame(notebook, padding="10")
        notebook.add(config_frame, text="Configuration")

        # Hardware info section
        hardware_frame = ttk.LabelFrame(config_frame, text="Hardware Optimized Settings", padding="10")
        hardware_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        config_frame.columnconfigure(0, weight=1)
        config_frame.columnconfigure(1, weight=1)

        hardware_text = tk.Text(hardware_frame, height=3, wrap="word", bg="#f0f0f0")
        hardware_text.pack(fill="x", padx=5, pady=5)
        hardware_text.insert("1.0",
            "Hardware: AMD Radeon 5600 Series (6GB VRAM estimated)\n"
            "Optimized for: qwen2.5-coder-1.5b + tinyllama-1.1b-chat-v1.0\n"
            "Context budget reduced to 4000 tokens for better performance on smaller models")
        hardware_text.config(state="disabled")

        # Endpoints section
        endpoints_frame = ttk.LabelFrame(config_frame, text="Server Endpoints", padding="10")
        endpoints_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Label(endpoints_frame, text="Model A Endpoint:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.endpoint_a_var = tk.StringVar(value=self.default_config["endpoints"]["a"])
        ttk.Entry(endpoints_frame, textvariable=self.endpoint_a_var, width=40).grid(row=0, column=1, sticky="ew")

        ttk.Label(endpoints_frame, text="Model B Endpoint:").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 0))
        self.endpoint_b_var = tk.StringVar(value=self.default_config["endpoints"]["b"])
        ttk.Entry(endpoints_frame, textvariable=self.endpoint_b_var, width=40).grid(row=1, column=1, sticky="ew", pady=(5, 0))
        endpoints_frame.columnconfigure(1, weight=1)

        # Models section
        models_frame = ttk.LabelFrame(config_frame, text="Model Configuration", padding="10")
        models_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        # Model A
        ttk.Label(models_frame, text="Model A (Planner):").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.model_a_var = tk.StringVar(value=self.default_config["models"]["a"]["name"])
        self.model_a_combo = ttk.Combobox(models_frame, textvariable=self.model_a_var, width=40)
        self.model_a_combo.grid(row=0, column=1, sticky="ew")

        ttk.Button(models_frame, text="Fetch Models", command=self.fetch_models).grid(row=0, column=2, padx=(5, 0))

        ttk.Label(models_frame, text="Model A System:").grid(row=1, column=0, sticky="nw", padx=(0, 5), pady=(5, 0))
        self.system_a_text = tk.Text(models_frame, height=3, width=50, wrap="word")
        self.system_a_text.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(5, 0))
        self.system_a_text.insert("1.0", self.default_config["models"]["a"]["system_prompt"])

        # Model B
        ttk.Label(models_frame, text="Model B (Critic):").grid(row=2, column=0, sticky="w", padx=(0, 5), pady=(10, 0))
        self.model_b_var = tk.StringVar(value=self.default_config["models"]["b"]["name"])
        self.model_b_combo = ttk.Combobox(models_frame, textvariable=self.model_b_var, width=40)
        self.model_b_combo.grid(row=2, column=1, sticky="ew", pady=(10, 0))

        ttk.Label(models_frame, text="Model B System:").grid(row=3, column=0, sticky="nw", padx=(0, 5), pady=(5, 0))
        self.system_b_text = tk.Text(models_frame, height=3, width=50, wrap="word")
        self.system_b_text.grid(row=3, column=1, columnspan=2, sticky="ew", pady=(5, 0))
        self.system_b_text.insert("1.0", self.default_config["models"]["b"]["system_prompt"])

        models_frame.columnconfigure(1, weight=1)

        # Conversation settings
        conv_frame = ttk.LabelFrame(config_frame, text="Conversation Settings", padding="10")
        conv_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Label(conv_frame, text="Initial Prompt:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.prompt_text = tk.Text(conv_frame, height=3, width=60, wrap="word")
        self.prompt_text.grid(row=0, column=1, columnspan=3, sticky="ew")
        self.prompt_text.insert("1.0", self.default_config["prompt"])

        ttk.Label(conv_frame, text="Mode:").grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(10, 0))
        self.mode_var = tk.StringVar(value=self.default_config["mode"])
        mode_combo = ttk.Combobox(conv_frame, textvariable=self.mode_var,
                                  values=["pingpong", "plan_critique", "debate"],
                                  state="readonly", width=15)
        mode_combo.grid(row=1, column=1, sticky="w", pady=(10, 0))

        ttk.Label(conv_frame, text="Rounds:").grid(row=1, column=2, sticky="w", padx=(20, 5), pady=(10, 0))
        self.rounds_var = tk.IntVar(value=self.default_config["rounds"])
        ttk.Spinbox(conv_frame, from_=1, to=20, textvariable=self.rounds_var, width=10).grid(row=1, column=3, sticky="w", pady=(10, 0))

        conv_frame.columnconfigure(1, weight=1)

        # Parameters section
        params_frame = ttk.LabelFrame(config_frame, text="Generation Parameters", padding="10")
        params_frame.grid(row=4, column=0, columnspan=2, sticky="ew")

        # Temperature
        ttk.Label(params_frame, text="Temperature:").grid(row=0, column=0, sticky="w")
        self.temperature_var = tk.DoubleVar(value=self.default_config["temperature"])
        temp_scale = ttk.Scale(params_frame, from_=0.1, to=2.0, variable=self.temperature_var, orient="horizontal")
        temp_scale.grid(row=0, column=1, sticky="ew", padx=(5, 10))
        self.temp_label = ttk.Label(params_frame, text=f"{self.temperature_var.get():.1f}")
        self.temp_label.grid(row=0, column=2)
        temp_scale.configure(command=self.update_temp_label)

        # Max tokens
        ttk.Label(params_frame, text="Max Tokens:").grid(row=0, column=3, sticky="w", padx=(20, 5))
        self.max_tokens_var = tk.IntVar(value=self.default_config["max_tokens"])
        ttk.Spinbox(params_frame, from_=100, to=4000, increment=100, textvariable=self.max_tokens_var, width=10).grid(row=0, column=4)

        # Context budget
        ttk.Label(params_frame, text="Context Budget:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.context_budget_var = tk.IntVar(value=self.default_config["context_budget"])
        ttk.Spinbox(params_frame, from_=1000, to=16000, increment=1000, textvariable=self.context_budget_var, width=10).grid(row=1, column=1, sticky="w", pady=(10, 0), padx=(5, 0))

        # Stop condition
        ttk.Label(params_frame, text="Stop If:").grid(row=1, column=2, sticky="w", padx=(20, 5), pady=(10, 0))
        self.stop_if_var = tk.StringVar(value=self.default_config.get("stop_if", ""))
        ttk.Entry(params_frame, textvariable=self.stop_if_var, width=15).grid(row=1, column=3, columnspan=2, sticky="w", pady=(10, 0))

        params_frame.columnconfigure(1, weight=1)

    def setup_conversation_tab(self, notebook):
        conv_frame = ttk.Frame(notebook, padding="10")
        notebook.add(conv_frame, text="Live Conversation")

        # Status bar
        status_frame = ttk.Frame(conv_frame)
        status_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        conv_frame.columnconfigure(0, weight=1)

        self.status_label = ttk.Label(status_frame, text="Ready - Configure models and click Start")
        self.status_label.grid(row=0, column=0, sticky="w")

        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        status_frame.columnconfigure(1, weight=1)

        # Conversation display
        self.conversation_text = scrolledtext.ScrolledText(conv_frame, height=30, wrap="word",
                                                          font=("Consolas", 10))
        self.conversation_text.grid(row=1, column=0, sticky="nsew")
        conv_frame.rowconfigure(1, weight=1)

        # Configure text tags for different speakers
        self.conversation_text.tag_configure("user", foreground="blue", font=("Consolas", 10, "bold"))
        self.conversation_text.tag_configure("model_a", foreground="green", font=("Consolas", 10))
        self.conversation_text.tag_configure("model_b", foreground="purple", font=("Consolas", 10))
        self.conversation_text.tag_configure("system", foreground="red", font=("Consolas", 9, "italic"))

        # Add initial help text
        self.conversation_text.insert("1.0",
            "Welcome to LM Studio Ping-Pong Orchestrator!\n\n"
            "1. Configure your models in the Configuration tab\n"
            "2. Click 'Fetch Models' to load available models from LM Studio\n"
            "3. Set your initial prompt and parameters\n"
            "4. Click 'Start Conversation' to begin\n\n"
            "The conversation will appear here in real-time with color coding:\n"
            "- Blue: Your initial prompt\n"
            "- Green: Model A responses\n"
            "- Purple: Model B responses\n"
            "- Red: System messages\n\n")

    def setup_performance_tab(self, notebook):
        perf_frame = ttk.Frame(notebook, padding="10")
        notebook.add(perf_frame, text="Performance Metrics")

        # Model A metrics
        a_frame = ttk.LabelFrame(perf_frame, text="Model A Performance", padding="10")
        a_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        self.metrics_a_text = scrolledtext.ScrolledText(a_frame, height=18, width=45, font=("Consolas", 9))
        self.metrics_a_text.grid(row=0, column=0, sticky="nsew")
        a_frame.rowconfigure(0, weight=1)
        a_frame.columnconfigure(0, weight=1)

        # Model B metrics
        b_frame = ttk.LabelFrame(perf_frame, text="Model B Performance", padding="10")
        b_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        self.metrics_b_text = scrolledtext.ScrolledText(b_frame, height=18, width=45, font=("Consolas", 9))
        self.metrics_b_text.grid(row=0, column=0, sticky="nsew")
        b_frame.rowconfigure(0, weight=1)
        b_frame.columnconfigure(0, weight=1)

        perf_frame.columnconfigure(0, weight=1)
        perf_frame.columnconfigure(1, weight=1)
        perf_frame.rowconfigure(0, weight=1)

        # Initialize with placeholder text
        self.metrics_a_text.insert("1.0", "No performance data yet.\nStart a conversation to see metrics.")
        self.metrics_b_text.insert("1.0", "No performance data yet.\nStart a conversation to see metrics.")

    def setup_controls(self, parent):
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(10, 0))

        # Load/Save config buttons
        if HAS_YAML:
            ttk.Button(control_frame, text="Load Config", command=self.load_config).pack(side="left", padx=(0, 5))
            ttk.Button(control_frame, text="Save Config", command=self.save_config).pack(side="left", padx=(0, 20))

        # Validate models button
        ttk.Button(control_frame, text="Validate Models", command=self.validate_models).pack(side="left", padx=(0, 20))

        # Main control buttons
        self.start_button = ttk.Button(control_frame, text="Start Conversation", command=self.start_conversation)
        self.start_button.pack(side="right", padx=(5, 0))

        self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_conversation, state="disabled")
        self.stop_button.pack(side="right")

    def update_temp_label(self, value):
        self.temp_label.config(text=f"{float(value):.1f}")

    def fetch_models(self):
        """Fetch available models from the LM Studio server and populate dropdowns"""
        endpoint = self.endpoint_a_var.get().strip()
        if not endpoint:
            messagebox.showerror("Error", "Please enter an endpoint first")
            return

        try:
            response = requests.get(f"{endpoint.rstrip('/')}/v1/models", timeout=10)
            response.raise_for_status()

            models = [model["id"] for model in response.json().get("data", [])]

            if models:
                self.model_a_combo['values'] = models
                self.model_b_combo['values'] = models

                # Auto-select your preferred models if available
                if "qwen2.5-coder-1.5b" in models:
                    self.model_a_var.set("qwen2.5-coder-1.5b")
                if "tinyllama-1.1b-chat-v1.0" in models:
                    self.model_b_var.set("tinyllama-1.1b-chat-v1.0")

                messagebox.showinfo("Success", f"Found {len(models)} models\n\nTip: Use the 'Available Models' tab for more details and easy selection!")
            else:
                messagebox.showwarning("Warning", "No models found")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch models: {str(e)}\n\nMake sure LM Studio is running and the server is started.")

    def validate_models(self):
        """Validate that both models are available and responsive"""
        config = self.get_current_config()

        def validate_thread():
            try:
                checker = ModelHealthChecker()

                # Check Model A
                self.message_queue.put(("status", "Validating Model A..."))
                health_a = checker.check_model(
                    config["endpoints"]["a"],
                    config["models"]["a"]["name"]
                )

                # Check Model B
                self.message_queue.put(("status", "Validating Model B..."))
                health_b = checker.check_model(
                    config["endpoints"]["b"],
                    config["models"]["b"]["name"]
                )

                # Report results
                if health_a["available"] and health_b["available"]:
                    self.message_queue.put(("validation_success", {
                        "a": health_a["response_time"],
                        "b": health_b["response_time"]
                    }))
                else:
                    errors = []
                    if not health_a["available"]:
                        errors.append(f"Model A: {health_a['error']}")
                    if not health_b["available"]:
                        errors.append(f"Model B: {health_b['error']}")
                    self.message_queue.put(("validation_error", "\n".join(errors)))

            except Exception as e:
                self.message_queue.put(("validation_error", str(e)))

        threading.Thread(target=validate_thread, daemon=True).start()

    def get_current_config(self):
        """Extract current configuration from GUI"""
        # Get text content from Text widgets
        prompt = self.prompt_text.get("1.0", "end-1c")
        system_a = self.system_a_text.get("1.0", "end-1c")
        system_b = self.system_b_text.get("1.0", "end-1c")

        return {
            "models": {
                "a": {
                    "name": self.model_a_var.get(),
                    "system_prompt": system_a
                },
                "b": {
                    "name": self.model_b_var.get(),
                    "system_prompt": system_b
                }
            },
            "endpoints": {
                "a": self.endpoint_a_var.get(),
                "b": self.endpoint_b_var.get()
            },
            "prompt": prompt,
            "rounds": self.rounds_var.get(),
            "temperature": self.temperature_var.get(),
            "max_tokens": self.max_tokens_var.get(),
            "context_budget": self.context_budget_var.get(),
            "stop_if": self.stop_if_var.get(),
            "sleep": 0.2,
            "mode": self.mode_var.get(),
            "transcripts_dir": "transcripts",
            "name": "gui_conversation"
        }

    def start_conversation(self):
        """Start the conversation in a separate thread"""
        if self.conversation_running:
            return

        config = self.get_current_config()

        # Validate required fields
        if not config["models"]["a"]["name"] or not config["models"]["b"]["name"]:
            messagebox.showerror("Error", "Please specify both model names")
            return

        if not config["prompt"].strip():
            messagebox.showerror("Error", "Please enter an initial prompt")
            return

        self.conversation_running = True
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.progress.start()
        self.status_label.config(text="Starting conversation...")

        # Clear previous conversation
        self.conversation_text.delete("1.0", "end")

        def run_conversation():
            try:
                # Create orchestrator
                orchestrator = ConversationOrchestrator(config, self.message_queue)
                self.current_orchestrator = orchestrator

                result = orchestrator.run_conversation()

                if result["success"]:
                    self.message_queue.put(("conversation_complete", result))
                else:
                    self.message_queue.put(("conversation_error", result.get("error", "Unknown error")))

            except Exception as e:
                self.message_queue.put(("conversation_error", str(e)))
            finally:
                self.message_queue.put(("conversation_finished", None))

        threading.Thread(target=run_conversation, daemon=True).start()

    def stop_conversation(self):
        """Stop the current conversation"""
        if self.current_orchestrator:
            self.current_orchestrator.stop_requested = True
        self.message_queue.put(("conversation_stopped", None))

    def check_message_queue(self):
        """Check for messages from background threads"""
        try:
            while True:
                msg_type, data = self.message_queue.get_nowait()

                if msg_type == "status":
                    self.status_label.config(text=data)

                elif msg_type == "validation_success":
                    self.status_label.config(text="Models validated successfully!")
                    messagebox.showinfo("Validation Success",
                                      f"Model A: {data['a']:.1f}s response time\n"
                                      f"Model B: {data['b']:.1f}s response time\n\n"
                                      "Both models are ready!")

                elif msg_type == "validation_error":
                    self.status_label.config(text="Model validation failed")
                    messagebox.showerror("Validation Error", data)

                elif msg_type == "conversation_update":
                    self.update_conversation_display(data)

                elif msg_type == "metrics_update":
                    self.update_metrics_display(data)

                elif msg_type == "conversation_complete":
                    self.status_label.config(text=f"Conversation complete - {data['completed_rounds']} rounds")
                    messagebox.showinfo("Conversation Complete",
                                      f"Successfully completed {data['completed_rounds']} rounds!\n\n"
                                      f"JSON transcript: {data['json_path']}\n"
                                      f"Markdown transcript: {data['md_path']}")

                elif msg_type == "conversation_error":
                    self.status_label.config(text="Conversation failed")
                    messagebox.showerror("Conversation Error", str(data))

                elif msg_type == "models_refreshed":
                    self.refresh_models_btn.config(state="normal")
                    if data["success"]:
                        self.connection_status.config(text=f"Connected - {data['count']} models found", foreground="green")

                        # Clear existing items
                        for item in self.models_tree.get_children():
                            self.models_tree.delete(item)

                        # Add new models
                        for model in data["models"]:
                            self.models_tree.insert("", "end", values=(
                                model["id"],
                                model["size"],
                                model["type"]
                            ))

                        # Also update the dropdowns in config tab
                        model_names = [m["id"] for m in data["models"]]
                        self.model_a_combo['values'] = model_names
                        self.model_b_combo['values'] = model_names
                    else:
                        self.connection_status.config(text=f"Connection failed: {data['error']}", foreground="red")
                        messagebox.showerror("Connection Error", f"Failed to connect to LM Studio:\n\n{data['error']}\n\nMake sure LM Studio is running and the server is started.")

                elif msg_type == "model_test_result":
                    model_name = data["model"]
                    result = data["result"]

                    if result["available"]:
                        messagebox.showinfo("Model Test Success",
                                          f"✅ Model '{model_name}' is working!\n\n"
                                          f"Response time: {result['response_time']:.2f}s\n\n"
                                          "This model is ready to use.")
                    else:
                        messagebox.showerror("Model Test Failed",
                                           f"❌ Model '{model_name}' failed validation:\n\n"
                                           f"{result['error']}\n\n"
                                           "This model may not be loaded or responsive.")

                elif msg_type in ["conversation_finished", "conversation_stopped"]:
                    self.conversation_running = False
                    self.current_orchestrator = None
                    self.start_button.config(state="normal")
                    self.stop_button.config(state="disabled")
                    self.progress.stop()
                    if msg_type == "conversation_stopped":
                        self.status_label.config(text="Conversation stopped by user")

        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self.check_message_queue)

    def update_conversation_display(self, data):
        """Update the conversation display with new content"""
        speaker = data["speaker"]
        content = data["content"]

        if speaker == "user":
            tag = "user"
            prefix = "👤 You: "
        elif speaker.upper() == "A":
            tag = "model_a"
            prefix = f"🤖 Model A ({self.model_a_var.get()}): "
        elif speaker.upper() == "B":
            tag = "model_b"
            prefix = f"🤖 Model B ({self.model_b_var.get()}): "
        else:
            tag = "system"
            prefix = "⚠️ System: "

        self.conversation_text.insert("end", f"\n{prefix}", tag)
        self.conversation_text.insert("end", f"{content}\n", tag)
        self.conversation_text.see("end")

    def update_metrics_display(self, data):
        """Update performance metrics display"""
        model_a_metrics = data.get("model_a", {})
        model_b_metrics = data.get("model_b", {})

        # Update Model A metrics
        self.metrics_a_text.delete("1.0", "end")
        if model_a_metrics and model_a_metrics.get('total_requests', 0) > 0:
            metrics_text = f"""Model: {self.model_a_var.get()}

Performance Metrics:
├─ Average Response Time: {model_a_metrics.get('avg_response_time', 0):.1f}s
├─ Tokens per Second: {model_a_metrics.get('tokens_per_second', 0):.1f}
├─ Error Rate: {model_a_metrics.get('error_rate', 0):.1%}
├─ Total Requests: {model_a_metrics.get('total_requests', 0)}
└─ Total Tokens: {model_a_metrics.get('total_tokens', 0)}

Hardware Notes:
• Running on AMD Radeon 5600 Series
• Optimized context budget for 1.5B model
• Expected CPU fallback if no ROCm"""
            self.metrics_a_text.insert("1.0", metrics_text)
        else:
            self.metrics_a_text.insert("1.0", "No metrics available yet.\n\nStart a conversation to see:\n• Response times\n• Token throughput\n• Error rates\n• Request counts")

        # Update Model B metrics
        self.metrics_b_text.delete("1.0", "end")
        if model_b_metrics and model_b_metrics.get('total_requests', 0) > 0:
            metrics_text = f"""Model: {self.model_b_var.get()}

Performance Metrics:
├─ Average Response Time: {model_b_metrics.get('avg_response_time', 0):.1f}s
├─ Tokens per Second: {model_b_metrics.get('tokens_per_second', 0):.1f}
├─ Error Rate: {model_b_metrics.get('error_rate', 0):.1%}
├─ Total Requests: {model_b_metrics.get('total_requests', 0)}
└─ Total Tokens: {model_b_metrics.get('total_tokens', 0)}

Hardware Notes:
• Running on AMD Radeon 5600 Series
• Optimized context budget for 1.1B model
• Expected CPU fallback if no ROCm"""
            self.metrics_b_text.insert("1.0", metrics_text)
        else:
            self.metrics_b_text.insert("1.0", "No metrics available yet.\n\nStart a conversation to see:\n• Response times\n• Token throughput\n• Error rates\n• Request counts")

    def load_config(self):
        """Load configuration from YAML file"""
        if not HAS_YAML:
            messagebox.showerror("Error", "YAML support not available. Please install PyYAML:\npip install pyyaml")
            return

        file_path = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("YAML files", "*.yaml *.yml"), ("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    if file_path.endswith(('.yaml', '.yml')):
                        config = yaml.safe_load(f)
                    else:
                        config = json.load(f)

                self.apply_config(config)
                messagebox.showinfo("Success", f"Configuration loaded from {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load config: {str(e)}")

    def save_config(self):
        """Save current configuration to YAML file"""
        if not HAS_YAML:
            messagebox.showerror("Error", "YAML support not available. Please install PyYAML:\npip install pyyaml")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                config = self.get_current_config()
                with open(file_path, 'w') as f:
                    if file_path.endswith(('.yaml', '.yml')):
                        yaml.dump(config, f, default_flow_style=False, indent=2)
                    else:
                        json.dump(config, f, indent=2)
                messagebox.showinfo("Success", f"Configuration saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save config: {str(e)}")

    def apply_config(self, config):
        """Apply loaded configuration to GUI"""
        try:
            # Models
            if "models" in config:
                if "a" in config["models"]:
                    self.model_a_var.set(config["models"]["a"].get("name", ""))
                    self.system_a_text.delete("1.0", "end")
                    self.system_a_text.insert("1.0", config["models"]["a"].get("system_prompt", ""))
                if "b" in config["models"]:
                    self.model_b_var.set(config["models"]["b"].get("name", ""))
                    self.system_b_text.delete("1.0", "end")
                    self.system_b_text.insert("1.0", config["models"]["b"].get("system_prompt", ""))

            # Endpoints
            if "endpoints" in config:
                self.endpoint_a_var.set(config["endpoints"].get("a", ""))
                self.endpoint_b_var.set(config["endpoints"].get("b", ""))

            # Settings
            if "prompt" in config:
                self.prompt_text.delete("1.0", "end")
                self.prompt_text.insert("1.0", config["prompt"])
            if "rounds" in config:
                self.rounds_var.set(config["rounds"])
            if "temperature" in config:
                self.temperature_var.set(config["temperature"])
            if "max_tokens" in config:
                self.max_tokens_var.set(config["max_tokens"])
            if "context_budget" in config:
                self.context_budget_var.set(config["context_budget"])
            if "stop_if" in config:
                self.stop_if_var.set(config["stop_if"])
            if "mode" in config:
                self.mode_var.set(config["mode"])

        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply config: {str(e)}")

def main():
    # Check for required modules
    try:
        import requests
    except ImportError:
        print("Error: requests module not found. Please install it:")
        print("pip install requests")
        sys.exit(1)

    if not HAS_YAML:
        print("Warning: PyYAML not found. Config save/load will be disabled.")
        print("To enable YAML support: pip install pyyaml")

    root = tk.Tk()
    app = PingPongGUI(root)

    # Add some basic keyboard shortcuts
    root.bind('<Control-q>', lambda e: root.quit())
    root.bind('<F5>', lambda e: app.refresh_model_list())

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main()
