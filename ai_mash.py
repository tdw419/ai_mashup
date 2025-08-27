#!/usr/bin/env python3
"""
ai_mash.py — Multi-AI chorus with provenance, per-agent overrides, reducer,
clipboard append, static viewer, and tiny HTTP server.

Commands:
  where                          -> print artifacts folder
  ask [--no-reducer] <prompt...> -> query callable agents; continue latest session
  append --agent X (--text|--text-file|--clipboard)
                                  -> add a manual response (e.g., Gemini) to latest round
  loop [--no-reducer] --rounds N <prompt...>
                                  -> start new session; run N iterative rounds
  export --mode {full|summary|roadmap|tasks}
                                  -> create exports from the latest round
  resubmit --mode <mode> <prompt...>
                                  -> Use an export as context for a new round
  index                          -> (re)generate mash_runs/index.html viewer
  serve [--port 8888]            -> serve mash_runs/ with a tiny HTTP server

Requires:
  pip install requests pyperclip
"""
import argparse
import json
import os
import pathlib
import shutil
import textwrap
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    import requests
    import pyperclip
except ImportError:
    raise SystemExit("Missing deps. Please run: pip install requests pyperclip")

# --- Globals and Configuration ---
ROOT = pathlib.Path(__file__).parent.resolve()
MASH_DIR = ROOT / "mash_runs"
CONFIG_FILE = ROOT / "ai_mash_config.json"

# Global toggle (set by CLI flags) to skip reducer per run
NO_REDUCER = False

def load_config() -> Dict[str, Any]:
    """Load agent, reducer, defaults, and limits from JSON config."""
    if not CONFIG_FILE.exists():
        raise SystemExit(f"Configuration file not found: {CONFIG_FILE}")
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        raise SystemExit(f"Failed to parse {CONFIG_FILE}: {e}")

CONFIG = load_config()

# --- Utilities ---
def now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def ensure_dir(path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)

def write_text(path: pathlib.Path, content: str):
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")

def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")

def new_session_dir() -> pathlib.Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    d = MASH_DIR / f"session_{ts}"
    ensure_dir(d)
    return d

def latest_session_dir() -> Optional[pathlib.Path]:
    if not MASH_DIR.exists():
        return None
    return max(MASH_DIR.glob("session_*"), default=None, key=lambda p: p.name)

def clamp_context(txt: str) -> str:
    """Clamp merged context to configured max length."""
    limit = int(CONFIG.get("limits", {}).get("max_context_chars", 48000))
    if len(txt) <= limit:
        return txt
    return txt[:limit] + "\n...[context truncated]..."

# --- Core AI Interaction ---
def run_export_agent(mode: str, context: str) -> str:
    """Runs the reducer agent with a special system prompt for exporting."""
    reducer_agent = CONFIG.get("reducer")
    if not (reducer_agent and reducer_agent.get("enabled") and reducer_agent.get("type") == "openai"):
        raise SystemExit("Reducer agent is not configured or enabled, which is required for this export mode.")

    print(f"Running export agent for mode: {mode}...")

    system_prompt = CONFIG.get("export_prompts", {}).get(mode)
    if not system_prompt:
        raise ValueError(f"Invalid export mode or prompt not configured: {mode}")

    # Create a temporary agent config with the export system prompt
    export_agent_config = reducer_agent.copy()
    export_agent_config["system"] = system_prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"<CONTEXT>\n{context}\n</CONTEXT>\n\nBased on the context above, perform your task."}
    ]

    return openai_chat(export_agent_config, messages)

def openai_chat(agent: Dict[str, Any], messages: List[Dict[str, str]]) -> str:
    """Send chat to an OpenAI-compatible endpoint (e.g., LM Studio)."""
    url = agent["base_url"].rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {agent.get('api_key','')}",
        "Content-Type": "application/json"
    }
    # Merge defaults with per-agent overrides
    params = CONFIG.get("defaults", {}).copy()
    params.update(agent.get("overrides", {}))
    body = {"model": agent["model"], "messages": messages, **params}

    try:
        r = requests.post(url, headers=headers, json=body, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        msg = f"[ERROR contacting {agent.get('name','unknown')}] {e}"
        print(f"\nWARNING: {msg}\n")
        return msg

def merge_round(responses_dir: pathlib.Path) -> str:
    """Merge all responses in a round into one markdown doc, preserving labels."""
    parts: List[str] = []
    for f in sorted(responses_dir.glob("*.md")):
        # Skip reducer summary files from the base merge; appended later
        if f.name.startswith("_") and f.name.endswith("_summary.md"):
            continue
        meta_file = f.with_suffix(".json")
        if meta_file.exists():
            try:
                meta = json.loads(read_text(meta_file))
                agent_name = meta.get("agent", f.stem)
            except Exception:
                agent_name = f.stem
        else:
            agent_name = f.stem
        parts.append(f"\n\n---\n# Response from: {agent_name}\n\n{read_text(f)}\n")
    header = f"# Multi-AI Mash (merged at {now_iso()})\n"
    return header + "".join(parts)

def run_round(session_dir: pathlib.Path, round_idx: int, prompt: str, prev_mash: Optional[str]):
    """Execute one round against all callable agents and write provenance."""
    round_dir = session_dir / f"round_{round_idx:02d}"
    responses_dir = round_dir / "responses"
    ensure_dir(responses_dir)

    print(f"\n=== Round {round_idx:02d} ===")

    # Query callable agents
    for agent in CONFIG.get("agents", []):
        if agent.get("type") != "openai":
            continue

        print(f"Querying {agent['name']}...")
        messages: List[Dict[str, str]] = []
        if agent.get("system"):
            messages.append({"role": "system", "content": agent["system"]})
        if prev_mash:
            context = clamp_context(prev_mash)
            messages.append({
                "role": "user",
                "content": (
                    "Based on the prior multi-AI context, continue the work.\n\n"
                    "<CONTEXT>\n" + context + "\n</CONTEXT>"
                )
            })
        messages.append({"role": "user", "content": prompt})

        content = openai_chat(agent, messages)
        out_md = responses_dir / f"{agent['name']}.md"
        write_text(out_md, content)

        effective_params = CONFIG.get("defaults", {}).copy()
        effective_params.update(agent.get("overrides", {}))
        meta = {
            "agent": agent["name"],
            "type": agent["type"],
            "model": agent.get("model"),
            "base_url": agent.get("base_url"),
            "timestamp": now_iso(),
            "prompt": prompt,
            "effective_params": effective_params
        }
        write_text(out_md.with_suffix(".json"), json.dumps(meta, indent=2))

    # Merge the round
    merged = merge_round(responses_dir)

    # Optional reducer
    reducer_agent = CONFIG.get("reducer")
    if (not NO_REDUCER and reducer_agent and reducer_agent.get("enabled")
            and reducer_agent.get("type") == "openai"):
        print(f"Running consensus reducer ({reducer_agent.get('name', 'reducer')})...")

        reducer_messages: List[Dict[str, str]] = []
        if reducer_agent.get("system"):
            reducer_messages.append({"role": "system", "content": reducer_agent["system"]})

        reducer_input = clamp_context(merged)
        reducer_prompt = (
            "Analyze the multi-AI responses below and produce a consensus summary with:\n"
            "1) Key Agreements\n2) Key Disagreements & Alternatives\n3) Open Questions\n"
            "4) Concrete Next Actions (bullet list, each action 1–2 lines).\n\n"
            "<RESPONSES>\n" + reducer_input + "\n</RESPONSES>"
        )
        reducer_messages.append({"role": "user", "content": reducer_prompt})

        summary = openai_chat(reducer_agent, reducer_messages)

        # --- Try to parse structured JSON from the reducer's output ---
        structured_summary = None
        json_marker = "\nJSON:"
        if json_marker in summary:
            parts = summary.split(json_marker, 1)
            human_readable_summary = parts[0].strip()
            json_string = parts[1].strip()
            try:
                structured_summary = json.loads(json_string)
                # If parsing succeeds, use the clean human-readable part only
                summary = human_readable_summary
            except json.JSONDecodeError:
                print("\nWARNING: Reducer produced invalid JSON. Keeping raw summary intact.\n")

        # Save reducer artifact and provenance
        reducer_out_md = responses_dir / f"_{reducer_agent['name']}_summary.md"
        write_text(reducer_out_md, summary)

        # Save structured JSON separately if it parsed cleanly
        if structured_summary:
            structured_out_path = reducer_out_md.with_name(f"_{reducer_agent['name']}_structured.json")
            write_text(structured_out_path, json.dumps(structured_summary, indent=2))

        reducer_effective = CONFIG.get("defaults", {}).copy()
        reducer_effective.update(reducer_agent.get("overrides", {}))
        write_text(
            reducer_out_md.with_suffix(".json"),
            json.dumps({
                "agent": reducer_agent["name"],
                "type": "reducer",
                "model": reducer_agent.get("model"),
                "base_url": reducer_agent.get("base_url"),
                "timestamp": now_iso(),
                "effective_params": reducer_effective
            }, indent=2)
        )

        # Append to merged for convenience
        merged += f"\n\n---\n# Consensus Summary\n\n{summary}\n"

    write_text(round_dir / "mash_merged.md", merged)
    print(f"Round {round_idx:02d} → {round_dir / 'mash_merged.md'}")

# --- Command Handlers ---
def cmd_where(_args):
    print(str(MASH_DIR))

def cmd_ask(args):
    global NO_REDUCER
    NO_REDUCER = bool(getattr(args, "no_reducer", False))

    session_dir = latest_session_dir() or new_session_dir()
    rounds = sorted(session_dir.glob("round_*"))
    if rounds:
        last = rounds[-1]
        prev_mash_path = last / "mash_merged.md"
        prev_mash = read_text(prev_mash_path) if prev_mash_path.exists() else None
        next_idx = int(last.name.split("_")[1]) + 1
    else:
        prev_mash, next_idx = None, 1

    prompt = " ".join(args.prompt)
    run_round(session_dir, next_idx, prompt, prev_mash)
    print(f"\nSuccess → {session_dir / f'round_{next_idx:02d}' / 'mash_merged.md'}")

def cmd_append(args):
    session_dir = latest_session_dir() or new_session_dir()
    rounds = sorted(session_dir.glob("round_*"))
    round_dir = rounds[-1] if rounds else (session_dir / "round_01")
    responses_dir = round_dir / "responses"
    ensure_dir(responses_dir)

    text = (
        args.text if args.text is not None else
        (read_text(pathlib.Path(args.text_file)) if args.text_file else None) or
        (pyperclip.paste() if args.clipboard else None)
    )
    if not text or not text.strip():
        raise SystemExit("No text provided (use --text, --text-file, or --clipboard).")

    agent_name = args.agent
    out_md = responses_dir / f"{agent_name}.md"
    write_text(out_md, text)
    meta = {"agent": agent_name, "type": "manual", "timestamp": now_iso()}
    write_text(out_md.with_suffix(".json"), json.dumps(meta, indent=2))

    merged = merge_round(responses_dir)
    write_text(round_dir / "mash_merged.md", merged)
    print(f"Appended '{agent_name}' → updated {round_dir / 'mash_merged.md'}")

def cmd_loop(args):
    global NO_REDUCER
    NO_REDUCER = bool(getattr(args, "no_reducer", False))

    session_dir = new_session_dir()
    prev_mash = None
    prompt = " ".join(args.prompt)

    for i in range(1, args.rounds + 1):
        run_round(session_dir, i, prompt, prev_mash)
        mash_path = session_dir / f"round_{i:02d}" / "mash_merged.md"
        prev_mash = read_text(mash_path)
        # After the first round, use a generic “continue” prompt to push progress.
        prompt = "Please review the context and provide the next logical step, refinement, or implementation."

    print(f"\nLoop complete → {session_dir / f'round_{args.rounds:02d}' / 'mash_merged.md'}")

def get_export_content(mode: str, mash_file_path: pathlib.Path) -> str:
    """Gets the content for a given export mode without writing to a file."""
    if not mash_file_path.exists():
        raise SystemExit(f"No merged mash file found in {mash_file_path.parent}")

    if mode == "full":
        return read_text(mash_file_path)
    elif mode in ["summary", "roadmap", "tasks"]:
        context = read_text(mash_file_path)
        return run_export_agent(mode, context)
    else:
        raise ValueError(f"Unknown export mode: {mode}")

def cmd_export(args):
    """Generate exports from the latest round's mash."""
    session_dir = latest_session_dir()
    if not session_dir:
        raise SystemExit("No sessions found. Run 'ask' or 'loop' first.")

    rounds = sorted(session_dir.glob("round_*"))
    if not rounds:
        raise SystemExit(f"No rounds found in session: {session_dir}")

    latest_round_dir = rounds[-1]

    # Allow specifying a round number
    round_dir = latest_round_dir
    if args.round is not None:
        round_dir = session_dir / f"round_{args.round:02d}"
        if not round_dir.exists():
            raise SystemExit(f"Round {args.round} not found in session {session_dir.name}")

    print(f"Exporting from session '{session_dir.name}', round '{round_dir.name}'...")

    mash_file = round_dir / "mash_merged.md"
    export_dir = round_dir / "exports"
    ensure_dir(export_dir)

    mode = args.mode
    content = get_export_content(mode, mash_file)

    file_extensions = {"full": "md", "summary": "md", "roadmap": "json", "tasks": "yaml"}
    out_path = export_dir / f"{mode}.{file_extensions[mode]}"
    write_text(out_path, content)
    print(f"'{mode}' export generated: {out_path}")

def cmd_resubmit(args):
    """Use an export as context for a new round of prompting."""
    global NO_REDUCER
    NO_REDUCER = bool(getattr(args, "no_reducer", False))

    session_dir = latest_session_dir()
    if not session_dir:
        raise SystemExit("No sessions found to resubmit to. Run 'ask' or 'loop' first.")

    rounds = sorted(session_dir.glob("round_*"))
    if not rounds:
        raise SystemExit(f"No rounds found in session: {session_dir}")

    latest_round_dir = rounds[-1]
    mash_file = latest_round_dir / "mash_merged.md"

    print(f"Generating content from mode '{args.mode}' to use as new context...")
    new_context = get_export_content(args.mode, mash_file)

    next_round_idx = int(latest_round_dir.name.split("_")[1]) + 1
    prompt = " ".join(args.prompt)

    run_round(session_dir, next_round_idx, prompt, new_context)
    print(f"\nResubmit successful. New round created → {session_dir / f'round_{next_round_idx:02d}'}")

def generate_index_html():
    """Create a lightweight static viewer at mash_runs/index.html."""
    ensure_dir(MASH_DIR)
    sessions = sorted([p for p in MASH_DIR.glob("session_*") if p.is_dir()], reverse=True)
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>AI Mash Viewer</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
body{font-family:system-ui,sans-serif;margin:0;display:flex;height:100vh}
.sidebar{width:350px;padding:16px;border-right:1px solid #ddd;overflow-y:auto;display:flex;flex-direction:column}
.main{flex-grow:1;padding:16px;overflow-y:auto}
h1,h2{margin:0 0 8px}
label{margin-top:12px;margin-bottom:4px;display:block}
select,button{font-size:14px;padding:6px 8px;width:100%;margin-bottom:8px}
#content{white-space:pre-wrap;font-family:Consolas,'Courier New',monospace;line-height:1.5}
#structured-summary{margin-top:16px}
details{border:1px solid #ddd;border-radius:4px;padding:8px;margin-bottom:8px}
summary{font-weight:bold;cursor:pointer}
ul{margin:8px 0 0;padding-left:20px}
</style>
</head>
<body>
<div class="sidebar">
  <h1>AI Mash Viewer</h1>
  <label>Session:</label>
  <select id="session"></select>
  <label>Round:</label>
  <select id="round"></select>
  <button id="load">Load</button>
  <div id="structured-summary"></div>
</div>
<div class="main">
  <h2>Mash Content</h2>
  <div id="content">Select a session/round and click Load.</div>
</div>
<script>
const sessions = %SESSIONS%;
const REDUCER_NAME = %REDUCER_NAME%;
const sessionSel = document.getElementById('session');
const roundSel = document.getElementById('round');
const content = document.getElementById('content');
const loadBtn = document.getElementById('load');
const structuredContainer = document.getElementById('structured-summary');

function populateSessions() {
  sessions.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.name;
    opt.textContent = s.name;
    sessionSel.appendChild(opt);
  });
  sessionSel.addEventListener('change', onSessionChange);
  onSessionChange();
}

function onSessionChange(){
  roundSel.innerHTML = '';
  const s = sessions.find(x => x.name === sessionSel.value);
  (s ? s.rounds : []).forEach(r => {
    const opt = document.createElement('option');
    opt.value = r;
    opt.textContent = r;
    roundSel.appendChild(opt);
  });
}

async function loadMash() {
  const s = sessionSel.value;
  const r = roundSel.value;
  if (!s || !r) { content.textContent = 'Please select a valid session and round.'; return; }
  const url = s + '/' + r + '/mash_merged.md';
  content.textContent = 'Loading ' + url + ' ...';
  structuredContainer.innerHTML = ''; // Clear previous summary
  try {
    const res = await fetch(url);
    if(!res.ok) throw new Error(res.status + ' ' + res.statusText);
    const txt = await res.text();
    content.textContent = txt;
    loadStructuredSummary(s, r);
  } catch(e) {
    content.textContent = 'Failed to load: ' + e;
  }
}

async function loadStructuredSummary(session, round) {
    if (!REDUCER_NAME) return;
    const url = `${session}/${round}/responses/_${REDUCER_NAME}_structured.json`;
    try {
        const res = await fetch(url);
        if (!res.ok) return; // File might not exist, which is fine
        const data = await res.json();
        let html = '<h2>Consensus Summary</h2>';
        for (const key in data) {
            if (Array.isArray(data[key]) && data[key].length > 0) {
                const title = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                html += `<details open><summary>${title}</summary><ul>`;
                data[key].forEach(item => {
                    const safeItem = String(item).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
                    html += `<li>${safeItem}</li>`;
                });
                html += '</ul></details>';
            }
        }
        structuredContainer.innerHTML = html;
    } catch (e) {
        // Fail silently if structured summary can't be loaded/parsed
        console.warn(`Could not load structured summary from ${url}:`, e);
    }
}

document.addEventListener('DOMContentLoaded', () => {
  populateSessions();
  loadBtn.addEventListener('click', loadMash);
});
</script>
</body>
</html>
"""
    data = []
    for s in sessions:
        rounds = sorted([p.name for p in s.glob("round_*") if (p / "mash_merged.md").exists()], reverse=True)
        if rounds:
            data.append({"name": s.name, "rounds": rounds})
    html = html.replace("%SESSIONS%", json.dumps(data))
    reducer_name = CONFIG.get("reducer", {}).get("name")
    html = html.replace("%REDUCER_NAME%", json.dumps(reducer_name))
    write_text(MASH_DIR / "index.html", html)

def cmd_index(_args):
    generate_index_html()
    print(f"Viewer written → {MASH_DIR / 'index.html'}")

def cmd_serve(args):
    generate_index_html()  # ensure index exists
    os.chdir(str(MASH_DIR))
    from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
    port = args.port
    print(f"Serving {MASH_DIR} on http://127.0.0.1:{port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), SimpleHTTPRequestHandler).serve_forever()

# --- CLI ---
def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate a multi-AI chorus with provenance, overrides, reducer, and a simple web viewer.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent('''
Export and Resubmit Modes:
  full:      The complete, raw merged mash file from a round.
  summary:   A concise, AI-generated summary of the mash.
  roadmap:   An AI-generated structured roadmap in JSON format.
  tasks:     An AI-generated list of actionable tasks in YAML format.

Example Resubmit Workflow:
  1. python ai_mash.py ask "Brainstorm features for a new app."
  2. python ai_mash.py resubmit --mode summary "From this summary, what is the most important feature to build first?"
''')
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_where = sub.add_parser("where", help="Show the artifacts directory.")
    p_where.set_defaults(func=cmd_where)

    p_ask = sub.add_parser("ask", help="Ask all callable agents; continue latest session.")
    p_ask.add_argument("prompt", nargs=argparse.REMAINDER, help="Prompt to send.")
    p_ask.add_argument("--no-reducer", action="store_true", help="Skip reducer for this run.")
    p_ask.set_defaults(func=cmd_ask)

    p_append = sub.add_parser("append", help="Append a manual response to latest round.")
    p_append.add_argument("--agent", required=True, help="Agent name, e.g., 'gemini'.")
    g = p_append.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="Raw text to append.")
    g.add_argument("--text-file", help="Path to file with text.")
    g.add_argument("--clipboard", action="store_true", help="Read text from clipboard.")
    p_append.set_defaults(func=cmd_append)

    p_loop = sub.add_parser("loop", help="Start a new session and run multiple rounds.")
    p_loop.add_argument("--rounds", type=int, default=3, help="Number of rounds.")
    p_loop.add_argument("prompt", nargs=argparse.REMAINDER, help="Initial prompt.")
    p_loop.add_argument("--no-reducer", action="store_true", help="Skip reducer for all rounds.")
    p_loop.set_defaults(func=cmd_loop)

    p_index = sub.add_parser("index", help="Generate mash_runs/index.html viewer.")
    p_index.set_defaults(func=cmd_index)

    p_serve = sub.add_parser("serve", help="Serve mash_runs/ with a tiny HTTP server.")
    p_serve.add_argument("--port", type=int, default=8888)
    p_serve.set_defaults(func=cmd_serve)

    p_export = sub.add_parser("export", help="Export artifacts from the latest round.")
    p_export.add_argument(
        "--mode",
        choices=["full", "summary", "roadmap", "tasks"],
        required=True,
        help="The type of export to generate."
    )
    p_export.add_argument(
        "--round",
        type=int,
        help="Specify a round number to export from (defaults to latest)."
    )
    p_export.set_defaults(func=cmd_export)

    p_resubmit = sub.add_parser("resubmit", help="Use an export as context for a new prompt.")
    p_resubmit.add_argument(
        "--mode",
        choices=["full", "summary", "roadmap", "tasks"],
        required=True,
        help="The export mode to use as context."
    )
    p_resubmit.add_argument("prompt", nargs=argparse.REMAINDER, help="Prompt to send with the new context.")
    p_resubmit.add_argument("--no-reducer", action="store_true", help="Skip reducer for this run.")
    p_resubmit.set_defaults(func=cmd_resubmit)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
