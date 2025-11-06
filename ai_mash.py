#!/usr/bin/env python3
"""
ai_mash.py — Multi-AI chorus with provenance, per-agent overrides, reducer,
clipboard append, static viewer, export pipelines, resubmit, and issue creation.
Commands:
  where                           -> print artifacts folder
  ask [--no-reducer] <prompt...> -> query callable agents; continue latest session
  append --agent X (--text|--text-file|--clipboard)
                                  -> add a manual response (e.g., Gemini) to latest round
  loop [--no-reducer] --rounds N <prompt...>
                                  -> start new session; run N iterative rounds
  export --mode {full|summary|roadmap|tasks} [--round N] [--session S]
                                  -> export results in specified format
  resubmit --mode {full|summary|roadmap|tasks} [--round N] [--session S] <prompt...>
                                  -> export and feed back into a new round
  issues [--owner-repo X] [--round N] [--session S]
                                  -> generate gh issue create commands from tasks
  index                           -> (re)generate mash_runs/index.html viewer
  serve [--port 8888]            -> serve mash_runs/ with a tiny HTTP server
Requires:
  pip install requests pyperclip
"""
import argparse
import json
import os
import pathlib
import textwrap
import zipfile
import shlex
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Optional
try:
    import requests
    import pyperclip
    import lancedb
except ImportError:
    raise SystemExit("Missing deps. Please run: pip install -r requirements.txt")

# --- Globals and Configuration ---
ROOT = pathlib.Path(__file__).parent.resolve()
MASH_DIR = ROOT / "mash_runs"
DB_DIR = ROOT / "lancedb_data"
CONFIG_FILE = ROOT / "ai_mash_config.json"
# Global toggle (set by CLI flags) to skip reducer per run
NO_REDUCER = False

def load_config() -> Dict[str, Any]:
    """Load agent, reducer, exporters, defaults, and limits from JSON config."""
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

def latest_round_dir(session_dir: pathlib.Path) -> Optional[pathlib.Path]:
    rounds = sorted(session_dir.glob("round_*"))
    return rounds[-1] if rounds else None

def pick_round_dir(session_dir: pathlib.Path, round_arg: Optional[int]) -> pathlib.Path:
    if round_arg is None:
        rd = latest_round_dir(session_dir)
        if not rd:
            raise SystemExit("No rounds found in session.")
        return rd
    rd = session_dir / f"round_{int(round_arg):02d}"
    if not rd.exists():
        raise SystemExit(f"Round {round_arg:02d} not found in {session_dir.name}.")
    return rd

def ensure_exports_dir(round_dir: pathlib.Path) -> pathlib.Path:
    exports = round_dir / "exports"
    ensure_dir(exports)
    return exports

def clamp_context(txt: str) -> str:
    """Clamp merged context to configured max length."""
    limit = int(CONFIG.get("limits", {}).get("max_context_chars", 48000))
    if len(txt) <= limit:
        return txt
    return txt[:limit] + "\n...[context truncated]..."

# --- Core AI Interaction ---
def openai_chat(agent: Dict[str, Any], messages: List[Dict[str, str]]) -> str:
    """Send chat to an OpenAI-compatible endpoint (e.g., LM Studio)."""
    url = agent["base_url"].rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {agent.get('api_key','')}",
        "Content-Type": "application/json"
    }
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
    merged = merge_round(responses_dir)
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
        structured_summary = None
        json_marker = "\nJSON:"
        if json_marker in summary:
            parts = summary.split(json_marker, 1)
            human_readable_summary = parts[0].strip()
            json_string = parts[1].strip()
            try:
                structured_summary = json.loads(json_string)
                summary = human_readable_summary
            except json.JSONDecodeError:
                print("\nWARNING: Reducer produced invalid JSON. Keeping raw summary intact.\n")
        reducer_out_md = responses_dir / f"_{reducer_agent['name']}_summary.md"
        write_text(reducer_out_md, summary)
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
        merged += f"\n\n---\n# Consensus Summary\n\n{summary}\n"
    write_text(round_dir / "mash_merged.md", merged)
    print(f"Round {round_idx:02d} → {round_dir / 'mash_merged.md'}")

# --- Export and Resubmit ---
def run_export(round_dir: pathlib.Path, mode: str) -> pathlib.Path:
    """Export a round's mash in the specified mode."""
    exports = ensure_exports_dir(round_dir)
    mash_path = round_dir / "mash_merged.md"
    if not mash_path.exists():
        raise SystemExit(f"Missing {mash_path}")
    merged = read_text(mash_path)
    if mode == "full":
        out = exports / "full.md"
        write_text(out, merged)
        print(f"Exported full → {out}")
        return out
    reducer_agent = CONFIG.get("reducer")
    if not (reducer_agent and reducer_agent.get("enabled") and reducer_agent.get("type") == "openai"):
        raise SystemExit("Reducer is disabled or not configured; cannot synthesize exports.")
    prompts = CONFIG.get("exporters", {})
    if mode not in prompts:
        raise SystemExit(f"No export prompt defined for mode '{mode}' in config.")
    prompt = prompts[mode]
    messages = [
        {"role": "system", "content": reducer_agent.get("system", "")},
        {"role": "user", "content": f"{prompt}\n\n<MERGED>\n{clamp_context(merged)}\n</MERGED>"}
    ]
    content = openai_chat(reducer_agent, messages)
    if mode == "summary":
        out = exports / "summary.md"
        write_text(out, content)
        print(f"Exported summary → {out}")
        return out
    if mode == "roadmap":
        out = exports / "roadmap.json"
        try:
            data = json.loads(content)
            write_text(out, json.dumps(data, indent=2))
        except json.JSONDecodeError:
            print("WARNING: Roadmap not valid JSON; saving raw text instead.")
            out = out.with_suffix(".txt")
            write_text(out, content)
        print(f"Exported roadmap → {out}")
        return out
    if mode == "tasks":
        struct_path = round_dir / "responses" / f"_{reducer_agent['name']}_structured.json"
        if struct_path.exists():
            try:
                s = json.loads(read_text(struct_path))
                tasks = s.get("next_actions", [])
                out = exports / "ai_tasks.json"
                write_text(out, json.dumps(tasks, indent=2))
                print(f"Exported tasks from structured reducer JSON → {out}")
                return out
            except Exception:
                print("WARNING: Could not parse structured reducer JSON; falling back to synthesis.")
        out = exports / "ai_tasks.json"
        try:
            tasks = json.loads(content)
            write_text(out, json.dumps(tasks, indent=2))
        except json.JSONDecodeError:
            print("WARNING: Tasks not valid JSON; saving raw text instead.")
            out = out.with_suffix(".txt")
            write_text(out, content)
        print(f"Exported tasks → {out}")
        return out
    raise SystemExit(f"Unknown export mode: {mode}")

def cmd_export(args):
    """Export artifacts from a round."""
    session_dir = MASH_DIR / args.session if args.session else latest_session_dir()
    if not session_dir or not session_dir.exists():
        raise SystemExit("No sessions found. Run 'ask' or 'loop' first.")
    round_dir = pick_round_dir(session_dir, args.round)
    run_export(round_dir, args.mode)

def cmd_resubmit(args):
    """Export then immediately run a new round seeded by the export artifact."""
    session_dir = MASH_DIR / args.session if args.session else latest_session_dir()
    if not session_dir or not session_dir.exists():
        raise SystemExit("No sessions found.")
    round_dir = pick_round_dir(session_dir, args.round)
    artifact = run_export(round_dir, args.mode)
    export_text = read_text(artifact)
    seed = f"Use this export as authoritative context for the next step.\n\n<EXPORT mode='{args.mode}'>\n{export_text}\n</EXPORT>"
    new_session = new_session_dir()
    prompt = " ".join(args.prompt) if args.prompt else "Continue based on the provided context."
    run_round(new_session, 1, prompt, seed)
    print(f"\nResubmitted export ({args.mode}) into new session {new_session.name}/round_01.")

def cmd_issues(args):
    """Generate GitHub CLI issue creation commands from tasks export."""
    session_dir = MASH_DIR / args.session if args.session else latest_session_dir()
    if not session_dir or not session_dir.exists():
        raise SystemExit("No sessions found.")
    round_dir = pick_round_dir(session_dir, args.round)
    tasks_path = round_dir / "exports" / "ai_tasks.json"
    if not tasks_path.exists():
        raise SystemExit(f"No tasks export found at {tasks_path}. Run 'export --mode tasks' first.")
    try:
        tasks = json.loads(read_text(tasks_path))
    except json.JSONDecodeError:
        raise SystemExit(f"Invalid JSON in {tasks_path}.")
    owner_repo = args.owner_repo or "user/repo"  # Placeholder; replace with actual repo
    commands = []
    for task in tasks:
        title = task.get("title", "Untitled Task")
        description = task.get("description", "")
        priority = task.get("priority", "medium")
        body = f"**Description**: {description}\n**Priority**: {priority}\n**Estimated Hours**: {task.get('estimated_hours', 'N/A')}"
        cmd = f"gh issue create --title {shlex.quote(title)} --body {shlex.quote(body)}"
        if owner_repo:
            cmd += f" --repo {owner_repo}"
        commands.append(cmd)
    print("\nGitHub CLI commands for creating issues:")
    for cmd in commands:
        print(cmd)
    print("\nYou can copy-paste these commands or pipe them to a shell (e.g., | bash).")

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
        prompt = "Please review the context and provide the next logical step, refinement, or implementation."
    print(f"\nLoop complete → {session_dir / f'round_{args.rounds:02d}' / 'mash_merged.md'}")


def execute_db_actions(db, actions):
    """Execute a list of database actions suggested by the LLM."""
    for action in actions:
        op = action.get("operation")
        try:
            if op == "rename_table":
                db.rename_table(action["from"], action["to"])
                print(f"Renamed table {action['from']} to {action['to']}")
            elif op == "drop_table":
                db.drop_table(action["table_name"])
                print(f"Dropped table {action['table_name']}")
            elif op == "create_table":
                # This is a simplified version. A real implementation would need to handle schema definitions.
                db.create_table(action["table_name"], data=[])
                print(f"Created table {action['table_name']}")
            elif op == "move_data":
                # This is a placeholder for a more complex data migration logic.
                print(f"Moving data from {action['from']} to {action['to']} (not implemented).")
            else:
                print(f"Unknown operation: {op}")
        except Exception as e:
            print(f"Error executing action {op}: {e}")

def init_organizer_db():
    """Initializes the SQLite DB for storing organization plans."""
    con = sqlite3.connect("mash_organizer.sqlite")
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS organization_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            llm_plan_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'proposed' -- proposed, executed, failed
        )
    """)
    con.commit()
    con.close()

def save_organization_plan(plan_json: str):
    """Saves a new organization plan from the LLM to the database."""
    init_organizer_db() # Ensure DB and table exist
    con = sqlite3.connect("mash_organizer.sqlite")
    cur = con.cursor()
    cur.execute(
        "INSERT INTO organization_plans (created_at, llm_plan_json) VALUES (?, ?)",
        (now_iso(), plan_json)
    )
    con.commit()
    con.close()

def cmd_database(args):
    """Manage the LanceDB database."""
    action = args.action
    db = lancedb.connect(DB_DIR)

    if action == "create":
        print(f"LanceDB database created at {DB_DIR}")
        if "documents" not in db.table_names():
            import pyarrow as pa
            schema = pa.schema([
                pa.field("vector", pa.list_(pa.float32(), 2)),
                pa.field("text", pa.string()),
                pa.field("id", pa.string())
            ])
            db.create_table("documents", schema=schema, data=[
                {'vector': [1.1, 1.2], 'text': 'hello world', 'id': '1'},
                {'vector': [0.5, 1.5], 'text': 'foo bar', 'id': '2'}
            ])
            print("Created dummy 'documents' table for demonstration.")

    elif action == "organize":
        print("Database organization loop started...")
        table_names = db.table_names()
        if not table_names:
            raise SystemExit("No tables found in the database to organize.")

        reducer_agent = CONFIG.get("reducer")
        if not (reducer_agent and reducer_agent.get("enabled")):
            raise SystemExit("Reducer agent is required for organization but is disabled.")

        for i in range(5):  # Limit iterations to prevent infinite loops
            print(f"\n--- Organization Iteration {i+1} ---")
            current_schema = {name: str(db.open_table(name).schema) for name in db.table_names()}
            schema_str = json.dumps(current_schema, indent=2)

            con = sqlite3.connect("mash_organizer.sqlite")
            cur = con.cursor()
            res = cur.execute("SELECT llm_plan_json FROM organization_plans ORDER BY id DESC LIMIT 1")
            last_plan = res.fetchone()
            con.close()

            previous_plan_prompt = ""
            if last_plan:
                previous_plan_prompt = f"The previous plan was:\n<PREVIOUS_PLAN>\n{last_plan[0]}\n</PREVIOUS_PLAN>\n\nCritique this plan and improve it. If no improvements are needed, respond with `{{\"status\": \"DONE\"}}`."

            prompt = (
                "You are a database architect iteratively refining a LanceDB organization. "
                f"Current schema:\n<SCHEMA>\n{schema_str}\n</SCHEMA>\n\n"
                f"{previous_plan_prompt}"
                "Propose a new organization scheme by extracting entities and relationships. "
                "Your output must be a JSON object with two keys: "
                "1. `plan`: A human-readable explanation of your proposed changes. "
                "2. `actions`: A list of actions to execute. Valid actions are: "
                "   - `{'operation': 'create_entity_table', 'table_name': '...', 'schema': {...}}` "
                "   - `{'operation': 'extract_and_move_data', 'source_table': '...', 'target_table': '...', 'entity_type': '...'}`"
                "\nIf the current organization is satisfactory, respond with `{\"status\": \"DONE\"}`."
            )
            messages = [{"role": "system", "content": reducer_agent.get("system", "")},
                        {"role": "user", "content": prompt}]
            response_str = openai_chat(reducer_agent, messages)
            print(f"LLM suggestion: {response_str}")

            try:
                response_json = json.loads(response_str)
                if response_json.get("status") == "DONE":
                    print("LLM is satisfied. Organization complete.")
                    break

                plan = response_json.get("plan")
                if plan:
                    print(f"LLM Plan: {plan}")

                actions = response_json.get("actions")
                if actions:
                    save_organization_plan(response_str)
                    print("LLM organization plan has been saved. Executing actions...")
                    execute_db_actions(db, actions)
                else:
                    print("No actions provided by LLM.")

            except json.JSONDecodeError:
                print("Invalid JSON from LLM. Aborting.")
                break
    else:
        raise SystemExit(f"Unknown database action: {action}")

def generate_index_html():
    """Create a sidebar viewer at mash_runs/index.html showing mash + structured reducer JSON."""
    ensure_dir(MASH_DIR)
    sessions = sorted([p for p in MASH_DIR.glob("session_*") if p.is_dir()], reverse=True)
    data = []
    for s in sessions:
        rounds = sorted([p.name for p in s.glob("round_*") if (p / "mash_merged.md").exists()], reverse=True)
        if rounds:
            data.append({"name": s.name, "rounds": rounds})
    reducer_name = CONFIG.get("reducer", {}).get("name", "")
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>AI Mash Viewer</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
  :root { --border:#ddd; }
  body{font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;margin:0;display:flex;height:100vh}
  .sidebar{width:360px;padding:16px;border-right:1px solid var(--border);overflow-y:auto}
  .main{flex-grow:1;padding:16px;overflow-y:auto}
  h1,h2{margin:0 0 8px}
  label{display:block;margin-top:8px;margin-bottom:4px;font-weight:600}
  select,button{font-size:14px;padding:6px 8px;width:100%;margin-bottom:8px}
  #content{white-space:pre-wrap;font-family:Consolas,'Courier New',monospace;line-height:1.5;border:1px solid var(--border);border-radius:6px;padding:12px}
  details{border:1px solid var(--border);border-radius:6px;padding:8px;margin-top:8px;background:#fafafa}
  summary{font-weight:700;cursor:pointer}
  ul{margin:8px 0 0;padding-left:20px}
  .muted{color:#666}
  .tabs{display:flex;border-bottom:1px solid var(--border);margin-bottom:16px}
  .tab-button{background:none;border:none;padding:10px 15px;cursor:pointer;font-size:16px}
  .tab-button.active{border-bottom:2px solid blue}
  .tab-content{display:none}
  .tab-content.active{display:block}
</style>
</head>
<body>
  <div class="sidebar">
    <h1>AI Mash Viewer</h1>
    <label>Session</label>
    <select id="session"></select>
    <label>Round</label>
    <select id="round"></select>
    <button id="load">Load</button>
    <div id="structured-summary"></div>
  </div>
  <div class="main">
    <div class="tabs">
      <button class="tab-button active" data-tab="mash">Mash Content</button>
      <button class="tab-button" data-tab="db">Database</button>
    </div>
    <div id="mash-content" class="tab-content active">
      <h2>Mash Content</h2>
      <div id="content" class="muted">Select a session and round, then click Load.</div>
    </div>
    <div id="db-content" class="tab-content">
      <h2>Database</h2>
      <p>Database Path: <code id="db-path"></code></p>
      <button id="organize-db">Organize Database</button>
      <div id="db-tables"></div>
    </div>
  </div>
<script>
const sessions = %SESSIONS%;
const REDUCER_NAME = %REDUCER_NAME%;
const DB_PATH = %DB_PATH%;
const sessionSel = document.getElementById('session');
const roundSel = document.getElementById('round');
const contentEl = document.getElementById('content');
const structuredEl = document.getElementById('structured-summary');
const loadBtn = document.getElementById('load');
function populateSessions() {
  sessionSel.innerHTML = '';
  sessions.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.name;
    opt.textContent = s.name;
    sessionSel.appendChild(opt);
  });
  sessionSel.addEventListener('change', onSessionChange);
  onSessionChange();
}
function onSessionChange() {
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
  if (!s || !r) {
    contentEl.textContent = 'Please select a valid session and round.';
    return;
  }
  structuredEl.innerHTML = '';
  const url = `${s}/${r}/mash_merged.md`;
  contentEl.textContent = 'Loading ' + url + ' ...';
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(res.status + ' ' + res.statusText);
    const txt = await res.text();
    contentEl.textContent = txt;
  } catch (e) {
    contentEl.textContent = 'Failed to load: ' + e;
  }
  loadStructuredSummary(s, r);
}
async function loadStructuredSummary(session, round) {
  if (!REDUCER_NAME) return;
  const url = `${session}/${round}/responses/_${REDUCER_NAME}_structured.json`;
  try {
    const res = await fetch(url);
    if (!res.ok) return;
    const data = await res.json();
    renderStructured(data);
  } catch(e) {
    console.warn('Could not load structured summary:', e);
  }
}
function titleize(key) {
  return key.replace(/_/g, ' ').replace(/\\b\\w/g, m => m.toUpperCase());
}
function renderStructured(data) {
  let html = '<h2>Consensus Summary</h2>';
  const sections = ['agreements','disagreements','open_questions','next_actions'];
  let hasAny = false;
  sections.forEach(key => {
    if (!data.hasOwnProperty(key)) return;
    const val = data[key];
    if (Array.isArray(val) && val.length > 0) {
      hasAny = true;
      html += `<details open><summary>${titleize(key)}</summary><ul>`;
      val.forEach(item => {
        if (typeof item === 'string') {
          html += `<li>${escapeHtml(item)}</li>`;
        } else {
          html += `<li><pre>${escapeHtml(JSON.stringify(item, null, 2))}</pre></li>`;
        }
      });
      html += '</ul></details>';
    }
  });
  structuredEl.innerHTML = hasAny ? html : '<p class="muted">No structured reducer data found.</p>';
}
function escapeHtml(s) {
  return String(s)
    .replaceAll('&','&amp;')
    .replaceAll('<','&lt;')
    .replaceAll('>','&gt;')
    .replaceAll('"','&quot;')
    .replaceAll("'",'&#39;');
}
document.addEventListener('DOMContentLoaded', () => {
  populateSessions();
  loadBtn.addEventListener('click', loadMash);

  document.getElementById('db-path').textContent = DB_PATH;

  document.querySelectorAll('.tab-button').forEach(button => {
    button.addEventListener('click', () => {
      document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
      button.classList.add('active');
      document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
      document.getElementById(button.dataset.tab + '-content').classList.add('active');
    });
  });

  const organizeBtn = document.getElementById('organize-db');
  const dbTablesDiv = document.getElementById('db-tables');

  organizeBtn.addEventListener('click', async () => {
      organizeBtn.textContent = 'Starting...';
      organizeBtn.disabled = true;
      await fetch('/api/organization/start', { method: 'POST' });
      organizeBtn.textContent = 'Organize Database';
      organizeBtn.disabled = false;
  });

  // Poll for status updates every 5 seconds
  setInterval(async () => {
      const response = await fetch('/api/organization/status');
      const plans = await response.json();

      dbTablesDiv.innerHTML = '<h3>Latest Organization Plans:</h3>';
      if (plans.length === 0) {
          dbTablesDiv.innerHTML += '<p>No organization plans yet.</p>';
      } else {
          plans.forEach(plan => {
              const planDiv = document.createElement('div');
              planDiv.style.border = '1px solid #ccc';
              planDiv.style.padding = '10px';
              planDiv.style.marginBottom = '10px';
              planDiv.innerHTML = `
                  <p><b>Time:</b> ${new Date(plan.time).toLocaleString()} | <b>Status:</b> ${plan.status}</p>
                  <pre><code>${JSON.stringify(JSON.parse(plan.plan), null, 2)}</code></pre>
              `;
              dbTablesDiv.appendChild(planDiv);
          });
      }
  }, 5000);
});
</script>
</body>
</html>
"""
    html = html.replace("%SESSIONS%", json.dumps(data))
    html = html.replace("%REDUCER_NAME%", json.dumps(reducer_name))
    html = html.replace("%DB_PATH%", json.dumps(str(DB_DIR.resolve())))
    write_text(MASH_DIR / "index.html", html)

def cmd_index(_args):
    generate_index_html()
    print(f"Viewer written → {MASH_DIR / 'index.html'}")

def cmd_serve(args):
    generate_index_html()
    from flask import Flask, jsonify, request
    import threading

    app = Flask(__name__, static_folder=str(MASH_DIR), static_url_path='')

    @app.route('/')
    def index():
        return app.send_static_file('index.html')

    @app.route('/api/organization/start', methods=['POST'])
    def start_organization():
        # Run the lengthy organization task in a background thread
        def run_org():
            print("Starting background organization task...")
            # Using Namespace to simulate the `args` object for the cmd
            org_args = argparse.Namespace(action="organize")
            cmd_database(org_args)
            print("Background organization task finished.")

        thread = threading.Thread(target=run_org)
        thread.start()
        return jsonify({"message": "Organization process started."})

    @app.route('/api/organization/status')
    def organization_status():
        # Read the latest plan from the SQLite DB to show status
        con = sqlite3.connect("mash_organizer.sqlite")
        cur = con.cursor()
        res = cur.execute("SELECT created_at, llm_plan_json, status FROM organization_plans ORDER BY id DESC LIMIT 5")
        plans = [{"time": row[0], "plan": row[1], "status": row[2]} for row in res.fetchall()]
        con.close()
        return jsonify(plans)

    port = args.port
    print(f"Serving UI on http://127.0.0.1:{port}")
    app.run(port=port, debug=False)

# --- CLI ---
def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate a multi-AI chorus with provenance, overrides, reducer, export, and web viewer.",
        formatter_class=argparse.RawTextHelpFormatter
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
    p_export = sub.add_parser("export", help="Export artifacts from a round.")
    p_export.add_argument("--mode", choices=["full", "summary", "roadmap", "tasks"], required=True,
                          help="Export mode: full, summary, roadmap, or tasks.")
    p_export.add_argument("--round", type=int, help="Round number (default: latest).")
    p_export.add_argument("--session", help="Session name (default: latest).")
    p_export.set_defaults(func=cmd_export)
    p_resubmit = sub.add_parser("resubmit", help="Export then create a new round using that export.")
    p_resubmit.add_argument("--mode", choices=["full", "summary", "roadmap", "tasks"], required=True,
                            help="Export mode to resubmit: full, summary, roadmap, or tasks.")
    p_resubmit.add_argument("--round", type=int, help="Round number (default: latest).")
    p_resubmit.add_argument("--session", help="Session name (default: latest).")
    p_resubmit.add_argument("prompt", nargs=argparse.REMAINDER, help="Prompt for the new round.")
    p_resubmit.set_defaults(func=cmd_resubmit)
    p_issues = sub.add_parser("issues", help="Generate GitHub CLI issue creation commands from tasks.")
    p_issues.add_argument("--owner-repo", help="GitHub owner/repo (e.g., 'user/repo').")
    p_issues.add_argument("--round", type=int, help="Round number (default: latest).")
    p_issues.add_argument("--session", help="Session name (default: latest).")
    p_issues.set_defaults(func=cmd_issues)
    p_index = sub.add_parser("index", help="Generate mash_runs/index.html viewer.")
    p_index.set_defaults(func=cmd_index)
    p_serve = sub.add_parser("serve", help="Serve mash_runs/ with a tiny HTTP server.")
    p_serve.add_argument("--port", type=int, default=8888)
    p_serve.set_defaults(func=cmd_serve)
    p_db = sub.add_parser("db", help="Manage the LanceDB database.")
    p_db.add_argument("action", choices=["create", "organize"], help="Database action.")
    p_db.set_defaults(func=cmd_database)
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
