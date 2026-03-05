"""Download eval trajectory data from WandB for viewing with `harbor view`.

Supports two modes:
1. harbor-job artifacts (new): Full ATIF trajectories, download directly as Harbor job dirs
2. HTML files (legacy): Parsed from trajectory HTML pages (truncated at 3000 chars)

Usage:
    # Download to Harbor-compatible job directory (viewable with `harbor view`)
    python download_wandb.py

    # Then view with:
    harbor view ./harbor_jobs
"""
import wandb
import json
import os
import re
import uuid
from collections import defaultdict
from pathlib import Path

ENTITY = "alchemxz"
PROJECT = "harbor"
RUN_ID = "5xnvoj3o"  # test-artifacts-0305-0718

OUTPUT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
HARBOR_JOBS_DIR = Path("/Users/lilyzhang/Documents/lilyzhng/RL_PostTrain/harbor/jobs")

# Legacy output for the custom viewer (index.html)
LEGACY_OUTPUT_FILE = OUTPUT_DIR / "data.json"


def download_harbor_job_artifacts(run):
    """Download harbor-job artifacts — these are ready to use with `harbor view`."""
    found_any = False

    for artifact in run.logged_artifacts():
        if artifact.type != "harbor-job":
            continue

        found_any = True
        step = artifact.metadata.get("step", 0)
        job_dir = HARBOR_JOBS_DIR / f"step-{step}"
        job_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading artifact: {artifact.name} -> {job_dir}")
        artifact.download(root=str(job_dir))

        # Create a minimal job config.json and result.json so harbor view recognizes it
        _write_job_metadata(job_dir, run, step)

        # Count what we got
        trials = [d for d in job_dir.iterdir() if d.is_dir() and (d / "agent" / "trajectory.json").exists()]
        print(f"  Step {step}: {len(trials)} trials with ATIF trajectories")

    return found_any


def _write_job_metadata(job_dir, run, step):
    """Write job-level config.json and result.json matching harbor view's expected format."""
    # Scan trial result.json files to extract agent/model info and rewards
    trial_dirs = [d for d in job_dir.iterdir() if d.is_dir() and (d / "result.json").exists()]
    trials_info = []
    agent_name = "terminus-2"
    model_name = "unknown"
    for td in trial_dirs:
        try:
            tr = json.loads((td / "result.json").read_text())
            reward = tr.get("verifier_result", {}).get("rewards", {}).get("reward", 0)
            trial_name = tr.get("trial_name", td.name)
            agent_name = tr.get("config", {}).get("agent", {}).get("name", agent_name)
            model_name = tr.get("config", {}).get("agent", {}).get("model_name", model_name)
            trials_info.append({"trial_name": trial_name, "reward": reward})
        except Exception:
            pass

    config_path = job_dir / "config.json"
    if not config_path.exists():
        config = {
            "job_name": job_dir.name,
            "jobs_dir": str(HARBOR_JOBS_DIR),
            "n_attempts": 1,
            "timeout_multiplier": 1.0,
            "debug": False,
            "orchestrator": {"type": "local", "n_concurrent_trials": 1, "quiet": False, "retry": {"max_retries": 0}, "kwargs": {}},
            "environment": {"type": "docker", "kwargs": {}},
            "verifier": {},
            "metrics": [],
            "agents": [{"name": agent_name, "model_name": model_name}],
            "datasets": [],
            "tasks": [],
            "artifacts": [],
        }
        config_path.write_text(json.dumps(config, indent=4))

    result_path = job_dir / "result.json"
    # Build reward_stats: group trial names by reward value
    reward_stats = defaultdict(list)
    for t in trials_info:
        reward_stats[str(t["reward"])].append(t["trial_name"])

    # Compute mean reward
    rewards = [t["reward"] for t in trials_info]
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    n_errors = sum(1 for t in trials_info if t["reward"] == 0)

    # Build eval key matching harbor's format: agent__model__dataset
    eval_key = f"{agent_name}__{model_name}__adhoc".replace("/", "_")

    result = {
        "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"skyrl-eval-step-{step}-{run.id}")),
        "started_at": run.created_at.replace("Z", "").replace("+00:00", ""),
        "finished_at": run.created_at.replace("Z", "").replace("+00:00", ""),
        "n_total_trials": len(trials_info),
        "stats": {
            "n_trials": len(trials_info),
            "n_errors": n_errors,
            "evals": {
                eval_key: {
                    "n_trials": len(trials_info),
                    "n_errors": n_errors,
                    "metrics": [{"mean": mean_reward}],
                    "reward_stats": {"reward": dict(reward_stats)},
                    "exception_stats": {},
                }
            },
        },
    }
    result_path.write_text(json.dumps(result, indent=4))


# ===== Legacy HTML parsing (for runs before the artifact change) =====

def parse_trajectory_html(html_str):
    """Parse a trajectory HTML file into structured data (legacy)."""
    header_match = re.search(
        r"TrajectoryID\(instance_id='(\d+)',\s*repetition_id=(\d+)\)\s*·\s*(\d+)\s*turns",
        html_str
    )
    reward_match = re.search(r"Reward:\s*([\d.]+)", html_str)

    if not header_match:
        return None

    instance_id = header_match.group(1)
    repetition_id = int(header_match.group(2))
    num_turns = int(header_match.group(3))
    reward = float(reward_match.group(1)) if reward_match else 0.0

    messages = []
    msg_pattern = re.compile(
        r'<div style="font-size:11px;color:#999;margin-bottom:4px">'
        r'[^<]*?(User|Assistant)[^<]*?</div>\s*'
        r'<div style="font-size:13px;line-height:1\.5;white-space:pre-wrap;word-break:break-word">'
        r'(.*?)</div>',
        re.DOTALL
    )

    for match in msg_pattern.finditer(html_str):
        role_text = match.group(1).lower()
        content_html = match.group(2)

        content = content_html
        content = content.replace('<br>', '\n').replace('<br/>', '\n').replace('<br />', '\n')
        content = content.replace('&lt;', '<').replace('&gt;', '>')
        content = content.replace('&amp;', '&').replace('&quot;', '"')
        content = content.replace('&#x27;', "'").replace('&#39;', "'")
        content = content.replace('<think>', '\x00THINK_OPEN\x00').replace('</think>', '\x00THINK_CLOSE\x00')
        content = re.sub(r'<[^>]+>', '', content)
        content = content.replace('\x00THINK_OPEN\x00', '<think>').replace('\x00THINK_CLOSE\x00', '</think>')
        content = content.strip()

        if role_text == "user":
            if messages and messages[-1]["role"] == "assistant" and len(messages) > 1:
                role = "tool"
            else:
                role = "user"
        else:
            role = "assistant"

        messages.append({"role": role, "content": content})

    return {
        "instance_id": instance_id,
        "repetition_id": repetition_id,
        "num_turns": num_turns,
        "reward": reward,
        "messages": messages,
    }


def download_and_build_legacy(run):
    """Download HTML files and build legacy data.json for the custom viewer."""
    files_dir = OUTPUT_DIR / "wandb_files"
    trajectory_files = defaultdict(list)

    for f in run.files():
        if "trajectories" not in f.name or not f.name.endswith(".html"):
            continue

        step_match = re.search(r"trajectories_(\d+)_", f.name)
        if not step_match:
            continue
        step = int(step_match.group(1))

        print(f"Downloading: {f.name} (step {step}, {f.size} bytes)")
        dl = f.download(root=str(files_dir), replace=True)

        with open(dl.name, "r") as fh:
            html = fh.read()

        parsed = parse_trajectory_html(html)
        if parsed:
            trajectory_files[step].append(parsed)
            print(f"  -> instance={parsed['instance_id']}, rep={parsed['repetition_id']}, "
                  f"reward={parsed['reward']}, turns={parsed['num_turns']}, "
                  f"messages={len(parsed['messages'])}")
        else:
            print(f"  -> FAILED to parse")

    # Build legacy data.json
    viewer_data = {"run_name": run.name, "model": "Qwen3-8B", "steps": {}}

    for step in sorted(trajectory_files.keys()):
        trajectories = trajectory_files[step]
        tasks_by_id = defaultdict(list)
        for traj in trajectories:
            tasks_by_id[traj["instance_id"]].append(traj)

        tasks = {}
        for instance_id in sorted(tasks_by_id.keys(), key=int):
            rollouts = sorted(tasks_by_id[instance_id], key=lambda t: t["repetition_id"])
            desc = ""
            if rollouts and rollouts[0]["messages"]:
                desc = rollouts[0]["messages"][0]["content"].split("\n")[0][:120]

            task_key = f"eval-task-{instance_id}"
            tasks[task_key] = {
                "task_name": f"eval-task-{instance_id}",
                "task_description": desc,
                "rollouts": [
                    {
                        "id": f"r{r['repetition_id']}",
                        "reward": r["reward"],
                        "num_turns": r["num_turns"],
                        "stop_reason": "complete",
                        "messages": r["messages"],
                    }
                    for r in rollouts
                ],
            }

        viewer_data["steps"][step] = {
            "label": f"Step {step}" + (" (before training)" if step == 0 else ""),
            "tasks": tasks,
        }

    with open(LEGACY_OUTPUT_FILE, "w") as f:
        json.dump(viewer_data, f, indent=2, ensure_ascii=False)

    print(f"\nWrote legacy {LEGACY_OUTPUT_FILE}")
    for step, data in viewer_data["steps"].items():
        print(f"  Step {step}: {len(data['tasks'])} tasks, "
              f"{sum(len(t['rollouts']) for t in data['tasks'].values())} rollouts")


def main():
    api = wandb.Api()
    run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")
    print(f"Run: {run.name} ({run.id})")

    # Try harbor-job artifacts first (new format)
    print("\n--- Checking for harbor-job artifacts ---")
    if download_harbor_job_artifacts(run):
        print(f"\nDone! View with: harbor view {HARBOR_JOBS_DIR}")
    else:
        # Fall back to legacy HTML parsing
        print("No harbor-job artifacts found, falling back to HTML parsing")
        download_and_build_legacy(run)
        print(f"\nLegacy viewer: open index.html (served at http://localhost:8770)")


if __name__ == "__main__":
    main()
