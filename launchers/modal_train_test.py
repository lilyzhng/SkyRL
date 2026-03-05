"""
Minimal Modal script to test SkyRL + Harbor training pipeline.
Runs generation-only mode with 1 sample on Qwen2.5-1.5B-Instruct.
"""

import modal
import os

# --- Modal setup ---
app = modal.App("skyrl-harbor-test")

# Persistent volume for model weights (avoid re-downloading)
model_volume = modal.Volume.from_name("skyrl-model-cache", create_if_missing=True)

# Build the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install("uv")
    # Install SkyRL with fsdp + harbor extras
    .run_commands(
        "uv pip install --system 'skyrl[fsdp] @ git+https://github.com/NovaSky-AI/SkyRL'",
        "uv pip install --system 'harbor @ git+https://github.com/laude-institute/harbor@8c040e1bb010201fd3c75bee3dede2407b9f57cd'",
    )
    # Pre-download the small model
    .run_commands(
        "python -c \"from transformers import AutoTokenizer, AutoModelForCausalLM; "
        "AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'); "
        "AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')\"",
    )
)

# Upload the dataset and SkyRL examples
skyrl_mount = modal.Mount.from_local_dir(
    "/Users/lilyzhang/Documents/lilyzhng/RL_PostTrain/SkyRL/examples",
    remote_path="/root/SkyRL/examples",
)

# Upload just 1 task from the dataset
dataset_mount = modal.Mount.from_local_dir(
    os.path.expanduser("~/data/harbor/CodeContests/code_contests-0000"),
    remote_path="/root/data/harbor/CodeContests/code_contests-0000",
)


@app.function(
    image=image,
    gpu="H100:1",  # Start with 1 GPU for testing
    timeout=1800,
    mounts=[skyrl_mount, dataset_mount],
    volumes={"/root/.cache": model_volume},
    secrets=[
        modal.Secret.from_dict({
            "DAYTONA_API_KEY": "dtn_41c7109e106fd349b61ee8c9258e427c69a5e514143743b2399c1d0e00909db8",
            "WANDB_API_KEY": "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",
        }),
    ],
)
def test_pipeline():
    """Test the SkyRL + Harbor pipeline with 1 sample."""
    import subprocess
    import sys

    # Verify GPU is available
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    print("=== GPU Info ===")
    print(result.stdout[:500])

    # Verify dependencies
    print("\n=== Checking dependencies ===")
    subprocess.run([sys.executable, "-c", "import skyrl; print(f'SkyRL: {skyrl.__version__}')"], check=True)
    subprocess.run([sys.executable, "-c", "import harbor; print('Harbor: OK')"], check=True)
    subprocess.run([sys.executable, "-c", "import vllm; print(f'vLLM: {vllm.__version__}')"], check=True)

    # Verify dataset
    print("\n=== Dataset ===")
    import os
    task_dir = "/root/data/harbor/CodeContests/code_contests-0000"
    print(f"Task dir exists: {os.path.exists(task_dir)}")
    print(f"Files: {os.listdir(task_dir)}")

    # Read the task instruction
    with open(f"{task_dir}/instruction.md") as f:
        print(f"Instruction preview: {f.read()[:200]}")

    print("\n=== Pipeline test complete ===")
    print("Dependencies OK, GPU available, dataset loaded.")
    print("Next step: run full generation with run_harbor_gen.sh")


@app.local_entrypoint()
def main():
    print("Launching SkyRL + Harbor test on Modal...")
    test_pipeline.remote()
    print("Done!")
