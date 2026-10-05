"""
Example usage:
    python -m ociv_agents.experiments.hyperparam_loop \
        --model_choice meta-llama/Llama-3.1-8B-Instruct \
        --batch_size 4 \
        --compute_history_ablation_scores \
        --positive_label_only \
        --use_neutral_prompt
    
    python -m ociv_agents.experiments.hyperparam_loop \
        --model_choice meta-llama/Llama-3.1-8B-Instruct \
        --batch_size 4 \
        --compute_label_ablation_scores \
        --positive_label_only
    
"""

# ---------------- Imports ----------------

import itertools
import subprocess
import argparse


# ---------------- Args ----------------

K_VALUES = [7]
SEEDS = [42, 43, 44]

BATCH_SIZE = 4


parser = argparse.ArgumentParser(
    description="OCIV path-dependent elicitation trajectory simulation."
)
parser.add_argument(
    "--model_choice",
    type=str,
    required=True,
    help="Observer model, e.g. meta-llama/Llama-3.1-8B-Instruct.",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=BATCH_SIZE,
    help="Batch size for OCIV scoring.",
)
parser.add_argument(
    "--compute_history_ablation_scores",
    action="store_true",
    help="Also score each candidate without dialogue history.",
)
parser.add_argument(
    "--positive_label_only",
    action="store_true",
    help="Score only the positive label instead of all valid labels.",
)
parser.add_argument(
    "--use_neutral_prompt",
    action="store_true",
    help="Use the neutral trajectory prompt instead of the standard prompt.",
)
parser.add_argument(
    "--compute_label_ablation_scores",
    action="store_true",
    help=(
        "Also score each candidate using the alternative "
        "Valuable/Not valuable/Uncertain labels."
    ),
)

args = parser.parse_args()

MODEL = args.model_choice
BATCH_SIZE = args.batch_size
COMPUTE_HISTORY_ABLATION_SCORES = args.compute_history_ablation_scores
POSITIVE_LABEL_ONLY = args.positive_label_only
USE_NEUTRAL_PROMPT = args.use_neutral_prompt
COMPUTE_LABEL_ABLATION_SCORES = args.compute_label_ablation_scores

configs = list(itertools.product(
    K_VALUES,
    SEEDS,
))

print(f"Running {len(configs)} configurations")

for i, (k, seed) in enumerate(configs, start=1):
    print(
        f"\n[{i}/{len(configs)}] "
        f"k={k}, seed={seed}"
    )

    command = [
        "python",
        "-m",
        "ociv_agents.experiments.trajectory_evaluate",
        "--model_choice", MODEL,
        "--topics_file", "ibm-rank-30k-topics",
        "--arguments_file", "ibm-rank-30k",
        "--trajectory_length", "100",
        "--num_trajectories", "100",
        "--candidate_k", str(k),
        "--success_prob", "0.50",
        "--seed", str(seed),
        "--batch_size", str(BATCH_SIZE),
    ]
    if COMPUTE_HISTORY_ABLATION_SCORES:
        command.append("--compute_history_ablation_scores")
    if POSITIVE_LABEL_ONLY:
        command.append("--positive_label_only")
    if USE_NEUTRAL_PROMPT:
        command.append("--use_neutral_prompt")
    if COMPUTE_LABEL_ABLATION_SCORES:
        command.append("--compute_label_ablation_scores")

    subprocess.run(command, check=True)

print("\nAll runs complete.")
