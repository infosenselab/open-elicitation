"""
Example usage:
    OCIV
    python -m ociv_agents.experiments.trajectory_evaluate \
        --selection_policy ociv \
        --model_choice meta-llama/Llama-3.1-8B-Instruct \
        --topics_file ibm-rank-30k-topics \
        --arguments_file ibm-rank-30k \
        --trajectory_length 100 \
        --num_trajectories 100 \
        --candidate_k 7 \
        --success_prob 0.50 \
        --seed 42 \
        --batch_size 4 \
        --compute_history_ablation_scores \
        --positive_label_only
        
    random
    python -m ociv_agents.experiments.trajectory_evaluate \
        --selection_policy random \
        --topics_file ibm-rank-30k-topics \
        --arguments_file ibm-rank-30k \
        --trajectory_length 100 \
        --num_trajectories 100 \
        --candidate_k 7 \
        --success_prob 0.50 \
        --seed 42
        
        
        
        
"""



# ---------------- Imports ----------------

import argparse
import json
import logging
import os
import random
import time

from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import yaml

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ociv_agents.utils.candidate_scoring import get_label_tokens, process_batch, format_valid_labels
from ociv_agents.utils.utils import read_jsonl #, write_jsonl


# ---------------- Defaults ----------------

BATCH_SIZE = 4

PRIMARY_POSITIVE_LABEL = "Yes"
PRIMARY_NEGATIVE_LABEL = "No"
PRIMARY_UNCERTAIN_LABEL = "Uncertain"

PRIMARY_VALID_LABELS = [
    PRIMARY_POSITIVE_LABEL,
    PRIMARY_NEGATIVE_LABEL,
    PRIMARY_UNCERTAIN_LABEL,
]

ABLATION_POSITIVE_LABEL = "Valuable"
ABLATION_NEGATIVE_LABEL = "Not valuable"
ABLATION_UNCERTAIN_LABEL = "Uncertain"

ABLATION_VALID_LABELS = [
    ABLATION_POSITIVE_LABEL,
    ABLATION_NEGATIVE_LABEL,
    ABLATION_UNCERTAIN_LABEL,
]

# Keep the existing names for the primary trajectory policy.
POSITIVE_LABEL = PRIMARY_POSITIVE_LABEL
NEGATIVE_LABEL = PRIMARY_NEGATIVE_LABEL
UNCERTAIN_LABEL = PRIMARY_UNCERTAIN_LABEL
VALID_LABELS = PRIMARY_VALID_LABELS


TRAJECTORY_LENGTH = 100 # total dialogue turns
NUM_TRAJECTORIES = 100

CANDIDATE_K = 5
HYPOTHESIS_SUCCESS_PROB = 0.50

RANDOM_SEED = 42

DATASET_OBJECTIVE = "Learn about an interview participant's views on common controversial topics"




# ---------------- Args ----------------

parser = argparse.ArgumentParser(
    description="OCIV path-dependent elicitation trajectory simulation."
)
parser.add_argument(
    "--selection_policy",
    type=str,
    choices=["ociv", "random"],
    default="ociv",
    help="Policy used to select a hypothesis from each candidate set.",
)
parser.add_argument(
    "--model_choice",
    type=str,
    default=None,
    help="Observer model. Required when --selection_policy=ociv.",
)
parser.add_argument(
    "--topics_file",
    type=str,
    required=True,
    help="Processed topics filename without .jsonl.",
)
parser.add_argument(
    "--arguments_file",
    type=str,
    required=True,
    help="Processed arguments filename without .jsonl.",
)
parser.add_argument(
    "--candidate_k",
    type=int,
    default=CANDIDATE_K,
    help="Number of hypotheses considered at each elicitation turn.",
)
parser.add_argument(
    "--trajectory_length",
    type=int,
    default=TRAJECTORY_LENGTH,
    help="Maximum total number of dialogue turns per trajectory.",
)
parser.add_argument(
    "--num_trajectories",
    type=int,
    default=NUM_TRAJECTORIES,
    help="Number of Monte Carlo trajectories.",
)
parser.add_argument(
    "--success_prob",
    type=float,
    default=HYPOTHESIS_SUCCESS_PROB,
    help="Probability that an investigated hypothesis is confirmed.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=BATCH_SIZE,
    help="Batch size for OCIV scoring.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=RANDOM_SEED,
    help="Random seed.",
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
        "Valuable/Not valuable/Uncertain labels. "
        "These scores do not affect trajectory selection."
    ),
)

args = parser.parse_args()

SELECTION_POLICY = args.selection_policy
MODEL_CHOICE = args.model_choice
TOPICS_FILE = f"{args.topics_file}.jsonl"
ARGUMENTS_FILE = f"{args.arguments_file}.jsonl"

CANDIDATE_K = args.candidate_k
TRAJECTORY_LENGTH = args.trajectory_length
NUM_TRAJECTORIES = args.num_trajectories
HYPOTHESIS_SUCCESS_PROB = args.success_prob
BATCH_SIZE = args.batch_size
RANDOM_SEED = args.seed
COMPUTE_HISTORY_ABLATION_SCORES = args.compute_history_ablation_scores
POSITIVE_LABEL_ONLY = args.positive_label_only
USE_NEUTRAL_PROMPT = args.use_neutral_prompt
COMPUTE_LABEL_ABLATION_SCORES = args.compute_label_ablation_scores

ACTIVE_TOPIC_CANDIDATES = CANDIDATE_K // 2 + 1
ALTERNATIVE_TOPIC_CANDIDATES = CANDIDATE_K - ACTIVE_TOPIC_CANDIDATES

if SELECTION_POLICY == "ociv":
    if USE_NEUTRAL_PROMPT:
        SYSTEM_PROMPT_FILE = "trajectory_prompt_w_history_neutral.txt"
    else:
        SYSTEM_PROMPT_FILE = "trajectory_prompt_w_history.txt"
else:
    SYSTEM_PROMPT_FILE = None



SCORING_LABELS = (
    [POSITIVE_LABEL]
    if POSITIVE_LABEL_ONLY
    else VALID_LABELS
)

ABLATION_SCORING_LABELS = (
    [ABLATION_POSITIVE_LABEL]
    if POSITIVE_LABEL_ONLY
    else ABLATION_VALID_LABELS
)

# ---------------- Validation ----------------

if TRAJECTORY_LENGTH <= 0:
    raise ValueError("--trajectory_length must be greater than 0.")

if NUM_TRAJECTORIES <= 0:
    raise ValueError("--num_trajectories must be greater than 0.")

if CANDIDATE_K < 2:
    raise ValueError("--candidate_k must be at least 2.")

if BATCH_SIZE <= 0:
    raise ValueError("--batch_size must be greater than 0.")

if not 0.0 <= HYPOTHESIS_SUCCESS_PROB <= 1.0:
    raise ValueError("--success_prob must be between 0 and 1.")


if SELECTION_POLICY == "random" and COMPUTE_HISTORY_ABLATION_SCORES:
    raise ValueError(
        "--compute_history_ablation_scores is not applicable to random selection."
    )

if SELECTION_POLICY == "random" and USE_NEUTRAL_PROMPT:
    raise ValueError("--use_neutral_prompt is not applicable to random selection.")


if SELECTION_POLICY == "random" and POSITIVE_LABEL_ONLY:
    raise ValueError("--positive_label_only is not applicable to random selection.")

if SELECTION_POLICY == "ociv" and MODEL_CHOICE is None:
    raise ValueError("--model_choice is required when --selection_policy=ociv.")

if SELECTION_POLICY == "random" and MODEL_CHOICE is not None:
    raise ValueError("--model_choice must not be provided for random selection.")

if SELECTION_POLICY == "random" and COMPUTE_LABEL_ABLATION_SCORES:
    raise ValueError(
        "--compute_label_ablation_scores is not applicable to random selection."
    )

# ---------------- Random seeds ----------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ---------------- Config ----------------

start_datetime = datetime.now().astimezone()
timestamp = start_datetime.strftime("%Y%m%dt%H%M%S")

with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

PROJ_STORE_PATH = config["paths"]["proj-store"]
MODELS_PATH = config["paths"]["models"]

DATA_FOLDER = os.path.join(PROJ_STORE_PATH, "data")

TOPICS_PATH = os.path.join(DATA_FOLDER, "ibm_rank_30k", "processed", TOPICS_FILE)

ARGUMENTS_PATH = os.path.join(DATA_FOLDER, "ibm_rank_30k", "processed", ARGUMENTS_FILE)



if SELECTION_POLICY == "ociv":
    SYSTEM_PROMPT_PATH = os.path.join("./ociv_agents", "prompts", SYSTEM_PROMPT_FILE)

    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read().strip()
else:
    SYSTEM_PROMPT_PATH = None
    SYSTEM_PROMPT = None


# ---------------- Model paths ----------------

#LLM_PATH = os.path.join(MODELS_PATH, MODEL_CHOICE)

LLM_PATH = (os.path.join(MODELS_PATH, MODEL_CHOICE) if SELECTION_POLICY == "ociv" else None)


# ---------------- Output paths ----------------


run_name = MODEL_CHOICE if SELECTION_POLICY == "ociv" else "random"
OUTPUT_FOLDER = os.path.join(PROJ_STORE_PATH, "experiments", "ociv-trajectory", args.arguments_file, run_name)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

history_ablation_suffix = "-historyablation" if COMPUTE_HISTORY_ABLATION_SCORES else ""
positive_only_suffix = "-positiveonly" if POSITIVE_LABEL_ONLY else ""
neutral_prompt_suffix = "-neutralprompt" if USE_NEUTRAL_PROMPT else ""
policy_suffix = f"-{SELECTION_POLICY}"
label_ablation_suffix = "-labelablation" if COMPUTE_LABEL_ABLATION_SCORES else ""
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, f"{timestamp}-{NUM_TRAJECTORIES}x{TRAJECTORY_LENGTH}-k{CANDIDATE_K}-p{HYPOTHESIS_SUCCESS_PROB}-s{RANDOM_SEED}{history_ablation_suffix}{positive_only_suffix}{neutral_prompt_suffix}{policy_suffix}{label_ablation_suffix}-trajectories.json")

#DIALOGUE_OUTPUT_PATH = OUTPUT_PATH.replace(".json", "-dialogue.json")


# ---------------- Logging ----------------

device = "cuda" if torch.cuda.is_available() else "cpu"

LOGS_DIR_BASE       = os.path.join(PROJ_STORE_PATH, "logs", "experiments", "ociv-trajectory", run_name)

LOGS_DIR            = os.path.join(LOGS_DIR_BASE, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
DIALOGUE_OUTPUT_DIR = os.path.join(LOGS_DIR_BASE, "dialogue-summary")
os.makedirs(DIALOGUE_OUTPUT_DIR, exist_ok=True)


LOG_FILENAME = os.path.join(LOGS_DIR, f"{timestamp}.md")
DIALOGUE_OUTPUT_PATH = os.path.join(DIALOGUE_OUTPUT_DIR, f"{timestamp}-dialogue-summary.json")



logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.FileHandler(LOG_FILENAME)],
    force=True,
)

logger = logging.getLogger(__name__)



# ---------------- Load models ----------------

model = None
tokenizer = None
label_tokens = None
ablation_label_tokens = None


if SELECTION_POLICY == "ociv":

    print(f"Now processing {MODEL_CHOICE}")


    # LLM
    if MODEL_CHOICE.startswith("mistralai/"):
        from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

        tokenizer = MistralCommonBackend.from_pretrained(LLM_PATH, trust_remote_code=True)
        model = Mistral3ForConditionalGeneration.from_pretrained(LLM_PATH, device_map="auto")

    else:
        tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=LLM_PATH, dtype=torch.float16, device_map="auto")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()



    # ---------------- Label tokenization ----------------

    logger.info("# Label Tokenization\n")

    label_tokens = get_label_tokens(
        tokenizer=tokenizer,
        valid_labels=SCORING_LABELS,
        logger=logger,
    )
    if COMPUTE_LABEL_ABLATION_SCORES:
        ablation_label_tokens = get_label_tokens(
            tokenizer=tokenizer,
            valid_labels=ABLATION_SCORING_LABELS,
            logger=logger,
        )
    
else:
    print("Now processing random baseline")



# ---------------- Load topics ----------------

topics = [
    row["topic"]
    for row in read_jsonl(TOPICS_PATH)
]

if not topics:
    raise ValueError(f"No topics found in {TOPICS_PATH}.")

topics = list(dict.fromkeys(topics))


# ---------------- Load arguments ----------------

arguments = []

for argument_id, row in enumerate(read_jsonl(ARGUMENTS_PATH)):
    arguments.append({
        "argument_id": argument_id,
        "argument": row["argument"],
        "original_topic": row["topic"],
        "WA": float(row["WA"]),
    })

if not arguments:
    raise ValueError(f"No arguments found in {ARGUMENTS_PATH}.")

#argument_texts = [row["argument"] for row in arguments]

logger.info(
    "# Dataset\n\n"
    f"- Topics: `{len(topics):,}`  \n"
    f"- Arguments: `{len(arguments):,}`  \n"
)


# ---------------- Argument indices by topic ----------------

indices_by_topic = defaultdict(list)

for idx, row in enumerate(arguments):
    indices_by_topic[row["original_topic"]].append(idx)



# ---------------- Candidate sampling helpers ----------------

def sample_indices(candidate_indices, n):

    candidate_indices = list(candidate_indices)

    if len(candidate_indices) <= n:
        return candidate_indices

    return random.sample(candidate_indices, n)


def sample_from_topic(state, topic, n):


    if n <= 0:
        return []

    available = [
        idx
        for idx in indices_by_topic.get(topic, [])
        if idx in state["available_indices"]
    ]

    return sample_indices(available, n)


def sample_from_distinct_topics(state, n, excluded_topics=None, excluded_indices=None):


    if n <= 0:
        return []

    excluded_topics = set(excluded_topics or [])
    excluded_indices = set(excluded_indices or [])

    topic_candidates = {}

    for topic, topic_indices in indices_by_topic.items():

        if topic in excluded_topics:
            continue

        available = [
            idx
            for idx in topic_indices
            if idx in state["available_indices"]
            and idx not in excluded_indices
        ]

        if available:
            topic_candidates[topic] = available

    candidate_topics = list(topic_candidates.keys())
    random.shuffle(candidate_topics)

    selected = []

    for topic in candidate_topics:
        selected.append(
            random.choice(topic_candidates[topic])
        )

        if len(selected) >= n:
            break

    return selected


def fill_candidate_set(state, selected, target_size):

    if len(selected) >= target_size:
        return selected[:target_size]

    selected = list(selected)
    selected_indices = set(selected)

    remaining = [
        idx
        for idx in state["available_indices"]
        if idx not in selected_indices
    ]

    random.shuffle(remaining)

    needed = target_size - len(selected)
    selected.extend(remaining[:needed])

    return selected


def sample_candidate_set(state):

    active_topic = state["active_topic"]

    if active_topic is None:
        selected = sample_from_distinct_topics(
            state=state,
            n=CANDIDATE_K,
        )

        return fill_candidate_set(
            state,
            selected,
            CANDIDATE_K,
        )

    active_candidates = sample_from_topic(
        state=state,
        topic=active_topic,
        n=ACTIVE_TOPIC_CANDIDATES,
    )

    alternative_candidates = sample_from_distinct_topics(
        state=state,
        n=ALTERNATIVE_TOPIC_CANDIDATES,
        excluded_topics={active_topic},
        excluded_indices=set(active_candidates),
    )

    selected = active_candidates + alternative_candidates

    return fill_candidate_set(state, selected, CANDIDATE_K)



# ---------------- build and format history ----------------

def build_history(state, topic):

    return [
        entry
        for entry in state["history"]
        if entry["topic"] == topic
    ]


def format_history(history):
    """
    Format history entries for insertion into the system prompt.
    """

    return "\n".join(
        f"- {entry['argument']} "
        f"[{'confirmed' if entry['confirmed'] else 'not confirmed'}]"
        for entry in history
    )
    
# ---------------- Hypothesis removal ----------------

def remove_hypothesis(state, selected_idx):
    """
    Remove the selected hypothesis from future consideration.
    """

    state["available_indices"].discard(selected_idx)


# ---------------- Candidate selection ----------------

def choose_hypothesis(candidate_results):
    if SELECTION_POLICY == "random":
        return int(np.random.randint(len(candidate_results)))
    
    scores = np.array([
        float(result["label_probs"][POSITIVE_LABEL])
        for result in candidate_results
    ])

    max_score = scores.max()
    best_positions = np.flatnonzero(np.isclose(scores, max_score))
    selected_position = int(np.random.choice(best_positions))

    return selected_position


# ---------------- Initialize trajectories ----------------

active_trajectories = []

for trajectory_id in range(NUM_TRAJECTORIES):
    active_trajectories.append({
        "trajectory_id": trajectory_id,

        # No topic is active before the first hypothesis is investigated.
        "active_topic": None,

        "available_indices": set(range(len(arguments))),

        "confirmed_hypotheses": 0,
        "unconfirmed_hypotheses": 0,
        "topic_switches": 0,

        "acquired_argument_ids": [],
        "history": [],
        "active_topic_history": [],

        "done": False,
    })


trajectory_buffers = defaultdict(list)


# ---------------- Configuration log ----------------

logger.info(
    "\n# Configuration\n\n"
    f"- **Selection policy:** `{SELECTION_POLICY}`  \n"
    f"- **Model:** `{MODEL_CHOICE if MODEL_CHOICE is not None else 'N/A'}`  \n"
    f"- **Arguments:** `{ARGUMENTS_FILE}`  \n"
    f"- **Topics:** `{TOPICS_FILE}`  \n"
    f"- **Start time:** "
    f"`{start_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')}`  \n"
    f"- **Device:** `{device}`  \n"
    f"- **Trajectories:** `{NUM_TRAJECTORIES}`  \n"
    f"- **Trajectory length:** `{TRAJECTORY_LENGTH}`  \n"
    f"- **Candidate k:** `{CANDIDATE_K}`  \n"
    f"- **Active-topic candidates:** "
    f"`{ACTIVE_TOPIC_CANDIDATES}`  \n"
    f"- **Alternative-topic candidates:** "
    f"`{ALTERNATIVE_TOPIC_CANDIDATES}`  \n"
    f"- **Compute history-ablation scores:** "
    f"`{COMPUTE_HISTORY_ABLATION_SCORES}`  \n"
    f"- **Positive-label-only scoring:** `{POSITIVE_LABEL_ONLY}`  \n"
    f"- **Scoring labels:** `{SCORING_LABELS}`  \n"
    f"- **Hypothesis success probability:** "
    f"`{HYPOTHESIS_SUCCESS_PROB}`  \n"
    f"- **Seed:** `{RANDOM_SEED}`  \n"
    f"- **Dataset objective:** "
    f"`{DATASET_OBJECTIVE}`  \n"
    f"- **Use neutral prompt:** `{USE_NEUTRAL_PROMPT}`  \n"
    f"- **System prompt file:** `{SYSTEM_PROMPT_FILE}`  \n"
    f"- **Compute label-ablation scores:** "
    f"`{COMPUTE_LABEL_ABLATION_SCORES}`  \n"
    f"- **Label-ablation prompt labels:** "
    f"`{ABLATION_VALID_LABELS if COMPUTE_LABEL_ABLATION_SCORES else 'N/A'}`  \n"
    f"- **Label-ablation scoring labels:** "
    f"`{ABLATION_SCORING_LABELS if COMPUTE_LABEL_ABLATION_SCORES else 'N/A'}`  \n"
    f"- **System prompt:**\n"
    f"```\n{SYSTEM_PROMPT}\n```"
)


def build_dialogue_turn(record, turn_id):
    """
    Convert one trajectory step into an elicitor/respondent exchange.
    """

    argument = record["selected_argument"]
    argument = argument[:1].upper() + argument[1:]

    return {
        "turn_id": turn_id,
        "elicitor": {
            "hypothesis": (
                f"Respondent believes: '{record['selected_argument']}'"
            ),
            "hypothesis_topic": record["selected_topic"],
            "hypothesis_score": record.get("selected_positive_prob_with_history"),
            "hypothesis_score_type": (
                "raw_sequence_probability"
                if SELECTION_POLICY == "ociv"
                else None
            ),
            "hypothesis_score_label": (
                POSITIVE_LABEL
                if SELECTION_POLICY == "ociv"
                else None
            ),
            "hypothesis_human_wa": record["selected_human_wa"],
        },
        "respondent": {
            "response": (
                argument
                if record["hypothesis_confirmed"]
                else "[Does not confirm]"
            ),
        },
    }


def build_dialogue_output():
    """
    Build the dialogue-only representation of all trajectories.
    """

    dialogue_output = {
        "metadata": {
            "dataset_objective": DATASET_OBJECTIVE,
            "model": MODEL_CHOICE,
            "candidate_k": CANDIDATE_K,
            #"history_h": HISTORY_H,
            "trajectory_length": TRAJECTORY_LENGTH,
            "num_trajectories": NUM_TRAJECTORIES,
            "success_probability": HYPOTHESIS_SUCCESS_PROB,
            "active_topic_candidates": ACTIVE_TOPIC_CANDIDATES,
            "alternative_topic_candidates": ALTERNATIVE_TOPIC_CANDIDATES,
            "seed": RANDOM_SEED,
            "positive_label": POSITIVE_LABEL,
            "negative_label": NEGATIVE_LABEL,
            "uncertain_label": UNCERTAIN_LABEL,
            "compute_history_ablation_scores": COMPUTE_HISTORY_ABLATION_SCORES,
            "positive_label_only": POSITIVE_LABEL_ONLY,
            "use_neutral_prompt": USE_NEUTRAL_PROMPT,
            "system_prompt_file": SYSTEM_PROMPT_FILE,
            "selection_policy": SELECTION_POLICY,
            "compute_label_ablation_scores": COMPUTE_LABEL_ABLATION_SCORES,
            "label_ablation": (
                {
                    "positive_label": ABLATION_POSITIVE_LABEL,
                    "negative_label": ABLATION_NEGATIVE_LABEL,
                    "uncertain_label": ABLATION_UNCERTAIN_LABEL,
                }
                if COMPUTE_LABEL_ABLATION_SCORES
                else None
            ),
        },
        "dialogues": [],
    }

    for trajectory_id in sorted(trajectory_buffers.keys()):

        turns = []

        turn_id = 0

        for record in trajectory_buffers[trajectory_id]:
            turns.append(
                build_dialogue_turn(
                    record=record,
                    turn_id=turn_id,
                )
            )

            turn_id += 1

        dialogue_output["dialogues"].append({
            "trajectory_id": trajectory_id,
            "turns": turns,
        })

    return dialogue_output


def write_dialogue_output(path):
    """
    Write the dialogue-only trajectory output as formatted JSON.
    """

    dialogue_output = build_dialogue_output()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(dialogue_output, f, ensure_ascii=False, indent=2)


# ---------------- Main simulation ----------------
def main():

    start_time = time.perf_counter()

    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)
    
    for step in tqdm(range(TRAJECTORY_LENGTH), desc="Trajectory steps"):

        batch_states = [state for state in active_trajectories if not state["done"]]

        if not batch_states:
            break

        # ------------------------------------------------------------
        # Construct candidate sets
        # ------------------------------------------------------------

        candidate_sets = {}

        for state in batch_states:

            candidates = sample_candidate_set(state)

            if not candidates:
                state["done"] = True
                continue

            candidate_sets[state["trajectory_id"]] = candidates

        batch_states = [state for state in batch_states if not state["done"]]

        if not batch_states:
            break

        # ------------------------------------------------------------
        # Prepare trajectory candidates for selection
        # ------------------------------------------------------------

        results_by_trajectory = defaultdict(list)

        if SELECTION_POLICY == "random":
            for state in batch_states:
                trajectory_id = state["trajectory_id"]

                for candidate_position, argument_idx in enumerate(
                    candidate_sets[trajectory_id]
                ):
                    argument_row = arguments[argument_idx]
                    candidate_topic = argument_row["original_topic"]

                    results_by_trajectory[trajectory_id].append({
                        "trajectory_id": trajectory_id,
                        "step": step,
                        "candidate_position": candidate_position,
                        "argument_idx": argument_idx,
                        "argument_id": argument_row["argument_id"],
                        "argument": argument_row["argument"],
                        "original_topic": candidate_topic,
                        "evaluation_topic": candidate_topic,
                        "history": build_history(state, candidate_topic),
                    })
        else:        
        
        
            evaluation_rows = []
            evaluation_rows_no_history = []
            evaluation_rows_label_ablation = []


            for state in batch_states:

                trajectory_id = state["trajectory_id"]
                active_topic = state["active_topic"]
                #full_history = list(state["history"])
                #prompt_history = build_history(state)
                #history_text = format_history(prompt_history)

                for candidate_position, argument_idx in enumerate(
                    candidate_sets[trajectory_id]
                ):
                    argument_row = arguments[argument_idx]
                    candidate_topic = argument_row["original_topic"]

                    prompt_history = build_history(state, candidate_topic)
                    history_text = format_history(prompt_history)

                
  

                    row = {
                        "trajectory_id": trajectory_id,
                        "step": step,
                        "candidate_position": candidate_position,
                        "argument_idx": argument_idx,
                        "argument_id": argument_row["argument_id"],
                        "argument": argument_row["argument"],
                        "original_topic": candidate_topic,
                        "evaluation_topic": candidate_topic,
                        "history": prompt_history,
                        "topic_match": (
                            active_topic is not None
                            and candidate_topic == active_topic
                        ),
                    }
                    
                    # ------------------------------------------------------------
                    # Primary labels: Yes / No / Uncertain
                    # ------------------------------------------------------------
                                  
                                        
                    system_instruction = SYSTEM_PROMPT.format(
                        dataset_objective=DATASET_OBJECTIVE.lower(),
                        topic=candidate_topic.lower(),
                        history=history_text,
                        valid_labels=format_valid_labels(VALID_LABELS),
                    )
                    
                    evaluation_rows.append({
                        **row,
                        "system_instruction": system_instruction,
                    })
                    
                    # ------------------------------------------------------------
                    # Label ablation: Valuable / Not valuable / Uncertain
                    # Same candidate and same history.
                    # ------------------------------------------------------------

                    

                    if COMPUTE_LABEL_ABLATION_SCORES:
                        ablation_system_instruction = SYSTEM_PROMPT.format(
                            dataset_objective=DATASET_OBJECTIVE.lower(),
                            topic=candidate_topic.lower(),
                            history=history_text,
                            valid_labels=format_valid_labels(ABLATION_VALID_LABELS),
                        )

                        evaluation_rows_label_ablation.append({
                            **row,
                            "system_instruction": ablation_system_instruction,
                        })
                    
                    

                    # ------------------------------------------------------------
                    # History ablation: primary labels, but no history
                    # ------------------------------------------------------------
                    if COMPUTE_HISTORY_ABLATION_SCORES:
                        system_instruction_no_history = SYSTEM_PROMPT.format(
                            dataset_objective=DATASET_OBJECTIVE.lower(),
                            topic=candidate_topic.lower(),
                            history="",
                            valid_labels=format_valid_labels(VALID_LABELS),
                        )

                        evaluation_rows_no_history.append({
                            **row,
                            "history": [],
                            "system_instruction": system_instruction_no_history,
                        })

            # ------------------------------------------------------------
            # Score candidates with history
            # ------------------------------------------------------------

            scored_rows = []

            for batch_start in range(0, len(evaluation_rows), BATCH_SIZE):
                batch_end = batch_start + BATCH_SIZE
                rows_chunk = evaluation_rows[batch_start:batch_end]

                results_chunk = process_batch(
                    rows=rows_chunk,
                    model=model,
                    tokenizer=tokenizer,
                    valid_labels=SCORING_LABELS,
                    label_tokens=label_tokens,
                    model_choice=MODEL_CHOICE,
                    logger=logger,
                )

                scored_rows.extend(results_chunk)



            # ------------------------------------------------------------
            # Score same candidates with label ablation
            # NOTE: These scores are recorded only. They do not determine which candidate is selected.
            # ------------------------------------------------------------

            scored_rows_label_ablation = []

            if COMPUTE_LABEL_ABLATION_SCORES:

                for batch_start in range(
                    0,
                    len(evaluation_rows_label_ablation),
                    BATCH_SIZE,
                ):
                    batch_end = batch_start + BATCH_SIZE
                    rows_chunk = evaluation_rows_label_ablation[
                        batch_start:batch_end
                    ]

                    results_chunk = process_batch(
                        rows=rows_chunk,
                        model=model,
                        tokenizer=tokenizer,
                        valid_labels=ABLATION_SCORING_LABELS,
                        label_tokens=ablation_label_tokens,
                        model_choice=MODEL_CHOICE,
                        logger=logger,
                    )

                    scored_rows_label_ablation.extend(results_chunk)


            # ------------------------------------------------------------
            # Score same candidates without history
            # ------------------------------------------------------------

            scored_rows_no_history = []
            
            if COMPUTE_HISTORY_ABLATION_SCORES:

                for batch_start in range(0, len(evaluation_rows_no_history), BATCH_SIZE):
                    batch_end = batch_start + BATCH_SIZE
                    rows_chunk = evaluation_rows_no_history[batch_start:batch_end]

                    results_chunk = process_batch(
                        rows=rows_chunk,
                        model=model,
                        tokenizer=tokenizer,
                        valid_labels=SCORING_LABELS,
                        label_tokens=label_tokens,
                        model_choice=MODEL_CHOICE,
                        logger=logger,
                    )

                    scored_rows_no_history.extend(results_chunk)




            # ------------------------------------------------------------
            # Build ablation lookup tables
            # ------------------------------------------------------------

            if COMPUTE_LABEL_ABLATION_SCORES:
                assert len(scored_rows_label_ablation) == len(scored_rows)

                label_ablation_scores = {
                    (
                        result["trajectory_id"],
                        result["candidate_position"],
                    ): result
                    for result in scored_rows_label_ablation
                }
                
                primary_keys = {
                    (
                        result["trajectory_id"],
                        result["candidate_position"],
                    )
                    for result in scored_rows
                }

                assert primary_keys == set(label_ablation_scores.keys())

            if COMPUTE_HISTORY_ABLATION_SCORES:
                assert len(scored_rows_no_history) == len(scored_rows)

                no_history_scores = {
                    (
                        result["trajectory_id"],
                        result["candidate_position"],
                    ): result
                    for result in scored_rows_no_history
                }










            #if COMPUTE_HISTORY_ABLATION_SCORES:
            #    no_history_scores = {
            #        (result["trajectory_id"], result["candidate_position"]): result
            #        for result in scored_rows_no_history
            #    }

            for result in scored_rows:
                
                result["positive_prob_with_history"] = float(result["label_probs"][POSITIVE_LABEL])

                if not POSITIVE_LABEL_ONLY:
                    result["negative_prob_with_history"] = float(result["label_probs"][NEGATIVE_LABEL])
                    result["uncertain_prob_with_history"] = float(result["label_probs"][UNCERTAIN_LABEL])
                
                
                # --------------------------------------------------------
                # Label ablation: same candidate, same history
                # --------------------------------------------------------

                if COMPUTE_LABEL_ABLATION_SCORES:
                    key = (
                        result["trajectory_id"],
                        result["candidate_position"],
                    )
                    ablation_result = label_ablation_scores[key]

                    result["label_ablation"] = {
                        "positive_label": ABLATION_POSITIVE_LABEL,
                        "positive_prob_with_history": float(
                            ablation_result["label_probs"][
                                ABLATION_POSITIVE_LABEL
                            ]
                        ),
                    }

                    if not POSITIVE_LABEL_ONLY:
                        result["label_ablation"].update({
                            "negative_label": ABLATION_NEGATIVE_LABEL,
                            "uncertain_label": ABLATION_UNCERTAIN_LABEL,
                            "negative_prob_with_history": float(
                                ablation_result["label_probs"][
                                    ABLATION_NEGATIVE_LABEL
                                ]
                            ),
                            "uncertain_prob_with_history": float(
                                ablation_result["label_probs"][
                                    ABLATION_UNCERTAIN_LABEL
                                ]
                            ),
                            "predicted_label": ablation_result["predicted_label"],
                        })

                # --------------------------------------------------------
                # History ablation
                # --------------------------------------------------------
                
                
                if COMPUTE_HISTORY_ABLATION_SCORES:
                    key = (result["trajectory_id"], result["candidate_position"])
                    no_history_result = no_history_scores[key]

                    result["positive_prob_without_history"] = float(no_history_result["label_probs"][POSITIVE_LABEL])
                    
                    if not POSITIVE_LABEL_ONLY:
                        result["negative_prob_without_history"] = float(no_history_result["label_probs"][NEGATIVE_LABEL])
                        result["uncertain_prob_without_history"] = float(no_history_result["label_probs"][UNCERTAIN_LABEL])


            results_by_trajectory = defaultdict(list)

            for result in scored_rows:
                results_by_trajectory[
                    result["trajectory_id"]
                ].append(result)

            # Restore candidate order after batching.
            for trajectory_id in results_by_trajectory:
                results_by_trajectory[trajectory_id].sort(
                    key=lambda row: row["candidate_position"]
                )

        # ------------------------------------------------------------
        # Elicitation decision and environment outcome
        # ------------------------------------------------------------

        for state in batch_states:

            trajectory_id = state["trajectory_id"]
            candidate_results = results_by_trajectory[trajectory_id]

            if not candidate_results:
                state["done"] = True
                continue

            active_topic_before = state["active_topic"]

            selected_position = choose_hypothesis(candidate_results)
            selected_result = candidate_results[selected_position]

            selected_idx = selected_result["argument_idx"]
            selected_row = arguments[selected_idx]

            
            if SELECTION_POLICY == "ociv":
                selected_positive_prob_with_history = selected_result["positive_prob_with_history"]
            #else:
            #    selected_positive_prob_with_history = None
    
            #selected_positive_prob_with_history = selected_result["positive_prob_with_history"]
            
            if SELECTION_POLICY == "ociv" and not POSITIVE_LABEL_ONLY:
                selected_negative_prob_with_history = selected_result["negative_prob_with_history"]
                selected_uncertain_prob_with_history = selected_result["uncertain_prob_with_history"]
            

            if SELECTION_POLICY == "ociv" and COMPUTE_HISTORY_ABLATION_SCORES:
                selected_positive_prob_without_history = selected_result["positive_prob_without_history"]
                
                if not POSITIVE_LABEL_ONLY:
                    selected_negative_prob_without_history = selected_result["negative_prob_without_history"]
                    selected_uncertain_prob_without_history = selected_result["uncertain_prob_without_history"]


            # --------------------------------------------------------
            # Environment independently determines whether the
            # participant holds the hypothesized view.
            # --------------------------------------------------------

            success_draw = float(np.random.random())

            hypothesis_confirmed = (success_draw < HYPOTHESIS_SUCCESS_PROB)

            # Once selected, remove the hypothesis from future consideration.
            
            remove_hypothesis(state, selected_idx)

            topic_switched = False

            selected_topic = selected_row["original_topic"]

            if (active_topic_before is not None and selected_topic != active_topic_before):
                state["topic_switches"] += 1
                topic_switched = True

            state["active_topic"] = selected_topic

            if hypothesis_confirmed:
                state["confirmed_hypotheses"] += 1

                state["acquired_argument_ids"].append(selected_row["argument_id"])
            else:
                state["unconfirmed_hypotheses"] += 1
            
            state["history"].append({
                "argument_id": selected_row["argument_id"],
                "argument": selected_row["argument"],
                "topic": selected_row["original_topic"],
                "confirmed": hypothesis_confirmed,
            })

            state["active_topic_history"].append(state["active_topic"])
            

            # --------------------------------------------------------
            # Store full candidate set so later analyses can examine
            # what alternatives were available to the elicitor.
            # --------------------------------------------------------

            candidate_records = []

            for candidate_result in candidate_results:
                candidate_idx = candidate_result["argument_idx"]
                candidate_row = arguments[candidate_idx]
                candidate_topic = candidate_row["original_topic"]

                candidate_record = {
                    "candidate_position": candidate_result["candidate_position"],
                    "argument_id": candidate_row["argument_id"],
                    "argument": candidate_row["argument"],
                    "topic": candidate_topic,
                    "human_wa": float(candidate_row["WA"]),
                    "selected_isfrom_active_topic": (active_topic_before is not None and candidate_topic == active_topic_before),
                    "selected_candidate": candidate_idx == selected_idx,
                    "history": candidate_result["history"],
                }

                if SELECTION_POLICY == "ociv":

                    candidate_record["positive_label"] = POSITIVE_LABEL
                    candidate_record["positive_prob_with_history"] = candidate_result["positive_prob_with_history"]
                    
                    #"negative_label": NEGATIVE_LABEL,
                    #"negative_prob_with_history": (candidate_result["negative_prob_with_history"]),
                    #"predicted_label": candidate_result["predicted_label"],
                    
                
                if SELECTION_POLICY == "ociv" and not POSITIVE_LABEL_ONLY:
                    candidate_record["negative_label"] = NEGATIVE_LABEL
                    candidate_record["uncertain_label"] = UNCERTAIN_LABEL
                    candidate_record["negative_prob_with_history"] = candidate_result["negative_prob_with_history"]
                    candidate_record["uncertain_prob_with_history"] = candidate_result["uncertain_prob_with_history"]
                    candidate_record["predicted_label"] = candidate_result["predicted_label"]
                                
                
                if SELECTION_POLICY == "ociv" and COMPUTE_HISTORY_ABLATION_SCORES:
                    candidate_record["positive_prob_without_history"] = (candidate_result["positive_prob_without_history"])
                    
                    if not POSITIVE_LABEL_ONLY:
                        candidate_record["negative_prob_without_history"] = (candidate_result["negative_prob_without_history"])
                        candidate_record["uncertain_prob_without_history"] = candidate_result["uncertain_prob_without_history"]
                
                if SELECTION_POLICY == "ociv" and COMPUTE_LABEL_ABLATION_SCORES:
                    candidate_record["label_ablation"] = candidate_result["label_ablation"]

                candidate_records.append(candidate_record)
                
            record = {
                "step": step,
    
                "candidate_k_requested": CANDIDATE_K,
                "candidate_k_actual": len(candidate_results),
                "candidates": candidate_records,


                "active_topic_before": active_topic_before,
                "active_topic_after": state["active_topic"],
                "selected_argument_id": selected_row["argument_id"],
                "selected_argument": selected_row["argument"],
                "selected_topic": selected_row["original_topic"],
                "selected_human_wa": float(selected_row["WA"]),
                

                "success_probability": HYPOTHESIS_SUCCESS_PROB,
                "success_draw": success_draw,
                "hypothesis_confirmed": hypothesis_confirmed,

                "information_acquired": hypothesis_confirmed,
                "topic_switched": topic_switched,

                "remaining_arguments": len(state["available_indices"]),

                "confirmed_hypotheses": (state["confirmed_hypotheses"]),
                "unconfirmed_hypotheses": (state["unconfirmed_hypotheses"]),
                "topic_switches": state["topic_switches"],

            }

            if SELECTION_POLICY == "ociv":
                record["selected_positive_prob_with_history"] = selected_positive_prob_with_history
                #"selected_negative_prob_with_history": (selected_negative_prob_with_history),
                                    

            if SELECTION_POLICY == "ociv" and not POSITIVE_LABEL_ONLY:
                record["selected_negative_prob_with_history"] = selected_negative_prob_with_history
                record["selected_uncertain_prob_with_history"] = selected_uncertain_prob_with_history
    
            if SELECTION_POLICY == "ociv" and COMPUTE_HISTORY_ABLATION_SCORES:
                record["selected_positive_prob_without_history"] = (selected_positive_prob_without_history)
                #record["selected_negative_prob_without_history"] = (selected_negative_prob_without_history)
                
                if not POSITIVE_LABEL_ONLY:
                    record["selected_negative_prob_without_history"] = selected_negative_prob_without_history
                    record["selected_uncertain_prob_without_history"] = selected_uncertain_prob_without_history

            trajectory_buffers[trajectory_id].append(record)

            if not state["available_indices"]:
                state["done"] = True

        logger.info(
            f"Completed step {step+1}. "
            f"Active trajectories: {sum(not s['done'] for s in active_trajectories)}\n\n"
        )

    # ------------------------------------------------------------
    # Write trajectories in trajectory/step order
    # ------------------------------------------------------------

    # JSONL
    #for trajectory_id in sorted(trajectory_buffers.keys()):
    #    for record in trajectory_buffers[trajectory_id]:
    #        write_jsonl(OUTPUT_PATH, record)
    
    # JSON
    trajectory_output = {
        "params": {
            "model": MODEL_CHOICE,
            "trajectory_length": TRAJECTORY_LENGTH,
            "num_trajectories": NUM_TRAJECTORIES,
            "candidate_k": CANDIDATE_K,
            "success_probability": HYPOTHESIS_SUCCESS_PROB,
            "active_topic_candidates": ACTIVE_TOPIC_CANDIDATES,
            "alternative_topic_candidates": ALTERNATIVE_TOPIC_CANDIDATES,
            "seed": RANDOM_SEED,
            "positive_label": POSITIVE_LABEL,
            "negative_label": NEGATIVE_LABEL,
            "uncertain_label": UNCERTAIN_LABEL,
            "compute_history_ablation_scores": COMPUTE_HISTORY_ABLATION_SCORES,
            "positive_label_only": POSITIVE_LABEL_ONLY,
            "use_neutral_prompt": USE_NEUTRAL_PROMPT,
            "system_prompt_file": SYSTEM_PROMPT_FILE,
            "selection_policy": SELECTION_POLICY,
            "compute_label_ablation_scores": COMPUTE_LABEL_ABLATION_SCORES,
            "label_ablation": (
                {
                    "positive_label": ABLATION_POSITIVE_LABEL,
                    "negative_label": ABLATION_NEGATIVE_LABEL,
                    "uncertain_label": ABLATION_UNCERTAIN_LABEL,
                }
                if COMPUTE_LABEL_ABLATION_SCORES
                else None
            ),
        },
        "dataset_objective": DATASET_OBJECTIVE,
        "trajectories": [],
    }

    for trajectory_id in sorted(trajectory_buffers.keys()):
        trajectory_output["trajectories"].append({
            "trajectory_id": trajectory_id,
            "steps": trajectory_buffers[trajectory_id],
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(trajectory_output, f, ensure_ascii=False, indent=2)



    # ------------------------------------------------------------
    # Write dialogue-only output
    # ------------------------------------------------------------

    write_dialogue_output(DIALOGUE_OUTPUT_PATH)


    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    final_successes = []
    final_failures = []
    final_switches = []
    unique_topics_visited = []

    for state in active_trajectories:
        final_successes.append(state["confirmed_hypotheses"])
        final_failures.append(state["unconfirmed_hypotheses"])
        final_switches.append(state["topic_switches"])

        visited = {
            topic
            for topic in state["active_topic_history"]
            if topic is not None
        }

        unique_topics_visited.append(len(visited))

    elapsed_time = time.perf_counter() - start_time

    days = int(elapsed_time // 86400)
    hours = int((elapsed_time % 86400) // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    logger.info(
        "\n# Summary\n\n"
        f"- **Trajectories:** `{NUM_TRAJECTORIES}`  \n"
        f"- **Mean confirmed hypotheses:** "
        f"`{np.mean(final_successes):.2f}`  \n"
        f"- **Mean unconfirmed hypotheses:** "
        f"`{np.mean(final_failures):.2f}`  \n"
        f"- **Mean topic switches:** "
        f"`{np.mean(final_switches):.2f}`  \n"
        f"- **Mean unique topics visited:** "
        f"`{np.mean(unique_topics_visited):.2f}`  \n"
        f"- **Elapsed time:** "
        f"`{days}d {hours}h {minutes}m {seconds:.2f}s`  \n"
        f"- **Output:** `{OUTPUT_PATH}`\n"
    )

    print(f"Done. Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()



