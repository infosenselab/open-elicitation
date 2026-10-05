"""
Example usage:
    ociv
    python -m ociv_agents.experiments.llm_consistency  \
    --model_choice meta-llama/Llama-3.1-8B-Instruct \
    --topics_file ibm-rank-30k-topics \
    --arguments_file ibm-rank-30k \
    --use_sampling 2 \
    --no_uncertain_label \
    --batch_size 4
    
    similarity
        python -m ociv_agents.experiments.llm_consistency \
        --model_choice cross-encoder/stsb-roberta-large \
        --topics_file ibm-rank-30k-topics \
        --arguments_file ibm-rank-30k \
        --use_sampling 2 \
        --semantic_similarity \
        --batch_size 32
"""


# ---------------- Imports ----------------
import logging
import os
import json
import time
import argparse
import random

from datetime import datetime

from tqdm import tqdm
import yaml
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from ociv_agents.utils.candidate_scoring import get_label_tokens, process_batch, format_valid_labels
from ociv_agents.utils.utils import read_jsonl, write_jsonl





# ---------------- Args ----------------

RANDOM_SEED = 42
SYSTEM_PROMPT_FILE = "trajectory_prompt_w_history.txt"

POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"
UNCERTAIN_LABEL = "Uncertain"
#VALID_LABELS = [POSITIVE_LABEL, NEGATIVE_LABEL, UNCERTAIN_LABEL]

DATASET_OBJECTIVE = (
    "Learn about an interview participant's views on common controversial topics"
)

parser = argparse.ArgumentParser(description="Training script with model choice")
parser.add_argument(
    "--model_choice",
    type=str,
    required=True,
    help="Fine-tuning model name (e.g. meta-llama/Llama-3.2-3B-Instruct). Must be in the models path specified in the config.yaml file."
)
parser.add_argument(
    "--topics_file",
    type=str,
    required=True,
    help="Topics to use."
)
parser.add_argument(
    "--arguments_file",
    type=str,
    required=True,
    help="Arguments to use."
)
parser.add_argument(
    "--use_sampling",
    type=int,
    default=None,
    metavar="N",
    help="Sample N arguments per topic instead of using all arguments."
)
parser.add_argument(
    "--no_uncertain_label",
    action="store_true",
    help="Ablate the Uncertain response label and score only Yes and No.",
)
parser.add_argument(
    "--semantic_similarity",
    action="store_true",
    help="Evaluate argument-topic semantic similarity instead of OCIV.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=2,
    help="Number of evaluations to process per batch.",
)

args = parser.parse_args()
MODEL_CHOICE = args.model_choice
TOPICS_FILE =  f"{args.topics_file}.jsonl"
ARGUMENTS_FILE =  f"{args.arguments_file}.jsonl"
SAMPLES_PER_TOPIC = args.use_sampling
USE_SAMPLING = SAMPLES_PER_TOPIC is not None
NO_UNCERTAIN_LABEL = args.no_uncertain_label
SEMANTIC_SIMILARITY = args.semantic_similarity
BATCH_SIZE = args.batch_size


if NO_UNCERTAIN_LABEL:
    VALID_LABELS = [POSITIVE_LABEL, NEGATIVE_LABEL]
else:
    VALID_LABELS = [POSITIVE_LABEL, NEGATIVE_LABEL, UNCERTAIN_LABEL]
    
IS_SIMILARITY_MODEL = MODEL_CHOICE.startswith("cross-encoder/")

print(f"Now processing {MODEL_CHOICE}")

# Val
if USE_SAMPLING and SAMPLES_PER_TOPIC < 1:
    parser.error("--use_sampling must be a positive integer.")

if SEMANTIC_SIMILARITY != IS_SIMILARITY_MODEL:
    parser.error(
        "--semantic_similarity must be used with a cross-encoder model, "
        "and cross-encoder models require --semantic_similarity."
    )

if SEMANTIC_SIMILARITY and NO_UNCERTAIN_LABEL:
    parser.error(
        "--no_uncertain_label cannot be used with --semantic_similarity."
    )

if BATCH_SIZE < 1:
    parser.error("--batch_size must be a positive integer.")
    
# ---------------- Config ----------------
start_datetime = datetime.now().astimezone()
timestamp = start_datetime.strftime("%Y%m%dt%H%M%S")

with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

PROJ_STORE_PATH = os.path.join(config["paths"]["proj-store"])
MODELS_PATH = os.path.join(config["paths"]["models"])

DATA_FOLDER = os.path.join(PROJ_STORE_PATH, "data")

TOPICS_PATH = os.path.join(DATA_FOLDER, "ibm_rank_30k", "processed", TOPICS_FILE)
ARGUMENTS_PATH = os.path.join(DATA_FOLDER, "ibm_rank_30k", "processed", ARGUMENTS_FILE)

arguments_output_name = args.arguments_file

if USE_SAMPLING:
    arguments_output_name = f"{arguments_output_name}-samp{SAMPLES_PER_TOPIC}"

if NO_UNCERTAIN_LABEL:
    arguments_output_name = f"{arguments_output_name}-no-uncertain"

OUTPUT_FOLDER = os.path.join(PROJ_STORE_PATH, "experiments", "llm-consistency", arguments_output_name, MODEL_CHOICE)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, f"{timestamp}-llm-consistency.jsonl")


SYSTEM_PROMPT_PATH = os.path.join("./ociv_agents", "prompts", SYSTEM_PROMPT_FILE)
with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()


# Models
CHOSEN_MODEL_PATH = os.path.join(MODELS_PATH, MODEL_CHOICE)


device = "cuda" if torch.cuda.is_available() else "cpu"


# Logging
LOGS_DIR = os.path.join(PROJ_STORE_PATH, "logs", "experiments", "llm-consistency", MODEL_CHOICE)
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILENAME = os.path.join(LOGS_DIR, f"{timestamp}.md")

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_FILENAME),
    ],
    force=True,
)

logger = logging.getLogger(__name__)


logger.info(
    "# Configuration\n\n"
    f"- **Model:** `{MODEL_CHOICE}`  \n"
    f"- **Arguments:** `{ARGUMENTS_FILE}`  \n"
    f"- **Topics:** `{TOPICS_FILE}`  \n"
    f"- **Sampling:** `{USE_SAMPLING}`  \n"
    f"- **Samples per topic:** `{SAMPLES_PER_TOPIC}`  \n"
    f"- **Start Time:** `{start_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')}`\n"
    f"- **Device:** `{device}`  \n"
    f"- Batch size: `{BATCH_SIZE}`\n"
    f"- **Semantic similarity:** `{SEMANTIC_SIMILARITY}`  \n"
)
if not SEMANTIC_SIMILARITY:
    logger.info(
        f"- **Valid labels:** `{VALID_LABELS}`  \n"
        f"- **No Uncertain ablation:** `{NO_UNCERTAIN_LABEL}`  \n"
        f"- System Prompt:\n"
        f"```\n"
        f"{SYSTEM_PROMPT}\n"
        f"```"
    )


# ---------------- Load model ----------------

if SEMANTIC_SIMILARITY:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(CHOSEN_MODEL_PATH, device=device)
    tokenizer = None
    label_tokens = None
    
    
elif MODEL_CHOICE.startswith("mistralai/"):
    from transformers import Mistral3ForConditionalGeneration,  MistralCommonBackend
    
    tokenizer = MistralCommonBackend.from_pretrained(CHOSEN_MODEL_PATH, trust_remote_code=True)
    
    model = Mistral3ForConditionalGeneration.from_pretrained(
        CHOSEN_MODEL_PATH,
        device_map="auto",
    )
    
    

else:
    tokenizer = AutoTokenizer.from_pretrained(CHOSEN_MODEL_PATH, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=CHOSEN_MODEL_PATH,
        dtype=torch.float16,
        device_map="auto"
    )

if not SEMANTIC_SIMILARITY:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()


    # ---------------- label tokenization ----------------

    logger.info(f"### Label Tokenization Check")

    label_tokens = get_label_tokens(tokenizer=tokenizer, valid_labels=VALID_LABELS, logger=logger)




# ---------------- Load topics ----------------

topics = [
    record["topic"]
    for record in read_jsonl(TOPICS_PATH)
]

if not topics:
    raise ValueError(f"No topics found in {TOPICS_PATH}.")



logger.info(
    f"\n### Load topics.\n\n"
    f"Loaded {len(topics):,} topics\n"
)



# ---------------- Load arguments ----------------

argument_rows = list(read_jsonl(ARGUMENTS_PATH))

if USE_SAMPLING:
    arguments_by_topic = {}

    for argument_id, argument_row in enumerate(argument_rows):
        topic = argument_row["topic"]
        arguments_by_topic.setdefault(topic, []).append(
            {
                "argument_id": argument_id,
                **argument_row,
            }
        )

    rng = random.Random(RANDOM_SEED)
    sampled_arguments = []

    for topic in topics:
        topic_arguments = arguments_by_topic.get(topic, [])

        if len(topic_arguments) < SAMPLES_PER_TOPIC:
            raise ValueError(
                f"Topic '{topic}' has only {len(topic_arguments)} arguments; "
                f"cannot sample {SAMPLES_PER_TOPIC}."
            )

        sampled_arguments.extend(
            rng.sample(topic_arguments, SAMPLES_PER_TOPIC)
        )

    evaluation_arguments = sampled_arguments

    logger.info(
        f"\n### Sample arguments.\n\n"
        f"- Arguments per topic: `{SAMPLES_PER_TOPIC}`  \n"
        f"- Topics: `{len(topics):,}`  \n"
        f"- Sampled arguments: `{len(evaluation_arguments):,}`  \n"
        f"- Random seed: `{RANDOM_SEED}`\n"
    )

else:
    evaluation_arguments = [
        {
            "argument_id": argument_id,
            **argument_row,
        }
        for argument_id, argument_row in enumerate(argument_rows)
    ]

    logger.info(
        f"\n### Load arguments.\n\n"
        f"- Sampling: `False`  \n"
        f"- Arguments: `{len(evaluation_arguments):,}`\n"
    )

# ---------------- Create argument x topic evaluations ----------------

def iter_evaluations():
    """
    Yield every argument evaluated under every topic.
    """

    for argument_row in evaluation_arguments:
        for evaluation_topic in topics:
            
            topic = evaluation_topic.strip().rstrip(".?!")

            if topic:
                topic = topic[0].lower() + topic[1:]

            system_instruction = SYSTEM_PROMPT.format(
                dataset_objective=DATASET_OBJECTIVE.lower(),
                topic=topic,
                history="",
                valid_labels=format_valid_labels(VALID_LABELS),
            )
            
            
            yield {
                "argument_id": argument_row["argument_id"],
                "argument": argument_row["argument"],
                "original_topic": argument_row["topic"],
                "evaluation_topic": evaluation_topic,
                "system_instruction": system_instruction,
                "topic_match": evaluation_topic == argument_row["topic"],
                "set": argument_row.get("set"),
                "WA": argument_row["WA"],
            }

def evaluate_batch(batch):
    if SEMANTIC_SIMILARITY:
        pairs = [
            (row["argument"], row["evaluation_topic"])
            for row in batch
        ]
        scores = model.predict(pairs)

        results = []

        for row, score in zip(batch, scores):
            result = dict(row)
            result.pop("system_instruction", None)
            result["semantic_similarity"] = float(score)
            result["model_choice"] = MODEL_CHOICE
            results.append(result)
            
        return results

    return process_batch(
        rows=batch,
        model=model,
        tokenizer=tokenizer,
        valid_labels=VALID_LABELS,
        label_tokens=label_tokens,
        model_choice=MODEL_CHOICE,
        logger=logger,
    )
# ---------------- Main ----------------

def main():


    start_time = time.perf_counter()

    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)



    num_arguments = len(evaluation_arguments)
    total_evaluations = num_arguments * len(topics)
    
    batch = []
    total = 0


    logger.info(f"\n# Run\n")
    
    with tqdm(total=total_evaluations, desc="Evaluations", unit="eval") as pbar:
        for row in iter_evaluations():

            batch.append(row)

            if len(batch) >= BATCH_SIZE:

                results = evaluate_batch(batch)

                for result in results:
                    write_jsonl(OUTPUT_PATH, result)
                
                total += len(results)
                pbar.update(len(results))

                #print(f"Processed {total:,}")
                batch = []

        # Remaining examples
        if batch:
            results = evaluate_batch(batch)

            for result in results:
                write_jsonl(OUTPUT_PATH, result)
                
            total += len(results)
            pbar.update(len(results))




    # Summary
    
    #end-of-run count check:

    assert total == total_evaluations, (
        f"Expected {total_evaluations:,} evaluations, got {total:,}."
    )

    elapsed_time = time.perf_counter() - start_time



    days = int(elapsed_time // 86400)
    hours = int((elapsed_time % 86400) // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    logger.info(
        f"\n# Summary\n\n"
        f"- **Evaluations:** `{total:,}`  \n"
        f"- **Topics:** `{len(topics):,}`  \n"
        f"- **Elapsed time:** `{days}d {hours}h {minutes}m {seconds:.2f}s`  \n"
        f"- **Output:** `{OUTPUT_PATH}`\n"
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
