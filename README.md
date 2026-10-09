# On Open-Ended Information Seeking for Information Elicitation Agents

## Abstract

Information elicitation is an open-ended information-seeking problem in which an interaction can unfold in many potentially valuable directions, requiring an elicitor to continually determine which information to pursue as new information emerges. In agentic elicitation, these decisions may be delegated to a foundation model, yet how model choice shapes the resulting information-seeking behavior remains understudied. We study how judgments about information value vary across LLMs and how these differences shape sequential information seeking. We first examine these judgments across 11 LLMs spanning multiple model families and parameter scales, using a shared set of information and elicitation objectives. We then develop a controlled elicitation simulation in which different models encounter the same information space and use the same selection rule, isolating these judgments from question generation and respondent behavior. Using this setting, we characterize the breadth--depth behavior that emerges from model-specific information-seeking preferences over the course of elicitation. We further examine how interaction history changes the evaluation and subsequent selection of prospective information. We test the robustness and boundaries of these findings through sensitivity analyses and ablations over the opportunities available to the elicitor, the response labels used to operationalize information-seeking preferences, the presence of interaction history, and whether redundancy is explicitly relevant to the assessment.


## Repository Structure

- `config/` - configuration files
- `open_elicitation/` — project source code
  - `evaluation/` - model evaluation and scoring
  - `prompts/` - prompts used for experiments
  - `experiments/` - model runs
  - `utils/` — support tools
- `datataset/` - contains the processed dataset used in the experiments.


For the trajectory files, click [here](output/).




## Getting Started

Create a file `config/config.yaml` with data and models paths as shown below.

```yaml
paths:
  proj-store: "/data/to/data/" # Actual path to data
  models: "/data/to/models" # Actual path to models
```

For evaluations with multiple models, complete the group required by the file in [here](open_elicitation/evaluation/result_groups).

## Typical Pipeline

- Run trajectory simulation: `open_elicitation/experiments/trajectory_evaluate.py`.
- Run diagnostic evaluations. E.g., `open_elicitation/evaluation/trajectory_evaluate_wa_metrics.ipynb`.

See each file for sample usage instructions listed at the top.




## Citing this work.

If you use this resource in your projects, please cite the following paper.

```bibtex
@misc{de_lima_open-ended_2026,
author = {De Lima, Victor and Yang, Grace Hui},
doi = {10.48550/arXiv.2610.07509},
title = {{On Open-Ended Information Seeking for Information Elicitation Agents}},
url = {https://arxiv.org/abs/2610.07509},
year = {2026}
}
```
