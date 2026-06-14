# AgentCV — Autonomous Computer Vision Solution Development

AgentCV is a multi-agent framework for iterative, automated development of computer vision solutions. Given a natural-language task description and a labelled dataset, the system generates, executes, evaluates, and refines Python code through a closed-loop pipeline orchestrated by [LangGraph](https://github.com/langchain-ai/langgraph). An optional **label-free improvement mode** replaces ground-truth validation with an LLM-as-judge, enabling unsupervised iterative refinement.

---

## Pipeline Overview

```
DataPreprocessor → DataAnalyser → [DatasetEnricher*] → [DLModelTrainer*]
    → Programmer → Runner → Evaluator → ImprovementSuggester → [loop]
                                                              → DemoBuilder
```

`*` optional stages controlled by feature flags.  
The Programmer–Runner–Evaluator–ImprovementSuggester loop repeats for up to `max_improvement_steps` steps per stage and `max_novel_solutions` stages.

---

## Agents

| Agent | Role | Model |
|---|---|---|
| **DataPreprocessorAgent** | Resolves dataset split paths; computes image and object counts per split. | — |
| **DataAnalyserAgent** | Analyses sample images to determine task type and desired output format (`line_segments`, `bounding_boxes`, `points`, …). | Qwen 2.5-VL 7B (Ollama) |
| **DatasetEnricherAgent** | Generates pseudo-labels for unlabelled data via a YOLO + SAM + Gemma3 pipeline. *(optional)* | YOLO11m · SAM3 · Gemma3 (Ollama) |
| **DLModelTrainerAgent** | Fine-tunes a YOLO model on the (enriched) dataset. *(optional)* | YOLO11 / YOLOv8 |
| **ProgrammerAgent** | Generates or refines a `Predictor` class in Python. Supports four reasoning modes: `initial_coding`, `bug_fixing`, `improving_based_on_suggestion`, `novelty_coding`. | Claude Sonnet 4.5 |
| **RunnerAgent** | Executes the generated script in an isolated workspace directory; captures stdout / stderr and return code. | — |
| **EvaluatorAgent** | Computes task-specific metrics (LINE\_F1, AP50, point distance) against ground-truth labels on val and test splits. In **label-free mode**, replaces val evaluation with an LLM-as-judge that scores prediction images directly. | — · Claude Opus 4.8 *(label-free judge)* |
| **ImprovementSuggesterAgent** | Analyses evaluation visualisations and source code; produces structured `VERIFIED_PROBLEMS` and `IMPROVEMENT_SUGGESTIONS` fed back to the Programmer. | Claude Sonnet 4.5 |
| **DemoBuilderAgent** | Selects the best-scoring solution from the workspace (by `val_metrics.json`), wraps it in a Tkinter GUI application, and packages it as a standalone `demo_app.exe` via PyInstaller. | — |

---

## Evaluation Modes

### Standard mode (`enable_label_free_improvement = False`)
Both val and test splits are evaluated against ground-truth annotations. Metrics: LINE\_F1 (line segments), AP50 (bounding boxes), point-distance (midpoints / keypoints). Visualisations colour-code detections as **TP** (green), **FP** (red), **FN** (yellow).

### Label-free mode (`enable_label_free_improvement = True`)
- **Test split** — evaluated normally against ground-truth (for experimental reference only).
- **Val split** — predictions are rendered in cyan on each image and sent to an LLM judge together with the task description. The judge returns a quality score in [0, 1] that is used as the optimisation signal throughout the improvement loop. No ground-truth labels are accessed during improvement.

---

## Configuration

All behaviour is controlled through `SystemSettings` in `src/config/settings.py`.

| Parameter | Default | Description |
|---|---|---|
| `enable_dataset_enricher` | `False` | Run pseudo-labelling on unlabelled data |
| `enable_dl_model_trainer` | `False` | Fine-tune a YOLO model before coding |
| `enable_novel_solution_search` | `False` | Generate architecturally distinct solutions across stages |
| `enable_label_free_improvement` | `True` | Replace val evaluation with LLM-as-judge |
| `max_runner_retries` | `3` | Maximum bug-fixing iterations per execution failure |
| `max_improvement_steps` | `5` | Maximum improvement steps per stage |
| `max_novel_solutions` | `2` | Number of independent solution stages |

### Models

| Setting | Model | Role |
|---|---|---|
| `programmer_llm` | Claude Sonnet 4.5 | Code generation and bug fixing |
| `improvement_llm` | Claude Sonnet 4.5 | Improvement suggestions |
| `judge_llm` | Claude Opus 4.8 | LLM-as-judge val scoring (label-free mode only) |
| `vision_llm` | Qwen 2.5-VL 7B | Dataset visual analysis (local, via Ollama) |

---

## Quick Start

### Environment setup

Create and activate a virtual environment, then install dependencies:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

```bash
pip install -r requirements.txt

python -m spacy download en_core_web_sm
```

Set the Anthropic API key:

```bash
# Windows
set SONNET_API_KEY=sk-ant-...

# Linux / macOS
export SONNET_API_KEY=sk-ant-...
```

Ollama with Qwen 2.5-VL must be running locally for dataset analysis:

```bash
ollama serve
ollama pull qwen2.5vl:7b
```

### Run

Edit `main.py` to point to your dataset and describe your task:

```python
settings = SystemSettings(
    enable_label_free_improvement=False,   # set True for unsupervised mode
    max_improvement_steps=3,
    max_novel_solutions=2,
)

initial_state = AgentState(
    user_prompt="Detect crop row lines in UAV RGB images using classical CV.",
    dl_dataset_path="data/data_structured/crop_line_uav/sugarbeet_3_charmont_2017_1_bboxes",
    eval_dataset_path="data/data_structured/crop_line_uav/sugarbeet_3_charmont_2017_1_lines",
)
```

Then run:

```bash
python main.py
```

### Outputs

| Path | Content |
|---|---|
| `workspace/stage_X_step_Y/generated_solution.py` | Generated Python predictor |
| `workspace/stage_X_step_Y/evaluation/metrics/` | Per-split JSON metric files |
| `workspace/stage_X_step_Y/evaluation/visualizations/` | Annotated prediction images |
| `workspace/demo_app.exe` | Standalone prediction demo application |

### Dataset Format

```
dataset_root/
├── images/
│   ├── train/
│   ├── val/
│   ├── test/
│   └── unlabelled/   # optional
└── labels/
    ├── train/
    ├── val/
    └── test/         # YOLO format (.txt per image)
```

---

## Project Structure

```
src/
├── agents/           # Agent implementations
├── config/           # SystemSettings
├── funcs/            # Stateless domain logic (evaluators, visualisers, loaders)
├── graph/            # LangGraph workflow definition
├── inference/        # LLM backend abstraction (Anthropic, Ollama)
├── prompts/          # Prompt templates per agent
└── state/            # Pydantic AgentState schema
workspace/            # Runtime artefacts (generated code, metrics, visualisations)
data/                 # Datasets
```
