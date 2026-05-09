
# Autonomous CV Agent System

Production-oriented prototype for autonomous Computer Vision prototyping using:

- LangGraph
- Ollama
- Coding LLM
- Vision/Reasoning MLLM
- Deterministic evaluation
- Iterative improvement loop
- Experiment tracking
- Sandbox runner
- Failure analysis

## Features

- Dataset understanding
- Annotation discovery
- Automatic evaluator generation
- Predictor generation
- Runner validation
- Iterative improvement
- Failure-driven optimization
- Structured logging
- Experiment memory
- LangSmith-ready hooks

## Recommended Ollama models

### Coding
```bash
ollama pull qwen2.5-coder:7b
```

### Vision / Reasoning
```bash
ollama pull qwen2.5vl:7b
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Tests

```bash
pytest
```
