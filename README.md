# LangGraph CV Agent System

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-workflow-orange.svg)
![Tests](https://img.shields.io/badge/tests-pytest-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

An agent-based system for iterative Computer Vision solution development using LangGraph orchestration.

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
python main.py
```

---

## 🧪 Tests

```bash
python -m pytest tests
```

---

# 🧠 How it works

The system is a pipeline of agents that:

- preprocess data
- analyze data
- optionally enrich dataset and train DL models
- generate solution code
- execute code
- evaluate results
- suggest improvements
- iterate until convergence or limit

---

# 🔄 LangGraph Flow

```mermaid
flowchart TD

A[Preprocessor] --> B[Analyzer]
B --> C{Enricher?}
C -->|yes| D[Dataset Enricher]
C -->|no| F[Programmer]
D --> E{DL Trainer?}
E -->|yes| G[DL Trainer]
E -->|no| F
G --> F
F --> H[Runner]
H -->|fail| F
H -->|success| I[Evaluator]
I --> J[Improver]
J --> K[State Update]
K -->|continue| F
K -->|stop| END
```
