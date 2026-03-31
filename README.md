# Learning DFA
A interactive tool for learning and visualizing finite automata (DFA) from positive example words.

## What it does

The tool implements an iterative learning strategy for acceptors (DFAs):

1. **Learns** a minimal DFA from a growing set of positive example words
2. **Visualizes** the learning process step by step
3. **Generalizes** optionally using heuristics (suffix, prefix, substring)
4. **Tests** arbitrary words against the learned hypothesis

- **acceptor.py**   Core DFA class
- **app.py**        Streamlit frontend

## Requirements

- Python 3.10+
- [Graphviz](https://graphviz.org/download/) installed on your system

Run the project inside a virtual environment and install the dependencies from `requirements.txt`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Getting Started

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`.

> Note: The project is intended to be run from an activated virtual environment.

## Usage

### Training
1. Enter words in the training input field, separated by newlines, commas, or semicolons
2. Optionally enable **heuristic generalization**
3. Click **Lernen & minimieren**

### Learning History
Use the **slider** to step through each learning iteration and see how the hypothesis evolves.

### Testing
Enter test words and click **Prüfen** to check whether they are accepted by the final hypothesis.

---

## Heuristic Generalization

When enabled, the tool tries to generalize beyond the exact training words:

| Heuristic | Recognized pattern | Example |
|-----------|-------------------|---------|
| Common suffix | Σ\*·suffix | `ab, bab` → Σ\*·`ab` |
| Common prefix | prefix·Σ\* | `abx, aby` → `ab`·Σ\* |
| Common substring | Σ\*·token·Σ\* | `xaby, kabz` → Σ\*·`ab`·Σ\* |
| None of the above | exact minimization | `{ab, ba}` |

Tie-break priority: **suffix > prefix > substring**, longer token wins.

> **Note:** Heuristic generalization may accept more words than the training set.
> Without negative examples, exact identification is impossible.

---
