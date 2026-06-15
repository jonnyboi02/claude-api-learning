# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Requires an `ANTHROPIC_API_KEY` in a `.env` file at the project root. All scripts call `load_dotenv()` automatically.

Install dependencies: `pip install anthropic python-dotenv`

## Running the pipeline

```bash
# Full pipeline: generate dataset → run compliance checks → grade → output report
python run_eval.py

# Regenerate dataset (delete existing first if needed)
python dataset.py

# Smoke test the compliance engine against a single hardcoded case
python compliance_prompt.py
```

`run_eval.py` skips dataset generation if `dataset.json` already exists — delete it to regenerate. Outputs are written to `output.json` and `output.html`.

## Architecture

Three-stage evaluation pipeline:

**Stage 1 — `dataset.py`**: Generates 20 synthetic test cases in batches of 3 (to stay within output token limits) using `claude-haiku-4-5` at `temperature=1.0`. Each test case includes a rule config, a current repo book, a proposed trade, and the expected compliance output with pre-computed net values.

**Stage 2 — `compliance_prompt.py`**: The system under test. Sends each task to `claude-haiku-4-5` at `temperature=0.0`. The model must compute post-trade net positions by dimension (counterparty or maturity_month), evaluate child rules against bounds, combine children via AND/OR logic at the parent level, and return a structured JSON verdict.

**Stage 3 — `evaluator.py`**: Iterates the dataset sequentially. Grades each compliance engine output against the expected result using a second `claude-haiku-4-5` call (LLM-as-judge). Uses prefill (assistant prefill `"```json"` + `stop_sequences=["```"]`) to extract clean JSON from grader responses. Generates `output.html` with color-coded scores (green ≥8, yellow 6-7, red ≤5) and `output.json` with the full per-case results.

## Domain: Repo trading compliance

- **Positive notional** = reverse repo (cash out, collateral in)
- **Negative notional** = repo (cash in, collateral out)
- Net values are computed **post-trade** across the full book
- Child rules aggregate by `counterparty` or `maturity_month` and check whether the net falls within `[lower_bound, upper_bound]`
- **OR parent**: breached if ANY child breaches
- **AND parent**: breached only if ALL children breach — partial breach does NOT trigger the parent (the critical AND edge case)
- `pm_override_required` is true if and only if any parent is breached

## Notebook

`001_prompt_evals copy.ipynb` is an earlier exploratory iteration with a generic code-generation evaluator (not repo-specific). The Python scripts in the root are the production refactor of that notebook.
