# CLAUDE.md

## Project Overview

Monitor Capture simulation — explores how LLM-based monitors get captured by adversarial agents they monitor. Demonstrates a "fourth dimension" of the AI control protocol tradeoff (robustness to monitor capture), grounded in the Myerson-Satterthwaite (1983) impossibility result.

A multi-agent marketplace runs for 10 rounds with benign traders, an adversarial agent (D0) trying to covertly move value to a passive sink (C), and an LLM-based compliance monitor that updates its few-shot examples each round.

## Project Structure

```
simulation.py              # Main simulation: agents, monitor, marketplace, metrics, plotting
scripts/analyze_results.py # Cross-model analysis of saved results
docs/                      # Theoretical framing, findings, result interpretation
results/                   # Output directory (per-model, auto-incrementing run dirs)
pyproject.toml             # Project config (Python >=3.14, openai + dotenv + matplotlib)
```

There is no `main.py` — `simulation.py` is the entrypoint.

## Running

```bash
# Requires OPENROUTER_API_KEY env var
python simulation.py
```

Models are configured in the `MODELS` list in `simulation.py`. The simulation uses OpenRouter as the API gateway (OpenAI-compatible client pointing at `openrouter.ai`).

## Key Concepts

- **Apparent safety**: fraction of actual D0->C relay chains the monitor flags (measured on real attack traffic)
- **True safety**: fraction of held-out red-team attacks the monitor flags (never enters monitor's example pool)
- **Capture gap**: apparent_safety - true_safety; grows as monitor overfits to seen attacks
- **Usefulness**: fraction of required legitimate trades (K_TRADES with Agent A) completed

## Architecture Notes

- All LLM agents (benign + adversarial + monitor) use the same OpenAI-compatible client via OpenRouter
- Benign agents act in parallel via `ThreadPoolExecutor`; multiple models run in parallel via `multiprocessing.Pool`
- Agent D0 can impersonate benign agents (send transactions with other agents' names as sender)
- Monitor uses in-context few-shot learning — its example pool grows each round with ground-truth labels
- Held-out attack set is fixed at simulation start (seeded with `random.seed(42)`) and never enters the monitor's training examples
- Results are saved to `results/{model_name}/{run_number}/` with `conversations.jsonl` and `results.png`

## Dependencies

Managed via `pyproject.toml` (use `uv` or `pip`):
- `openai` — OpenAI-compatible API client (used with OpenRouter)
- `python-dotenv` — env var loading
- `matplotlib` — result plotting
