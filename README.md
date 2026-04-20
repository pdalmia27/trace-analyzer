# Trace Analyzer

Standalone Chrome/Perfetto trace analysis utilities for multi-turn RL traces.

The repo contains two entrypoints:

- `tools/multi_turn_trace_analyzer.py`
- `tools/multi_turn_trace_summary_visualizer.py`

The analyzer reads one trace JSON and writes normalized CSV, JSON, and Markdown
artifacts. The visualizer reads those analyzer outputs and renders a single-file
HTML summary.

## CLI

Analyze one trace:

```bash
python tools/multi_turn_trace_analyzer.py \
  --trace /path/to/trace.json \
  --outdir /tmp/trace_out
```

Render the HTML summary:

```bash
python tools/multi_turn_trace_summary_visualizer.py \
  --input-dir /tmp/trace_out \
  --output-html /tmp/trace_summary.html
```

## Analyzer outputs

The analyzer writes:

- `events.csv`
- `tool_events.csv`
- `llm_events.csv`
- `trajectory_summary.csv`
- `task_summary.csv`
- `turn_summary.csv`
- `concurrency_intervals.csv`
- `concurrency_summary.csv`
- `summary.json`
- `report.md`

## Tests

Run:

```bash
pytest -q tests/test_multi_turn_trace_analyzer.py tests/test_multi_turn_trace_summary_visualizer.py
```

The real-trace test uses a local trace fixture path when available and skips
cleanly when that fixture is absent.
