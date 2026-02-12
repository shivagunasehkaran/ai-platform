#!/usr/bin/env python3
"""Reset metrics.json to empty state."""

import json
from pathlib import Path

METRICS_FILE = Path(__file__).parent.parent / "metrics.json"

empty_metrics = {
    "chat_metrics": [],
    "retrieval_metrics": [],
    "agent_metrics": [],
}

METRICS_FILE.write_text(json.dumps(empty_metrics, indent=2))
print(f"✅ Reset {METRICS_FILE}")
