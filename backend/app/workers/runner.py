#!/usr/bin/env python3
"""In-process background worker.

Used when no Celery broker is configured. Runs the same task functions on the
same schedule using a thread, so `docker compose up` without a broker still gets
feature refreshes, sentiment scoring and cache warming.

    python -m app.workers.runner            # run the scheduler loop
    python -m app.workers.runner --once     # run every task once and exit
    python -m app.workers.runner --task refresh_search_index
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import settings  # noqa: E402
from app.core.logging_config import configure_logging, get_logger  # noqa: E402
from app.workers.tasks import ALL_TASKS, SCHEDULE, run_task  # noqa: E402

logger = get_logger("worker.runner")
_stop = threading.Event()


def _handle_signal(signum, _frame) -> None:  # pragma: no cover - signal path
    logger.info("worker_stopping", signal=signum)
    _stop.set()


def run_forever() -> int:
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    next_run = {name: time.monotonic() + min(interval, 15) for name, (_, interval) in SCHEDULE.items()}
    logger.info("worker_started", tasks=sorted(SCHEDULE), mode="in-process")

    while not _stop.is_set():
        now = time.monotonic()
        for name, (_, interval) in SCHEDULE.items():
            if now >= next_run[name]:
                run_task(name)
                next_run[name] = time.monotonic() + interval
        _stop.wait(1.0)
    logger.info("worker_stopped")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="E-Commerce Intelligence background worker")
    parser.add_argument("--once", action="store_true", help="run every task once and exit")
    parser.add_argument("--task", help="run a single named task and exit")
    parser.add_argument("--list", action="store_true", help="list available tasks")
    args = parser.parse_args()

    configure_logging(settings.LOG_LEVEL, json_output=settings.LOG_JSON)

    if args.list:
        print(json.dumps({name: f"every {interval}s" for name, (_, interval) in SCHEDULE.items()}, indent=2))
        return 0
    if args.task:
        if args.task not in ALL_TASKS:
            print(f"Unknown task '{args.task}'. Available: {', '.join(sorted(ALL_TASKS))}", file=sys.stderr)
            return 2
        print(json.dumps(run_task(args.task), indent=2, default=str))
        return 0
    if args.once:
        results = {name: run_task(name) for name in SCHEDULE}
        print(json.dumps(results, indent=2, default=str))
        return 0
    return run_forever()


if __name__ == "__main__":
    raise SystemExit(main())
