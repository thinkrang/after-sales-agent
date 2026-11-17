"""Resume persisted plans without calling an LLM. Waiting approvals stay waiting."""
import argparse
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import orchestrator
import tickets
import orders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    tickets.init_tickets_table()
    orders.init_order_db(seed_demo=False)
    if args.once:
        print(f"Scanned {orchestrator.recover_once()} unfinished plans")
    else:
        try:
            orchestrator.worker(threading.Event())
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
