"""Run-safety lifecycle (repo_design §11): signals, stall + heartbeat watchdogs, disk guard.

The two watchdogs are complements, not duplicates: `StallWatchdog` watches games-PROGRESS
from inside the loop (it catches a live-but-unproductive run), `HeartbeatWatchdog` watches
per-source liveness from its own thread (it catches a wedged one). Both end in `os._exit`.
"""
