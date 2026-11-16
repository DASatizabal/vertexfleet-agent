"""Runs the loop: observe > update KB > reason > entail > plan > approve > act > observe result.

Bounded by config agent.max_iterations and stopped immediately by the kill switch.
"""
from settings import kill_switch_on, load_config


def run_cycle(raw_inputs):
    if kill_switch_on():
        return {"status": "halted", "reason": "kill switch is on"}
    _max_iter = load_config()["agent"]["max_iterations"]
    raise NotImplementedError("orchestrator.run_cycle is filled in during Phase 4")
