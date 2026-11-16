"""Stage: ACTION

Purpose: Run only allow-listed tools; log every ToolResult; respect the kill switch.
Main output schema: ToolResult (defined in Phase 2, Step 3)
"""


def run(*args, **kwargs):
    raise NotImplementedError("actions.run is filled in during Phase 4")
