"""Controlled failures for the agent loop and tool validation."""


class AgentLoopError(RuntimeError):
    """Base class for controlled agent-loop failures."""


class AgentTimeoutError(AgentLoopError):
    """Raised when one user turn exceeds its wall-clock deadline."""


class MaxTurnsExceededError(AgentLoopError):
    """Raised when the model does not finish within the configured turn limit."""


class ToolProtocolError(AgentLoopError):
    """Raised when a model tool call cannot be correlated safely."""


class ToolArgumentError(ValueError):
    """Raised when tool arguments do not satisfy the declared input schema."""


class ApprovalDeniedError(AgentLoopError):
    """Raised when a side-effecting tool call is not approved."""
