import logging
from typing import List, Dict, Any

class BaseAgent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.logs: List[str] = []
        self.logger = logging.getLogger(f"graph_rag.agents.{name.lower().replace(' ', '_')}")

    def log(self, message: str) -> None:
        """Records agent actions and prints them to system logs."""
        formatted_message = f"[{self.name}] {message}"
        self.logs.append(formatted_message)
        self.logger.info(message)

    def clear_logs(self) -> None:
        """Clears the trace logs for a new run."""
        self.logs.clear()

    def get_logs(self) -> List[str]:
        """Returns the transcript of the agent's work."""
        return self.logs
