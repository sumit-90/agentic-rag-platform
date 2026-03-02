class ConversationMemory:
    """Sliding-window conversation history for multi-turn RAG sessions.

    History is stored as a list of OpenAI-compatible message dicts
    (``{"role": ..., "content": ...}``).  When the window is full, the
    oldest user+assistant turn pair is dropped to keep memory bounded.
    """

    def __init__(self, max_turns: int = 20) -> None:
        self._max_turns = max_turns
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, role: str, content: str) -> None:
        """Append a single message and evict the oldest turn if over capacity."""
        self._history.append({"role": role, "content": content})
        # Each turn = 2 messages (user + assistant).  Drop the oldest pair.
        max_messages = self._max_turns * 2
        if len(self._history) > max_messages:
            self._history = self._history[2:]

    def get_messages(self) -> list[dict]:
        """Return a shallow copy of the history for safe external use."""
        return list(self._history)

    def clear(self) -> None:
        """Wipe all history."""
        self._history = []

    def __len__(self) -> int:
        return len(self._history)
