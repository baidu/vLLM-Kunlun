"""Optional low-overhead diagnostics for TP logits all-gather."""
import functools
import os
import time
from typing import Any

_CALL_ID = 0


def _enabled() -> bool:
    return os.getenv("KUNLUN_DSV4_DEBUG", "0") == "1"


def _every() -> int:
    return max(1, int(os.getenv("KUNLUN_DSV4_DEBUG_EVERY", "1")))


def _log(event: str, call_id: int, **fields: Any) -> None:
    if not _enabled() or (event == "end" and call_id % _every()):
        return
    rank = os.getenv("RANK", os.getenv("LOCAL_RANK", "?"))
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(
        f"[DSV4_DEBUG] rank={rank} pid={os.getpid()} call={call_id} "
        f"stage=logits_allgather event={event} {details}",
        flush=True,
    )


def apply(module: object) -> None:
    """Wrap the community logits boundary without modifying vLLM sources."""
    global _CALL_ID
    processor = module.LogitsProcessor
    original = processor._gather_logits
    if getattr(original, "_kunlun_dsv4_debug", False):
        return

    @functools.wraps(original)
    def wrapped(self: object, logits: Any) -> Any:
        global _CALL_ID
        _CALL_ID += 1
        call_id = _CALL_ID
        _log(
            "begin",
            call_id,
            shape=tuple(logits.shape),
            numel=logits.numel(),
            use_all_gather=int(self.use_all_gather),
        )
        started = time.monotonic()
        result = original(self, logits)
        _log(
            "end",
            call_id,
            elapsed_ms=round((time.monotonic() - started) * 1000, 3),
            shape=tuple(result.shape),
            numel=result.numel(),
        )
        return result

    wrapped._kunlun_dsv4_debug = True
    processor._gather_logits = wrapped
