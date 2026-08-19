"""Optional execution-boundary diagnostics for the V1 GPU model runner."""
import functools
import os
import time
from typing import Any

_CALL_ID = 0


def _enabled() -> bool:
    return os.getenv("KUNLUN_DSV4_DEBUG", "0") == "1"


def _log(stage: str, event: str, call_id: int, **fields: Any) -> None:
    if not _enabled() or (event == "end" and call_id % max(1, int(os.getenv("KUNLUN_DSV4_DEBUG_EVERY", "1")))):
        return
    rank = os.getenv("RANK", os.getenv("LOCAL_RANK", "?"))
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    print(
        f"[DSV4_DEBUG] rank={rank} pid={os.getpid()} call={call_id} "
        f"stage={stage} event={event} {details}",
        flush=True,
    )


def _next_call_id() -> int:
    global _CALL_ID
    _CALL_ID += 1
    return _CALL_ID


def _shape(value: Any) -> Any:
    shape = getattr(value, "shape", None)
    return tuple(shape) if shape is not None else "?"


def _runner_metadata(runner: Any, scheduler_output: Any) -> dict[str, Any]:
    input_batch = getattr(runner, "input_batch", None)
    total_tokens = getattr(scheduler_output, "total_num_scheduled_tokens", None)
    if total_tokens is None:
        total_tokens = getattr(scheduler_output, "total_num_sch", "?")
    return {
        "num_reqs": getattr(input_batch, "num_reqs", "?"),
        "scheduled_tokens": total_tokens,
        "graph_dispatcher": type(getattr(runner, "cudagraph_dispatcher", None)).__name__,
    }


def apply(module: object) -> None:
    """Wrap GPUModelRunner boundaries without modifying community vLLM."""
    runner = module.GPUModelRunner
    execute_model = runner.execute_model
    model_forward = runner._model_forward
    if getattr(execute_model, "_kunlun_dsv4_debug", False):
        return

    @functools.wraps(execute_model)
    def execute_model_wrapped(self: Any, scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
        call_id = _next_call_id()
        _log("execute_model", "begin", call_id, **_runner_metadata(self, scheduler_output))
        started = time.monotonic()
        try:
            result = execute_model(self, scheduler_output, *args, **kwargs)
        except BaseException as exc:
            _log("execute_model", "error", call_id, error=type(exc).__name__)
            raise
        _log(
            "execute_model",
            "end",
            call_id,
            elapsed_ms=round((time.monotonic() - started) * 1000, 3),
            result=type(result).__name__,
        )
        return result

    @functools.wraps(model_forward)
    def model_forward_wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        call_id = _next_call_id()
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        positions = kwargs.get("positions", args[1] if len(args) > 1 else None)
        _log(
            "model_forward",
            "begin",
            call_id,
            input_shape=_shape(input_ids),
            position_shape=_shape(positions),
            model=type(getattr(self, "model", None)).__name__,
        )
        started = time.monotonic()
        try:
            result = model_forward(self, *args, **kwargs)
        except BaseException as exc:
            _log("model_forward", "error", call_id, error=type(exc).__name__)
            raise
        _log(
            "model_forward",
            "end",
            call_id,
            elapsed_ms=round((time.monotonic() - started) * 1000, 3),
            result=type(result).__name__,
        )
        return result

    execute_model_wrapped._kunlun_dsv4_debug = True
    model_forward_wrapped._kunlun_dsv4_debug = True
    runner.execute_model = execute_model_wrapped
    runner._model_forward = model_forward_wrapped
