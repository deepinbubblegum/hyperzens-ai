#!/usr/bin/env python3
"""Sandboxed local tools for the FFF agent loop.

The distilled student (``train_fff_agent.py``) was trained on Hermes-style
function calling. This module is the **runtime** counterpart:

* Parse ``<tool_call>...</tool_call>`` blocks (JSON, XML, Qwen arg tags).
* Execute a small, workspace-bound tool set.
* Truncate payloads so tool traces cannot balloon the ChatML context.

Tools
-----
``read_file``   — UTF-8 file slice under the workspace root.
``list_dir``    — directory listing (names + sizes, no recursive dump).
``run_shell``   — allowlisted, no-shell subprocess in the workspace cwd.
``sys_stats``   — CPU / RAM / disk / optional GPU snapshot (psutil).

Security
--------
Paths are resolved against ``workspace``; symlink escapes are rejected.
``run_shell`` never uses ``shell=True``. Only an allowlisted binary set
(plus a git read-only subcommand set) may run, with a hard timeout.

Shapes / budgets
----------------
Every tool returns a :class:`ToolResult` whose ``payload`` is already
clipped to ``max_output_chars`` (default 2400 ≈ 500–700 tokens). The
agent loop injects ``format_tool_response_message`` as a single user
turn and **drops** the trace after the final answer.
"""

from __future__ import annotations

import inspect
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Budgets — keep tool I/O cheap for low-resource PCs
# ---------------------------------------------------------------------------

DEFAULT_MAX_OUTPUT_CHARS: int = 2400
DEFAULT_MAX_FILE_BYTES: int = 256 * 1024
DEFAULT_COMMAND_TIMEOUT_S: float = 8.0
DEFAULT_MAX_DIR_ENTRIES: int = 80
DEFAULT_READ_LINE_LIMIT: int = 200

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    flags=re.DOTALL | re.IGNORECASE,
)
_FUNCTION_XML_RE = re.compile(
    r"<function\s*=\s*([^>\s]+)\s*>(.*?)</function>",
    flags=re.DOTALL | re.IGNORECASE,
)
_PARAM_XML_RE = re.compile(
    r"<parameter\s*=\s*([^>\s]+)\s*>(.*?)</parameter>",
    flags=re.DOTALL | re.IGNORECASE,
)
_ARG_PAIR_RE = re.compile(
    r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>",
    flags=re.DOTALL | re.IGNORECASE,
)


# Read-only / inspect binaries. Arguments that look like paths are still
# constrained to the workspace (see :func:`_reject_outside_path_args`).
_SHELL_ALLOWLIST: frozenset[str] = frozenset(
    {
        "ls",
        "pwd",
        "cat",
        "head",
        "tail",
        "wc",
        "df",
        "du",
        "uname",
        "date",
        "whoami",
        "hostname",
        "uptime",
        "file",
        "stat",
        "which",
        "git",
        "python",
        "python3",
        "pip",
        "pip3",
        "nvidia-smi",
        "sysctl",
        "sw_vers",
        "ps",
        "env",
        "echo",
        "find",
        "rg",
        "grep",
    }
)

_GIT_ALLOW_SUBCOMMANDS: frozenset[str] = frozenset(
    {
        "status",
        "log",
        "diff",
        "show",
        "branch",
        "rev-parse",
        "ls-files",
        "describe",
        "remote",
        "shortlog",
        "blame",
    }
)

_FIND_DENIED_FLAGS: frozenset[str] = frozenset({"-delete", "-exec", "-execdir", "-ok"})


class ToolError(RuntimeError):
    """User-facing tool failure (bad args, sandbox, timeout, allowlist)."""


@dataclass(frozen=True)
class ToolCall:
    """One parsed Hermes-style invocation.

    Attributes
    ----------
    name:
        Tool identifier (``read_file``, ``list_dir``, …).
    arguments:
        JSON-object kwargs. Missing / invalid values are coerced by the tool.
    raw:
        Original ``<tool_call>`` inner text (debug / logging).
    """

    name: str
    arguments: dict[str, Any]
    raw: str = ""


@dataclass
class ToolResult:
    """Clipped tool outcome ready to inject as ``<tool_response>``.

    Attributes
    ----------
    name:
        Tool that ran.
    ok:
        ``False`` on sandbox / timeout / I/O errors (payload still holds the
        error string so the model can recover).
    payload:
        Already truncated text (never exceed ``max_output_chars``).
    truncated:
        Whether ``payload`` was clipped.
    elapsed_ms:
        Wall time of the tool itself (not model decode).
    """

    name: str
    ok: bool
    payload: str
    truncated: bool = False
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class ToolSpec:
    """JSON-schema-ish description shown in the system prompt."""

    name: str
    description: str
    parameters: dict[str, str]
    handler: Callable[..., str]


def clip_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Clip ``text`` to ``max_chars``, appending a truncation marker.

    Returns
    -------
    tuple[str, bool]
        ``(clipped, was_truncated)``.
    """
    max_chars = max(int(max_chars), 64)
    if len(text) <= max_chars:
        return text, False
    keep = max_chars - 48
    omitted = len(text) - keep
    clipped = text[:keep] + f"\n…[truncated {omitted} chars]"
    return clipped, True


def resolve_under_workspace(workspace: Path, user_path: str | os.PathLike[str]) -> Path:
    """Resolve ``user_path`` inside ``workspace``; reject symlink / ``..`` escapes.

    Relative paths are joined to the workspace root. Absolute paths are
    allowed only when they remain inside that root after ``Path.resolve()``.
    """
    root = workspace.expanduser().resolve()
    raw = Path(str(user_path)).expanduser()
    candidate = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    root_s = str(root)
    cand_s = str(candidate)
    if cand_s != root_s and not cand_s.startswith(root_s + os.sep):
        raise ToolError(f"path escapes workspace: {user_path}")
    return candidate


def isolate_tool_phase(text: str) -> str:
    """Drop hallucinated ``<tool_response>`` (and anything after the last call).

    If the model emits a tool call then continues into a fake result, keep
    only the call block(s) so the runtime can inject the real payload.
    """
    response_m = re.search(r"<tool_response\b", text, flags=re.IGNORECASE)
    head = text[: response_m.start()] if response_m else text
    closes = list(re.finditer(r"</tool_call>", head, flags=re.IGNORECASE))
    if not closes:
        return text.strip()
    return head[: closes[-1].end()].strip()


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract tool calls from a model reply (Hermes JSON / XML / Qwen tags).

    Incomplete (unclosed) ``<tool_call>`` blocks are ignored so a truncated
    generation does not execute a partial JSON object.
    """
    calls: list[ToolCall] = []
    for match in _TOOL_CALL_RE.finditer(text):
        inner = match.group(1).strip()
        parsed = _parse_tool_inner(inner)
        if parsed is not None:
            calls.append(parsed)
    return calls


def _parse_tool_inner(inner: str) -> ToolCall | None:
    """Parse the body of one ``<tool_call>`` block into a :class:`ToolCall`."""
    xml_fn = _FUNCTION_XML_RE.search(inner)
    if xml_fn:
        name = xml_fn.group(1).strip()
        args = {
            k.strip(): _maybe_json_value(v.strip())
            for k, v in _PARAM_XML_RE.findall(xml_fn.group(2))
        }
        if name:
            return ToolCall(name=name, arguments=args, raw=inner)

    pairs = _ARG_PAIR_RE.findall(inner)
    if pairs:
        first_line = inner.splitlines()[0].strip() if inner else ""
        name = first_line if first_line and "<" not in first_line else ""
        if not name:
            # Qwen sometimes puts the name on its own line after whitespace.
            for line in inner.splitlines():
                line = line.strip()
                if line and not line.startswith("<"):
                    name = line
                    break
        args = {k.strip(): _maybe_json_value(v.strip()) for k, v in pairs}
        if name:
            return ToolCall(name=name, arguments=args, raw=inner)

    blob = _extract_json_object(inner)
    if blob is not None:
        name = str(blob.get("name") or blob.get("tool") or "").strip()
        args = blob.get("arguments") or blob.get("parameters") or blob.get("args") or {}
        if isinstance(args, str):
            args = _maybe_json_value(args)
        if not isinstance(args, dict):
            args = {"input": args}
        if name:
            return ToolCall(name=name, arguments=args, raw=inner)
    return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort JSON object parse (handles leading/trailing prose)."""
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _maybe_json_value(raw: str) -> Any:
    """Parse a parameter value as JSON when possible; else keep the string."""
    raw = raw.strip()
    if not raw:
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def format_tool_response_message(results: list[ToolResult]) -> str:
    """Pack one or more results into a single ChatML user turn.

    Training folds tool turns as ``role=user`` with ``<tool_response>`` tags
    (see ``train_fff_agent.py``). Multiple results share one message so the
    context grows by *one* turn per tool round, not one per tool.
    """
    parts: list[str] = []
    for r in results:
        status = "ok" if r.ok else "error"
        header = f"tool={r.name} status={status}"
        if r.truncated:
            header += " truncated=true"
        parts.append(f"<tool_response>\n{header}\n{r.payload}\n</tool_response>")
    return "\n".join(parts)


def tools_schema_prompt(specs: list[ToolSpec], workspace: Path) -> str:
    """Compact tool card for the system prompt (avoid dumping full JSON Schema)."""
    lines = [
        "You can call local tools. Emit ONLY a <tool_call> block (no fake "
        "<tool_response>). Wait for the real result, then answer.",
        f"Workspace root: {workspace}",
        "Tools:",
    ]
    for spec in specs:
        params = ", ".join(f"{k}: {v}" for k, v in spec.parameters.items())
        lines.append(f"- {spec.name}({params}) — {spec.description}")
    lines.append(
        'Format:\n<tool_call>\n{"name": "TOOL", "arguments": {}}\n</tool_call>'
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _read_file(
    *,
    workspace: Path,
    path: str,
    offset: int = 0,
    limit: int = DEFAULT_READ_LINE_LIMIT,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> str:
    """Read a UTF-8 text file slice. ``offset``/``limit`` are 1-based line numbers."""
    target = resolve_under_workspace(workspace, path)
    if not target.exists():
        raise ToolError(f"not found: {path}")
    if not target.is_file():
        raise ToolError(f"not a file: {path}")
    size = target.stat().st_size
    if size > max_file_bytes:
        raise ToolError(
            f"file too large ({size} bytes > {max_file_bytes}); "
            "use offset/limit or a smaller file"
        )
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ToolError(f"not a UTF-8 text file: {path}") from exc
    if "\x00" in text:
        raise ToolError(f"binary file refused: {path}")

    lines = text.splitlines()
    off = max(int(offset), 0)
    lim = max(int(limit), 1)
    start = off if off > 0 else 1
    # offset=0 means "from the first line".
    start_idx = start - 1 if start >= 1 else 0
    chunk = lines[start_idx : start_idx + lim]
    numbered = [f"{start_idx + i + 1:>5}| {line}" for i, line in enumerate(chunk)]
    header = (
        f"path={target.relative_to(workspace)} lines={len(lines)} "
        f"showing={start_idx + 1}-{start_idx + len(chunk)}"
    )
    return header + "\n" + "\n".join(numbered)


def _list_dir(
    *,
    workspace: Path,
    path: str = ".",
    max_entries: int = DEFAULT_MAX_DIR_ENTRIES,
) -> str:
    """List a directory under the workspace (non-recursive)."""
    target = resolve_under_workspace(workspace, path)
    if not target.exists():
        raise ToolError(f"not found: {path}")
    if not target.is_dir():
        raise ToolError(f"not a directory: {path}")
    max_entries = max(int(max_entries), 1)
    try:
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except OSError as exc:
        raise ToolError(f"cannot list {path}: {exc}") from exc

    rows: list[str] = []
    for i, entry in enumerate(entries):
        if i >= max_entries:
            rows.append(f"… ({len(entries) - max_entries} more)")
            break
        kind = "dir " if entry.is_dir() else "file"
        try:
            size = "-" if entry.is_dir() else str(entry.stat().st_size)
        except OSError:
            size = "?"
        rel = entry.relative_to(workspace)
        rows.append(f"{kind}  {size:>10}  {rel}")
    header = f"path={target.relative_to(workspace) if target != workspace else '.'} n={len(entries)}"
    return header + "\n" + "\n".join(rows)


def _reject_outside_path_args(workspace: Path, argv: list[str]) -> None:
    """Refuse absolute / ``../`` path arguments that leave the workspace."""
    root = workspace.resolve()
    for arg in argv[1:]:
        if arg.startswith("-"):
            continue
        looks_like_path = (
            arg.startswith(".")
            or arg.startswith("/")
            or arg.startswith("~")
            or "/" in arg
            or "\\" in arg
        )
        if not looks_like_path:
            continue
        try:
            resolve_under_workspace(root, arg)
        except ToolError:
            raise ToolError(f"argument escapes workspace: {arg}") from None


def _run_shell(
    *,
    workspace: Path,
    command: str,
    timeout_s: float = DEFAULT_COMMAND_TIMEOUT_S,
) -> str:
    """Run an allowlisted command with ``cwd=workspace`` (no shell interpolation)."""
    command = str(command).strip()
    if not command:
        raise ToolError("empty command")
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise ToolError(f"cannot parse command: {exc}") from exc
    if not argv:
        raise ToolError("empty command")

    binary = Path(argv[0]).name
    if binary not in _SHELL_ALLOWLIST:
        raise ToolError(
            f"command not allowlisted: {binary}. "
            f"allowed={sorted(_SHELL_ALLOWLIST)}"
        )
    if binary == "git":
        sub = argv[1] if len(argv) > 1 else ""
        if sub.startswith("-"):
            # e.g. git -C … — still require a real subcommand later.
            pass
        elif sub not in _GIT_ALLOW_SUBCOMMANDS:
            raise ToolError(
                f"git subcommand not allowlisted: {sub or '(none)'}. "
                f"allowed={sorted(_GIT_ALLOW_SUBCOMMANDS)}"
            )
    if binary == "find" and _FIND_DENIED_FLAGS.intersection(argv):
        raise ToolError("find -delete/-exec is not allowed")

    _reject_outside_path_args(workspace, argv)

    exe = shutil.which(argv[0])
    if exe is None:
        raise ToolError(f"executable not found on PATH: {argv[0]}")
    argv = [exe, *argv[1:]]

    try:
        proc = subprocess.run(
            argv,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            check=False,
            env=_sanitized_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"timed out after {timeout_s:.1f}s: {command}") from exc
    except OSError as exc:
        raise ToolError(f"failed to exec: {exc}") from exc

    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    out = out.strip() or "(no output)"
    return f"exit={proc.returncode}\n{out}"


def _sanitized_env() -> dict[str, str]:
    """Pass a minimal environment (keep PATH / locale; drop secrets-looking keys)."""
    keep_prefixes = ("PATH", "HOME", "USER", "LANG", "LC_", "TERM", "TMPDIR", "TMP")
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if any(upper.startswith(p) for p in keep_prefixes):
            env[key] = value
    env.setdefault("PATH", "/usr/bin:/bin:/usr/local/bin")
    return env


def _sys_stats(*, workspace: Path) -> str:
    """Snapshot host resources. Uses psutil when installed; else /proc-less fallbacks."""
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cwd": str(workspace),
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["cpu"] = {
            "count": psutil.cpu_count(logical=True),
            "percent": psutil.cpu_percent(interval=0.15),
        }
        info["ram"] = {
            "used_mb": round(vm.used / (1024 * 1024), 1),
            "total_mb": round(vm.total / (1024 * 1024), 1),
            "percent": vm.percent,
        }
        du = shutil.disk_usage(str(workspace))
        info["disk"] = {
            "used_gb": round(du.used / (1024**3), 2),
            "total_gb": round(du.total / (1024**3), 2),
            "percent": round(100.0 * du.used / max(du.total, 1), 1),
        }
        proc = psutil.Process(os.getpid())
        info["this_process"] = {
            "rss_mb": round(proc.memory_info().rss / (1024 * 1024), 1),
            "cpu_percent": proc.cpu_percent(interval=0.05),
        }
    except ImportError:
        info["note"] = "psutil not installed — limited stats"
        info["cpu_count"] = os.cpu_count()

    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(idx) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(idx) / (1024 * 1024)
            total = torch.cuda.get_device_properties(idx).total_memory / (1024 * 1024)
            info["gpu"] = {
                "name": torch.cuda.get_device_name(idx),
                "allocated_mb": round(allocated, 1),
                "reserved_mb": round(reserved, 1),
                "total_mb": round(total, 1),
            }
    except Exception:  # noqa: BLE001 — stats must never crash the agent
        pass

    return json.dumps(info, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Dispatch table bound to one workspace + output budget.

    Parameters
    ----------
    workspace:
        Sandbox root (default: process cwd).
    max_output_chars:
        Hard cap on every :class:`ToolResult`.payload.
    command_timeout_s:
        ``run_shell`` timeout.
    """

    def __init__(
        self,
        workspace: Path | str | None = None,
        *,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
        command_timeout_s: float = DEFAULT_COMMAND_TIMEOUT_S,
        max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    ) -> None:
        self.workspace: Path = Path(workspace or Path.cwd()).expanduser().resolve()
        self.max_output_chars: int = int(max_output_chars)
        self.command_timeout_s: float = float(command_timeout_s)
        self.max_file_bytes: int = int(max_file_bytes)
        self._specs: dict[str, ToolSpec] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(
            ToolSpec(
                name="read_file",
                description="Read a UTF-8 text file under the workspace (line slice).",
                parameters={
                    "path": "str",
                    "offset": "int=0",
                    "limit": "int=200",
                },
                handler=self._wrap_read_file,
            )
        )
        self.register(
            ToolSpec(
                name="list_dir",
                description="List a workspace directory (non-recursive).",
                parameters={"path": "str='.'", "max_entries": "int=80"},
                handler=self._wrap_list_dir,
            )
        )
        self.register(
            ToolSpec(
                name="run_shell",
                description=(
                    "Run an allowlisted inspect command in the workspace "
                    "(ls, git status/log/diff, python3 --version, …)."
                ),
                parameters={"command": "str"},
                handler=self._wrap_run_shell,
            )
        )
        self.register(
            ToolSpec(
                name="sys_stats",
                description="CPU, RAM, disk, this-process RSS, optional CUDA VRAM.",
                parameters={},
                handler=self._wrap_sys_stats,
            )
        )

    def register(self, spec: ToolSpec) -> None:
        """Add or replace a tool by name."""
        self._specs[spec.name] = spec

    def specs(self) -> list[ToolSpec]:
        """Stable-order tool specs for the system prompt."""
        return [self._specs[k] for k in sorted(self._specs)]

    def schema_prompt(self) -> str:
        """Compact tool card (see :func:`tools_schema_prompt`)."""
        return tools_schema_prompt(self.specs(), self.workspace)

    def names(self) -> list[str]:
        return sorted(self._specs)

    def execute(self, call: ToolCall) -> ToolResult:
        """Run ``call`` and always return a clipped :class:`ToolResult`."""
        t0 = time.perf_counter()
        spec = self._specs.get(call.name)
        if spec is None:
            payload, truncated = clip_text(
                f"unknown tool {call.name!r}. available={self.names()}",
                self.max_output_chars,
            )
            return ToolResult(
                name=call.name,
                ok=False,
                payload=payload,
                truncated=truncated,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            )
        try:
            raw = spec.handler(**_filter_kwargs(spec.handler, call.arguments))
            if not isinstance(raw, str):
                raw = json.dumps(raw, ensure_ascii=False, default=str)
            ok = True
        except TypeError as exc:
            raw = f"bad arguments for {call.name}: {exc}"
            ok = False
        except ToolError as exc:
            raw = str(exc)
            ok = False
        except Exception as exc:  # noqa: BLE001 — keep the agent loop alive
            raw = f"internal error in {call.name}: {type(exc).__name__}: {exc}"
            ok = False
        payload, truncated = clip_text(raw, self.max_output_chars)
        return ToolResult(
            name=call.name,
            ok=ok,
            payload=payload,
            truncated=truncated,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def execute_all(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Execute calls in order (no parallelism — keeps FS side effects sane)."""
        return [self.execute(c) for c in calls]

    def _wrap_read_file(
        self,
        path: str = "",
        offset: int = 0,
        limit: int = DEFAULT_READ_LINE_LIMIT,
        **_extra: Any,
    ) -> str:
        if not path:
            raise ToolError("read_file requires path")
        return _read_file(
            workspace=self.workspace,
            path=str(path),
            offset=int(offset),
            limit=int(limit),
            max_file_bytes=self.max_file_bytes,
        )

    def _wrap_list_dir(
        self,
        path: str = ".",
        max_entries: int = DEFAULT_MAX_DIR_ENTRIES,
        **_extra: Any,
    ) -> str:
        return _list_dir(
            workspace=self.workspace,
            path=str(path),
            max_entries=int(max_entries),
        )

    def _wrap_run_shell(self, command: str = "", **_extra: Any) -> str:
        if not command:
            raise ToolError("run_shell requires command")
        return _run_shell(
            workspace=self.workspace,
            command=str(command),
            timeout_s=self.command_timeout_s,
        )

    def _wrap_sys_stats(self, **_extra: Any) -> str:
        return _sys_stats(workspace=self.workspace)


def _filter_kwargs(fn: Callable[..., Any], args: dict[str, Any]) -> dict[str, Any]:
    """Pass known parameters; ignore extras the model hallucinated."""
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(args)
    allowed = {
        name
        for name, p in sig.parameters.items()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {k: v for k, v in args.items() if k in allowed}


def build_system_prompt(base: str, registry: ToolRegistry | None) -> str:
    """Join the persona prompt with a compact tool card (if tools are enabled)."""
    base = base.strip()
    if registry is None:
        return base
    return base + "\n\n" + registry.schema_prompt()
