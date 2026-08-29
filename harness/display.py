"""Terminal rendering and token-usage metrics."""

from __future__ import annotations

import io
import os
import select
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, TextIO

from rich.console import Console
from rich.markdown import Markdown

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - non-POSIX
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]

SCREEN_STYLE = "white on black"
_BLACK_SCREEN = False
_METRICS_BELOW_PROMPT = False
_LAST_METRICS: dict[str, Any] | None = None
_SCREEN_SIZE = (80, 24)
_PROMPT_ROWS = 5
_RECORD_TRANSCRIPT = True
_TRANSCRIPT: _TranscriptStream | None = None
_WHEEL_LINES = 3
_ESC_FOLLOW_TIMEOUT = 0.05
_CSI_MAX = 64
_RAW_READ_CHUNK = 256
_H_MARGIN = 1
_INPUT_PREFIX = " > "
_INPUT_RIGHT_PAD = " "

_ENTER_ALT_SCREEN = "\033[?1049h"
_LEAVE_ALT_SCREEN = "\033[?1049l"
_CURSOR_HOME = "\033[H"
_ERASE_DISPLAY = "\033[2J"
_RESET_SCROLL_REGION = "\033[r"
_ENABLE_MOUSE = "\033[?1006h\033[?1000h"
_DISABLE_MOUSE = "\033[?1000l\033[?1006l"
_SGR_RESET = "\033[0m"
_SGR_WHITE_ON_BLACK = "\033[40;37m"
_SGR_DIM = "\033[2m"
_OSC_SET_BACKGROUND = "\033]11;#000000\007"
_OSC_SET_FOREGROUND = "\033]10;#FFFFFF\007"
_OSC_RESET_BACKGROUND = "\033]111\007"
_OSC_RESET_FOREGROUND = "\033]110\007"
_SAVE_CURSOR = "\0337"
_RESTORE_CURSOR = "\0338"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
_BOX_TL = "╭"
_BOX_TR = "╮"
_BOX_BL = "╰"
_BOX_BR = "╯"
_BOX_H = "─"
_BOX_V = "│"


def _supports_fullscreen(stream: TextIO | None = None) -> bool:
    """Return True when the stream can host an alternate-screen TUI."""
    target = sys.stdout if stream is None else stream
    isatty = getattr(target, "isatty", None)
    if not callable(isatty):
        return False
    try:
        if not isatty():
            return False
    except ValueError:
        return False
    term = os.environ.get("TERM", "")
    return term.lower() not in {"dumb", "unknown"}


def _scroll_region_height(height: int | None = None) -> int:
    """Rows reserved for conversation output above the pinned prompt."""
    total = _SCREEN_SIZE[1] if height is None else height
    if total >= _PROMPT_ROWS + 2:
        return total - _PROMPT_ROWS - 1
    return max(total - 1, 1)


def _content_width(width: int | None = None) -> int:
    """Usable columns after the left and right one-space margins."""
    total = _SCREEN_SIZE[0] if width is None else width
    return max(total - 2 * _H_MARGIN, 1)


def _prompt_origin() -> int:
    """1-based column where boxed prompt and transcript text start."""
    return _H_MARGIN + 1


def _enter_fullscreen(stream: TextIO) -> None:
    global _SCREEN_SIZE
    size = shutil.get_terminal_size(fallback=(80, 24))
    width = max(size.columns, 1)
    height = max(size.lines, 1)
    _SCREEN_SIZE = (width, height)
    stream.write(
        f"{_ENTER_ALT_SCREEN}{_OSC_SET_BACKGROUND}{_OSC_SET_FOREGROUND}"
        f"{_SGR_WHITE_ON_BLACK}{_ERASE_DISPLAY}{_CURSOR_HOME}"
    )
    blank = " " * width
    for row in range(height):
        stream.write(f"\033[{row + 1};1H{blank}")
    if height >= _PROMPT_ROWS + 2:
        stream.write(f"\033[1;{_scroll_region_height(height)}r")
    stream.write(f"{_ENABLE_MOUSE}\033[1;{_prompt_origin()}H")
    stream.flush()


def _leave_fullscreen(stream: TextIO) -> None:
    stream.write(
        f"{_SHOW_CURSOR}{_DISABLE_MOUSE}{_RESET_SCROLL_REGION}{_SGR_RESET}"
        f"{_OSC_RESET_BACKGROUND}{_OSC_RESET_FOREGROUND}{_LEAVE_ALT_SCREEN}"
    )
    stream.flush()


def _take_ansi(data: str, index: int) -> tuple[str, int]:
    """Return the ANSI sequence starting at index and the next index."""
    length = len(data)
    if index + 1 >= length:
        return data[index], index + 1
    nxt = data[index + 1]
    if nxt == "[":
        cursor = index + 2
        while cursor < length:
            char = data[cursor]
            cursor += 1
            if "@" <= char <= "~":
                break
        return data[index:cursor], cursor
    if nxt == "]":
        cursor = index + 2
        while cursor < length:
            if data[cursor] == "\x07":
                cursor += 1
                break
            if data[cursor] == "\x1b" and cursor + 1 < length and data[cursor + 1] == "\\":
                cursor += 2
                break
            cursor += 1
        return data[index:cursor], cursor
    if nxt == "O" and index + 2 < length:
        return data[index : index + 3], index + 3
    return data[index : index + 2], index + 2


def _visible_width(text: str) -> int:
    width = 0
    index = 0
    while index < len(text):
        if text[index] == "\x1b":
            _seq, index = _take_ansi(text, index)
            continue
        if text[index] < " ":
            index += 1
            continue
        width += 1
        index += 1
    return width


class _TranscriptStream:
    """Pass-through stdout that keeps visual rows for scrollback."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self.lines: list[str] = []
        self._partial = ""
        self.offset = 0
        self._bol = True

    def write(self, data: str) -> int:
        if not isinstance(data, str):
            data = str(data)
        if _RECORD_TRANSCRIPT:
            jumped = self.offset != 0
            if jumped:
                self.offset = 0
            self._ingest(data)
            if jumped:
                _redraw_transcript(self)
                self._bol = not self._partial
                return len(data)
            return self._stream.write(self._pad_outgoing(data))
        return self._stream.write(data)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        isatty = getattr(self._stream, "isatty", None)
        return bool(callable(isatty) and isatty())

    def fileno(self) -> int:
        return self._stream.fileno()

    @property
    def encoding(self) -> str | None:
        return getattr(self._stream, "encoding", "utf-8")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def rows(self) -> list[str]:
        if self._partial:
            return self.lines + [self._partial]
        return self.lines

    def _pad_outgoing(self, data: str) -> str:
        """Inset live transcript writes by one column on the left."""
        if _H_MARGIN <= 0:
            return data
        pad = " " * _H_MARGIN
        out: list[str] = []
        index = 0
        while index < len(data):
            char = data[index]
            if char == "\x1b":
                seq, index = _take_ansi(data, index)
                out.append(seq)
                continue
            index += 1
            if char in {"\n", "\r"}:
                out.append(char)
                self._bol = True
                continue
            if self._bol and char >= " ":
                out.append(pad)
                self._bol = False
            out.append(char)
        return "".join(out)

    def _ingest(self, data: str) -> None:
        width = _content_width()
        index = 0
        while index < len(data):
            char = data[index]
            if char == "\x1b":
                seq, index = _take_ansi(data, index)
                if seq.endswith("m"):
                    self._partial += seq
                continue
            if char == "\r":
                index += 1
                if index < len(data) and data[index] == "\n":
                    index += 1
                    self.lines.append(self._partial)
                    self._partial = ""
                else:
                    self._partial = ""
                continue
            if char == "\n":
                self.lines.append(self._partial)
                self._partial = ""
                index += 1
                continue
            self._partial += char
            index += 1
            if _visible_width(self._partial) >= width:
                self.lines.append(self._partial)
                self._partial = ""


def _redraw_transcript(
    transcript: _TranscriptStream | None = None,
    reveal_cursor: bool = True,
) -> None:
    target = _TRANSCRIPT if transcript is None else transcript
    if target is None:
        return
    global _RECORD_TRANSCRIPT
    viewport = _scroll_region_height()
    rows = target.rows()
    start = max(0, len(rows) - viewport - target.offset)
    visible = rows[start : start + viewport]
    previous = _RECORD_TRANSCRIPT
    _RECORD_TRANSCRIPT = False
    out = getattr(target, "_stream", sys.stdout)
    try:
        origin = _prompt_origin()
        parts = [_HIDE_CURSOR, _SAVE_CURSOR]
        for index in range(viewport):
            line = visible[index] if index < len(visible) else ""
            parts.append(
                f"\033[{index + 1};1H{_SGR_WHITE_ON_BLACK}\033[K"
                f"\033[{index + 1};{origin}H{line}"
            )
        parts.append(_RESTORE_CURSOR)
        if reveal_cursor:
            parts.append(_SHOW_CURSOR)
        out.write("".join(parts))
        out.flush()
    finally:
        _RECORD_TRANSCRIPT = previous


def _scroll_transcript(delta: int, reveal_cursor: bool = True) -> None:
    """Move the transcript viewport. Positive delta shows older rows."""
    if _TRANSCRIPT is None or delta == 0:
        return
    viewport = _scroll_region_height()
    max_offset = max(0, len(_TRANSCRIPT.rows()) - viewport)
    _TRANSCRIPT.offset = min(max_offset, max(0, _TRANSCRIPT.offset + delta))
    _redraw_transcript(_TRANSCRIPT, reveal_cursor=reveal_cursor)


@contextmanager
def fullscreen_session(stream: TextIO | None = None) -> Iterator[None]:
    """Take the whole terminal with a black background; restore it on exit."""
    global _BLACK_SCREEN, _TRANSCRIPT
    target = sys.stdout if stream is None else stream
    if not _supports_fullscreen(target):
        yield
        return
    _enter_fullscreen(target)
    _BLACK_SCREEN = True
    original_stdout = sys.stdout
    restore_stdout = False
    if target is original_stdout:
        transcript = _TranscriptStream(original_stdout)
        sys.stdout = transcript
        _TRANSCRIPT = transcript
        restore_stdout = True
    try:
        yield
    finally:
        if restore_stdout:
            sys.stdout = original_stdout
            _TRANSCRIPT = None
        _BLACK_SCREEN = False
        _leave_fullscreen(target)


@contextmanager
def prompt_status_session() -> Iterator[None]:
    """Show token metrics under the prompt instead of after the answer."""
    global _METRICS_BELOW_PROMPT
    _METRICS_BELOW_PROMPT = True
    try:
        yield
    finally:
        _METRICS_BELOW_PROMPT = False


def last_response_metrics() -> dict[str, Any] | None:
    """Return metrics from the most recent printed model response."""
    return _LAST_METRICS


def _cached_token_count(usage: Any) -> int | None:
    prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
    return getattr(prompt_tokens_details, "cached_tokens", None)


def response_metrics(
    response: Any, latency_seconds: float, context_window: int
) -> dict[str, Any]:
    """Extract token counts and derive context utilization."""
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    cached_tokens = _cached_token_count(usage)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    if (
        total_tokens is None
        and prompt_tokens is not None
        and completion_tokens is not None
    ):
        total_tokens = prompt_tokens + completion_tokens

    context_percent = (
        round(total_tokens / context_window * 100, 4)
        if total_tokens is not None
        else None
    )
    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_ms": round(latency_seconds * 1_000, 2),
        "context_window_tokens": context_window,
        "context_used_percent": context_percent,
    }


def _combined_metrics(
    responses: list[Any], latency_seconds: float, context_window: int
) -> dict[str, Any]:
    def usage_total(attribute: str) -> int | None:
        values = [
            getattr(getattr(response, "usage", None), attribute, None)
            for response in responses
        ]
        present = [value for value in values if value is not None]
        return sum(present) if present else None

    prompt_tokens = usage_total("prompt_tokens")
    cached_token_counts = [
        _cached_token_count(getattr(response, "usage", None)) for response in responses
    ]
    present_cached_token_counts = [
        count for count in cached_token_counts if count is not None
    ]
    cached_tokens = (
        sum(present_cached_token_counts) if present_cached_token_counts else None
    )
    completion_tokens = usage_total("completion_tokens")
    total_tokens = usage_total("total_tokens")
    if (
        total_tokens is None
        and prompt_tokens is not None
        and completion_tokens is not None
    ):
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "latency_ms": round(latency_seconds * 1_000, 2),
        "context_window_tokens": context_window,
        "context_used_percent": (
            round(total_tokens / context_window * 100, 4)
            if total_tokens is not None
            else None
        ),
        "model_calls": len(responses),
    }


def format_metric(value: Any, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value}{suffix}"


def format_token_count(value: Any) -> str:
    return "n/a" if value is None else f"{value:,}".replace(",", ".")


def empty_metrics(context_window: int | None = None) -> dict[str, Any]:
    """Placeholder metrics shown before the first model response."""
    return {
        "prompt_tokens": None,
        "cached_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "latency_ms": None,
        "context_window_tokens": context_window,
        "context_used_percent": None,
    }


def format_metrics_line(metrics: dict[str, Any] | None) -> str:
    """Render token, latency, and context-window utilization as one line."""
    data = empty_metrics()
    if metrics:
        data.update(metrics)
    return (
        f"prompt={format_metric(data['prompt_tokens'])} tokens | "
        f"cached={format_metric(data['cached_tokens'])} tokens | "
        f"completion={format_metric(data['completion_tokens'])} tokens | "
        f"total={format_metric(data['total_tokens'])} tokens | "
        f"latency={format_metric(data['latency_ms'], ' ms')} | "
        f"context={format_metric(data['context_used_percent'], '%')} of "
        f"{format_token_count(data['context_window_tokens'])} tokens"
    )


def prompt_status_lines(metrics: dict[str, Any] | None) -> tuple[str, str]:
    """Token and context stats shown under the boxed prompt."""
    data = empty_metrics()
    if metrics:
        data.update(metrics)
    return (
        (
            f"prompt={format_metric(data['prompt_tokens'])} tokens | "
            f"cached={format_metric(data['cached_tokens'])} tokens | "
            f"completion={format_metric(data['completion_tokens'])} tokens | "
            f"total={format_metric(data['total_tokens'])} tokens"
        ),
        (
            f"latency={format_metric(data['latency_ms'], ' ms')} | "
            f"context={format_metric(data['context_used_percent'], '%')} of "
            f"{format_token_count(data['context_window_tokens'])} tokens"
        ),
    )


def _output_console() -> Console:
    if _BLACK_SCREEN:
        return Console(style=SCREEN_STYLE, width=_content_width())
    return Console()


def print_user_input(content: str) -> None:
    """Render a submitted user message in the conversation transcript."""
    console = _output_console()
    console.print()
    console.print("you> ", style="bold green", end="")
    console.print(content)


def print_response(content: str, metrics: dict[str, Any]) -> None:
    """Render the Markdown answer and its request metrics."""
    global _LAST_METRICS
    _LAST_METRICS = metrics
    console = _output_console()
    console.print()
    console.print("assistant> ", style="bold cyan", end="")
    console.print(Markdown(content))
    if not _METRICS_BELOW_PROMPT:
        console.print("metrics> " + format_metrics_line(metrics), style="dim")


def model_label(model: str, reasoning_effort: str | None = None) -> str:
    """Right-side prompt label: GPT model id and optional reasoning effort."""
    name = (model or "").strip() or "gpt-oss-120b"
    effort = (reasoning_effort or "").strip()
    return f"{name} ({effort})" if effort else name


def _box_width() -> int:
    if _BLACK_SCREEN:
        columns = _SCREEN_SIZE[0]
    else:
        columns = shutil.get_terminal_size(fallback=(80, 24)).columns
    return max(columns - 2 * _H_MARGIN, 24)


def _fit(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width == 1:
        return "…"
    return text[: width - 1] + "…"


def _horizontal_border(width: int, left: str, right: str, label: str = "") -> str:
    inner = max(width - 2, 0)
    if label:
        title = f" {label} "
        if len(title) <= inner:
            return f"{left}{_BOX_H * (inner - len(title))}{title}{right}"
    return f"{left}{_BOX_H * inner}{right}"


def _top_border(width: int, label: str = "") -> str:
    return _horizontal_border(width, _BOX_TL, _BOX_TR, label)


def _bottom_border(width: int) -> str:
    return _horizontal_border(width, _BOX_BL, _BOX_BR)


def _visible_typed_text(text: str, label: str, width: int) -> tuple[str, str]:
    """Return the clipped input text and model label that fit in the box."""
    inner = max(width - 2, 0)
    prefix = _INPUT_PREFIX
    fitted_label = label
    reserved = len(_INPUT_RIGHT_PAD)
    if fitted_label:
        reserved += len(fitted_label) + 1
    text_budget = inner - len(prefix) - reserved
    if text_budget < 1:
        fitted_label = ""
        reserved = len(_INPUT_RIGHT_PAD)
        text_budget = max(inner - len(prefix) - reserved, 0)
    display = text
    if len(display) > text_budget:
        if text_budget <= 1:
            display = display[-text_budget:]
        else:
            display = "…" + display[-(text_budget - 1) :]
    return display, fitted_label


def _input_row(text: str, label: str, width: int) -> str:
    """Build the boxed input line with the model name on the right."""
    inner = max(width - 2, 0)
    prefix = _INPUT_PREFIX
    display, fitted_label = _visible_typed_text(text, label, width)
    pad = inner - len(prefix) - len(display) - len(fitted_label) - len(
        _INPUT_RIGHT_PAD
    )
    if pad < 0:
        pad = 0
    return (
        f"{_BOX_V}{prefix}{display}{' ' * pad}{fitted_label}"
        f"{_INPUT_RIGHT_PAD}{_BOX_V}"
    )


def _status_lines(metrics: dict[str, Any] | None, width: int) -> tuple[str, str]:
    first, second = prompt_status_lines(metrics)
    return _fit(first, width), _fit(second, width)


def _supports_live_prompt() -> bool:
    if not _BLACK_SCREEN or termios is None or tty is None:
        return False
    isatty = getattr(sys.stdin, "isatty", None)
    if not callable(isatty):
        return False
    try:
        if not isatty():
            return False
        termios.tcgetattr(sys.stdin.fileno())
    except (OSError, ValueError, AttributeError, termios.error):
        return False
    return True


def _prompt_style() -> tuple[str, str]:
    dim = _SGR_DIM
    reset = _SGR_WHITE_ON_BLACK if _BLACK_SCREEN else _SGR_RESET
    return dim, reset


def _styled_prompt_line(line: str, row: int | None) -> str:
    dim, reset = _prompt_style()
    styled = f"{dim}{line}{reset}"
    indent = " " * _H_MARGIN
    if row is None:
        return f"{indent}{styled}\n"
    origin = _prompt_origin()
    return f"\033[{row};1H\033[K\033[{row};{origin}H{styled}"


def _prompt_chrome_ansi(
    text: str,
    label: str,
    metrics: dict[str, Any] | None,
    width: int,
    rows: tuple[int, ...] | None,
    *,
    chrome: bool = True,
) -> str:
    """ANSI to paint the boxed prompt. chrome=False redraws only the input row."""
    if chrome:
        status = _status_lines(metrics, width)
        lines = (
            _top_border(width),
            _input_row(text, label, width),
            _bottom_border(width),
            *status,
        )
        positions = rows if rows is not None else (None,) * len(lines)
        return "".join(
            _styled_prompt_line(line, row) for row, line in zip(positions, lines)
        )
    input_row = None if rows is None else rows[1]
    return _styled_prompt_line(_input_row(text, label, width), input_row)


def _write_prompt_widget(
    text: str,
    label: str,
    metrics: dict[str, Any] | None,
    width: int,
    rows: tuple[int, ...] | None,
) -> None:
    _emit_prompt(_prompt_chrome_ansi(text, label, metrics, width, rows))


def _cursor_column(text: str, label: str, width: int) -> int:
    display, _fitted = _visible_typed_text(text, label, width)
    return _H_MARGIN + len(_BOX_V) + len(_INPUT_PREFIX) + len(display) + 1


def _cursor_ansi(
    text: str, label: str, width: int, rows: tuple[int, ...]
) -> str:
    return f"\033[{rows[1]};{_cursor_column(text, label, width)}H"


def _emit_prompt(data: str, flush: bool = False) -> None:
    """Write prompt chrome without recording it into scrollback."""
    global _RECORD_TRANSCRIPT
    previous = _RECORD_TRANSCRIPT
    _RECORD_TRANSCRIPT = False
    try:
        sys.stdout.write(data)
        if flush:
            sys.stdout.flush()
    finally:
        _RECORD_TRANSCRIPT = previous


def _refresh_live_prompt(
    text: str,
    label: str,
    metrics: dict[str, Any] | None,
    width: int,
    rows: tuple[int, ...],
    *,
    chrome: bool = True,
) -> None:
    """Paint the prompt and reveal the cursor on the typed text in one flush."""
    _emit_prompt(
        (
            f"{_HIDE_CURSOR}"
            f"{_prompt_chrome_ansi(text, label, metrics, width, rows, chrome=chrome)}"
            f"{_cursor_ansi(text, label, width, rows)}"
            f"{_SHOW_CURSOR}"
        ),
        flush=True,
    )


def _utf8_width(byte: int) -> int:
    if byte < 0x80:
        return 1
    if 0xC2 <= byte <= 0xDF:
        return 2
    if 0xE0 <= byte <= 0xEF:
        return 3
    if 0xF0 <= byte <= 0xF4:
        return 4
    return 1


def _is_csi_final(char: str) -> bool:
    return len(char) == 1 and "@" <= char <= "~"


def _os_read(fd: int, count: int) -> bytes:
    while True:
        try:
            return os.read(fd, count)
        except InterruptedError:
            continue


class _RawStdin:
    """Byte-buffered TTY reader so CSI bursts are not lost to TextIO + select."""

    def __init__(self, fd: int) -> None:
        self._fd = fd
        self._buf = bytearray()

    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        return self._fd

    def peek(self, n: int = 1) -> str:
        if self._buf:
            return "\0" * min(max(n, 1), len(self._buf))
        try:
            ready, _, _ = select.select([self._fd], [], [], 0)
        except (OSError, ValueError):
            return ""
        return "\0" if ready else ""

    def read(self, n: int = 1) -> str:
        chars: list[str] = []
        while len(chars) < n:
            char = self._read_one()
            if char == "":
                break
            chars.append(char)
        return "".join(chars)

    def read_byte(self) -> str:
        if not self._fill(1):
            return ""
        byte = self._buf[0]
        del self._buf[0]
        return chr(byte)

    def _fill(self, needed: int) -> bool:
        while len(self._buf) < needed:
            try:
                chunk = _os_read(self._fd, max(_RAW_READ_CHUNK, needed - len(self._buf)))
            except OSError:
                return bool(self._buf)
            if not chunk:
                return bool(self._buf)
            self._buf.extend(chunk)
        return True

    def _read_one(self) -> str:
        if not self._fill(1):
            return ""
        lead = self._buf[0]
        if lead < 0xC2 or lead > 0xF4:
            del self._buf[0]
            return chr(lead)
        width = _utf8_width(lead)
        if not self._fill(width):
            return self.read_byte()
        raw = bytes(self._buf[:width])
        try:
            char = raw.decode("utf-8")
        except UnicodeDecodeError:
            return self.read_byte()
        del self._buf[:width]
        return char


def _key_source(stdin: TextIO) -> TextIO:
    if isinstance(stdin, _RawStdin):
        return stdin
    try:
        if isinstance(stdin, io.TextIOWrapper) and stdin.isatty():
            return _RawStdin(stdin.fileno())
    except (OSError, ValueError, AttributeError, TypeError):
        pass
    return stdin


def _input_pending(stdin: TextIO) -> bool:
    """True when the next read(1) can return without waiting on the fd."""
    peek = getattr(stdin, "peek", None)
    if callable(peek):
        try:
            if peek(1):
                return True
        except (OSError, TypeError, ValueError, io.UnsupportedOperation):
            pass
    decoded = getattr(stdin, "_decoded_chars", None)
    used = getattr(stdin, "_decoded_chars_used", 0)
    if isinstance(decoded, str) and isinstance(used, int) and used < len(decoded):
        return True
    buf = getattr(stdin, "buffer", None)
    buf_peek = getattr(buf, "peek", None)
    if callable(buf_peek):
        try:
            if buf_peek(1):
                return True
        except (OSError, TypeError, ValueError, io.UnsupportedOperation):
            pass
    try:
        ready, _, _ = select.select([stdin.fileno()], [], [], 0)
        return bool(ready)
    except (OSError, TypeError, ValueError, io.UnsupportedOperation, AttributeError):
        return False


def _read_char(stdin: TextIO, timeout: float | None = None) -> str:
    if timeout is not None:
        try:
            isatty = getattr(stdin, "isatty", None)
            if callable(isatty) and isatty() and not _input_pending(stdin):
                ready, _, _ = select.select([stdin.fileno()], [], [], timeout)
                if not ready:
                    return ""
        except (OSError, TypeError, ValueError, io.UnsupportedOperation, AttributeError):
            pass
    char = stdin.read(1)
    return char or ""


def _read_raw_unit(stdin: TextIO) -> str:
    read_byte = getattr(stdin, "read_byte", None)
    if callable(read_byte):
        try:
            return read_byte() or ""
        except (OSError, TypeError, ValueError):
            return ""
    return _read_char(stdin)


def _decode_sgr_mouse(payload: str, event: str) -> tuple[str, str]:
    if event == "m":
        return "ignore", ""
    button_text = payload.split(";", 1)[0]
    try:
        button = int(button_text)
    except ValueError:
        return "ignore", ""
    if button & 64:
        return ("scroll_up" if button % 2 == 0 else "scroll_down"), ""
    return "ignore", ""


def _decode_x10_mouse(button_char: str) -> tuple[str, str]:
    if not button_char:
        return "ignore", ""
    button = ord(button_char) - 32
    if button & 64:
        return ("scroll_up" if button % 2 == 0 else "scroll_down"), ""
    return "ignore", ""


def _decode_csi(seq: str) -> tuple[str, str]:
    if seq.endswith("A"):
        return "scroll_up", ""
    if seq.endswith("B"):
        return "scroll_down", ""
    if seq.startswith("[5") and seq.endswith("~"):
        return "page_up", ""
    if seq.startswith("[6") and seq.endswith("~"):
        return "page_down", ""
    if seq.startswith("[<") and seq[-1:] in {"M", "m"}:
        return _decode_sgr_mouse(seq[2:-1], seq[-1])
    return "ignore", ""


def _read_csi(stdin: TextIO) -> tuple[str, str]:
    """Consume a CSI sequence without per-byte timeouts so payloads cannot leak."""
    first = _read_char(stdin)
    if not first:
        return "ignore", ""
    if first == "M":
        button = _read_raw_unit(stdin)
        _read_raw_unit(stdin)
        _read_raw_unit(stdin)
        return _decode_x10_mouse(button)
    body = [first]
    if not _is_csi_final(first):
        while len(body) < _CSI_MAX:
            end = _read_char(stdin)
            if not end:
                break
            body.append(end)
            if _is_csi_final(end):
                break
    return _decode_csi("[" + "".join(body))


def _read_key(stdin: TextIO) -> tuple[str, str]:
    """Read one key or pointer event. Wheel/arrows never count as EOF."""
    char = _read_char(stdin)
    if char == "":
        return "eof", ""
    if char in {"\n", "\r"}:
        return "enter", ""
    if char in {"\x7f", "\b"}:
        return "backspace", ""
    if char == "\x04":
        return "ctrl-d", ""
    if char == "\x9b":
        return _read_csi(stdin)
    if char != "\x1b":
        if char >= " ":
            return "char", char
        return "ignore", ""

    nxt = _read_char(stdin, timeout=_ESC_FOLLOW_TIMEOUT)
    if not nxt:
        return "ignore", ""
    if nxt == "O":
        arrow = _read_char(stdin)
        if arrow in {"A", "a"}:
            return "scroll_up", ""
        if arrow in {"B", "b"}:
            return "scroll_down", ""
        return "ignore", ""
    if nxt == "[":
        return _read_csi(stdin)
    return "ignore", ""


def _place_prompt_cursor(text: str, label: str, width: int, rows: tuple[int, ...]) -> None:
    """Move the visible cursor onto the typed text without redrawing chrome."""
    _emit_prompt(
        f"{_HIDE_CURSOR}{_cursor_ansi(text, label, width, rows)}{_SHOW_CURSOR}",
        flush=True,
    )


def _read_live_prompt(label: str, metrics: dict[str, Any]) -> str:
    global _RECORD_TRANSCRIPT
    height = _SCREEN_SIZE[1]
    box_width = max(_content_width(), 24)
    top_row = max(height - _PROMPT_ROWS + 1, 1)
    rows = tuple(top_row + index for index in range(_PROMPT_ROWS))
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    text = ""
    previous = _RECORD_TRANSCRIPT
    _RECORD_TRANSCRIPT = False
    sys.stdout.write(_SAVE_CURSOR)
    sys.stdout.flush()
    try:
        tty.setcbreak(fd)
        keys = _key_source(sys.stdin)
        _refresh_live_prompt(text, label, metrics, box_width, rows)
        while True:
            kind, value = _read_key(keys)
            if kind == "eof" or kind == "ctrl-d":
                if not text:
                    raise EOFError
                if kind == "eof":
                    break
                continue
            if kind == "enter":
                break
            if kind == "scroll_up":
                _scroll_transcript(_WHEEL_LINES, reveal_cursor=False)
                _place_prompt_cursor(text, label, box_width, rows)
                continue
            if kind == "scroll_down":
                _scroll_transcript(-_WHEEL_LINES, reveal_cursor=False)
                _place_prompt_cursor(text, label, box_width, rows)
                continue
            if kind == "page_up":
                _scroll_transcript(
                    max(_scroll_region_height() - 1, 1), reveal_cursor=False
                )
                _place_prompt_cursor(text, label, box_width, rows)
                continue
            if kind == "page_down":
                _scroll_transcript(
                    -max(_scroll_region_height() - 1, 1), reveal_cursor=False
                )
                _place_prompt_cursor(text, label, box_width, rows)
                continue
            if kind == "ignore":
                continue
            if kind == "backspace":
                text = text[:-1]
            elif kind == "char":
                text += value
            else:
                continue
            _refresh_live_prompt(
                text, label, metrics, box_width, rows, chrome=False
            )
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write(f"{_SHOW_CURSOR}{_RESTORE_CURSOR}")
        sys.stdout.flush()
        _RECORD_TRANSCRIPT = previous
    return text


def _read_fallback_prompt(label: str, metrics: dict[str, Any]) -> str:
    width = _box_width()
    indent = " " * _H_MARGIN
    print()
    print(indent + _top_border(width, label))
    try:
        text = input(f"{indent}{_BOX_V}{_INPUT_PREFIX}")
    except (EOFError, KeyboardInterrupt):
        print()
        print(indent + _bottom_border(width))
        for line in _status_lines(metrics, width):
            print(indent + line)
        raise
    print(indent + _bottom_border(width))
    for line in _status_lines(metrics, width):
        print(indent + line)
    return text


def read_prompt(
    model: str,
    reasoning_effort: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> str:
    """Read a line from a Grok-style boxed prompt with metrics underneath."""
    label = model_label(model, reasoning_effort)
    snapshot = empty_metrics() if metrics is None else metrics
    if _supports_live_prompt():
        return _read_live_prompt(label, snapshot)
    return _read_fallback_prompt(label, snapshot)
