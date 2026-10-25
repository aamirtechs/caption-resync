#!/usr/bin/env python3
"""caption-resync: batch-fix out-of-sync subtitles from the command line.

Shift, stretch, convert (SRT <-> WebVTT), strip styling tags, renumber and
validate whole folders of subtitle files. Zero dependencies, Python 3.9+.

Typical use: your subtitles lag the audio by a constant offset, so every
cue arrives late. One command fixes the whole season:

    caption-resync --shift -2.5s --dir subtitles/ --glob "*.srt"

Outputs land next to the inputs with a ``.resync`` suffix by default, so the
originals are never touched.
"""

from __future__ import annotations

import argparse
import glob as glob_module
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

__version__ = "1.0.0"

# A timestamp: [hh:]mm:ss,mmm or [hh:]mm:ss.mmm (1 to 3 millisecond digits).
TS_RE = re.compile(
    r"(?:(\d+):)?(\d{1,2}):(\d{1,2})[,.](\d{1,3})"
)


@dataclass
class ParseIssue:
    """Something the parser noticed while reading a subtitle file."""

    line: int
    severity: str  # "error" or "warning"
    message: str

    def __str__(self) -> str:
        return "line %d [%s] %s" % (self.line, self.severity, self.message)


@dataclass
class Cue:
    """One subtitle cue. Times are integer milliseconds."""

    start: int
    end: int
    text: str
    cue_id: Optional[str] = None
    settings: str = ""  # WebVTT cue settings, e.g. "align:start position:0%"


@dataclass
class SubtitleFile:
    """Parsed subtitle document plus any issues found while parsing."""

    cues: List[Cue] = field(default_factory=list)
    fmt: str = "srt"  # "srt" or "vtt"
    header_blocks: List[str] = field(default_factory=list)  # VTT STYLE/REGION
    # VTT NOTE comments as (cue_index, text): the note sits before the cue at
    # that index, so an index equal to ``len(cues)`` means "after the last cue".
    # Only WebVTT output can carry them; SRT has no comment syntax.
    notes: List[Tuple[int, str]] = field(default_factory=list)
    issues: List[ParseIssue] = field(default_factory=list)


# --------------------------------------------------------------------------
# Timestamps and offsets
# --------------------------------------------------------------------------

def parse_timestamp(text: str) -> int:
    """Parse ``hh:mm:ss,mmm`` / ``hh:mm:ss.mmm`` / ``mm:ss.mmm`` into ms."""
    text = text.strip()
    m = TS_RE.fullmatch(text)
    if not m:
        raise ValueError("bad timestamp: %r" % text)
    hours, minutes, seconds, millis = m.groups()
    hours = int(hours) if hours else 0
    millis = (millis + "000")[:3]  # pad "5" to "500"
    total = ((hours * 3600) + int(minutes) * 60 + int(seconds)) * 1000
    return total + int(millis)


def format_srt_ts(ms: int) -> str:
    """Format milliseconds as ``HH:MM:SS,mmm``."""
    if ms < 0:
        ms = 0
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, millis = divmod(rem, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, millis)


def format_vtt_ts(ms: int) -> str:
    """Format milliseconds as ``HH:MM:SS.mmm``."""
    if ms < 0:
        ms = 0
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, millis = divmod(rem, 1000)
    return "%02d:%02d:%02d.%03d" % (h, m, s, millis)


OFFSET_RE = re.compile(r"^(?P<sign>[+-]?)(?P<body>.*)$")


def parse_offset(text: str) -> int:
    """Parse a --shift offset into integer milliseconds.

    Accepted forms (sign optional): ``2500`` or ``2500ms`` (milliseconds),
    ``2.5s`` (seconds), ``hh:mm:ss,mmm`` / ``hh:mm:ss.mmm`` (timestamp).
    """
    text = text.strip()
    m = OFFSET_RE.match(text)
    assert m is not None
    sign = -1 if m.group("sign") == "-" else 1
    body = m.group("body").strip()
    if not body:
        raise ValueError("empty offset")
    try:
        if re.fullmatch(r"\d+", body):
            return sign * int(body)
        if body.lower().endswith("ms") and re.fullmatch(r"\d+ms", body.lower()):
            return sign * int(body[:-2])
        if body.lower().endswith("s"):
            return sign * int(round(float(body[:-1]) * 1000))
        # Fall through to timestamp parsing.
        return sign * parse_timestamp(body)
    except ValueError:
        raise ValueError(
            "bad offset %r: use ms (2500), seconds (2.5s), "
            "or hh:mm:ss,mmm" % text
        )


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def _split_blocks(lines: List[str]) -> List[Tuple[int, List[str]]]:
    """Split lines into blank-line-separated blocks. Returns (start_line, lines)."""
    blocks: List[Tuple[int, List[str]]] = []
    current: List[str] = []
    start = 1
    for i, line in enumerate(lines, start=1):
        if line.strip() == "":
            if current:
                blocks.append((start, current))
                current = []
        else:
            if not current:
                start = i
            current.append(line)
    if current:
        blocks.append((start, current))
    return blocks


def _split_timestamps(ts_line: str) -> Optional[Tuple[int, int, str]]:
    """Split a ``start --> end [settings]`` line into (start, end, settings)."""
    if "-->" not in ts_line:
        return None
    left, right = ts_line.split("-->", 1)
    right = right.strip()
    # The end timestamp is the first token; anything after is cue settings.
    parts = right.split(None, 1)
    settings = parts[1] if len(parts) > 1 else ""
    try:
        start = parse_timestamp(left)
        end = parse_timestamp(parts[0])
    except ValueError:
        return None
    return start, end, settings


def parse_srt(text: str, name: str = "<input>") -> SubtitleFile:
    """Parse SubRip text. Returns the cues plus any issues found."""
    doc = SubtitleFile(fmt="srt")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for start_line, block in _split_blocks(lines):
        ts_index = 0
        cue_id: Optional[str] = None
        if len(block) >= 2 and "-->" in block[1] and "-->" not in block[0]:
            cue_id = block[0].strip()
            ts_index = 1
        elif "-->" not in block[0]:
            doc.issues.append(ParseIssue(
                start_line, "error",
                "block %s is not a cue (no timestamp line)" % (cue_id or "?"),
            ))
            continue
        parsed = _split_timestamps(block[ts_index])
        if parsed is None:
            doc.issues.append(ParseIssue(
                start_line + ts_index, "error",
                "malformed timestamp line: %r" % block[ts_index].strip(),
            ))
            continue
        start, end, _settings = parsed
        body = "\n".join(block[ts_index + 1:]).strip()
        if not body:
            doc.issues.append(ParseIssue(
                start_line, "warning", "cue has no text; keeping it",
            ))
        doc.cues.append(Cue(start=start, end=end, text=body, cue_id=cue_id))
    return doc


def parse_vtt(text: str, name: str = "<input>") -> SubtitleFile:
    """Parse WebVTT text (header, NOTE/STYLE/REGION blocks, cues)."""
    doc = SubtitleFile(fmt="vtt")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    # Find the WEBVTT header line (first non-blank line).
    first = next((l for l in lines if l.strip()), "")
    if not first.startswith("WEBVTT"):
        doc.issues.append(ParseIssue(
            1, "error", "missing WEBVTT header; treating as WebVTT anyway",
        ))
        header_end = 0
    else:
        header_end = lines.index(first)
    for start_line, block in _split_blocks(lines[header_end + 1:]):
        head = block[0].strip()
        if head == "NOTE" or head.startswith("NOTE "):
            # Comments carry no timing, but losing them loses author metadata
            # on a WebVTT -> WebVTT pass; keep them for the renderer to place.
            doc.notes.append((len(doc.cues), "\n".join(line.rstrip() for line in block)))
            continue
        if head == "STYLE" or head == "REGION":
            doc.header_blocks.append("\n".join(block).strip())
            continue
        ts_index = 0
        cue_id: Optional[str] = None
        if len(block) >= 2 and "-->" not in block[0] and "-->" in block[1]:
            cue_id = block[0].strip()
            ts_index = 1
        elif "-->" not in block[0]:
            doc.issues.append(ParseIssue(
                start_line, "error",
                "block is not a cue (no timestamp line)",
            ))
            continue
        parsed = _split_timestamps(block[ts_index])
        if parsed is None:
            doc.issues.append(ParseIssue(
                start_line + ts_index, "error",
                "malformed timestamp line: %r" % block[ts_index].strip(),
            ))
            continue
        start, end, settings = parsed
        body = "\n".join(block[ts_index + 1:]).strip()
        doc.cues.append(Cue(
            start=start, end=end, text=body, cue_id=cue_id, settings=settings,
        ))
    return doc


def parse_text(text: str, name: str = "<input>") -> SubtitleFile:
    """Auto-detect SRT vs WebVTT and parse."""
    first = next((l for l in text.splitlines() if l.strip()), "")
    if first.startswith("WEBVTT"):
        return parse_vtt(text, name)
    return parse_srt(text, name)


def read_file(path: str) -> SubtitleFile:
    """Read a subtitle file (BOM-tolerant UTF-8) and parse it."""
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        text = fh.read()
    return parse_text(text, path)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def render_srt(doc: SubtitleFile) -> str:
    out: List[str] = []
    for i, cue in enumerate(doc.cues, start=1):
        out.append(cue.cue_id if cue.cue_id is not None else str(i))
        out.append("%s --> %s" % (format_srt_ts(cue.start), format_srt_ts(cue.end)))
        out.append(cue.text)
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n" if out else ""


def render_vtt(doc: SubtitleFile) -> str:
    out: List[str] = ["WEBVTT", ""]
    for block in doc.header_blocks:
        out.append(block)
        out.append("")
    notes_by_index: Dict[int, List[str]] = {}
    for index, note in doc.notes:
        notes_by_index.setdefault(index, []).append(note)
    for index, cue in enumerate(doc.cues):
        for note in notes_by_index.get(index, []):
            out.append(note)
            out.append("")
        if cue.cue_id is not None:
            out.append(cue.cue_id)
        ts = "%s --> %s" % (format_vtt_ts(cue.start), format_vtt_ts(cue.end))
        if cue.settings:
            ts += " " + cue.settings
        out.append(ts)
        out.append(cue.text)
        out.append("")
    # Notes that sat after the last cue keep their place after it.
    for note in notes_by_index.get(len(doc.cues), []):
        out.append(note)
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n" if len(out) > 2 else "WEBVTT\n"


def render(doc: SubtitleFile, fmt: str) -> str:
    if fmt == "vtt":
        return render_vtt(doc)
    return render_srt(doc)


# --------------------------------------------------------------------------
# Operations
# --------------------------------------------------------------------------

def op_shift(doc: SubtitleFile, offset_ms: int) -> int:
    """Shift every cue by offset_ms. Times clamp at zero; returns clamp count."""
    clamped = 0
    for cue in doc.cues:
        start, end = cue.start + offset_ms, cue.end + offset_ms
        if start < 0 or end < 0:
            clamped += 1
        cue.start = max(0, start)
        cue.end = max(0, end)
    return clamped


def op_stretch(doc: SubtitleFile, factor: float) -> None:
    """Multiply every timestamp by factor (fixes framerate drift)."""
    if factor <= 0:
        raise ValueError("stretch factor must be positive, got %r" % factor)
    for cue in doc.cues:
        cue.start = int(round(cue.start * factor))
        cue.end = int(round(cue.end * factor))


TAG_RE = re.compile(r"<[^<>]*>")
BRACE_RE = re.compile(r"\{[^{}]*\}")


def strip_styles(text: str) -> str:
    """Remove HTML-like tags (<i>, <b>, <font ...>) and {..} override tags."""
    text = BRACE_RE.sub("", text)
    text = TAG_RE.sub("", text)
    # Collapse whitespace left behind by tag removal.
    text = re.sub(r"[ \t]{2,}", " ", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def op_strip_styles(doc: SubtitleFile) -> None:
    for cue in doc.cues:
        cue.text = strip_styles(cue.text)
        cue.settings = ""  # position/align settings are styling too


def op_renumber(doc: SubtitleFile) -> None:
    """Renumber numeric cue identifiers sequentially from 1."""
    n = 0
    for cue in doc.cues:
        if cue.cue_id is None or cue.cue_id.isdigit():
            n += 1
            cue.cue_id = str(n)


# --------------------------------------------------------------------------
# Validation (--check)
# --------------------------------------------------------------------------

@dataclass
class CheckIssue:
    severity: str  # "error" or "warning"
    message: str

    def __str__(self) -> str:
        return "[%s] %s" % (self.severity.upper(), self.message)


def check_doc(doc: SubtitleFile, path: str) -> List[CheckIssue]:
    """Validate cue timing: malformed cues, bad durations, overlaps, order."""
    found: List[CheckIssue] = []
    for issue in doc.issues:
        found.append(CheckIssue(issue.severity, "%s: %s" % (path, issue)))
    prev: Optional[Cue] = None
    for i, cue in enumerate(doc.cues, start=1):
        label = "cue %s" % (cue.cue_id if cue.cue_id is not None else i)
        if cue.end <= cue.start:
            found.append(CheckIssue(
                "error",
                "%s: %s has zero or negative duration (%d ms)" % (path, label, cue.end - cue.start),
            ))
        if prev is not None:
            if cue.start < prev.start:
                found.append(CheckIssue(
                    "error",
                    "%s: %s starts before the previous cue (out of order)" % (path, label),
                ))
            elif cue.start < prev.end:
                found.append(CheckIssue(
                    "error",
                    "%s: %s overlaps the previous cue by %d ms" % (path, label, prev.end - cue.start),
                ))
        prev = cue
    return found


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def collect_inputs(args: argparse.Namespace) -> List[str]:
    paths: List[str] = list(args.files or [])
    if args.dir:
        if not os.path.isdir(args.dir):
            raise SystemExit("error: --dir is not a directory: %s" % args.dir)
        for pattern in args.glob.split(","):
            pattern = pattern.strip()
            if pattern:
                paths.extend(glob_module.glob(os.path.join(args.dir, pattern)))
    # De-duplicate, keep order.
    seen = set()
    unique: List[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return sorted(unique)


def output_path(input_path: str, args: argparse.Namespace, out_fmt: str) -> str:
    base = os.path.basename(input_path)
    stem, _ext = os.path.splitext(base)
    new_ext = ".vtt" if out_fmt == "vtt" else ".srt"
    name = stem + args.suffix + new_ext
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        return os.path.join(args.out_dir, name)
    return os.path.join(os.path.dirname(input_path), name)


def describe_ops(args: argparse.Namespace) -> List[str]:
    ops: List[str] = []
    if args.shift is not None:
        ops.append("shift %+d ms" % args.shift_ms)
    if args.stretch is not None:
        ops.append("stretch x%s" % args.stretch)
    if args.to:
        ops.append("convert to %s" % args.to.upper())
    if args.strip_styles:
        ops.append("strip styles")
    if args.renumber:
        ops.append("renumber")
    return ops


def process_file(path: str, args: argparse.Namespace, out_fmt: str) -> Tuple[bool, int]:
    """Process one file. Returns (ok, clamped_count)."""
    doc = read_file(path)
    parse_errors = [i for i in doc.issues if i.severity == "error"]
    if parse_errors and not args.force:
        print("%s: skipped, %d parse error(s) (use --check for detail, "
              "--force to process anyway)" % (path, len(parse_errors)))
        return False, 0

    clamped = 0
    if args.shift is not None:
        clamped = op_shift(doc, args.shift_ms)
    if args.stretch is not None:
        op_stretch(doc, args.stretch)
    if args.strip_styles:
        op_strip_styles(doc)
    if args.renumber:
        op_renumber(doc)

    text = render(doc, out_fmt)
    dest = output_path(path, args, out_fmt)
    if args.dry_run:
        first = doc.cues[0] if doc.cues else None
        print("%s -> %s [%s, %d cue(s)]" % (
            path, dest, ", ".join(describe_ops(args)), len(doc.cues)))
        if first is not None:
            print("  first cue now: %s --> %s" % (
                format_srt_ts(first.start), format_srt_ts(first.end)))
        return True, clamped

    with open(dest, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    if clamped and not args.quiet:
        print("warning: %d cue(s) clamped at 00:00:00 in %s" % (clamped, path))
    if not args.quiet:
        print("%s -> %s [%d cue(s)]" % (path, dest, len(doc.cues)))
    return True, clamped


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="caption-resync",
        description="Batch-fix out-of-sync subtitles: shift, stretch, convert "
                    "SRT/WebVTT, strip styles, renumber, validate.",
    )
    p.add_argument("files", nargs="*", help="subtitle files to process")
    p.add_argument("--dir", help="process every matching file in a directory")
    p.add_argument("--glob", default="*.srt,*.vtt",
                   help="comma-separated patterns for --dir (default: *.srt,*.vtt)")
    p.add_argument("--shift",
                   help="shift all cues: ms (2500), seconds (2.5s), "
                        "or hh:mm:ss,mmm; negative moves earlier")
    p.add_argument("--stretch", type=float,
                   help="multiply timestamps, e.g. 1.0427 to fix 23.976 -> 25 fps drift")
    p.add_argument("--to", choices=["srt", "vtt"],
                   help="convert output format")
    p.add_argument("--strip-styles", action="store_true",
                   help="remove <i>/<b>/<font> tags, {..} tags, cue position settings")
    p.add_argument("--renumber", action="store_true",
                   help="renumber cue identifiers sequentially from 1")
    p.add_argument("--suffix", default=".resync",
                   help="output filename suffix (default: .resync)")
    p.add_argument("--out-dir", help="write outputs here instead of next to inputs")
    p.add_argument("--dry-run", action="store_true",
                   help="show what would change without writing files")
    p.add_argument("--check", action="store_true",
                   help="validate files and report timing problems (exit 1 on errors)")
    p.add_argument("--force", action="store_true",
                   help="process files even when parsing found errors")
    p.add_argument("--quiet", "-q", action="store_true", help="less output")
    p.add_argument("--version", action="version", version="%(prog)s " + __version__)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.shift is not None:
        try:
            args.shift_ms = parse_offset(args.shift)
        except ValueError as exc:
            print("error: %s" % exc)
            return 2

    paths = collect_inputs(args)
    if not paths:
        print("error: no input files (pass files or use --dir)")
        return 2

    # --check mode: validate only.
    if args.check:
        problems: List[CheckIssue] = []
        total_cues = 0
        for path in paths:
            try:
                doc = read_file(path)
            except OSError as exc:
                problems.append(CheckIssue("error", "%s: cannot read: %s" % (path, exc)))
                continue
            except UnicodeDecodeError as exc:
                problems.append(CheckIssue("error", "%s: not valid UTF-8: %s" % (path, exc)))
                continue
            total_cues += len(doc.cues)
            problems.extend(check_doc(doc, path))
        errors = [x for x in problems if x.severity == "error"]
        warnings = [x for x in problems if x.severity == "warning"]
        for issue in problems:
            print(issue)
        print("%d file(s), %d cue(s): %d error(s), %d warning(s)" % (
            len(paths), total_cues, len(errors), len(warnings)))
        return 1 if errors else 0

    if not describe_ops(args):
        print("error: nothing to do (add --shift, --stretch, --to, "
              "--strip-styles, --renumber, or use --check)")
        return 2

    out_fmt = args.to or None
    ok_count = 0
    for path in paths:
        fmt = out_fmt or ("vtt" if path.lower().endswith(".vtt") else "srt")
        try:
            ok, _clamped = process_file(path, args, fmt)
        except OSError as exc:
            print("%s: error: %s" % (path, exc))
            continue
        if ok:
            ok_count += 1
    if not args.quiet:
        print("done: %d/%d file(s) processed" % (ok_count, len(paths)))
    return 0 if ok_count == len(paths) else 1


if __name__ == "__main__":
    sys.exit(main())
