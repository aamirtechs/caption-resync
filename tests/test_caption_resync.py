"""Tests for caption_resync. Run with: python3 -m unittest discover tests -v"""

import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import caption_resync
from caption_resync import (
    check_doc,
    format_srt_ts,
    format_vtt_ts,
    main,
    op_renumber,
    op_shift,
    op_stretch,
    op_strip_styles,
    parse_offset,
    parse_text,
    stats_for_doc,
    parse_timestamp,
    render,
    render_srt,
    render_vtt,
    strip_styles,
)

MODULE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = [sys.executable, os.path.join(MODULE, "caption_resync.py")]

SRT_SAMPLE = """1
00:00:01,000 --> 00:00:04,000
Hello world

2
00:00:05,500 --> 00:00:08,250
Second line
with two rows

3
00:01:10,000 --> 00:01:12,000
Third cue
"""

VTT_SAMPLE = """WEBVTT

00:00:01.000 --> 00:00:04.000
Hello world

cue-two
00:00:05.500 --> 00:00:08.250 align:start position:0%
Second line
"""


def parse_ok(text):
    doc = parse_text(text)
    errors = [i for i in doc.issues if i.severity == "error"]
    assert errors == [], "unexpected errors: %s" % errors
    return doc


class TestTimestampParsing(unittest.TestCase):
    def test_comma_milliseconds(self):
        self.assertEqual(parse_timestamp("00:00:01,000"), 1000)

    def test_dot_milliseconds(self):
        self.assertEqual(parse_timestamp("00:00:01.000"), 1000)

    def test_hours(self):
        self.assertEqual(parse_timestamp("01:02:03,456"),
                         (3600 + 120 + 3) * 1000 + 456)

    def test_no_hours_vtt_style(self):
        self.assertEqual(parse_timestamp("02:03.500"), 123500)

    def test_short_millis_padded(self):
        self.assertEqual(parse_timestamp("00:00:01,5"), 1500)

    def test_bad_timestamp_raises(self):
        with self.assertRaises(ValueError):
            parse_timestamp("not a time")

    def test_format_srt(self):
        self.assertEqual(format_srt_ts(3723456), "01:02:03,456")

    def test_format_vtt(self):
        self.assertEqual(format_vtt_ts(3723456), "01:02:03.456")


class TestOffsetParsing(unittest.TestCase):
    def test_plain_ms(self):
        self.assertEqual(parse_offset("2500"), 2500)

    def test_negative_ms(self):
        self.assertEqual(parse_offset("-1500"), -1500)

    def test_ms_suffix(self):
        self.assertEqual(parse_offset("750ms"), 750)

    def test_seconds(self):
        self.assertEqual(parse_offset("2.5s"), 2500)

    def test_negative_seconds(self):
        self.assertEqual(parse_offset("-2.5s"), -2500)

    def test_timestamp_offset(self):
        self.assertEqual(parse_offset("00:00:02,500"), 2500)

    def test_negative_timestamp(self):
        self.assertEqual(parse_offset("-00:01:00.000"), -60000)

    def test_bad_offset_raises(self):
        with self.assertRaises(ValueError):
            parse_offset("banana")


class TestSrtParsing(unittest.TestCase):
    def test_basic(self):
        doc = parse_ok(SRT_SAMPLE)
        self.assertEqual(len(doc.cues), 3)
        self.assertEqual(doc.fmt, "srt")

    def test_cue_fields(self):
        doc = parse_ok(SRT_SAMPLE)
        cue = doc.cues[0]
        self.assertEqual((cue.start, cue.end), (1000, 4000))
        self.assertEqual(cue.text, "Hello world")
        self.assertEqual(cue.cue_id, "1")

    def test_multiline_text(self):
        doc = parse_ok(SRT_SAMPLE)
        self.assertEqual(doc.cues[1].text, "Second line\nwith two rows")

    def test_bom_tolerated(self):
        doc = parse_ok("﻿" + SRT_SAMPLE)
        self.assertEqual(len(doc.cues), 3)

    def test_malformed_timestamp_is_error(self):
        doc = parse_text("1\n00:00:01,000 -> 00:00:04,000\nHi\n")
        errors = [i for i in doc.issues if i.severity == "error"]
        self.assertTrue(errors)

    def test_block_without_timestamp_is_error(self):
        doc = parse_text("just some words\n\nmore words\n")
        errors = [i for i in doc.issues if i.severity == "error"]
        self.assertEqual(len(errors), 2)


class TestVttParsing(unittest.TestCase):
    def test_basic(self):
        doc = parse_ok(VTT_SAMPLE)
        self.assertEqual(len(doc.cues), 2)
        self.assertEqual(doc.fmt, "vtt")

    def test_cue_identifier(self):
        doc = parse_ok(VTT_SAMPLE)
        self.assertEqual(doc.cues[1].cue_id, "cue-two")

    def test_cue_settings(self):
        doc = parse_ok(VTT_SAMPLE)
        self.assertEqual(doc.cues[1].settings, "align:start position:0%")

    def test_note_block_skipped(self):
        doc = parse_ok("WEBVTT\n\nNOTE this is a comment\n\n"
                       "00:00:01.000 --> 00:00:02.000\nHi\n")
        self.assertEqual(len(doc.cues), 1)

    def test_note_blocks_are_stored_with_their_position(self):
        doc = parse_ok(
            "WEBVTT\n\nNOTE first note\n\n"
            "00:00:01.000 --> 00:00:02.000\nhi\n\n"
            "NOTE between cues\n\n"
            "00:00:03.000 --> 00:00:04.000\nbye\n\n"
            "NOTE trailing\n"
        )
        self.assertEqual(
            doc.notes,
            [(0, "NOTE first note"), (1, "NOTE between cues"), (2, "NOTE trailing")],
        )
        self.assertEqual(len(doc.cues), 2)

    def test_note_blocks_round_trip_in_place(self):
        source = (
            "WEBVTT\n\nNOTE first note\n\n"
            "00:00:01.000 --> 00:00:02.000\nhi\n\n"
            "NOTE between cues\n\n"
            "00:00:03.000 --> 00:00:04.000\nbye\n\n"
            "NOTE trailing\n"
        )
        self.assertEqual(render_vtt(parse_ok(source)), source)

    def test_notes_are_dropped_on_vtt_to_srt(self):
        doc = parse_ok(
            "WEBVTT\n\nNOTE keep me out of srt\n\n"
            "00:00:01.000 --> 00:00:02.000\nhi\n"
        )
        self.assertEqual(len(doc.notes), 1)
        self.assertNotIn("NOTE", render_srt(doc))

    def test_strip_styles_leaves_notes_alone(self):
        doc = parse_ok(
            "WEBVTT\n\nNOTE keep me\n\n"
            "00:00:01.000 --> 00:00:02.000 align:start\n<i>hi</i>\n"
        )
        op_strip_styles(doc)
        self.assertEqual(doc.notes, [(0, "NOTE keep me")])
        self.assertIn("NOTE keep me", render_vtt(doc))

    def test_style_block_preserved(self):
        doc = parse_ok("WEBVTT\n\nSTYLE\n::cue { color: lime }\n\n"
                       "00:00:01.000 --> 00:00:02.000\nHi\n")
        self.assertEqual(len(doc.header_blocks), 1)
        self.assertIn("::cue", doc.header_blocks[0])

    def test_header_with_title(self):
        doc = parse_ok("WEBVTT - my video\n\n00:00:01.000 --> 00:00:02.000\nHi\n")
        self.assertEqual(len(doc.cues), 1)


class TestStats(unittest.TestCase):
    def test_counts_words_and_average_wpm(self):
        doc = parse_ok(
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:06.000\n"
            + " ".join(["word"] * 10) + "\n\n"
            "00:00:06.000 --> 00:00:09.000\ntwo words\n"
        )
        stats = stats_for_doc(doc)
        self.assertEqual(stats["cues"], 2)
        self.assertEqual(stats["words"], 12)
        # 100 WPM and 40 WPM -> mean 70
        self.assertAlmostEqual(stats["wpm"], 70.0)
        self.assertEqual(stats["over"], 0)

    def test_flags_cues_above_the_readable_speed(self):
        doc = parse_ok(
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\n"
            + " ".join(["word"] * 10) + "\n"
        )
        stats = stats_for_doc(doc)
        self.assertEqual(stats["over"], 1, "10 words in 1s is 600 WPM")

    def test_zero_duration_cue_is_skipped_for_speed(self):
        doc = parse_ok(
            "WEBVTT\n\n00:00:01.000 --> 00:00:01.000\nthree words here\n"
        )
        stats = stats_for_doc(doc)
        self.assertEqual(stats["cues"], 1)
        self.assertEqual(stats["words"], 3)
        self.assertEqual(stats["wpm"], 0.0)


class TestOperations(unittest.TestCase):
    def test_shift_positive(self):
        doc = parse_ok(SRT_SAMPLE)
        op_shift(doc, 2500)
        self.assertEqual(doc.cues[0].start, 3500)
        self.assertEqual(doc.cues[0].end, 6500)

    def test_shift_negative_clamped(self):
        doc = parse_ok(SRT_SAMPLE)
        clamped = op_shift(doc, -2500)
        self.assertEqual(clamped, 1)
        self.assertEqual(doc.cues[0].start, 0)
        self.assertEqual(doc.cues[0].end, 1500)
        # Later cues keep their relative position, moved by the same offset.
        self.assertEqual(doc.cues[1].start, 3000)


    def test_shift_far_negative_keeps_duration(self):
        doc = parse_ok(SRT_SAMPLE)
        op_shift(doc, -5000)
        # 整条被推到零之前：起点截到 0，但时长保留（不能变成 0 ms）
        self.assertEqual(doc.cues[0].start, 0)
        self.assertEqual(doc.cues[0].end, doc.cues[0].start + 3000)
        self.assertGreater(doc.cues[0].end, doc.cues[0].start)

    def test_stretch(self):
        doc = parse_ok(SRT_SAMPLE)
        op_stretch(doc, 2.0)
        self.assertEqual(doc.cues[0].start, 2000)
        self.assertEqual(doc.cues[0].end, 8000)

    def test_stretch_bad_factor(self):
        doc = parse_ok(SRT_SAMPLE)
        with self.assertRaises(ValueError):
            op_stretch(doc, 0)

    def test_strip_styles(self):
        self.assertEqual(strip_styles("<i>Hello</i> <b>world</b>"),
                         "Hello world")
        self.assertEqual(strip_styles("{\\an8}Top text"), "Top text")
        self.assertEqual(
            strip_styles('<font color="red">Red</font>'), "Red")

    def test_op_strip_styles_clears_settings(self):
        doc = parse_ok(VTT_SAMPLE)
        op_strip_styles(doc)
        self.assertEqual(doc.cues[1].settings, "")
        self.assertEqual(doc.cues[1].text, "Second line")

    def test_renumber(self):
        doc = parse_text("7\n00:00:01,000 --> 00:00:02,000\nA\n\n"
                         "9\n00:00:03,000 --> 00:00:04,000\nB\n")
        op_renumber(doc)
        self.assertEqual([c.cue_id for c in doc.cues], ["1", "2"])


class TestRendering(unittest.TestCase):
    def test_render_srt_shape(self):
        doc = parse_ok(SRT_SAMPLE)
        out = render_srt(doc)
        self.assertTrue(out.startswith("1\n00:00:01,000 --> 00:00:04,000\n"))
        self.assertTrue(out.endswith("\n"))

    def test_render_vtt_shape(self):
        doc = parse_ok(SRT_SAMPLE)
        out = render_vtt(doc)
        self.assertTrue(out.startswith("WEBVTT\n\n"))
        self.assertIn("00:00:01.000 --> 00:00:04.000", out)

    def test_vtt_keeps_identifier_and_settings(self):
        doc = parse_ok(VTT_SAMPLE)
        out = render(doc, "vtt")
        self.assertIn("cue-two\n00:00:05.500 --> 00:00:08.250 "
                      "align:start position:0%", out)

    def test_srt_to_vtt_to_srt_round_trip(self):
        first = parse_ok(SRT_SAMPLE)
        vtt_text = render(first, "vtt")
        second = parse_ok(vtt_text)
        srt_text = render(second, "srt")
        third = parse_ok(srt_text)
        self.assertEqual(len(first.cues), len(third.cues))
        for a, b in zip(first.cues, third.cues):
            self.assertLessEqual(abs(a.start - b.start), 1)
            self.assertLessEqual(abs(a.end - b.end), 1)
            self.assertEqual(a.text, b.text)


class TestCheck(unittest.TestCase):
    def test_clean_file(self):
        doc = parse_ok(SRT_SAMPLE)
        self.assertEqual(check_doc(doc, "ok.srt"), [])

    def test_overlap_detected(self):
        doc = parse_ok("1\n00:00:01,000 --> 00:00:05,000\nA\n\n"
                       "2\n00:00:04,000 --> 00:00:06,000\nB\n")
        issues = check_doc(doc, "x.srt")
        self.assertTrue(any("overlap" in i.message for i in issues))
        self.assertTrue(any(i.severity == "error" for i in issues))

    def test_out_of_order_detected(self):
        doc = parse_ok("1\n00:00:05,000 --> 00:00:06,000\nA\n\n"
                       "2\n00:00:01,000 --> 00:00:02,000\nB\n")
        issues = check_doc(doc, "x.srt")
        self.assertTrue(any("out of order" in i.message for i in issues))

    def test_zero_duration_detected(self):
        doc = parse_ok("1\n00:00:05,000 --> 00:00:05,000\nA\n")
        issues = check_doc(doc, "x.srt")
        self.assertTrue(any("duration" in i.message for i in issues))


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def write(self, name, text):
        path = os.path.join(self.tmp, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def run_cli(self, *args):
        return subprocess.run(CLI + list(args), capture_output=True, text=True,
                              cwd=self.tmp)

    def test_shift_end_to_end(self):
        src = self.write("a.srt", SRT_SAMPLE)
        proc = self.run_cli("--shift=-2.5s", "--quiet", src)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        dest = os.path.join(self.tmp, "a.resync.srt")
        self.assertTrue(os.path.exists(dest))
        doc = parse_ok(open(dest, encoding="utf-8").read())
        # First cue clamped at zero; second cue moved earlier by 2500 ms.
        self.assertEqual(doc.cues[0].start, 0)
        self.assertEqual(doc.cues[1].start, 3000)

    def test_convert_to_vtt(self):
        src = self.write("a.srt", SRT_SAMPLE)
        proc = self.run_cli("--to", "vtt", "--quiet", src)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        dest = os.path.join(self.tmp, "a.resync.vtt")
        text = open(dest, encoding="utf-8").read()
        self.assertTrue(text.startswith("WEBVTT"))

    def test_dir_batch_and_out_dir(self):
        self.write("one.srt", SRT_SAMPLE)
        self.write("two.srt", SRT_SAMPLE)
        out = os.path.join(self.tmp, "out")
        proc = self.run_cli("--dir", self.tmp, "--shift", "1000",
                        "--out-dir", out, "--quiet")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(os.path.exists(os.path.join(out, "one.resync.srt")))
        self.assertTrue(os.path.exists(os.path.join(out, "two.resync.srt")))

    def test_dry_run_writes_nothing(self):
        src = self.write("a.srt", SRT_SAMPLE)
        proc = self.run_cli("--shift", "1000", "--dry-run", src)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("a.srt ->", proc.stdout)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "a.resync.srt")))

    def test_check_clean_exit_zero(self):
        src = self.write("a.srt", SRT_SAMPLE)
        proc = self.run_cli("--check", src)
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("0 error(s)", proc.stdout)

    def test_check_overlap_exit_one(self):
        src = self.write("bad.srt",
                         "1\n00:00:01,000 --> 00:00:05,000\nA\n\n"
                         "2\n00:00:04,000 --> 00:00:06,000\nB\n")
        proc = self.run_cli("--check", src)
        self.assertEqual(proc.returncode, 1)
        self.assertIn("overlap", proc.stdout)

    def test_stats_prints_a_report_and_writes_nothing(self):
        src = self.write("a.srt", SRT_SAMPLE)
        proc = self.run_cli("--stats", src)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("CUES", proc.stdout)
        self.assertIn("AVG WPM", proc.stdout)
        self.assertIn("a.srt", proc.stdout)
        self.assertEqual(
            sorted(os.listdir(self.tmp)),
            ["a.srt"],
            "--stats must not write output files",
        )

    def test_stats_dir_mode_prints_one_row_per_file_and_a_total(self):
        self.write("a.srt", SRT_SAMPLE)
        self.write("b.vtt", VTT_SAMPLE)
        proc = self.run_cli("--stats", "--dir", ".")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        rows = [line for line in proc.stdout.splitlines() if line.strip()]
        self.assertTrue(any("a.srt" in line for line in rows), rows)
        self.assertTrue(any("b.vtt" in line for line in rows), rows)
        self.assertTrue(any(line.startswith("TOTAL") for line in rows), rows)
        self.assertEqual(sorted(os.listdir(self.tmp)), ["a.srt", "b.vtt"])


    def test_no_op_is_error(self):
        src = self.write("a.srt", SRT_SAMPLE)
        proc = self.run_cli(src)
        self.assertEqual(proc.returncode, 2)

    def test_bad_shift_is_error(self):
        src = self.write("a.srt", SRT_SAMPLE)
        proc = self.run_cli("--shift", "banana", src)
        self.assertEqual(proc.returncode, 2)


if __name__ == "__main__":
    unittest.main()
