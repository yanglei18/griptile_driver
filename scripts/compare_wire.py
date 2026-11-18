#!/usr/bin/env python3
"""Compare row44 decoding and rolling snapshots offline. Only hashes and counts leave the replay."""

import argparse
from collections import Counter
from dataclasses import fields
import hashlib
import importlib
import io
from itertools import groupby, zip_longest
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
REVISION = "74e861c23aa54d104503c6e521d739ec25163c2b"
PACKAGE = "ros2_ws/src/vt_tactile_capture"
MODULES = ("__init__", "protocol", "assembler", "grid", "serial_source", "serial_access", "host_clock")
READ_SIZE = 257  # Deliberately splits headers, payloads and 44-byte frame boundaries.
MAX_AGE_NS = 100_000_000
CHANNEL = "right"  # A valid identifier in both schemas. No physical side is inferred.


def load_reference(checkout, destination):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(checkout), *args])

    # Read Git objects directly, never uncommitted files in the reference checkout.
    package = destination / "vt_tactile_capture"
    package.mkdir()
    for module in MODULES:
        blob = f"{REVISION}:{PACKAGE}/vt_tactile_capture/{module}.py"
        (package / f"{module}.py").write_bytes(git("show", blob))
    sys.path.insert(0, str(destination))
    return git("rev-parse", f"{REVISION}:{PACKAGE}").decode().strip()


def samples(blocks):
    return [(block.row, block.adc) for block in blocks]


def canonical(record):
    result = {field.name: getattr(record, field.name) for field in fields(record)}
    if "hand" in result:
        result["sensor_id"] = result.pop("hand")
    if "blocks" in result:
        result["blocks"] = samples(result["blocks"])
    return result


def counters(parser):
    return (parser.discarded_bytes_total, parser.malformed_packets_total, parser.buffered_bytes)


def clocks(sequence):
    # Raw wire bytes carry no clocks. Use identical reproducible read-completion tags.
    offset = sequence * 1_000_000
    return (1_700_000_000_000_000_000 + offset, 1_000_000_000 + offset, True)


class Replay:
    def __init__(self, package_name, id_field):
        protocol = importlib.import_module(package_name + ".protocol")
        self.file_parser = protocol.Parser()
        self.live_parser = protocol.StreamParser()
        self.batch_type = importlib.import_module(package_name + ".serial_source").Batch
        self.assembler = importlib.import_module(package_name + ".assembler").Assembler(CHANNEL)
        self.id_field = id_field
        self.packet_index = self.sequence = self.file_rows = 0
        self.seen_rows = set()

    def feed(self, data, tag):
        parsed = self.file_parser.feed(data)
        self.file_rows += len(parsed)
        self.seen_rows.update(block.row for block in parsed)
        yield "file_rows", samples(parsed)
        yield "file_counters", counters(self.file_parser)
        admitted = self.live_parser.feed(data, tag)
        yield "tagged_live_rows", [(block.row, block.adc, read_tag) for block, read_tag in admitted]
        yield "live_counters", (*counters(self.live_parser), self.live_parser.confirmed)
        for read_tag, group in groupby(admitted, key=lambda pair: pair[1]):
            blocks = tuple(block for block, _ in group)
            batch = self.batch_type(
                **{self.id_field: CHANNEL}, session_id="wire-parity", stream_epoch=1,
                sequence=self.sequence, first_packet_index=self.packet_index, blocks=blocks,
                host_realtime_ns=read_tag[0], host_monotonic_ns=read_tag[1], time_valid=True,
                connected=True, reason="ok", producer_dropped_total=0,
                discarded_bytes_total=self.live_parser.discarded_bytes_total,
                malformed_packets_total=self.live_parser.malformed_packets_total)
            self.sequence += 1
            self.packet_index += len(blocks)
            yield "batches", canonical(batch)
            self.assembler.apply(batch)
            yield "snapshots", canonical(self.assembler.snapshot(tag[1], MAX_AGE_NS))

    def finish(self, tag):
        yield "expired_snapshot", canonical(self.assembler.snapshot(tag[1] + MAX_AGE_NS + 1, MAX_AGE_NS))
        self.file_parser.reset()
        self.live_parser.reset()
        yield "eof_reset_counters", [counters(self.file_parser), counters(self.live_parser)]


class Comparison:
    def __init__(self):
        self.digests = {}
        self.event_counts = Counter()

    def check(self, candidate, reference, read_number):
        # Compare every complete event before hashing. A mismatch never dumps private samples.
        for actual, expected in zip_longest(candidate, reference or (), fillvalue=None):
            if reference is not None and actual != expected:
                stage = (actual or expected)[0]
                raise ValueError(f"reference mismatch in {stage} at read {read_number}")
            kind, value = actual
            if kind not in self.digests:
                self.digests[kind] = hashlib.sha256()
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
            self.digests[kind].update(encoded + b"\n")
            self.event_counts[kind] += 1


def synthetic_wire():
    return b"".join(struct.pack(">2sH20H", b"\xa5\x5a", row, *(
        scan * 400 + row * 20 + col for col in range(20)))
        for scan in range(4) for row in range(20))


def run(args):
    sys.path.insert(0, str(ROOT / "src" / "griptile_core"))
    candidate = Replay("griptile_core", "sensor_id")
    reference = None
    report = {"schema": 1, "read_size_bytes": READ_SIZE, "max_age_ns": MAX_AGE_NS,
              "clocks": "Synthetic monotonic and realtime tags at 1 ms per read",
              "input": "user-supplied wire capture" if args.wire else "built-in synthetic frames"}
    with tempfile.TemporaryDirectory() as tmp:
        if args.baseline_source:
            tree = load_reference(args.baseline_source, Path(tmp))
            report["baseline"] = {"commit": REVISION, "package": PACKAGE, "package_tree": tree}
            reference = Replay("vt_tactile_capture", "hand")
        counts = Comparison()
        wire_hash = hashlib.sha256()
        reads = size = 0
        with args.wire.open("rb") if args.wire else io.BytesIO(synthetic_wire()) as stream:
            while data := stream.read(READ_SIZE):
                reads += 1
                size += len(data)
                wire_hash.update(data)
                tag = clocks(reads)
                counts.check(candidate.feed(data, tag), reference.feed(data, tag) if reference else None, reads)
        if not candidate.packet_index:
            raise ValueError("no confirmed row44 stream in this input")
        report["live_before_eof"] = dict(zip(
            ("discarded_bytes_total", "malformed_packets_total", "buffered_bytes"), counters(candidate.live_parser)))
        counts.check(candidate.finish(clocks(reads)), reference.finish(clocks(reads)) if reference else None, reads)
    report.update({
        "result": "equal" if reference else "fingerprints only; no reference supplied",
        "input_bytes": size, "input_sha256": wire_hash.hexdigest(), "reads": reads,
        "file_rows": candidate.file_rows, "distinct_file_row_addresses": len(candidate.seen_rows),
        "live_rows": candidate.packet_index, "live_batches": candidate.sequence,
        "compared_fields": "All parser outputs, counters, original live read tags, batch and state fields. "
                           "The hand field is renamed sensor_id for comparison. "
                           "Each data batch produces one snapshot including 400 ADC cells, valid and updated "
                           "masks, 20 row clocks of each kind, loss count and status. A final age-expiry "
                           "snapshot and parser resets are checked. Queue loss is zero in this synchronous replay.",
        "trace_events": dict(counts.event_counts),
        "trace_sha256": {kind: digest.hexdigest() for kind, digest in counts.digests.items()},
    })
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wire", type=Path, help="Private raw serial capture, or omit for synthetic frames")
    parser.add_argument("--baseline-source", type=Path, help="Trusted VisionTactile-Dataset Git checkout "
                        "containing the pinned reference commit")
    parser.add_argument("--output", type=Path, help="Write counts and SHA-256 digests as JSON")
    args = parser.parse_args()
    report = json.dumps(run(args), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
