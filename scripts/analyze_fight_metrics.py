"""Aggregate technique_events.csv files into "most-used technique" statistics.

Reads one or more ``technique_events.csv`` files written by
``fight_analysis.py`` (schema: fighter_id, technique, family,
detection_method, start_frame, end_frame, start_time_s, end_time_s,
duration_s, confidence, dtw_score, frame_count, source) and reports counts
per fighter/technique/family, optionally bucketed by round via a sidecar CSV.

Usage:
    python scripts/analyze_fight_metrics.py --events-csv data/runs/run_.../technique_events.csv
    python scripts/analyze_fight_metrics.py --runs-root data/runs --plot both
    python scripts/analyze_fight_metrics.py --events-csv events.csv --rounds-csv rounds.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _resolve_project_path(path_value: str) -> Path:
    p = Path(path_value)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _as_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class EventRow:
    fighter_id: str
    technique: str
    family: str
    detection_method: str
    start_time_s: float
    end_time_s: float
    duration_s: float
    confidence: float
    dtw_score: float | None
    source: str
    round: int | None = None


@dataclass
class FighterTechniqueSummary:
    fighter_id: str
    technique: str
    family: str
    count: int
    mean_confidence: float
    mean_duration_s: float
    hybrid_count: int
    classifier_only_count: int


def load_events(events_csv: Path) -> list[EventRow]:
    with events_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    events: list[EventRow] = []
    for r in rows:
        dtw_score = r.get("dtw_score", "")
        events.append(
            EventRow(
                fighter_id=r.get("fighter_id", ""),
                technique=r.get("technique", ""),
                family=r.get("family", "") or "",
                detection_method=r.get("detection_method", ""),
                start_time_s=_as_float(r.get("start_time_s", "")),
                end_time_s=_as_float(r.get("end_time_s", "")),
                duration_s=_as_float(r.get("duration_s", "")),
                confidence=_as_float(r.get("confidence", "")),
                dtw_score=_as_float(dtw_score) if dtw_score else None,
                source=r.get("source", ""),
            )
        )
    return events


def discover_events_csvs(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        return []
    return sorted(runs_root.rglob("technique_events.csv"))


def load_round_boundaries(rounds_csv: Path) -> list[tuple[int, float, float]]:
    """Load ``round,start_time_s,end_time_s`` rows, sorted by start time."""
    with rounds_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    boundaries = [
        (int(r["round"]), _as_float(r["start_time_s"]), _as_float(r["end_time_s"])) for r in rows if r.get("round")
    ]
    return sorted(boundaries, key=lambda b: b[1])


def assign_rounds(events: list[EventRow], boundaries: list[tuple[int, float, float]]) -> None:
    """Stamp ``event.round`` in place from the round whose window contains its start time."""
    for event in events:
        for round_no, start, end in boundaries:
            if start <= event.start_time_s < end:
                event.round = round_no
                break


def summarize_by_fighter_technique(events: list[EventRow], exclude: set[str]) -> list[FighterTechniqueSummary]:
    groups: dict[tuple[str, str], list[EventRow]] = defaultdict(list)
    for event in events:
        if event.technique in exclude:
            continue
        groups[(event.fighter_id, event.technique)].append(event)

    summaries: list[FighterTechniqueSummary] = []
    for (fighter_id, technique), rows in groups.items():
        confidences = [r.confidence for r in rows if r.confidence == r.confidence]  # drop NaN
        durations = [r.duration_s for r in rows if r.duration_s == r.duration_s]
        summaries.append(
            FighterTechniqueSummary(
                fighter_id=fighter_id,
                technique=technique,
                family=rows[0].family,
                count=len(rows),
                mean_confidence=(sum(confidences) / len(confidences)) if confidences else float("nan"),
                mean_duration_s=(sum(durations) / len(durations)) if durations else float("nan"),
                hybrid_count=sum(1 for r in rows if r.detection_method == "hybrid"),
                classifier_only_count=sum(1 for r in rows if r.detection_method == "classifier_only"),
            )
        )
    summaries.sort(key=lambda s: (s.fighter_id, -s.count))
    return summaries


def summarize_by_round(events: list[EventRow], exclude: set[str]) -> list[dict[str, object]]:
    groups: dict[tuple[int | None, str, str], int] = Counter()
    for event in events:
        if event.technique in exclude or event.round is None:
            continue
        groups[(event.round, event.fighter_id, event.technique)] += 1
    rows = [
        {"round": round_no, "fighter_id": fighter_id, "technique": technique, "count": count}
        for (round_no, fighter_id, technique), count in groups.items()
    ]
    rows.sort(key=lambda r: (r["round"], r["fighter_id"], -r["count"]))
    return rows


def print_console_summary(summaries: list[FighterTechniqueSummary], print_top: int) -> None:
    by_fighter: dict[str, list[FighterTechniqueSummary]] = defaultdict(list)
    for s in summaries:
        by_fighter[s.fighter_id].append(s)

    for fighter_id in sorted(by_fighter):
        rows = by_fighter[fighter_id]
        total = sum(s.count for s in rows)
        print(f"\n{fighter_id}: {total} technique event(s) across {len(rows)} technique(s)")
        for s in rows[:print_top]:
            print(
                f"  {s.technique:20s} x{s.count:<4d} "
                f"(hybrid {s.hybrid_count}, classifier-only {s.classifier_only_count}, "
                f"mean conf {s.mean_confidence:.2f})"
            )


def write_outputs(
    summaries: list[FighterTechniqueSummary],
    round_summary: list[dict[str, object]],
    events_csvs: list[Path],
    output_csv: Path,
    output_json: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fighter_id",
                "technique",
                "family",
                "count",
                "mean_confidence",
                "mean_duration_s",
                "hybrid_count",
                "classifier_only_count",
            ],
        )
        writer.writeheader()
        for s in summaries:
            writer.writerow(asdict(s))

    payload = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "events_csvs": [str(p) for p in events_csvs],
        "by_fighter_technique": [asdict(s) for s in summaries],
        "by_round": round_summary,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if round_summary:
        round_csv = output_csv.with_name(output_csv.stem + "_by_round.csv")
        with round_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "fighter_id", "technique", "count"])
            writer.writeheader()
            writer.writerows(round_summary)
        print(f"wrote per-round breakdown to: {round_csv}")


def plot_bar(summaries: list[FighterTechniqueSummary], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        raise ImportError("--plot bar requires pandas and matplotlib: pip install pandas matplotlib") from exc

    df = pd.DataFrame([asdict(s) for s in summaries])
    if df.empty:
        print("no events to plot")
        return
    pivot = df.pivot_table(index="technique", columns="fighter_id", values="count", fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    ax = pivot.plot(kind="bar", figsize=(max(8, len(pivot) * 0.6), 6))
    ax.set_ylabel("technique events")
    ax.set_xlabel("technique")
    ax.set_title("Technique usage by fighter")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"wrote bar chart to: {output_path}")


def plot_timeline(events: list[EventRow], exclude: set[str], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError as exc:
        raise ImportError("--plot timeline requires matplotlib: pip install matplotlib") from exc

    kept = [e for e in events if e.technique not in exclude]
    if not kept:
        print("no events to plot")
        return
    fighters = sorted({e.fighter_id for e in kept})
    techniques = sorted({e.technique for e in kept})
    palette = cm.get_cmap("tab20", max(len(techniques), 1))
    color_by_technique = {t: palette(i) for i, t in enumerate(techniques)}

    fig, ax = plt.subplots(figsize=(14, 1.2 * len(fighters) + 2))
    for row, fighter_id in enumerate(fighters):
        spans = [(e.start_time_s, max(e.duration_s, 0.15)) for e in kept if e.fighter_id == fighter_id]
        colors = [color_by_technique[e.technique] for e in kept if e.fighter_id == fighter_id]
        for (start, width), color in zip(spans, colors):
            ax.broken_barh([(start, width)], (row - 0.4, 0.8), facecolors=color)

    ax.set_yticks(range(len(fighters)))
    ax.set_yticklabels(fighters)
    ax.set_xlabel("fight time (s)")
    ax.set_title("Technique timeline")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_by_technique[t]) for t in techniques]
    ax.legend(handles, techniques, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"wrote timeline to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events-csv", nargs="+", default=None, help="one or more technique_events.csv files")
    parser.add_argument("--runs-root", type=str, default="data/runs", help="discover technique_events.csv under here")
    parser.add_argument("--rounds-csv", type=str, default=None, help="optional round,start_time_s,end_time_s sidecar")
    parser.add_argument(
        "--exclude-technique",
        nargs="*",
        default=[],
        help="technique keys to drop from the summary (e.g. fighting_stance)",
    )
    parser.add_argument("--output-csv", type=str, default=None, help="default: data/runs/fight_summary_<timestamp>.csv")
    parser.add_argument("--output-json", type=str, default=None, help="default: alongside --output-csv")
    parser.add_argument("--plot", choices=["none", "bar", "timeline", "both"], default="none")
    parser.add_argument("--print-top", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.events_csv:
        events_csvs = [_resolve_project_path(p) for p in args.events_csv]
        missing = [p for p in events_csvs if not p.exists()]
        if missing:
            print(f"error: events csv not found: {missing}")
            return 2
    else:
        events_csvs = discover_events_csvs(_resolve_project_path(args.runs_root))
        if not events_csvs:
            print(f"no technique_events.csv found under: {_resolve_project_path(args.runs_root)}")
            return 1

    events: list[EventRow] = []
    for path in events_csvs:
        events.extend(load_events(path))
    if not events:
        print("no events found in the given csv(s)")
        return 1

    if args.rounds_csv:
        boundaries = load_round_boundaries(_resolve_project_path(args.rounds_csv))
        assign_rounds(events, boundaries)

    exclude = set(args.exclude_technique)
    summaries = summarize_by_fighter_technique(events, exclude=exclude)
    round_summary = summarize_by_round(events, exclude=exclude) if args.rounds_csv else []

    print_console_summary(summaries, print_top=args.print_top)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = (
        _resolve_project_path(args.output_csv) if args.output_csv else _resolve_project_path(f"data/runs/fight_summary_{ts}.csv")
    )
    output_json = _resolve_project_path(args.output_json) if args.output_json else output_csv.with_suffix(".json")
    write_outputs(summaries, round_summary, events_csvs, output_csv, output_json)
    print(f"\nwrote summary csv to: {output_csv}")
    print(f"wrote summary json to: {output_json}")

    if args.plot in {"bar", "both"}:
        plot_bar(summaries, output_csv.with_name(output_csv.stem + "_bar.png"))
    if args.plot in {"timeline", "both"}:
        plot_timeline(events, exclude, output_csv.with_name(output_csv.stem + "_timeline.png"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
