#!/usr/bin/env python3
import argparse
import gzip
import html
import json
import shutil
from collections import Counter
from pathlib import Path


def load_json_maybe_gzip(path: Path):
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    except OSError:
        return json.loads(path.read_text(encoding="utf-8"))


def find_metric_files(input_path: Path):
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("metrics_ep_ids_*.json.gz"))


def load_task_titles(dataset_path: Path):
    if not dataset_path.exists():
        return {}
    payload = load_json_maybe_gzip(dataset_path)
    titles = {}
    for episode in payload.get("episodes", []):
        episode_id = str(episode.get("episode_id"))
        instructions = episode.get("instructions", {})
        for task_id, instruction in instructions.items():
            titles[(episode_id, task_id)] = instruction.get("lang", task_id)
    return titles


def find_episode_videos(input_path: Path):
    if input_path.is_file():
        search_root = input_path.parent
    else:
        search_root = input_path

    videos = {}
    for video_path in search_root.rglob("trajectory.mp4"):
        episode_id = video_path.parent.name
        videos[episode_id] = video_path.resolve()
    return videos


def stage_episode_videos(episode_videos, output_path: Path):
    if not episode_videos:
        return {}

    report_dir = output_path.resolve().parent
    videos_dir = report_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    staged = {}
    for episode_id, video_path in episode_videos.items():
        episode_dir = videos_dir / str(episode_id)
        episode_dir.mkdir(parents=True, exist_ok=True)
        dest_path = episode_dir / video_path.name
        shutil.copy2(video_path, dest_path)
        staged[episode_id] = dest_path
    return staged


def to_float(value):
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def pct(numerator, denominator):
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_rows(metric_files, task_titles, episode_videos):
    rows = []
    for metric_file in metric_files:
        payload = load_json_maybe_gzip(metric_file)
        episode_id = metric_file.name.replace("metrics_ep_ids_", "").replace(".json.gz", "")
        for task_id, metrics in payload.items():
            row = {
                "episode_id": episode_id,
                "task_id": task_id,
                "task_title": task_titles.get((episode_id, task_id), task_id),
                "episode_video": episode_videos.get(episode_id, ""),
                "predicate_task_success": metrics.get("predicate_task_success", False),
                "high_level_goal_success": metrics.get("high_level_goal_success", False),
                "high_level_dtg_success": metrics.get("high_level_dtg_success", False),
                "high_level_sem_cov_success": metrics.get("high_level_sem_cov_success", False),
                "findingdory_spl": to_float(metrics.get("findingdory_spl")),
                "findingdory_high_level_spl": to_float(metrics.get("findingdory_high_level_spl")),
                "num_steps": metrics.get("num_steps"),
                "subgoal_count": metrics.get("subgoal_count"),
                "target_iou_coverage": to_float(metrics.get("target_iou_coverage")),
                "force_terminate": metrics.get("force_terminate", False),
                "bad_called_terminate": metrics.get("bad_called_terminate", False),
                "vlm_failure_modes": metrics.get("vlm_failure_modes", {}),
                "low_level_failure_modes": metrics.get("c", {}),
                "raw_metrics": metrics,
            }
            rows.append(row)
    return rows


def summarize(rows):
    total = len(rows)
    failure_counts = Counter()
    low_level_counts = Counter()

    for row in rows:
        for key, value in row["vlm_failure_modes"].items():
            if value:
                failure_counts[key] += 1
        for key, value in row["low_level_failure_modes"].items():
            if value:
                low_level_counts[key] += 1

    summary = {
        "total_tasks": total,
        "episodes": len({row["episode_id"] for row in rows}),
        "task_success_rate": pct(sum(bool(r["predicate_task_success"]) for r in rows), total),
        "high_level_goal_success_rate": pct(sum(bool(r["high_level_goal_success"]) for r in rows), total),
        "high_level_dtg_success_rate": pct(sum(bool(r["high_level_dtg_success"]) for r in rows), total),
        "high_level_sem_cov_success_rate": pct(sum(bool(r["high_level_sem_cov_success"]) for r in rows), total),
        "mean_spl": mean([r["findingdory_spl"] for r in rows]),
        "mean_high_level_spl": mean([r["findingdory_high_level_spl"] for r in rows]),
        "mean_steps": mean([to_float(r["num_steps"]) for r in rows]),
        "mean_subgoals": mean([to_float(r["subgoal_count"]) for r in rows]),
        "mean_iou_coverage": mean([r["target_iou_coverage"] for r in rows]),
        "failure_counts": failure_counts,
        "low_level_counts": low_level_counts,
    }
    return summary


def metric_card(label, value, suffix=""):
    return f"""
    <div class="card">
      <div class="label">{html.escape(label)}</div>
      <div class="value">{value}{html.escape(suffix)}</div>
    </div>
    """


def bar_row(label, count, total):
    width = pct(count, total)
    return f"""
    <div class="bar-row">
      <div class="bar-label">{html.escape(label)}</div>
      <div class="bar-wrap">
        <div class="bar-fill" style="width: {width:.2f}%"></div>
      </div>
      <div class="bar-value">{count} / {total} ({width:.1f}%)</div>
    </div>
    """


def render_html(rows, summary, source_path):
    failure_rows = "\n".join(
        bar_row(key, summary["failure_counts"].get(key, 0), summary["total_tasks"])
        for key in sorted(
            {
                *summary["failure_counts"].keys(),
                "misidentification",
                "response_error",
                "out_of_bounds_pred",
                "num_targets_error",
                "oracle_agent_timeout",
            }
        )
    )
    low_level_rows = "\n".join(
        bar_row(key, summary["low_level_counts"].get(key, 0), summary["total_tasks"])
        for key in sorted(
            {
                *summary["low_level_counts"].keys(),
                "low_level_policy_timeout",
                "incorrect_stop",
                "tgt_in_view_but_too_far",
                "tgt_in_view_slightly_far",
                "tgt_not_in_view_slightly_far",
                "tgt_not_in_view_too_far",
                "low_level_policy_failure",
            }
        )
    )

    episode_video_cards = []
    seen_episodes = set()
    for row in rows:
        episode_id = row["episode_id"]
        if episode_id in seen_episodes:
            continue
        seen_episodes.add(episode_id)
        video_path = row["episode_video"]
        if not video_path:
            continue
        episode_video_cards.append(
            f"""
            <div class="video-card">
              <div class="video-title">Episode {html.escape(episode_id)}</div>
              <video controls preload="metadata" src="{html.escape(str(video_path))}"></video>
              <div class="video-path">{html.escape(str(video_path))}</div>
            </div>
            """
        )

    task_rows = []
    for row in rows:
        video_cell = ""
        if row["episode_video"]:
            video_cell = f'<a href="{html.escape(str(row["episode_video"]))}">video</a>'
        task_rows.append(
            f"""
            <tr>
              <td>{html.escape(row['episode_id'])}</td>
              <td>{html.escape(row['task_id'])}</td>
              <td>{html.escape(row['task_title'])}</td>
              <td>{video_cell}</td>
              <td>{'1' if row['predicate_task_success'] else '0'}</td>
              <td>{'1' if row['high_level_goal_success'] else '0'}</td>
              <td>{'1' if row['high_level_dtg_success'] else '0'}</td>
              <td>{'1' if row['high_level_sem_cov_success'] else '0'}</td>
              <td>{row['findingdory_spl']:.3f}</td>
              <td>{row['findingdory_high_level_spl']:.3f}</td>
              <td>{row['num_steps']}</td>
              <td>{row['subgoal_count']}</td>
              <td><details><summary>view</summary><pre>{html.escape(json.dumps(row['raw_metrics'], indent=2))}</pre></details></td>
            </tr>
            """
        )

    cards = "\n".join(
        [
            metric_card("Episodes", summary["episodes"]),
            metric_card("Tasks", summary["total_tasks"]),
            metric_card("Task SR", f"{summary['task_success_rate']:.1f}", "%"),
            metric_card("HL Goal SR", f"{summary['high_level_goal_success_rate']:.1f}", "%"),
            metric_card("HL DTG SR", f"{summary['high_level_dtg_success_rate']:.1f}", "%"),
            metric_card("HL SemCov SR", f"{summary['high_level_sem_cov_success_rate']:.1f}", "%"),
            metric_card("Mean SPL", f"{summary['mean_spl']:.3f}"),
            metric_card("Mean HL SPL", f"{summary['mean_high_level_spl']:.3f}"),
            metric_card("Mean Steps", f"{summary['mean_steps']:.1f}"),
            metric_card("Mean Subgoals", f"{summary['mean_subgoals']:.2f}"),
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>FindingDory Metrics Viewer</title>
  <style>
    body {{
      font-family: ui-sans-serif, system-ui, sans-serif;
      background: #f5f1e8;
      color: #1f2933;
      margin: 0;
      padding: 24px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
    }}
    .subtitle {{
      margin-bottom: 20px;
      color: #52606d;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-bottom: 28px;
    }}
    .card {{
      background: #fffdf7;
      border: 1px solid #d9d2c3;
      border-radius: 14px;
      padding: 14px 16px;
      box-shadow: 0 8px 24px rgba(20, 28, 36, 0.05);
    }}
    .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #7b8794;
      margin-bottom: 8px;
    }}
    .value {{
      font-size: 28px;
      font-weight: 700;
    }}
    .section {{
      background: #fffdf7;
      border: 1px solid #d9d2c3;
      border-radius: 16px;
      padding: 18px;
      margin-bottom: 24px;
      box-shadow: 0 8px 24px rgba(20, 28, 36, 0.05);
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 240px 1fr 160px;
      gap: 12px;
      align-items: center;
      margin: 10px 0;
    }}
    .bar-wrap {{
      height: 12px;
      background: #e8e1d1;
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar-fill {{
      height: 100%;
      background: linear-gradient(90deg, #c05621, #f59e0b);
    }}
    .video-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .video-card {{
      border: 1px solid #e5dfd0;
      border-radius: 12px;
      padding: 12px;
      background: #fffaf0;
    }}
    .video-title {{
      font-weight: 700;
      margin-bottom: 10px;
    }}
    .video-card video {{
      width: 100%;
      border-radius: 10px;
      background: #000;
    }}
    .video-path {{
      margin-top: 8px;
      color: #7b8794;
      font-size: 12px;
      word-break: break-all;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid #e5dfd0;
      padding: 10px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #fffdf7;
    }}
    details pre {{
      white-space: pre-wrap;
      margin: 8px 0 0;
      font-size: 12px;
      background: #f8f5ee;
      border-radius: 8px;
      padding: 10px;
    }}
  </style>
</head>
<body>
  <h1>FindingDory Metrics Viewer</h1>
  <div class="subtitle">Source: {html.escape(str(source_path))}</div>

  <div class="cards">
    {cards}
  </div>

  <div class="section">
    <h2>VLM Failure Modes</h2>
    {failure_rows}
  </div>

  <div class="section">
    <h2>Low-Level Failure Modes</h2>
    {low_level_rows}
  </div>

  <div class="section">
    <h2>Episode Videos</h2>
    <div class="video-grid">
      {''.join(episode_video_cards) if episode_video_cards else '<div>No trajectory videos found next to the metrics files.</div>'}
    </div>
  </div>

  <div class="section">
    <h2>Per-Task Details</h2>
    <table>
      <thead>
        <tr>
          <th>Episode</th>
          <th>Task</th>
          <th>Title</th>
          <th>Video</th>
          <th>Task SR</th>
          <th>HL Goal</th>
          <th>HL DTG</th>
          <th>HL SemCov</th>
          <th>SPL</th>
          <th>HL SPL</th>
          <th>Steps</th>
          <th>Subgoals</th>
          <th>Raw</th>
        </tr>
      </thead>
      <tbody>
        {''.join(task_rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""

def relativize_video_paths(rows, html_output_path: Path):
    html_dir = html_output_path.resolve().parent
    updated_rows = []
    for row in rows:
        updated = dict(row)
        video_path = row["episode_video"]
        if video_path:
            updated["episode_video"] = Path(
                __import__("os").path.relpath(video_path, html_dir)
            )
        updated_rows.append(updated)
    return updated_rows


def main():
    parser = argparse.ArgumentParser(description="Render FindingDory metrics to an HTML report.")
    parser.add_argument(
        "input",
        help="Metrics file or directory containing metrics_ep_ids_*.json.gz files.",
    )
    parser.add_argument(
        "--output",
        default="findingdory_metrics_view.html",
        help="Output HTML file path.",
    )
    parser.add_argument(
        "--dataset",
        default="findingdory/data/datasets/findingdory-habitat/findingdory/val/episodes.json.gz",
        help="Dataset file used to resolve task ids to natural-language instruction titles.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    metric_files = find_metric_files(input_path)
    if not metric_files:
        raise SystemExit(f"No metrics files found under {input_path}")

    task_titles = load_task_titles(Path(args.dataset))
    episode_videos = find_episode_videos(input_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged_videos = stage_episode_videos(episode_videos, output_path)
    rows = build_rows(metric_files, task_titles, staged_videos)
    summary = summarize(rows)
    html_rows = relativize_video_paths(rows, output_path)
    html_report = render_html(html_rows, summary, input_path)
    output_path.write_text(html_report, encoding="utf-8")
    print(f"Wrote metrics viewer to {output_path.resolve()}")


if __name__ == "__main__":
    main()
