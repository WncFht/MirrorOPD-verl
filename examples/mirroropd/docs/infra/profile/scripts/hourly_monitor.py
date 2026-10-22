# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Hourly follow-up in the existing Codex thread, until an explicit reviewed DONE.

The daemon only queues prompts. The existing conversation performs the review;
GPU allocation remains owned by the experiment runner. One unacknowledged prompt
is allowed, preventing a growing backlog while the conversation is busy.
"""

import argparse
import fcntl
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def review_prompt(root, tick):
    return f"""【用户授权的每小时 Full-KL 检查，编号 {tick}】
用户要求每小时看一眼，直到你判断能做的工作已经完成。继续本会话已有 Full-KL profile/优化任务，不是新的无关任务。
运行目录：{root}
收到后先写 {root / "hourly-ack.json"}，内容为 {{"tick": {tick}}}，表示已开始处理这一条检查。
检查 status.json、runner 进程、各任务 exit.json/stdout.log、GPU 0–3 的实际占用。
只在前四卡空闲且没有其他任务时做 GPU 实验，不碰后四卡，不结束他人的进程。
如果队列还在运行，审阅已完成结果并做独立的 CPU 分析。
若失败或卡住，找根因、修复、测试并在新证据目录重试；不要覆盖失败证据或修改运行中的冻结源码。
避免启动重复队列，继续使用共享 GPU 锁。
结果齐全后，分析原始 trace 和显存，检查数值/梯度、固定输入管线及端到端证据，做有依据的进一步优化和小实验。
仅有脚本退出或候选算子变快，不算整项工作完成。没有收益的方案也应记录；不要为了优化而无休止扫描。
文档、图和复现说明更新在 examples/mirroropd/docs/infra/profile；按阶段提交，只提交本任务修改。
保留其他任务的工作区与暂存区。遵守本会话最新用户指示，尤其停止/暂停要求。
若只是 GPU 仍忙，保留定时检查，简短报告状态，不把等待写成完成。
若本机可做的工作已完成且结果已审阅、必要验证完成、文档图表与提交已整理，
写 {root / "REVIEW_DONE.json"}，记录完成依据、结果文档和提交号。
这会停止每小时触发，并在本会话给用户最终总结。
用户要求暂停或取消时写 {root / "MONITOR_STOP"}。
不要创建第二个定时器，不要仅回复收到后结束；完成这一轮实际检查与可执行工作。"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--thread", required=True)
    p.add_argument("--codex", required=True)
    p.add_argument("--interval", type=int, default=3600)
    args = p.parse_args()
    if args.interval < 60:
        p.error("interval must be at least 60 seconds")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "hourly-monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_file = root / "hourly-monitor.json"
        state = json.loads(state_file.read_text()) if state_file.exists() else {"tick": 0, "pending": False}
        state.update(pid=os.getpid(), thread=args.thread, interval_seconds=args.interval, state="scheduled")
        # Persist the cadence across restarts; never enqueue catch-up bursts.
        next_at = max(time.time(), state.get("next_at_unix", time.time() + args.interval))
        state.update(next_at_unix=next_at, updated_at=stamp())
        write(state_file, state)
        while True:
            if (root / "REVIEW_DONE.json").exists() or (root / "MONITOR_STOP").exists():
                state.update(
                    state="finished" if (root / "REVIEW_DONE.json").exists() else "stopped", updated_at=stamp()
                )
                write(state_file, state)
                return
            if time.time() < next_at:
                time.sleep(min(30, next_at - time.time()))
                continue
            ack_path = root / "hourly-ack.json"
            try:
                ack = json.loads(ack_path.read_text()) if ack_path.exists() else {}
            except (OSError, ValueError):
                ack = {}
            if state["pending"] and ack.get("tick", -1) < state["tick"]:
                state.update(state="awaiting_previous_review", updated_at=stamp())
            else:
                tick = state["tick"] + 1
                try:
                    result = subprocess.run(
                        [args.codex, "queue", "--thread", args.thread, "--message", review_prompt(root, tick)],
                        capture_output=True,
                        text=True,
                        timeout=90,
                    )
                    accepted = result.returncode == 0 and "Queued message" in result.stdout
                    if accepted:
                        match = re.search(r"Queued message ([\w-]+)", result.stdout)
                        state.update(
                            tick=tick,
                            pending=True,
                            state="queued",
                            queued_at=stamp(),
                            message_id=match.group(1) if match else None,
                        )
                    else:
                        state.update(state="queue_failed", last_error=(result.stderr or result.stdout)[-2000:])
                except (OSError, subprocess.TimeoutExpired) as exc:
                    state.update(state="queue_failed", last_error=str(exc))
                state["updated_at"] = stamp()
            next_at += args.interval
            if next_at <= time.time():
                next_at = time.time() + args.interval
            state["next_at_unix"] = next_at
            write(state_file, state)
            with (root / "hourly-history.jsonl").open("a") as log:
                log.write(json.dumps(state, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
