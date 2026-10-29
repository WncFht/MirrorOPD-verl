# Copyright 2026 MirrorOPD-verl contributors
# SPDX-License-Identifier: Apache-2.0
"""Shared task metrics and identical-request transport retries."""

import asyncio
import json
import time


def aggregate(rows, samples):
    import numpy as np

    from verl.utils.metric.evaluation import dataset_slug, evaluation_metrics

    sources = [r["data_source"] for r in rows]
    ids = [r["prompt_id"] for r in rows]
    infos = {key: [r.get(key) for r in rows] for key in ["score", "acc", "format_score", "pred"]}
    infos["__response_length__"] = [r.get("response_length") for r in rows]
    infos["__finish_reason__"] = [r.get("finish_reason") for r in rows]
    expected = {(r["data_source"], r["prompt_id"]): samples for r in rows}
    metrics = evaluation_metrics(
        sources, ids, infos, expected, sample_indices=[r["sample_index"] for r in rows], prompt_ids=ids
    )
    for source in sorted(set(sources)):
        subset = [r for r in rows if r["data_source"] == source and r["status"] == "ok"]
        if not subset:
            continue
        prefix = f"eval/{dataset_slug(source)}/generation/"
        metrics[prefix + "repeat8_mean"] = float(np.mean([r["repeat8"] for r in subset]))
        metrics[prefix + "empty_text_fraction"] = float(np.mean([not r["output"].strip() for r in subset]))
    return metrics


async def request(session, url, body, retry_log, prompt_id, index):
    import aiohttp

    for attempt in range(1, 4):
        try:
            async with session.post(url, json=body) as response:
                value = await response.json()
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: {value}")
            return value, attempt
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            with retry_log.open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            prompt_id=prompt_id,
                            sample_index=index,
                            attempt=attempt,
                            seed=body.get("seed"),
                            error=repr(error),
                            time=time.time(),
                        )
                    )
                    + "\n"
                )
            if attempt == 3:
                raise
            await asyncio.sleep(attempt)
    raise AssertionError("unreachable")
