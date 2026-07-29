"""
make_wmt24pp_split.py
─────────────────────
Generate the FIXED test/selection split for the WMT24++ dataset and write it
to data/wmt24pp_split.json. Run ONCE; the JSON is committed so every model /
language / k / reasoning run reads the exact same split forever. Do NOT re-run
with a different seed after experiments have started.

Verifies (via the HuggingFace datasets-server API, no dataset download):
  • each config used has exactly 998 rows,
  • English sources + is_bad_source flags are identical across configs
    (they are source-side properties, so one shared split works for all
    language pairs),
  • counts the is_bad_source rows (row 0 is a canary GUID) and excludes them
    from BOTH the test set and the selection pool.

Inputs:  none (network access to datasets-server.huggingface.co).
Outputs: data/wmt24pp_split.json with test_segment_ids (100) and
         pool_segment_ids (remaining good rows), plus verification metadata.
"""

import json
import os
import random
import time
import urllib.request

# SEED: fixed RNG seed for the 100-sentence test draw. Never change after runs exist.
SEED = 12345
# N_TEST: number of test sentences to draw from the good (non-bad-source) rows.
N_TEST = 100
# EXPECTED_ROWS: row count per config claimed by the dataset card; verified below.
EXPECTED_ROWS = 998
# CONFIGS: every WMT24++ config our experiments touch (superset is fine — the
# split is over segment_ids, which are shared across configs).
CONFIGS = ["en-ca_ES", "en-zu_ZA", "en-ml_IN", "en-sk_SK", "en-is_IS"]
# OUT_PATH: where the committed split file lands (repo-relative).
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "wmt24pp_split.json")

BASE = "https://datasets-server.huggingface.co"
DS = "google/wmt24pp"
PAGE = 100


def _get(url: str, retries: int = 6) -> dict:
    """
    Purpose: GET a datasets-server URL and parse JSON, retrying transient errors.
    Inputs: url string, retries count.
    Outputs: parsed JSON dict.
    """
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "wmt24pp-split"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            last = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"GET failed after {retries} attempts: {url}") from last


def fetch_config_rows(config: str) -> list:
    """
    Purpose: Page through all rows of one config, keeping the split-relevant fields.
    Inputs: config name string (e.g. "en-sw_TZ").
    Outputs: list of dicts with segment_id / source / is_bad_source, ordered by row index.
    """
    rows = []
    offset = 0
    while True:
        data = _get(f"{BASE}/rows?dataset={DS}&config={config}&split=train&offset={offset}&length={PAGE}")
        batch = data["rows"]
        if not batch:
            break
        for r in batch:
            row = r["row"]
            rows.append({
                "segment_id": row["segment_id"],
                "source": row["source"],
                "is_bad_source": bool(row["is_bad_source"]),
            })
        offset += len(batch)
        if offset >= EXPECTED_ROWS:
            break
    return rows


def main() -> None:
    """
    Purpose: Verify WMT24++ structure, build the fixed split, write the JSON.
    Inputs: module constants.
    Outputs: data/wmt24pp_split.json.
    """
    ref_config = CONFIGS[0]
    ref_rows = fetch_config_rows(ref_config)
    if len(ref_rows) != EXPECTED_ROWS:
        raise AssertionError(f"{ref_config}: expected {EXPECTED_ROWS} rows, got {len(ref_rows)}")
    print(f"[ok] {ref_config}: {len(ref_rows)} rows")

    for cfg in CONFIGS[1:]:
        rows = fetch_config_rows(cfg)
        if len(rows) != EXPECTED_ROWS:
            raise AssertionError(f"{cfg}: expected {EXPECTED_ROWS} rows, got {len(rows)}")
        for a, b in zip(ref_rows, rows):
            if a["segment_id"] != b["segment_id"] or a["source"] != b["source"] or a["is_bad_source"] != b["is_bad_source"]:
                raise AssertionError(f"{cfg}: source-side mismatch vs {ref_config} at segment_id={a['segment_id']}")
        print(f"[ok] {cfg}: rows + sources + bad flags identical to {ref_config}")

    bad_ids = [r["segment_id"] for r in ref_rows if r["is_bad_source"]]
    good_ids = [r["segment_id"] for r in ref_rows if not r["is_bad_source"]]
    print(f"[ok] good rows: {len(good_ids)}   bad-source rows (excluded): {len(bad_ids)}")

    rng = random.Random(SEED)
    test_ids = sorted(rng.sample(good_ids, N_TEST))
    test_set = set(test_ids)
    pool_ids = [sid for sid in good_ids if sid not in test_set]

    payload = {
        "dataset": DS,
        "seed": SEED,
        "expected_rows_per_config": EXPECTED_ROWS,
        "configs_verified": CONFIGS,
        "n_bad_source_excluded": len(bad_ids),
        "bad_source_segment_ids": bad_ids,
        "n_test": len(test_ids),
        "n_pool": len(pool_ids),
        "test_segment_ids": test_ids,
        "pool_segment_ids": pool_ids,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"[ok] wrote {OUT_PATH}   test={len(test_ids)} pool={len(pool_ids)}")


if __name__ == "__main__":
    main()
