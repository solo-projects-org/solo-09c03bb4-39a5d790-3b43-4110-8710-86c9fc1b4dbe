#!/usr/bin/env python3
"""End-to-end self test for the ensemble rainfall probability API.

By default the script spawns TWO uvicorn processes sharing one SQLite file
(dual-instance deployment), waits for health, and exercises every key flow:

  * registration / member-count / threshold / datetime validation
  * idempotent resubmit (same revision+content) and conflicting resubmit rejection
  * revision ordering, grid-shape and non-negativity checks
  * publish blocked until every member is present
  * probability / denominator / all-missing null semantics and used revisions
  * correction immediately expires the old product while history stays queryable
  * two-instance interleaved publish vs. correction -> only consistent products
  * parallel duplicate publishes collapse to one idempotent product
  * injected compute failure leaves no half-written product (atomicity)
  * slow computation never blocks another forecast time's ingestion
  * Swagger UI / OpenAPI availability

Only the Python standard library is required.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS_COUNT = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS_COUNT
    if not cond:
        raise AssertionError(f"FAIL: {name} {detail}")
    PASS_COUNT += 1
    print(f"  PASS  {name}")


def http(method: str, base: str, path: str, body: Optional[dict] = None) -> Tuple[int, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"raw": raw}


def approx(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def matrix_approx(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        for x, y in zip(ra, rb):
            if not approx(x, y):
                return False
    return True


# --------------------------------------------------------------------------- #
# dual-instance process management
# --------------------------------------------------------------------------- #
def spawn_servers() -> Tuple[list, str, str, str, str]:
    tmp = tempfile.mkdtemp(prefix="rain-selftest-")
    db_path = os.path.join(tmp, "rain.db")
    fail_marker = os.path.join(tmp, "fail.marker")
    env = dict(os.environ)
    env.update(
        RAIN_DB_PATH=db_path,
        RAIN_TEST_PUBLISH_DELAY_S="0.6",
        RAIN_TEST_SENSITIVITY_DELAY_S="0.6",
        RAIN_TEST_FAIL_MARKER=fail_marker,
        PYTHONPATH=ROOT,
    )
    procs = []
    for port in (8091, 8092):
        log = open(os.path.join(tmp, f"uvicorn-{port}.log"), "w")
        procs.append(
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app",
                 "--host", "127.0.0.1", "--port", str(port)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        )
    base_a, base_b = "http://127.0.0.1:8091", "http://127.0.0.1:8092"
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            s1, _ = http("GET", base_a, "/health")
            s2, _ = http("GET", base_b, "/health")
            if s1 == 200 and s2 == 200:
                return procs, base_a, base_b, fail_marker, tmp
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("servers did not become healthy")


def shutdown(procs: list) -> None:
    for p in procs:
        p.send_signal(signal.SIGINT)
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


def main() -> int:
    base_a = os.environ.get("RAIN_BASE_A")
    base_b = os.environ.get("RAIN_BASE_B")
    fail_marker = os.environ.get("RAIN_TEST_FAIL_MARKER", "")
    spawned = False
    procs: list = []
    if not base_a or not base_b:
        procs, base_a, base_b, fail_marker, _ = spawn_servers()
        spawned = True
        print(f"[setup] spawned dual instances {base_a} / {base_b}")
    hooks_enabled = bool(fail_marker)
    try:
        run(base_a, base_b, fail_marker, hooks_enabled)
    finally:
        if spawned:
            shutdown(procs)
    print(f"\nALL TESTS PASSED ({PASS_COUNT} assertions)")
    return 0


def register(base: str, members: int = 3, rows: int = 2, cols: int = 2,
             thresholds=(5.0, 10.0), names=None) -> dict:
    if names is None:
        names = [f"m{i}" for i in range(1, members + 1)]
    body = {
        "forecast_time": "2026-11-16T00:00:00Z",
        "rows": rows, "cols": cols,
        "member_names": names,
        "thresholds": list(thresholds),
    }
    status, data = http("POST", base, "/groups", body)
    assert status == 201, (status, data)
    return data


def run(base_a: str, base_b: str, fail_marker: str, hooks_enabled: bool) -> None:
    print("\n[1] validation on registration")
    bad = {"forecast_time": "2026-11-16T00:00:00Z", "rows": 2, "cols": 2,
           "member_names": ["only-one"], "thresholds": [5.0]}
    s, d = http("POST", base_a, "/groups", bad)
    check("1 member rejected (need 2-8)", s == 422, str(d))
    bad["member_names"] = [f"m{i}" for i in range(9)]
    s, d = http("POST", base_a, "/groups", bad)
    check("9 members rejected", s == 422)
    bad["member_names"] = ["a", "b"]
    bad["thresholds"] = [-1.0]
    s, d = http("POST", base_a, "/groups", bad)
    check("negative threshold rejected", s == 422)
    bad["thresholds"] = [5.0]
    bad["forecast_time"] = "2026-11-16T00:00:00"  # naive, no timezone
    s, d = http("POST", base_a, "/groups", bad)
    check("naive datetime rejected", s == 422)

    print("\n[2] happy-path group + member submission semantics")
    g = register(base_a, members=3, thresholds=[5.0, 10.0])
    gid = g["group_id"]
    check("registered 3 members", len(g["members"]) == 3)
    s, d = http("GET", base_b, f"/groups/{gid}")
    check("group visible from instance B", s == 200 and d["all_members_present"] is False)

    rain1 = [[0.0, 10.0], [20.0, 0.0]]
    s, d = http("PUT", base_a, f"/groups/{gid}/members/m1",
                {"rain": rain1, "revision": 1})
    check("m1 rev1 accepted", s == 200 and d["duplicate"] is False, str(d))
    s, d2 = http("PUT", base_b, f"/groups/{gid}/members/m1",
                 {"rain": rain1, "revision": 1})
    check("same revision+content resent is idempotent", s == 200 and d2["duplicate"] is True)
    check("idempotent hash stable", d["content_hash"] == d2["content_hash"])
    s, d = http("PUT", base_a, f"/groups/{gid}/members/m1",
                {"rain": [[0.0, 10.0], [20.0, 1.0]], "revision": 1})
    check("same revision different content rejected (异文拒收)",
          s == 409 and d["error"]["code"] == "REVISION_CONTENT_CONFLICT", str(d))
    s, d = http("PUT", base_a, f"/groups/{gid}/members/ghost",
                {"rain": rain1, "revision": 1})
    check("unknown member -> 404", s == 404 and d["error"]["code"] == "MEMBER_NOT_FOUND")
    s, d = http("PUT", base_a, "/groups/nope/members/m1", {"rain": rain1, "revision": 1})
    check("unknown group -> 404", s == 404 and d["error"]["code"] == "GROUP_NOT_FOUND")

    print("\n[3] revision ordering and grid validation on a second group")
    g2 = register(base_b, members=2, thresholds=[5.0])
    gid2 = g2["group_id"]
    s, d = http("PUT", base_a, f"/groups/{gid2}/members/m1",
                {"rain": [[1.0, 2.0], [3.0, 4.0]], "revision": 1})
    check("g2 m1 rev1", s == 200)
    s, d = http("PUT", base_b, f"/groups/{gid2}/members/m1",
                {"rain": [[9.0, 9.0], [9.0, 9.0]], "revision": 2})
    check("g2 m1 rev2", s == 200)
    s, d = http("PUT", base_a, f"/groups/{gid2}/members/m1",
                {"rain": [[1.0, 2.0], [3.0, 4.0]], "revision": 1})
    check("lower revision rejected", s == 409 and d["error"]["code"] == "REVISION_CONFLICT")
    s, d = http("PUT", base_a, f"/groups/{gid2}/members/m1",
                {"rain": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "revision": 3})
    check("grid shape mismatch rejected", s == 422 and d["error"]["code"] == "GRID_MISMATCH")
    s, d = http("PUT", base_a, f"/groups/{gid2}/members/m1",
                {"rain": [[-1.0, 2.0], [3.0, 4.0]], "revision": 3})
    check("negative rain rejected", s == 422 and d["error"]["code"] == "NEGATIVE_RAIN")
    s, d = http("PUT", base_a, f"/groups/{gid2}/members/m1",
                {"rain": [[1.0, 2.0], [3.0, 4.0]], "missing": [[1, 0, 0]], "revision": 3})
    check("missing grid shape mismatch rejected",
          s == 422 and d["error"]["code"] == "GRID_MISMATCH", str(d))

    print("\n[4] publishing requires every member")
    s, d = http("POST", base_a, f"/groups/{gid}/publish")
    check("incomplete group cannot publish",
          s == 409 and d["error"]["code"] == "MEMBERS_INCOMPLETE", str(d))

    print("\n[5] complete group and verify probability math")
    rain2 = [[10.0, 10.0], [0.0, 0.0]]
    rain3 = [[5.0, 0.0], [99.0, 0.0]]
    miss3 = [[0, 0], [1, 1]]  # m3 missing on second row
    s, _ = http("PUT", base_b, f"/groups/{gid}/members/m2",
                {"rain": rain2, "revision": 1})
    check("m2 rev1 via instance B", s == 200)
    s, _ = http("PUT", base_a, f"/groups/{gid}/members/m3",
                {"rain": rain3, "missing": miss3, "revision": 1})
    check("m3 rev1 with missing grid", s == 200)

    s, p1 = http("POST", base_a, f"/groups/{gid}/publish")
    check("publish succeeds when complete", s == 200, str(p1))
    check("product marked current", p1["is_current"] is True)
    check("3 member revisions recorded",
          {m["member_id"]: m["revision"] for m in p1["members_used"]}
          == {"m1": 1, "m2": 1, "m3": 1})
    by_thr = {t["threshold"]: t for t in p1["thresholds"]}
    check("denominators exclude missing members",
          by_thr[5.0]["denominator"] == [[3, 3], [2, 2]],
          str(by_thr[5.0]["denominator"]))
    check("probability threshold=5",
          matrix_approx(by_thr[5.0]["probability"],
                        [[2 / 3, 2 / 3], [0.5, 0.0]]),
          str(by_thr[5.0]["probability"]))
    check("probability threshold=10",
          matrix_approx(by_thr[10.0]["probability"],
                        [[1 / 3, 2 / 3], [0.5, 0.0]]),
          str(by_thr[10.0]["probability"]))
    s, p1b = http("POST", base_b, f"/groups/{gid}/publish")
    check("republish identical revision set is idempotent",
          s == 200 and p1b["product_id"] == p1["product_id"])
    s, cur = http("GET", base_a, f"/groups/{gid}/products/current")
    check("current product query", s == 200 and cur["product_id"] == p1["product_id"])
    s, listing = http("GET", base_b, f"/groups/{gid}/products")
    check("one product in history listing", s == 200 and len(listing["products"]) == 1)
    s, d = http("GET", base_a, f"/groups/{gid2}/products/{p1['product_id']}")
    check("product from other group -> 404", s == 404)

    print("\n[6] all-missing cell yields null probability and zero denominator")
    g3 = register(base_a, members=2, rows=1, cols=2, thresholds=[5.0])
    gid3 = g3["group_id"]
    http("PUT", base_a, f"/groups/{gid3}/members/m1",
         {"rain": [[5.0, 9.0]], "missing": [[0, 1]], "revision": 1})
    http("PUT", base_b, f"/groups/{gid3}/members/m2",
         {"rain": [[10.0, 9.0]], "missing": [[0, 1]], "revision": 1})
    s, p3 = http("POST", base_a, f"/groups/{gid3}/publish")
    t3 = p3["thresholds"][0]
    check("all-missing cell null", t3["probability"] == [[1.0, None]], str(t3))
    check("all-missing cell denominator 0", t3["denominator"] == [[2, 0]])

    print("\n[7] correction expires old product immediately, history retained")
    s, _ = http("PUT", base_b, f"/groups/{gid}/members/m1",
                {"rain": [[0.0, 10.0], [20.0, 5.0]], "revision": 2})
    check("correction rev2 accepted", s == 200)
    s, d = http("GET", base_a, f"/groups/{gid}/products/current")
    check("no stale product served as current warning basis",
          s == 404 and d["error"]["code"] == "NO_CURRENT_PRODUCT", str(d))
    s, listing = http("GET", base_b, f"/groups/{gid}/products")
    check("old product retained but superseded",
          len(listing["products"]) == 1 and listing["products"][0]["is_current"] is False)
    s, hist = http("GET", base_a, f"/groups/{gid}/products/{p1['product_id']}")
    check("historical product still queryable", s == 200 and hist["is_current"] is False)
    s, p2 = http("POST", base_b, f"/groups/{gid}/publish")
    check("republish after correction", s == 200 and p2["product_id"] != p1["product_id"])
    check("new product uses rev2 for corrected member",
          {m["member_id"]: m["revision"] for m in p2["members_used"]}
          == {"m1": 2, "m2": 1, "m3": 1})
    s, cur = http("GET", base_b, f"/groups/{gid}/products/current")
    check("current points at new product", cur["product_id"] == p2["product_id"])

    print("\n[8] dual-instance interleaving: publish vs concurrent correction")
    g4 = register(base_a, members=3, rows=64, cols=64, thresholds=[5.0, 20.0])
    gid4 = g4["group_id"]
    for i, inst in enumerate((base_a, base_b, base_a)):
        name = f"m{i + 1}"
        grid = [[float((i + 1) * (r + c) % 25) for c in range(64)] for r in range(64)]
        s, _ = http("PUT", inst, f"/groups/{gid4}/members/{name}",
                    {"rain": grid, "revision": 1})
        assert s == 200

    result: Dict[str, Any] = {}

    def worker_publish() -> None:
        result["status"], result["body"] = http("POST", base_a, f"/groups/{gid4}/publish")

    t = threading.Thread(target=worker_publish)
    t.start()
    time.sleep(0.15)  # let the publish enter its slow compute phase
    t0 = time.time()
    s, corr = http("PUT", base_b, f"/groups/{gid4}/members/m2",
                   {"rain": [[7.0] * 64 for _ in range(64)], "revision": 2})
    write_elapsed = time.time() - t0
    check("correction write not blocked by in-flight compute", s == 200, str(corr))
    if hooks_enabled:
        check("ingest stayed fast while compute was running", write_elapsed < 0.5,
              f"{write_elapsed:.2f}s")
    t.join()
    check("in-flight publish eventually returns 200", result["status"] == 200, str(result))

    s, status4 = http("GET", base_a, f"/groups/{gid4}")
    latest = {m["member_id"]: m["latest_revision"] for m in status4["members"]}
    s, cur4 = http("GET", base_b, f"/groups/{gid4}/products/current")
    check("a current product exists after interleaving", s == 200)
    cur_map = {m["member_id"]: m["revision"] for m in cur4["members_used"]}
    check("current product matches latest member revisions (no mixed revision group)",
          cur_map == latest, f"{cur_map} vs {latest}")
    s, listing4 = http("GET", base_a, f"/groups/{gid4}/products")
    for prod in listing4["products"]:
        check(f"product {prod['product_id']} has complete revision set",
              len(prod["revision_map"]) == 3)
        if prod["is_current"]:
            check("only one current product",
                  sum(1 for x in listing4["products"] if x["is_current"]) == 1)
    check("current listing entry consistent",
          [x for x in listing4["products"] if x["is_current"]][0]["product_id"]
          == cur4["product_id"])

    print("\n[9] parallel publishes of the same revision set collapse to one product")
    g5 = register(base_b, members=2, rows=8, cols=8, thresholds=[5.0])
    gid5 = g5["group_id"]
    http("PUT", base_a, f"/groups/{gid5}/members/m1",
         {"rain": [[6.0] * 8 for _ in range(8)], "revision": 1})
    http("PUT", base_b, f"/groups/{gid5}/members/m2",
         {"rain": [[0.0] * 8 for _ in range(8)], "revision": 1})
    outs: Dict[str, Any] = {}

    def pub(instance: str) -> None:
        outs[instance] = http("POST", instance, f"/groups/{gid5}/publish")

    threads = [threading.Thread(target=pub, args=(base_a,)),
               threading.Thread(target=pub, args=(base_b,))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    ids = {body["product_id"] for st, body in outs.values() if st == 200}
    check("both parallel publishes succeed", all(st == 200 for st, _ in outs.values()))
    check("both return the same idempotent product", len(ids) == 1, str(ids))
    s, listing5 = http("GET", base_a, f"/groups/{gid5}/products")
    check("exactly one product row for the revision set", len(listing5["products"]) == 1)

    print("\n[10] compute failure leaves no half-written product")
    g6 = register(base_a, members=2, rows=4, cols=4, thresholds=[5.0])
    gid6 = g6["group_id"]
    http("PUT", base_a, f"/groups/{gid6}/members/m1",
         {"rain": [[6.0] * 4 for _ in range(4)], "revision": 1})
    http("PUT", base_b, f"/groups/{gid6}/members/m2",
         {"rain": [[4.0] * 4 for _ in range(4)], "revision": 1})
    if hooks_enabled:
        with open(fail_marker, "w") as fh:
            fh.write("fail")
        s, d = http("POST", base_a, f"/groups/{gid6}/publish")
        os.remove(fail_marker)
        check("injected failure surfaces as 500", s == 500 and d["error"]["code"] == "COMPUTE_FAILED")
        s, d = http("GET", base_b, f"/groups/{gid6}/products/current")
        check("no current product after failed compute", s == 404)
        s, listing6 = http("GET", base_a, f"/groups/{gid6}/products")
        check("no product rows at all after failed compute", listing6["products"] == [])
        s, p6 = http("POST", base_b, f"/groups/{gid6}/publish")
        check("publish succeeds after failure cleared", s == 200 and p6["is_current"])
    else:
        print("  SKIP  failure injection (RAIN_TEST_FAIL_MARKER not configured)")

    print("\n[11] slow compute on one time does not block another time's ingestion")
    g7 = register(base_a, members=2, rows=16, cols=16, thresholds=[5.0])
    gid7 = g7["group_id"]
    http("PUT", base_a, f"/groups/{gid7}/members/m1",
         {"rain": [[6.0] * 16 for _ in range(16)], "revision": 1})
    http("PUT", base_a, f"/groups/{gid7}/members/m2",
         {"rain": [[6.0] * 16 for _ in range(16)], "revision": 1})

    bg = {}

    def slow_publish() -> None:
        bg["result"] = http("POST", base_a, f"/groups/{gid7}/publish")

    th = threading.Thread(target=slow_publish)
    th.start()
    time.sleep(0.15)
    t0 = time.time()
    g8 = register(base_b, members=2, rows=3, cols=3, thresholds=[5.0])
    http("PUT", base_b, f"/groups/{g8['group_id']}/members/m1",
         {"rain": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], "revision": 1})
    s, _ = http("PUT", base_b, f"/groups/{g8['group_id']}/members/m2",
                {"rain": [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]], "revision": 1})
    ingest_elapsed = time.time() - t0
    check("other time registration+ingest succeeded during slow compute", s == 200)
    if hooks_enabled:
        check("other time ingest stayed fast", ingest_elapsed < 0.5, f"{ingest_elapsed:.2f}s")
    th.join()
    check("slow publish completed", bg["result"][0] == 200, str(bg))

    print("\n[12] Swagger / OpenAPI")
    s, openapi = http("GET", base_a, "/openapi.json")
    check("openapi.json served", s == 200 and "paths" in openapi)
    for path in ("/groups", "/groups/{group_id}/members/{member_id}",
                 "/groups/{group_id}/publish",
                 "/groups/{group_id}/products/current",
                 "/groups/{group_id}/sensitivity"):
        check(f"path {path} documented", path in openapi["paths"])
    req = urllib.request.Request(base_a + "/docs")
    with urllib.request.urlopen(req, timeout=10) as resp:
        check("Swagger UI served", resp.status == 200 and b"swagger" in resp.read().lower())

    print("\n[13] leave-one-out sensitivity: math, attribution and validation")
    gs = register(base_a, members=3, rows=2, cols=3, thresholds=[5.0, 10.0])
    gids = gs["group_id"]
    s, _ = http("PUT", base_a, f"/groups/{gids}/members/m1",
                {"rain": [[10, 10, 10], [0, 10, 0]],
                 "missing": [[0, 0, 0], [1, 0, 0]], "revision": 1})
    check("s m1 rev1", s == 200)
    s, _ = http("PUT", base_b, f"/groups/{gids}/members/m2",
                {"rain": [[0, 10, 0], [0, 0, 0]],
                 "missing": [[0, 0, 0], [1, 1, 0]], "revision": 1})
    check("s m2 rev1", s == 200)
    s, _ = http("PUT", base_a, f"/groups/{gids}/members/m3",
                {"rain": [[0, 0, 0], [0, 0, 7]],
                 "missing": [[0, 0, 1], [1, 1, 0]], "revision": 1})
    check("s m3 rev1", s == 200)

    s, d = http("POST", base_a, f"/groups/{gids}/sensitivity",
                {"threshold": 5.0, "alert_probability": 0.5})
    check("sensitivity before publish -> 404 NO_CURRENT_PRODUCT",
          s == 404 and d["error"]["code"] == "NO_CURRENT_PRODUCT", str(d))

    s, ps = http("POST", base_a, f"/groups/{gids}/publish")
    check("publish for sensitivity group", s == 200, str(ps))

    s, d = http("POST", base_a, f"/groups/{gids}/sensitivity",
                {"threshold": 7.0, "alert_probability": 0.5})
    check("unregistered threshold rejected",
          s == 422 and d["error"]["code"] == "THRESHOLD_NOT_REGISTERED", str(d))
    s, d = http("POST", base_a, f"/groups/{gids}/sensitivity",
                {"threshold": 5.0, "alert_probability": 1.5})
    check("alert line > 1 rejected", s == 422 and d["error"]["code"] == "VALIDATION_ERROR")
    s, d = http("POST", base_a, f"/groups/{gids}/sensitivity",
                {"threshold": 5.0, "alert_probability": -0.1})
    check("negative alert line rejected", s == 422)
    s, d = http("POST", base_a, "/groups/nope/sensitivity",
                {"threshold": 5.0, "alert_probability": 0.5})
    check("unknown group -> 404", s == 404 and d["error"]["code"] == "GROUP_NOT_FOUND")

    s, sens = http("POST", base_b, f"/groups/{gids}/sensitivity",
                   {"threshold": 5.0, "alert_probability": 0.5})
    check("sensitivity computed via instance B", s == 200, str(sens))
    check("sensitivity bound to current product", sens["product_id"] == ps["product_id"])
    check("members_used reflects product revisions",
          {m["member_id"]: m["revision"] for m in sens["members_used"]}
          == {"m1": 1, "m2": 1, "m3": 1})
    flips = sens["flips"]
    got = {(f["row"], f["col"], f["member_id"]) for f in flips}
    expected = {(0, 0, "m2"), (0, 0, "m3"), (0, 2, "m1"), (1, 2, "m1"), (1, 2, "m2")}
    check("exactly the expected (cell, member) flips", got == expected, str(got))
    check("flip counts", sens["flip_count"] == 5 and sens["flipped_cell_count"] == 3,
          f'{sens["flip_count"]}/{sens["flipped_cell_count"]}')
    f = next(x for x in flips if (x["row"], x["col"], x["member_id"]) == (0, 0, "m2"))
    check("flip record probabilities",
          approx(f["original_probability"], 1 / 3) and approx(f["loo_probability"], 0.5), str(f))
    check("flip record warning direction",
          f["original_warning"] is False and f["loo_warning"] is True)
    check("flip record valid member counts",
          f["original_valid_members"] == 3 and f["loo_valid_members"] == 2)
    f = next(x for x in flips if (x["row"], x["col"], x["member_id"]) == (0, 2, "m1"))
    check("removing the only hitting member flips warning off",
          approx(f["original_probability"], 0.5) and approx(f["loo_probability"], 0.0)
          and f["original_warning"] is True and f["loo_warning"] is False
          and f["loo_valid_members"] == 1, str(f))
    check("member missing at a cell is never attributed there",
          all(f["member_id"] != "m3" or (f["row"], f["col"]) != (0, 2) for f in flips))
    check("null-probability and leave-one-out-all-missing cells never judged",
          all((f["row"], f["col"]) not in {(1, 0), (1, 1)} for f in flips))

    s, sens10 = http("POST", base_a, f"/groups/{gids}/sensitivity",
                     {"threshold": 10.0, "alert_probability": 0.5})
    got10 = {(f["row"], f["col"], f["member_id"]) for f in sens10["flips"]}
    check("second registered threshold analyzed independently",
          s == 200 and got10 == {(0, 0, "m2"), (0, 0, "m3"), (0, 2, "m1")}
          and sens10["flip_count"] == 3, str(got10))

    s, listing = http("GET", base_a, f"/groups/{gids}/products")
    check("sensitivity is read-only (no products created)", len(listing["products"]) == 1)

    print("\n[14] correction invalidates the sensitivity basis; republish restores it")
    s, _ = http("PUT", base_b, f"/groups/{gids}/members/m2",
                {"rain": [[10, 10, 10], [10, 10, 10]],
                 "missing": [[0, 0, 0], [1, 1, 0]], "revision": 2})
    check("correction rev2 accepted", s == 200)
    s, d = http("POST", base_a, f"/groups/{gids}/sensitivity",
                {"threshold": 5.0, "alert_probability": 0.5})
    check("no stale conclusions while no current product",
          s == 404 and d["error"]["code"] == "NO_CURRENT_PRODUCT", str(d))
    s, ps2 = http("POST", base_a, f"/groups/{gids}/publish")
    check("republish after correction", s == 200 and ps2["product_id"] != ps["product_id"])
    s, sens2 = http("POST", base_b, f"/groups/{gids}/sensitivity",
                    {"threshold": 5.0, "alert_probability": 0.5})
    check("sensitivity uses the new product", s == 200 and sens2["product_id"] == ps2["product_id"])
    check("members_used shows corrected revision",
          {m["member_id"]: m["revision"] for m in sens2["members_used"]}
          == {"m1": 1, "m2": 2, "m3": 1})
    check("conclusions recomputed from new grids (no flips remain)",
          sens2["flip_count"] == 0, str(sens2["flips"]))

    print("\n[15] sensitivity vs concurrent correction: stale conclusions never delivered")
    gi = register(base_a, members=3, rows=16, cols=16, thresholds=[5.0])
    gidi = gi["group_id"]
    for i, inst in enumerate((base_a, base_b, base_a)):
        grid = [[float((i + 1) * (r * 3 + c) % 17) for c in range(16)] for r in range(16)]
        s, _ = http("PUT", inst, f"/groups/{gidi}/members/m{i + 1}",
                    {"rain": grid, "revision": 1})
        assert s == 200
    s, pi = http("POST", base_a, f"/groups/{gidi}/publish")
    assert s == 200

    sens_result: Dict[str, Any] = {}

    def worker_sensitivity() -> None:
        sens_result["result"] = http("POST", base_a, f"/groups/{gidi}/sensitivity",
                                     {"threshold": 5.0, "alert_probability": 0.5})

    t = threading.Thread(target=worker_sensitivity)
    t.start()
    time.sleep(0.15)  # let the sensitivity request enter its slow recompute phase
    t0 = time.time()
    s, corr = http("PUT", base_b, f"/groups/{gidi}/members/m2",
                   {"rain": [[3.0] * 16 for _ in range(16)], "revision": 2})
    corr_elapsed = time.time() - t0
    check("correction accepted during sensitivity recompute", s == 200, str(corr))
    if hooks_enabled:
        check("correction write not blocked by the recompute", corr_elapsed < 0.5,
              f"{corr_elapsed:.2f}s")
    t0 = time.time()
    g_other = register(base_b, members=2, rows=3, cols=3, thresholds=[5.0])
    http("PUT", base_b, f"/groups/{g_other['group_id']}/members/m1",
         {"rain": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], "revision": 1})
    s, _ = http("PUT", base_b, f"/groups/{g_other['group_id']}/members/m2",
                {"rain": [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]], "revision": 1})
    ingest_elapsed = time.time() - t0
    check("other forecast time kept ingesting during the recompute", s == 200)
    if hooks_enabled:
        check("other time ingest stayed fast", ingest_elapsed < 0.5, f"{ingest_elapsed:.2f}s")
    t.join()
    st, body = sens_result["result"]
    if hooks_enabled:
        check("stale conclusions not delivered (404 after mid-compute correction)",
              st == 404 and body["error"]["code"] == "NO_CURRENT_PRODUCT", str(sens_result))
    else:
        check("sensitivity answered consistently", st in (200, 404, 409), str(sens_result))
    s, pi2 = http("POST", base_a, f"/groups/{gidi}/publish")
    check("republish after interleaved correction", s == 200)
    s, sens3 = http("POST", base_b, f"/groups/{gidi}/sensitivity",
                    {"threshold": 5.0, "alert_probability": 0.5})
    check("post-republish sensitivity bound to the new product",
          s == 200 and sens3["product_id"] == pi2["product_id"], str(sens3))
    check("post-republish sensitivity uses the corrected revision",
          any(m["member_id"] == "m2" and m["revision"] == 2 for m in sens3["members_used"]))


if __name__ == "__main__":
    sys.exit(main())
