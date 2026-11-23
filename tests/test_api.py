"""End-to-end API tests: versioning, correction update, conflicts, predict."""
import random

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


def make_rows(n, speed_mu, press_mu, coef, intercept, noise, seed,
              press_shift=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        s = int(round(rng.gauss(speed_mu, 25)))
        p = int(round(rng.gauss(press_mu, 12)))
        y = coef[0] * s + coef[1] * (p + press_shift) + intercept + rng.gauss(0, noise)
        rows.append((s, p, int(round(y))))
    return rows


def payload(factory_id, rows, correction=0):
    n = len(rows)
    return {
        "factory_id": factory_id,
        "n": n,
        "sums": [sum(r[i] for r in rows) for i in range(3)],
        "outer": [[sum(r[i] * r[j] for r in rows) for j in range(3)]
                  for i in range(3)],
        "correction": correction,
    }


def demo_factories():
    rows = [
        make_rows(400, 1_000_000 + 30 * i, 500_000 + 8 * i, (0.02, 0.011),
                  1500, 4, seed=200 + i)
        for i in range(4)
    ]
    # F3's pressure sensor is really off by +25 but reports 0
    rows[2] = make_rows(400, 1_000_060, 500_016, (0.02, 0.011), 1500, 4,
                        seed=999, press_shift=25)
    return [payload(f"F{i+1}", r) for i, r in enumerate(rows)]


@pytest.fixture()
def client():
    app = create_app(":memory:")
    with TestClient(app) as c:
        yield c


def create(client, factories=None):
    r = client.post("/versions", json={"factories": factories or demo_factories()})
    assert r.status_code == 201, r.text
    return r.json()


def test_create_and_query_current(client):
    body = create(client)
    assert body["version"] == 1
    assert body["lambda"] in (0.01, 0.1, 1.0)
    assert set(body["factory_errors"]) == {"F1", "F2", "F3", "F4"}
    assert set(body["lambda_scores"]) == {"0.01", "0.1", "1.0"}
    cur = client.get("/versions/current").json()
    assert cur == body
    by_id = client.get("/versions/1").json()
    assert by_id == body


def test_predict_uses_current_version(client):
    create(client)
    r = client.post("/predict", json={"factory_id": "F1", "speed": 1_000_030,
                                      "pressure": 500_008})
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == 1
    assert isinstance(body["power"], float)


def test_correction_update_atomic_and_versioned(client):
    before = create(client)
    err_before = before["factory_errors"]["F3"]
    pred_before = client.post("/predict", json={
        "factory_id": "F3", "speed": 1_000_060, "pressure": 500_016
    }).json()

    r = client.post("/versions/1/correction",
                    json={"factory_id": "F3", "correction": 25})
    assert r.status_code == 200, r.text
    after = r.json()
    assert after["version"] == 2
    assert after["corrections"]["F3"] == 25
    # errors + model changed together with the correction
    assert after["factory_errors"]["F3"] < err_before
    assert after["model"] != before["model"]

    cur = client.get("/versions/current").json()
    assert cur["version"] == 2
    assert cur["factory_errors"] == after["factory_errors"]

    pred_after = client.post("/predict", json={
        "factory_id": "F3", "speed": 1_000_060, "pressure": 500_016
    }).json()
    assert pred_after["version"] == 2
    assert pred_after["power"] != pred_before["power"]

    # old version is immutable history
    assert client.get("/versions/1").json()["corrections"]["F3"] == 0


def test_conflict_keeps_current_version(client):
    create(client)
    client.post("/versions/1/correction",
                json={"factory_id": "F3", "correction": 25})
    # replay against stale version 1
    r = client.post("/versions/1/correction",
                    json={"factory_id": "F3", "correction": 99})
    assert r.status_code == 409
    cur = client.get("/versions/current").json()
    assert cur["version"] == 2
    assert cur["corrections"]["F3"] == 25


def test_update_unknown_factory(client):
    create(client)
    r = client.post("/versions/1/correction",
                    json={"factory_id": "NOPE", "correction": 1})
    assert r.status_code == 404
    assert client.get("/versions/current").json()["version"] == 1


def test_update_unknown_version(client):
    create(client)
    r = client.post("/versions/99/correction",
                    json={"factory_id": "F1", "correction": 1})
    assert r.status_code == 404


def test_predict_unknown_factory(client):
    create(client)
    r = client.post("/predict", json={"factory_id": "NOPE", "speed": 1,
                                      "pressure": 2})
    assert r.status_code == 404


def test_predict_without_model(client):
    r = client.post("/predict", json={"factory_id": "F1", "speed": 1,
                                      "pressure": 2})
    assert r.status_code == 503


def test_predict_during_recompute_uses_current_version(client, monkeypatch):
    import threading

    import app.main as main_mod

    create(client)
    orig_train = main_mod.train
    started = threading.Event()
    release = threading.Event()

    def slow_train(stats):
        started.set()
        assert release.wait(10)
        return orig_train(stats)

    monkeypatch.setattr(main_mod, "train", slow_train)
    outcome = {}

    def do_update():
        r = client.post("/versions/1/correction",
                        json={"factory_id": "F3", "correction": 500})
        outcome["status"] = r.status_code

    t = threading.Thread(target=do_update)
    t.start()
    assert started.wait(10)
    # recomputation is blocked mid-flight: predictions keep using version 1
    for _ in range(5):
        r = client.post("/predict", json={"factory_id": "F3",
                                          "speed": 1_000_060,
                                          "pressure": 500_016})
        assert r.status_code == 200
        assert r.json()["version"] == 1
    assert client.get("/versions/current").json()["version"] == 1
    release.set()
    t.join(10)
    assert outcome["status"] == 200
    assert client.get("/versions/current").json()["version"] == 2


def test_factory_count_bounds(client):
    fs = demo_factories()
    assert client.post("/versions", json={"factories": fs[:2]}).status_code == 422
    too_many = [dict(f, factory_id=f"X{i}") for i, f in enumerate(fs * 3)]
    assert client.post("/versions",
                       json={"factories": too_many}).status_code == 422


def test_duplicate_factory_ids_rejected(client):
    fs = demo_factories()
    fs[1]["factory_id"] = fs[0]["factory_id"]
    assert client.post("/versions", json={"factories": fs}).status_code == 422


def test_inconsistent_stats_rejected(client):
    fs = demo_factories()
    fs[0]["outer"][0][0] = 1  # impossible: n*sum(s^2) < sum(s)^2
    r = client.post("/versions", json={"factories": fs})
    assert r.status_code == 422
    assert client.get("/versions/current").status_code == 404


def test_huge_base_values_variance_preserved(client):
    body = create(client)  # demo factories sit around 1e6 / 5e5
    assert body["model"]["std"]["0"] > 0
    assert body["model"]["std"]["1"] > 0
    # model explains the data well: worst LOFO error far below power variance
    assert max(body["factory_errors"].values()) < 1e4


def test_beyond_int64_values_keep_variance(client):
    # readings ~1e12: sums ~1e15, outer sums ~1e27 -- beyond float64's exact
    # integer range; variance of a few units must still survive.
    rng = random.Random(42)
    factories = []
    for i in range(3):
        rows = []
        for _ in range(60):
            s = 10**12 + 1000 * i + rng.randint(-4, 4)
            p = 10**12 + rng.randint(-3, 3)
            y = 3 * (s - 10**12) + 2 * (p - 10**12) + 5000 + rng.randint(-2, 2)
            rows.append((s, p, y))
        factories.append(payload(f"G{i}", rows))
    body = create(client, factories)
    assert body["model"]["std"]["0"] > 0
    assert body["model"]["std"]["1"] > 0
    errs = body["factory_errors"]
    # interpolating factory fits almost perfectly -> variance really usable
    assert errs["G1"] < 100.0
    # edge factories extrapolate in LOFO, but if variance had collapsed to
    # zero the speed feature would be dropped and errors would be ~1e6+
    assert max(errs.values()) < 1e5


# ---------------------------------------------------------------- gear API
GEAR_CONDITIONS = {
    "pressure_min": 500_000,
    "pressure_max": 500_030,
    "speed_min": 999_900,
    "speed_max": 1_000_200,
    "power_limit": 27_007,
}


def select(client, **overrides):
    body = {"factory_id": "F3", **GEAR_CONDITIONS, **overrides}
    return client.post("/gear/select", json=body)


def review(client, gear, **overrides):
    body = {"factory_id": "F3", "gear": gear, **GEAR_CONDITIONS, **overrides}
    return client.post("/gear/review", json=body)


def test_gear_select_returns_highest_usable(client):
    create(client)
    r = select(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["available"] is True
    assert body["version"] == 1
    assert body["correction"] == 0
    assert isinstance(body["gear"], int)
    assert body["max_power"] <= GEAR_CONDITIONS["power_limit"]
    assert body["max_distance"] <= 3.0
    # with an unconstrained cap only the distance bound is left: the picked
    # gear is the highest one satisfying it, and one more is rejected.
    top = select(client, power_limit=1e9).json()
    assert top["gear"] >= body["gear"]
    over = review(client, top["gear"] + 1, power_limit=1e9)
    assert over.status_code == 409
    assert "mahalanobis" in over.json()["reasons"]


def test_gear_select_uses_raw_pressure_and_one_version(client):
    create(client)
    before = select(client).json()
    # reversing the raw endpoints must not change the answer
    reversed_ = select(client, pressure_min=500_030,
                       pressure_max=500_000).json()
    assert reversed_["gear"] == before["gear"]
    assert reversed_["version"] == before["version"]

    after_corr = client.post(
        "/versions/1/correction", json={"factory_id": "F3", "correction": 500}
    )
    assert after_corr.status_code == 200
    new = select(client).json()
    assert new["version"] == 2
    assert new["correction"] == 500
    # the correction changes the summary mean and the model together
    assert new["max_power"] != before["max_power"]


def test_gear_select_no_usable_gear(client):
    create(client)
    r = select(client, power_limit=1)
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["gear"] is None
    # pressure far outside the factory's operating range
    far = select(client, pressure_min=900_000, pressure_max=900_001).json()
    assert far["available"] is False and far["gear"] is None


def test_gear_select_unknown_factory_and_no_model(client):
    assert select(client).status_code == 503
    create(client)
    r = select(client, factory_id="NOPE")
    assert r.status_code == 404


def test_gear_select_rejects_bad_ranges(client):
    create(client)
    assert select(client, speed_min=10, speed_max=5).status_code == 422
    assert select(client, power_limit=0).status_code == 422
    assert select(client, pressure_min=1.5).status_code == 422


def test_gear_review_accepts_same_version_selection(client):
    create(client)
    chosen = select(client).json()["gear"]
    r = review(client, chosen)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["accepted"] is True
    assert body["reasons"] == []
    assert body["version"] == 1


def test_gear_review_rejects_after_version_change_with_suggestion(client):
    create(client)
    # Unconstrained cap: the distance bound alone decides the highest gear.
    top = select(client, power_limit=1e9).json()
    chosen = top["gear"]
    # Set the cap just at that gear's worst predicted power (tiny slack for
    # the float serialisation), so it passes under version 1 with no margin.
    limit = top["max_power"] * (1 + 1e-9)
    r = review(client, chosen, power_limit=limit)
    assert r.status_code == 200
    assert r.json()["accepted"] is True

    client.post("/versions/1/correction",
                json={"factory_id": "F3", "correction": 500})
    r = review(client, chosen, power_limit=limit)
    assert r.status_code == 409, r.text
    body = r.json()
    assert body["accepted"] is False
    assert body["version"] == 2
    assert body["correction"] == 500
    assert "power_limit" in body["reasons"]
    assert body["suggestion"]["available"] is True
    assert body["suggestion"]["gear"] < chosen
    # the suggestion is the current version's own selection
    assert body["suggestion"]["gear"] == select(
        client, power_limit=limit).json()["gear"]


def test_gear_review_same_conditions_rechecked_after_retrain(client):
    create(client)
    top = select(client, power_limit=1e9).json()
    limit = top["max_power"] * (1 + 1e-9)
    client.post("/versions/1/correction",
                json={"factory_id": "F3", "correction": 500})
    # the current version's own gear passes its own conditions
    fresh = select(client, power_limit=limit).json()["gear"]
    ok = review(client, fresh, power_limit=limit)
    assert ok.status_code == 200
    assert ok.json()["version"] == 2
    assert ok.json()["gear"] == fresh


def test_gear_review_out_of_range_gear(client):
    create(client)
    r = review(client, 5)
    assert r.status_code == 409
    assert "speed_range" in r.json()["reasons"]
    assert "suggestion" in r.json()


def test_gear_select_during_recompute_uses_current_version(client,
                                                          monkeypatch):
    import threading

    import app.main as main_mod

    create(client)
    first = select(client).json()
    orig_train = main_mod.train
    started = threading.Event()
    release = threading.Event()

    def slow_train(stats):
        started.set()
        assert release.wait(10)
        return orig_train(stats)

    monkeypatch.setattr(main_mod, "train", slow_train)
    outcome = {}

    def do_update():
        r = client.post("/versions/1/correction",
                        json={"factory_id": "F3", "correction": 500})
        outcome["status"] = r.status_code

    t = threading.Thread(target=do_update)
    t.start()
    assert started.wait(10)
    # mid-recompute, selection stays on committed version 1: correction,
    # summary and model all from that one version
    for _ in range(5):
        body = select(client).json()
        assert body["version"] == 1
        assert body["correction"] == 0
        assert body["gear"] == first["gear"]
    release.set()
    t.join(10)
    assert outcome["status"] == 200
    after = select(client).json()
    assert after["version"] == 2 and after["correction"] == 500
