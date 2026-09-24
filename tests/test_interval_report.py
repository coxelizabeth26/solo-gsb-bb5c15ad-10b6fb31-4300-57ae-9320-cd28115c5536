"""数值区间预测报告测试：GET /events/{event_id}/interval-report。

SQLite 文件库 + 模拟时钟，覆盖：
  - 未结算 -> 409 INTERVAL_REPORT_NOT_SETTLED；未知事件 -> 404
  - 已结算但标准结果不是有限十进制数 -> 422 NON_FINITE_STANDARD_RESULT，
    既有结算结果与计数不变
  - 「下界..上界」解析：合法区间、指数/负数/小数、边界闭区间命中
  - INVALID_FORMAT（缺分隔符、空边界、非数字、空白、下划线、Infinity/NaN）
  - INVALID_ORDER（下界严格大于上界）
  - 命中排名：宽度升序、同宽按承诺提交时间、再按承诺 ID；名次连续
  - 未命中者列出但 rank=null，排在命中者之后
  - 未揭示承诺单独计数；非法原文单独计数且不参与命中
  - 十进制精确计算（0.1+0.2 类边界不因浮点误判）
  - 报告只读：既有逐字符结算计数不变
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from uuid import UUID

_tmpdir = tempfile.mkdtemp(prefix="sealed_predictions_interval_report_test_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/interval_report.db"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import app.clock as clock_module  # noqa: E402
import app.main as main_module  # noqa: E402
from app.hashing import compute_commitment  # noqa: E402

T0 = datetime(2030, 7, 1, 0, 0, 0, tzinfo=timezone.utc)
COMMIT_DL = T0 + timedelta(hours=1)
REVEAL_DL = T0 + timedelta(hours=2)


class MutableClock:
    def __init__(self, start: datetime):
        self.t = start

    def __call__(self) -> datetime:
        return self.t


clock = MutableClock(T0)

client = TestClient(main_module.app)


@pytest.fixture(autouse=True)
def _fixed_clock(monkeypatch):
    clock.t = T0
    monkeypatch.setattr(clock_module, "now_utc", clock)
    with client:  # 触发 lifespan 建表（幂等）
        yield


def create_event() -> str:
    r = client.post("/events", json={
        "title": "区间预测事件",
        "commit_deadline": COMMIT_DL.isoformat(),
        "reveal_deadline": REVEAL_DL.isoformat(),
    })
    assert r.status_code == 201, r.text
    return r.json()["id"]


def commit(event_id: str, text: str, salt: str) -> str:
    h = compute_commitment(text, salt)
    r = client.post(f"/events/{event_id}/commitments", json={"commitment_hash": h})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def reveal(event_id: str, text: str, salt: str) -> None:
    r = client.post(f"/events/{event_id}/reveals",
                    json={"original_text": text, "salt": salt})
    assert r.status_code == 201, r.text


def settle(event_id: str, standard_result: str) -> dict:
    r = client.post(f"/events/{event_id}/settlement",
                    json={"standard_result": standard_result})
    assert r.status_code == 200, r.text
    return r.json()


def report(event_id: str, expected: int = 200) -> dict:
    r = client.get(f"/events/{event_id}/interval-report")
    assert r.status_code == expected, r.text
    return r.json()


def by_id(entries: list[dict]) -> dict[str, dict]:
    return {e["commitment_id"]: e for e in entries}


# --------------------------------------------------------------------------- #
# 门禁
# --------------------------------------------------------------------------- #

def test_report_rejected_before_settlement():
    event_id = create_event()
    body = report(event_id, expected=409)
    assert body["detail"]["code"] == "INTERVAL_REPORT_NOT_SETTLED"
    # 拒绝后仍可正常结算，不留任何状态
    clock.t = REVEAL_DL
    settle(event_id, "5")


def test_report_404_for_missing_event():
    missing = "00000000-0000-0000-0000-000000000000"
    r = client.get(f"/events/{missing}/interval-report")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "EVENT_NOT_FOUND"


def test_report_422_when_standard_result_not_finite_decimal():
    # 空串在结算请求体层即 422（standard_result 要求非空），不进入本场景
    for bad in ("abc", "Infinity", "-Infinity", "NaN", " 5", "5 ", "1..2"):
        clock.t = T0
        event_id = create_event()
        clock.t = REVEAL_DL
        settled = settle(event_id, bad)
        body = report(event_id, expected=422)
        assert body["detail"]["code"] == "NON_FINITE_STANDARD_RESULT"
        # 既有结算结果与计数不变
        r = client.get(f"/events/{event_id}/settlement")
        assert r.json() == settled


# --------------------------------------------------------------------------- #
# 解析与命中
# --------------------------------------------------------------------------- #

def test_boundaries_are_inclusive_and_decimal_exact():
    event_id = create_event()
    # 0.3 恰为区间端点：浮点下 0.1 + 0.2 != 0.3，十进制精确比较必须命中。
    # 各承诺使用不同提交时刻，避免同刻回退到承诺 ID 次序
    clock.t = T0 + timedelta(seconds=10)
    c_edge = commit(event_id, "0.1..0.3", "s-edge")
    clock.t = T0 + timedelta(seconds=20)
    c_point = commit(event_id, "0.3..0.3", "s-point")
    clock.t = T0 + timedelta(seconds=30)
    c_out = commit(event_id, "0.31..0.9", "s-out")
    clock.t = T0 + timedelta(seconds=40)
    c_exp = commit(event_id, "1e-1..3E0", "s-exp")       # 0.1 .. 3
    clock.t = T0 + timedelta(seconds=50)
    c_neg = commit(event_id, "-1.50..-0.50", "s-neg")    # 不含 -0.3
    clock.t = COMMIT_DL
    reveal(event_id, "0.1..0.3", "s-edge")
    reveal(event_id, "0.3..0.3", "s-point")
    reveal(event_id, "0.31..0.9", "s-out")
    reveal(event_id, "1e-1..3E0", "s-exp")
    reveal(event_id, "-1.50..-0.50", "s-neg")
    clock.t = REVEAL_DL
    settle(event_id, "0.3")

    body = report(event_id)
    assert body["status"] == "settled"
    assert body["standard_result"] == "0.3"
    assert body["hit_count"] == 3
    assert body["miss_count"] == 2
    assert body["invalid_count"] == 0
    assert body["unrevealed_count"] == 0

    entries = by_id(body["entries"])
    edge = entries[c_edge]
    assert edge["hit"] is True and edge["rank"] == 2
    assert edge["lower"] == "0.1" and edge["upper"] == "0.3"
    assert Decimal(edge["width"]) == Decimal("0.2")
    # 零点宽度的点区间与端点同值 -> 命中，宽度最小故排名第 1
    point = entries[c_point]
    assert point["hit"] is True and point["rank"] == 1
    assert Decimal(point["width"]) == 0
    # 指数写法正常解析并命中（0.1 <= 0.3 <= 3），宽 2.9 排名第 3
    assert entries[c_exp]["hit"] is True and entries[c_exp]["rank"] == 3
    assert Decimal(entries[c_exp]["lower"]) == Decimal("0.1")
    assert Decimal(entries[c_exp]["upper"]) == Decimal("3")
    # 严格落在区间外 -> 未命中、不排名
    out = entries[c_out]
    assert out["hit"] is False and out["rank"] is None
    neg = entries[c_neg]
    assert neg["hit"] is False and neg["rank"] is None
    # 未命中者排在命中者之后
    assert [e["commitment_id"] for e in body["entries"]][-2:] == [c_out, c_neg]


@pytest.mark.parametrize(
    "text",
    [
        "1",            # 缺分隔符
        "1..",          # 空上界
        "..10",         # 空下界
        "1..2..3",      # 多个分隔符
        "1..abc",       # 非数字
        "abc..10",
        "1 .. 10",      # 含空白
        " 1..10",
        "1..10 ",
        "1_0..20",      # 下划线
        "Inf..10",      # 非有限值
        "Infinity..10",
        "NaN..10",
        "1..NaN",
        "１２..30",      # Unicode 全角数字不属于 ASCII 十进制字面量
        "",             # 空串
        "..",
    ],
)
def test_invalid_format_listed_separately(text):
    event_id = create_event()
    cid = commit(event_id, text, f"salt-{text!r}")
    clock.t = COMMIT_DL
    reveal(event_id, text, f"salt-{text!r}")
    clock.t = REVEAL_DL
    settle(event_id, "5")

    body = report(event_id)
    assert body["entries"] == []
    assert body["hit_count"] == 0 and body["miss_count"] == 0
    assert body["invalid_count"] == 1
    invalid = body["invalids"][0]
    assert invalid["commitment_id"] == cid
    assert invalid["original_text"] == text
    assert invalid["reason"] == "INVALID_FORMAT"
    assert invalid["committed_at"] is not None


def test_lower_greater_than_upper_is_invalid_order():
    event_id = create_event()
    cid = commit(event_id, "10..1", "s")
    clock.t = COMMIT_DL
    reveal(event_id, "10..1", "s")
    clock.t = REVEAL_DL
    settle(event_id, "5")

    body = report(event_id)
    assert body["entries"] == []
    assert body["invalid_count"] == 1
    assert body["invalids"][0] == {
        **body["invalids"][0],
        "commitment_id": cid,
        "original_text": "10..1",
        "reason": "INVALID_ORDER",
    }


# --------------------------------------------------------------------------- #
# 排名
# --------------------------------------------------------------------------- #

def test_hit_ranking_by_width_then_commit_time_then_id():
    event_id = create_event()
    # 提交时间刻意与宽度优劣不一致：宽区间最先提交，窄区间后提交
    clock.t = T0 + timedelta(minutes=1)
    c_wide = commit(event_id, "0..100", "s-wide")       # 宽 100
    clock.t = T0 + timedelta(minutes=2)
    c_mid = commit(event_id, "9..11", "s-mid")          # 宽 2
    # 同一提交时刻的两个等宽（宽 1）区间：次序按承诺 ID
    clock.t = T0 + timedelta(minutes=3)
    c_tie_a = commit(event_id, "9.5..10.5", "s-tie-a")
    c_tie_b = commit(event_id, "9.500..10.500", "s-tie-b")
    clock.t = T0 + timedelta(minutes=4)
    c_miss_early = commit(event_id, "11..20", "s-miss-1")  # 未命中
    clock.t = T0 + timedelta(minutes=5)
    c_miss_late = commit(event_id, "-5..9", "s-miss-2")
    c_bad = commit(event_id, "oops", "s-bad")              # 非法
    c_unrevealed = commit(event_id, "0..10", "s-unrev")    # 未揭示

    clock.t = COMMIT_DL
    reveal(event_id, "0..100", "s-wide")
    reveal(event_id, "9..11", "s-mid")
    reveal(event_id, "9.5..10.5", "s-tie-a")
    reveal(event_id, "9.500..10.500", "s-tie-b")
    reveal(event_id, "11..20", "s-miss-1")
    reveal(event_id, "-5..9", "s-miss-2")
    reveal(event_id, "oops", "s-bad")
    clock.t = REVEAL_DL
    settle(event_id, "10")

    body = report(event_id)
    assert body["hit_count"] == 4
    assert body["miss_count"] == 2
    assert body["invalid_count"] == 1
    assert body["unrevealed_count"] == 1
    assert body["invalids"][0]["commitment_id"] == c_bad
    assert body["invalids"][0]["reason"] == "INVALID_FORMAT"

    entries = body["entries"]
    # 前四名均为命中，名次连续
    assert [e["rank"] for e in entries[:4]] == [1, 2, 3, 4]
    assert all(e["hit"] for e in entries[:4])
    # 未命中者在后且不排名
    assert [e["rank"] for e in entries[4:]] == [None, None]
    assert all(not e["hit"] for e in entries[4:])

    ranked = by_id(entries[:4])
    # 宽 1 的两个等宽区间并列最前：同宽按承诺提交时间（相同）再按承诺 ID
    tie_order = sorted([c_tie_a, c_tie_b], key=lambda c: UUID(c))
    assert entries[0]["commitment_id"] == tie_order[0]
    assert Decimal(ranked[tie_order[0]]["width"]) == 1
    assert entries[1]["commitment_id"] == tie_order[1]
    # 其后为宽 2、宽 100
    assert entries[2]["commitment_id"] == c_mid
    assert entries[3]["commitment_id"] == c_wide
    assert Decimal(ranked[c_wide]["width"]) == 100

    # 未命中者按承诺提交时间、承诺 ID 排序
    assert [e["commitment_id"] for e in entries[4:]] == [c_miss_early, c_miss_late]

    # 四类计数之和等于承诺总数（4 命中 + 2 未命中 + 1 非法 + 1 未揭示）
    assert (
        body["hit_count"] + body["miss_count"]
        + body["invalid_count"] + body["unrevealed_count"]
    ) == 8


def test_zero_hits_has_no_ranks():
    event_id = create_event()
    c1 = commit(event_id, "11..20", "s1")
    c2 = commit(event_id, "-5..9", "s2")
    clock.t = COMMIT_DL
    reveal(event_id, "11..20", "s1")
    reveal(event_id, "-5..9", "s2")
    clock.t = REVEAL_DL
    settle(event_id, "10")

    body = report(event_id)
    assert body["hit_count"] == 0
    assert [e["rank"] for e in body["entries"]] == [None, None]
    # 两个承诺在同一模拟时刻提交：回退按承诺 ID 排序
    assert [e["commitment_id"] for e in body["entries"]] == sorted([c1, c2], key=UUID)


def test_report_is_read_only_and_keeps_settlement_tally():
    event_id = create_event()
    # 结算计数仍按揭示原文与标准结果逐字符精确匹配：
    # 只有原文恰为 "10" 的承诺算正确，"10..20" 不算
    c_exact = commit(event_id, "10", "s-exact")
    commit(event_id, "10..20", "s-range")
    commit(event_id, "never", "s-unrev")
    clock.t = COMMIT_DL
    reveal(event_id, "10", "s-exact")
    reveal(event_id, "10..20", "s-range")
    clock.t = REVEAL_DL
    settled = settle(event_id, "10")
    assert settled["correct_count"] == 1
    assert settled["incorrect_count"] == 1
    assert settled["unrevealed_count"] == 1

    body = report(event_id)
    entries = by_id(body["entries"])
    # 原文 "10" 无法解析为区间 -> 非法格式，不算命中也不影响结算计数
    assert body["invalid_count"] == 1
    assert body["invalids"][0]["commitment_id"] == c_exact
    assert body["invalids"][0]["reason"] == "INVALID_FORMAT"
    assert body["hit_count"] == 1
    assert body["unrevealed_count"] == 1

    # 结算固化计数保持不变
    r = client.get(f"/events/{event_id}/settlement")
    after = r.json()
    assert after["correct_count"] == 1
    assert after["incorrect_count"] == 1
    assert after["unrevealed_count"] == 1
    assert after["settled_at"] == settled["settled_at"]

    # 报告可重复获取，内容一致（generated_at 为报告生成时刻）
    body_again = report(event_id)
    for key in (
        "event_id", "status", "standard_result", "settled_at",
        "hit_count", "miss_count", "invalid_count", "unrevealed_count",
        "entries", "invalids",
    ):
        assert body_again[key] == body[key]


def test_zero_commitments_event_report():
    event_id = create_event()
    clock.t = REVEAL_DL
    settle(event_id, "0")
    body = report(event_id)
    assert body["hit_count"] == 0
    assert body["miss_count"] == 0
    assert body["invalid_count"] == 0
    assert body["unrevealed_count"] == 0
    assert body["entries"] == []
    assert body["invalids"] == []
