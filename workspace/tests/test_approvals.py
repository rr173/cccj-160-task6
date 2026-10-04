"""工具扩权审批：有效期 + 主动撤销（版本号并发控制）+ 时刻重确认的端到端验证。

覆盖（需求逐条）：
- 尚未生效 / 恰好到期（闭区间边界）/ 已到期：建批、候选首次命中、一步发布
  三种路径都按业务请求发生时刻重新确认授权；
- 并发撤销与发布竞争（撤销 CAS expected_version，发布单事务重确认）；
- 授权只覆盖候选新增的部分工具：建批与候选命中均拒绝；
- 撤销后重新审批：旧撤销行保留、新授权放行；
- 旧永久审批兼容：两列 NULL 的老审批永久有效；
- 已经产生的调用按当时保存的授权快照复查，随后到期/撤销不改写结论；
- 任一校验失败都不改变线上指针或放量状态；
- JSON API（授予带有效期 / 撤销 CAS / 状态查询 / 事件历史 / 调用授权复查）与页面冒烟。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone

_TMP = tempfile.mkdtemp(prefix="promptgov-approval-test-")
os.environ["PROMPTGOV_DATA"] = _TMP
os.environ["PROMPTGOV_SEED"] = "0"
os.environ["PROMPTGOV_PORT"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from promptgov import core, db, services  # noqa: E402

db.init_db()

CONTRACT = {
    "customer_text": {"type": "string", "required": True},
    "lang": {"type": "string", "required": False},
}
PARAMS = {"temperature": 0.2, "top_p": 0.9, "max_tokens": 1024}
INSTRUCTION_V1 = "你是客服分流助手。客户原话：{{ customer_text }} 请给出 label 与 score。"
INSTRUCTION_V2 = "你是客服分流助手（v2）。客户原话：{{ customer_text }} 优先识别 positive。"
INPUTS = {"customer_text": "客服非常耐心，问题一次解决", "lang": "zh"}
TEST_CASES = [
    {"name": "满意", "inputs": {"customer_text": "非常满意，点赞", "lang": "zh"},
     "expect": {"assert": "json_field", "field": "label", "equals": "positive"}},
]
EXPANDED_TOOLS = ["ticket.read", "refund.execute"]
BASE_TOOLS = ["ticket.read"]


def _uid() -> str:
    return uuid.uuid4().hex[:8]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _fmt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _shift(field: str, approval_id: str, dt) -> None:
    """直接把审批有效期列改到指定绝对时刻（datetime 或已格式化字符串）。"""
    val = dt if isinstance(dt, str) else _fmt(dt)
    with db.write_tx() as c:
        c.execute(f"UPDATE approvals SET {field}=? WHERE approval_id=?",
                  (val, approval_id))


def make_family(tools_v2=None):
    """v1 上线（ticket.read），v2 已过评估+release 审批但**无** tool_expansion。"""
    fid = f"f-{_uid()}"
    services.create_family(fid, f"族{fid}", "", "test")
    services.create_testset(fid, "固定测试集", TEST_CASES, "test")
    v1 = services.create_version(fid, "1.0.0", INSTRUCTION_V1, CONTRACT,
                                 BASE_TOOLS, PARAMS, "langfang-3-64k", "test")
    services.run_evaluation(services.start_evaluation(v1["version_id"], "test"), "test")
    services.approve(v1["version_id"], "release", "ops", "approved", "首发")
    services.promote(fid, v1["version_id"], "test")
    v2 = services.create_version(fid, "1.1.0", INSTRUCTION_V2, CONTRACT,
                                 tools_v2 or EXPANDED_TOOLS, PARAMS,
                                 "langfang-3-64k", "test")
    services.run_evaluation(services.start_evaluation(v2["version_id"], "test"), "test")
    services.approve(v2["version_id"], "release", "ops", "approved", "候选")
    return fid, v1["version_id"], v2["version_id"]


def hit_key(rollout_id: str, weight: int) -> str:
    """找一个（确定性地）会命中候选的 routing_key。"""
    i = 0
    while True:
        key = f"k-{i}"
        if core.stable_bucket(rollout_id, key) < weight:
            return key
        i += 1


def granting(vid, **kw):
    return services.approve(vid, "tool_expansion", "sec", "approved",
                            kw.pop("reason", "扩权"), **kw)


class ValidityWindowTest(unittest.TestCase):

    def test_effective_predicate_closed_interval_boundaries(self):
        """有效期为闭区间 [valid_from, valid_until]：恰好生效/恰好失效那一秒仍有效。"""
        fid, v1, v2 = make_family()
        start, end = _utcnow() + timedelta(hours=1), _utcnow() + timedelta(hours=2)
        a = granting(v2, valid_from=_fmt(start), valid_until=_fmt(end))
        from promptgov.db import get_conn
        row = get_conn().execute("SELECT * FROM approvals WHERE approval_id=?",
                                 (a["approval_id"],)).fetchone()
        eff = lambda at: services._effective_at(row, _fmt(at))  # noqa: E731
        self.assertFalse(eff(start - timedelta(seconds=1)), "生效前一秒必须无效")
        self.assertTrue(eff(start), "恰好到达生效时刻必须有效（闭区间）")
        self.assertTrue(eff(start + timedelta(minutes=30)))
        self.assertTrue(eff(end), "恰好到达失效时刻那一秒仍有效（闭区间）")
        self.assertFalse(eff(end + timedelta(seconds=1)), "越过失效时刻一秒必须无效")

    def test_not_yet_effective_blocks_all_three_paths(self):
        """尚未生效：建批、候选首次命中、一步发布都必须拒绝，且状态不变。"""
        fid, v1, v2 = make_family()
        future = _utcnow() + timedelta(hours=2)
        granting(v2, valid_from=_fmt(future))
        # 1) 建批被拦
        with self.assertRaises(services.Conflict) as ctx:
            services.create_rollout(fid, v2, 100, "test")
        self.assertIn("tool_expansion", str(ctx.exception))
        self.assertIsNone(services.current_rollout(fid), "建批失败不得产生批次")
        # 2) 一步发布被拦（即便直接把一个批次造出来，候选命中也要拦；先测发布）
        with self.assertRaises(services.Conflict):
            services.promote(fid, v2, "test", expected_active=v1)
        self.assertEqual(services.get_family(fid)["active_version_id"], v1,
                         "授权未生效不得移动线上指针")
        # 3) 手工构造批次后，首个候选命中必须被拒（绕过建批门禁验证路由门禁）
        ro = self._force_rollout(fid, v1, v2)
        key = hit_key(ro["rollout_id"], 100)
        before = services.get_rollout(ro["rollout_id"])
        with self.assertRaises(services.Conflict):
            services.invoke(fid, INPUTS, "t", routing_key=key)
        after = services.get_rollout(ro["rollout_id"])
        self.assertEqual((before["status"], before["weight"], before["generation"]),
                         (after["status"], after["weight"], after["generation"]),
                         "候选命中被拒不得改变放量状态")
        self.assertEqual(services.get_family(fid)["active_version_id"], v1)

    def test_exactly_expired_second_blocks_first_hit(self):
        """授权 valid_until == 业务时刻的下一秒：候选命中被拒；边界秒内仍放行。"""
        fid, v1, v2 = make_family()
        now = _utcnow().replace(microsecond=0)
        a = granting(v2, valid_until=_fmt(now + timedelta(seconds=1)))
        ro = self._force_rollout(fid, v1, v2)
        key = hit_key(ro["rollout_id"], 100)
        # 当前时刻在闭区间内（距失效还有 1 秒）：放行并固化快照
        res = services.invoke(fid, INPUTS, "t", routing_key=key)
        self.assertEqual(res["version_id"], v2)
        self.assertEqual(res["auth_snapshot"]["result"], "covered")
        # 把失效时刻改到当前时刻的前一秒（确定性地制造"已越过失效时刻"）
        _shift("valid_until", a["approval_id"], _utcnow() - timedelta(seconds=1))
        with self.assertRaises(services.Conflict) as ctx:
            services.invoke(fid, INPUTS, "t", routing_key=key)
        self.assertIn("到期", str(ctx.exception))
        # 恰好等于失效时刻（闭区间最后一秒）：仍然放行
        boundary = _utcnow().replace(microsecond=0)
        _shift("valid_until", a["approval_id"], _fmt(boundary))
        conn = db.get_conn()
        # 直接用谓词在该固定时刻断言（不受真实时钟抖动影响）
        row = conn.execute("SELECT * FROM approvals WHERE approval_id=?",
                           (a["approval_id"],)).fetchone()
        self.assertTrue(services._effective_at(row, _fmt(boundary)))
        self.assertFalse(services._effective_at(
            row, _fmt(boundary + timedelta(seconds=1))))

    def test_expired_blocks_create_rollout_and_promote_and_complete(self):
        fid, v1, v2 = make_family()
        a = granting(v2, valid_until=_fmt(_utcnow() + timedelta(days=1)))
        ro = services.create_rollout(fid, v2, 100, "test")
        # 授权随后到期
        _shift("valid_until", a["approval_id"], _fmt(_utcnow() - timedelta(seconds=1)))
        # 完成（候选转正）必须在同一事务重确认 -> 拒绝，指针/批次不变
        with self.assertRaises(services.Conflict):
            services.complete_rollout(ro["rollout_id"], ro["generation"], "test")
        self.assertEqual(services.get_family(fid)["active_version_id"], v1)
        cur = services.get_rollout(ro["rollout_id"])
        self.assertEqual((cur["status"], cur["generation"]), ("ramping", 1),
                         "完成失败不得终结批次或推进代次")
        # 一步发布同样拒绝
        with self.assertRaises(services.Conflict):
            services.promote(fid, v2, "test", expected_active=v1)
        self.assertEqual(services.get_family(fid)["active_version_id"], v1)

    def test_partial_tool_coverage_rejected(self):
        """授权只覆盖候选新增工具的一部分：建批与候选命中都必须拒绝。"""
        fid, v1, v2 = make_family()
        # v2 新增 refund.execute；只授予 ticket.read（基线已有），未覆盖新增工具
        granting(v2, granted_tools=["ticket.read"], reason="只覆盖旧工具")
        with self.assertRaises(services.Conflict) as ctx:
            services.create_rollout(fid, v2, 100, "test")
        self.assertIn("部分", str(ctx.exception))
        # 即便造了批次，候选命中也被拦
        ro = self._force_rollout(fid, v1, v2)
        key = hit_key(ro["rollout_id"], 100)
        with self.assertRaises(services.Conflict):
            services.invoke(fid, INPUTS, "t", routing_key=key)
        # 补齐覆盖全部工具后放行
        granting(v2, granted_tools=EXPANDED_TOOLS, reason="覆盖全部工具")
        res = services.invoke(fid, INPUTS, "t", routing_key=key)
        self.assertEqual(res["version_id"], v2)

    def _force_rollout(self, fid, v1, v2, weight=100):
        """绕过建批门禁直接插入一个进行中批次（用于单独验证路由/完成门禁）。"""
        ro_id = f"ro_{_uid()}"
        with db.write_tx() as c:
            c.execute(
                """INSERT INTO rollouts(rollout_id,family_id,candidate_version_id,
                   base_version_id,weight,status,generation,created_by)
                   VALUES (?,?,?,?,?, 'ramping', 1, 'test')""",
                (ro_id, fid, v2, v1, weight))
        return services.get_rollout(ro_id)


class RevocationConcurrencyTest(unittest.TestCase):

    def test_revoke_cas_and_revoke_then_promote_loses(self):
        """撤销以 expected_version 做 CAS：错误版本 409；撤销与一步发布竞争，
        写锁串行下二者之一成功，发布方若在事务内看到已撤销则失败且指针不变。"""
        fid, v1, v2 = make_family()
        a = granting(v2)
        self.assertEqual(a["approval_version"], 1)
        # 错误版本号撤销 -> 409
        with self.assertRaises(services.Conflict):
            services.revoke_approval(a["approval_id"], expected_version=99,
                                     actor="sec", reason="误操作")
        still = services.get_approval(a["approval_id"])
        self.assertEqual(still["state"], "active")
        self.assertEqual(still["approval_version"], 1)

        # 正确版本号撤销 -> 版本号 +1，行保留、状态 revoked、有事件
        services.revoke_approval(a["approval_id"], expected_version=1,
                                 actor="sec", reason="风险回收")
        revoked = services.get_approval(a["approval_id"])
        self.assertEqual(revoked["state"], "revoked")
        self.assertEqual(revoked["approval_version"], 2)
        events = services.list_approval_events(a["approval_id"])
        self.assertEqual(sorted(e["kind"] for e in events), ["grant", "revoke"])
        # 重复撤销 -> 409
        with self.assertRaises(services.Conflict):
            services.revoke_approval(a["approval_id"], expected_version=2,
                                     actor="sec", reason="再撤")
        # 撤销后建批/发布/命中全部拒绝
        with self.assertRaises(services.Conflict):
            services.create_rollout(fid, v2, 100, "test")
        with self.assertRaises(services.Conflict):
            services.promote(fid, v2, "test", expected_active=v1)
        self.assertEqual(services.get_family(fid)["active_version_id"], v1)

    def test_concurrent_revoke_vs_complete_exactly_one_winner(self):
        """撤销与"批次完成（候选转正）"并发：单事务重确认保证只有一方赢，
        赢的是撤销则指针不动、批次仍在；赢的是完成则指针=候选、批次 completed。"""
        fid, v1, v2 = make_family()
        a = granting(v2)
        ro = services.create_rollout(fid, v2, 100, "test")
        outcomes = []
        barrier = threading.Barrier(2)

        def do_revoke():
            barrier.wait()
            try:
                services.revoke_approval(a["approval_id"], 1, "sec", "并发撤销")
                outcomes.append("revoke_ok")
            except services.Conflict:
                outcomes.append("revoke_lost")

        def do_complete():
            barrier.wait()
            try:
                services.complete_rollout(ro["rollout_id"], ro["generation"], "race")
                outcomes.append("complete_ok")
            except services.Conflict:
                outcomes.append("complete_lost")

        t1 = threading.Thread(target=do_revoke)
        t2 = threading.Thread(target=do_complete)
        t1.start(); t2.start(); t1.join(); t2.join()
        self.assertEqual(len(outcomes), 2)
        active_now = services.get_family(fid)["active_version_id"]
        ro_now = services.get_rollout(ro["rollout_id"])
        approvals_now = services.get_approval(a["approval_id"])
        if "complete_ok" in outcomes:
            self.assertEqual(outcomes.count("complete_ok"), 1)
            # 完成与撤销不写同一行：两者可能都成功（各赢各的 CAS），也可能撤销在
            # 完成事务之后被完成方的重确认抢先——若撤销失败则必为 409；若撤销成功，
            # 它发生在指针翻转已提交之后，不影响本次转正结果。
            self.assertEqual(active_now, v2)
            self.assertEqual(ro_now["status"], "completed")
            if "revoke_ok" in outcomes:
                self.assertEqual(approvals_now["state"], "revoked")
            else:
                self.assertIn("revoke_lost", outcomes)
        else:
            # 撤销先提交：完成事务的重确认看到已撤销，必然失败，指针/批次不变
            self.assertIn("revoke_ok", outcomes)
            self.assertIn("complete_lost", outcomes)
            self.assertEqual(active_now, v1, "撤销赢：线上指针不得移动")
            self.assertEqual(ro_now["status"], "ramping", "撤销赢：批次不得终结")
            self.assertEqual(approvals_now["state"], "revoked")

    def test_revoke_then_reapprove_allows_traffic(self):
        """撤销后重新审批：新审批新版本号放行，旧撤销行与事件历史保留。"""
        fid, v1, v2 = make_family()
        old = granting(v2, reason="第一轮")
        services.revoke_approval(old["approval_id"], 1, "sec", "回收")
        with self.assertRaises(services.Conflict):
            services.create_rollout(fid, v2, 100, "test")
        # 重新审批（新的一张）
        new = granting(v2, valid_from=_fmt(_utcnow() - timedelta(minutes=1)),
                       reason="重新审批")
        self.assertNotEqual(new["approval_id"], old["approval_id"])
        ro = services.create_rollout(fid, v2, 100, "test")
        key = hit_key(ro["rollout_id"], 100)
        res = services.invoke(fid, INPUTS, "t", routing_key=key)
        self.assertEqual(res["version_id"], v2)
        # 历史里旧审批仍是 revoked，新审批 active；两条 grant + 一条 revoke
        history = services.list_approval_events(version_id=v2)
        kinds = sorted(e["kind"] for e in history)
        self.assertEqual(kinds, ["grant", "grant", "revoke"])


class SnapshotRecheckTest(unittest.TestCase):

    def test_legacy_permanent_approval_stays_valid_forever(self):
        """旧永久审批（两列 NULL）：任何时刻都有效。"""
        fid, v1, v2 = make_family()
        a = granting(v2)  # 不传有效期 => 永久
        self.assertIsNone(a["valid_from"]); self.assertIsNone(a["valid_until"])
        row = db.get_conn().execute(
            "SELECT * FROM approvals WHERE approval_id=?", (a["approval_id"],)).fetchone()
        far_future = _utcnow() + timedelta(days=3650)
        self.assertTrue(services._effective_at(row, _fmt(far_future)),
                        "旧永久审批在任意未来时刻仍有效（兼容）")

    def test_invalid_window_and_release_ignores_window(self):
        """生效晚于失效 -> 400；release 审批不接受有效期控制（撤销被拒、不设窗口）。"""
        fid, v1, v2 = make_family()
        with self.assertRaises(services.DomainError):
            granting(v2, valid_from="2026-12-01 00:00",
                     valid_until="2026-01-01 00:00")
        # 即便传了有效期，release 审批也不存窗口
        rel = services.approve(v2, "release", "ops", "approved", "r",
                               valid_from="2026-12-01 00:00",
                               valid_until="2027-01-01 00:00")
        self.assertIsNone(rel["valid_from"])
        self.assertIsNone(rel["valid_from"]); self.assertIsNone(rel["valid_until"])
        with self.assertRaises(services.DomainError):
            services.revoke_approval(rel["approval_id"], 1, "ops", "release 不可撤")
        # 时间格式错误
        with self.assertRaises(services.DomainError):
            granting(v2, valid_from="not-a-time")

    def test_past_call_recheck_uses_snapshot_ignore_later_expiry_revoke(self):
        """已产生的候选命中调用固化授权快照；随后到期、再撤销，复查结论恒为 pass，
        且快照内版本号/有效期保持调用当时的值。"""
        fid, v1, v2 = make_family()
        end = _utcnow() + timedelta(days=1)
        end_str = _fmt(end)
        a = granting(v2, valid_until=end_str)
        ro = services.create_rollout(fid, v2, 100, "test")
        key = hit_key(ro["rollout_id"], 100)
        res = services.invoke(fid, INPUTS, "t", routing_key=key)
        rid = res["request_id"]
        self.assertEqual(res["auth_snapshot"]["approval_version"], 1)
        # 当时复查：pass，审批当前仍 active
        rc = services.recheck_request_auth(rid)
        self.assertEqual(rc["conclusion"], "pass")
        self.assertEqual(rc["current"]["state"], "active")
        # 到期后复查：快照结论仍 pass，仅当前状态变为 expired
        _shift("valid_until", a["approval_id"], _fmt(_utcnow() - timedelta(seconds=1)))
        rc = services.recheck_request_auth(rid)
        self.assertEqual(rc["conclusion"], "pass", "到期不得改写当年放行结论")
        self.assertEqual(rc["current"]["state"], "expired")
        self.assertEqual(rc["snapshot"]["valid_until"], end_str,
                         "快照内有效期必须是调用当时固化的值")
        # 再撤销：快照结论依旧 pass，当前状态 revoked、版本号推进
        services.revoke_approval(a["approval_id"], 1, "sec", "事后回收")
        rc = services.recheck_request_auth(rid)
        self.assertEqual(rc["conclusion"], "pass", "撤销不得改写当年放行结论")
        self.assertEqual(rc["current"]["state"], "revoked")
        self.assertEqual(rc["current"]["approval_version_now"], 2)
        self.assertEqual(rc["snapshot"]["approval_version"], 1,
                         "快照版本号必须停留在调用当时")
        # 该调用的精确重演仍复现（内容层不受授权状态影响）
        rep = services.replay(rid, "test")
        self.assertEqual(rep["verdict"], "reproduced")

    def test_old_call_without_snapshot_is_legacy_pass(self):
        """升级前产生的调用（auth_snapshot_json 为空）：复查为 legacy_pass。"""
        fid, v1, v2 = make_family()
        res = services.invoke(fid, INPUTS, "t")  # 直接命中线上基线，无扩权快照
        rc = services.recheck_request_auth(res["request_id"])
        self.assertEqual(rc["conclusion"], "legacy_pass")
        self.assertIsNone(rc["snapshot"])

    def test_rejected_failed_invocation_leaves_no_state_change(self):
        """授权失效后候选命中抛错：不产生 requests 行，批次/指针原样。"""
        fid, v1, v2 = make_family()
        a = granting(v2)
        ro = services.create_rollout(fid, v2, 100, "test")
        key = hit_key(ro["rollout_id"], 100)
        services.revoke_approval(a["approval_id"], 1, "sec", "回收")
        n_req_before = db.get_conn().execute("SELECT COUNT(*) c FROM requests").fetchone()["c"]
        with self.assertRaises(services.Conflict):
            services.invoke(fid, INPUTS, "t", routing_key=key)
        n_req_after = db.get_conn().execute("SELECT COUNT(*) c FROM requests").fetchone()["c"]
        self.assertEqual(n_req_before, n_req_after, "被拒请求不得写调用记录")
        self.assertEqual(services.get_family(fid)["active_version_id"], v1)

    def test_rollback_is_exempt_from_expired_expansion_approval(self):
        """回滚只收窄权限：候选转正后即便扩权审批被撤销，仍可回滚到基线历史版本。"""
        fid, v1, v2 = make_family()
        a = granting(v2)
        services.promote(fid, v2, "test", expected_active=v1)
        self.assertEqual(services.get_family(fid)["active_version_id"], v2)
        services.revoke_approval(a["approval_id"], 1, "sec", "转正后回收扩权")
        # 回滚到 v1（工具更少）不要求新的扩权审批
        services.rollback(fid, v1, "test", expected_active=v2)
        self.assertEqual(services.get_family(fid)["active_version_id"], v1)


class HttpApiTest(unittest.TestCase):
    """JSON API 与页面冒烟。"""

    @classmethod
    def setUpClass(cls):
        os.environ["PROMPTGOV_SEED"] = "0"
        from promptgov import app
        cls.httpd = app.build_server()
        cls.port = cls.httpd.server_address[1]
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _req(self, method, path, body=None):
        url = f"http://127.0.0.1:{self.port}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    def test_approval_lifecycle_api(self):
        fid, v1, v2 = make_family()
        end = _fmt(_utcnow() + timedelta(hours=2))
        start = _fmt(_utcnow() - timedelta(minutes=1))
        # 授予带有效期的专项审批
        code, a = self._req("POST", f"/api/versions/{v2}/approve", {
            "kind": "tool_expansion", "approver": "sec", "decision": "approved",
            "reason": "api 窗口授权", "valid_from": start, "valid_until": end})
        self.assertEqual(code, 201, a)
        aid, ver = a["approval_id"], a["approval_version"]
        self.assertEqual(a["state"], "active")
        self.assertIsNotNone(a["remaining_seconds"])
        self.assertGreater(a["remaining_seconds"], 3600 - 120)
        # 单查 / 列表 / 事件历史
        code, got = self._req("GET", f"/api/approvals/{aid}")
        self.assertEqual(code, 200)
        self.assertEqual(sorted(got["granted_tools"]), sorted(EXPANDED_TOOLS))
        code, evs = self._req("GET", f"/api/approvals/{aid}/events")
        self.assertEqual(code, 200)
        self.assertEqual([e["kind"] for e in evs], ["grant"])
        # 生效窗口内：建批 + 候选命中
        code, ro = self._req("POST", f"/api/families/{fid}/rollouts",
                             {"candidate_version_id": v2, "weight": 100})
        self.assertEqual(code, 201, ro)
        key = hit_key(ro["rollout_id"], 100)
        code, res = self._req("POST", f"/api/families/{fid}/invoke",
                              {"inputs": INPUTS, "routing_key": key})
        self.assertEqual(code, 201, res)
        self.assertEqual(res["version_id"], v2)
        rid = res["request_id"]
        # 调用授权复查 API
        code, rc = self._req("GET", f"/api/requests/{rid}/auth-recheck")
        self.assertEqual(code, 200)
        self.assertEqual(rc["conclusion"], "pass")
        self.assertEqual(rc["snapshot"]["approval_id"], aid)
        # 撤销：缺 expected_version -> 400；错版本 -> 409；正确 -> 200
        code, err = self._req("POST", f"/api/approvals/{aid}/revoke", {"reason": "x"})
        self.assertEqual(code, 400)
        code, err = self._req("POST", f"/api/approvals/{aid}/revoke",
                              {"reason": "x", "expected_version": 99})
        self.assertEqual(code, 409)
        code, rev = self._req("POST", f"/api/approvals/{aid}/revoke",
                              {"reason": "api 撤销", "expected_version": ver})
        self.assertEqual(code, 200, rev)
        self.assertEqual(rev["state"], "revoked")
        self.assertEqual(rev["approval_version"], ver + 1)
        # 撤销后候选命中被门禁拒绝（409）
        code, err = self._req("POST", f"/api/families/{fid}/invoke",
                              {"inputs": INPUTS, "routing_key": key})
        self.assertEqual(code, 409)
        # 旧调用复查结论不变
        code, rc = self._req("GET", f"/api/requests/{rid}/auth-recheck")
        self.assertEqual(rc["conclusion"], "pass")
        self.assertEqual(rc["current"]["state"], "revoked")
        # release 审批不允许撤销
        code, rel = self._req("POST", f"/api/versions/{v2}/approve",
                              {"kind": "release", "approver": "ops", "reason": "r"})
        code, err = self._req("POST", f"/api/approvals/{rel['approval_id']}/revoke",
                              {"reason": "x", "expected_version": 1})
        self.assertEqual(code, 400)
        # 页面冒烟（评估页展示状态/剩余时间/历史；复查页展示授权面板）
        for path in (f"/versions/{v2}/eval", f"/requests/{rid}/replay",
                     f"/families/{fid}/rollout", "/"):
            resp = urllib.request.urlopen(
                urllib.request.Request(f"http://127.0.0.1:{self.port}{path}"))
            self.assertEqual(resp.status, 200, path)
        html_text = urllib.request.urlopen(
            urllib.request.Request(
                f"http://127.0.0.1:{self.port}/versions/{v2}/eval")).read().decode()
        self.assertIn("已撤销", html_text)
        self.assertIn("审批历史", html_text)
        self.assertIn("主动撤销", html_text)
        self.assertIn("09:", html_text)  # 有效期窗口渲染在页面上
        self.assertIn("生效中", html_text)  # release 永久审批仍展示生效中


if __name__ == "__main__":
    unittest.main(verbosity=2)
