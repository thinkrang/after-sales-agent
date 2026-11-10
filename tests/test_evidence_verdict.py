"""证据核查：用 schema 约束结论，任何异常都保守地按"证据不足"处理。"""
import pytest

import evidence

RESULTS = [{"source": "return_policy.txt", "content": "签收后 7 天内可无理由退货。"}]


class FakeStructured:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def invoke(self, messages):
        if self.error:
            raise self.error
        return self.result


class FakeLLM:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def with_structured_output(self, schema):
        return FakeStructured(self.result, self.error)


def test_no_documents_short_circuits_without_calling_model(monkeypatch):
    monkeypatch.setattr(evidence, "get_llm", lambda: FakeLLM(error=AssertionError("不应调用模型")))
    verdict = evidence.verify_evidence("多久能退？", [])
    assert verdict.sufficient is False
    assert "未检索" in verdict.missing


def test_sufficient_verdict_passes_through(monkeypatch):
    expected = evidence.EvidenceVerdict(sufficient=True, quote="签收后 7 天内可无理由退货。")
    monkeypatch.setattr(evidence, "get_llm", lambda: FakeLLM(result=expected))
    verdict = evidence.verify_evidence("多久能退？", RESULTS)
    assert verdict.sufficient is True
    assert verdict.quote == expected.quote


def test_claim_of_evidence_without_quote_is_downgraded(monkeypatch):
    # 声称有证据却不给原文，等于凭印象回答：按不足处理。
    monkeypatch.setattr(evidence, "get_llm", lambda: FakeLLM(
        result=evidence.EvidenceVerdict(sufficient=True, quote="  ")))
    verdict = evidence.verify_evidence("多久能退？", RESULTS)
    assert verdict.sufficient is False
    assert "原文" in verdict.missing


def test_verifier_failure_is_conservative_not_permissive(monkeypatch):
    monkeypatch.setattr(evidence, "get_llm", lambda: FakeLLM(error=TimeoutError("下游超时")))
    verdict = evidence.verify_evidence("多久能退？", RESULTS)
    assert verdict.sufficient is False
    assert "核查器不可用" in verdict.missing


def test_legacy_string_keeps_old_protocol_for_eval_scripts():
    assert evidence.legacy_string(
        evidence.EvidenceVerdict(sufficient=True, quote="原句")
    ) == "YES|原句"
    assert evidence.legacy_string(
        evidence.EvidenceVerdict(sufficient=False, missing="缺少金额")
    ) == "NO|缺少金额"
