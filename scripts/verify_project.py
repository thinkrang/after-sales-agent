"""隔离数据目录的一键核验。默认不调用付费模型；--with-model 显式开启。

python scripts/verify_project.py --output work/verification
python scripts/verify_project.py --with-model --output work/verification-model

保留各步骤日志与 JSON/XML，不覆盖业务数据库、索引或旧报告。
Docker 只检测 daemon，不启动/重建容器；本地验收不等同于 Docker 验收。
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def source_manifest():
    paths = (list(ROOT.glob("*.py")) + list((ROOT / "scripts").glob("*.py"))
             + list((ROOT / "tests").glob("*.py")) + list((ROOT / "skills").rglob("*.md"))
             + list((ROOT / "skills").rglob("*.py"))
             + [ROOT / "requirements.txt", ROOT / "docs/evaluation/voucher-policy-fixtures.json"])
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-model", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "verification")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=output))
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               AGENT_DATA_DIR=str(run_dir / "data"), AGENT_VECTOR_DB_DIR=str(run_dir / "vector_db"),
               AGENT_UPLOAD_DIR=str(run_dir / "uploads"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               AUTH_SEED_DEMO_USERS="1")
    from task_eval import provenance
    report = dict(date=datetime.now(timezone.utc).isoformat(), provenance=provenance(),
                  python=sys.version, executable=sys.executable, platform=platform.platform(),
                  directory=str(run_dir), with_model=args.with_model, stages=[])
    manifest = source_manifest()
    (run_dir / "source-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def save():
        (run_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(name, command, timeout=1200):
        start = time.perf_counter()
        stage_provenance = provenance()
        before = source_manifest()
        print(f"RUN {name}", flush=True)
        try:
            with (run_dir / f"{name}.log").open("w", encoding="utf-8") as log:
                proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=timeout)
            code = proc.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            code = -1
            with (run_dir / f"{name}.log").open("a", encoding="utf-8") as log:
                log.write(f"\n{type(exc).__name__}: {exc}")
        report["stages"].append(dict(name=name, command=command, exit_code=code,
                                      seconds=round(time.perf_counter() - start, 2),
                                      provenance=stage_provenance,
                                      source_unchanged=before == source_manifest()))
        save()
        print(f"{'PASS' if code == 0 else 'FAIL'} {name} ({report['stages'][-1]['seconds']}s)", flush=True)
        return code == 0

    py = [sys.executable, "-X", "utf8"]
    run("dependencies", py + ["-m", "pip", "check"])
    run("packages", py + ["-m", "pip", "freeze"])
    if not run("fresh-index", py + ["-c", "import rag; rag.build_index()"]):
        print(f"索引重建失败，详见 {run_dir}", flush=True)
        return 1
    run("pytest", py + ["-m", "pytest", "-o", "addopts=", "-m", "not llm", "-q",
                           f"--junitxml={run_dir / 'pytest.xml'}"])
    run("actions", py + ["action_eval.py", "--output", str(run_dir / "actions.json")])
    run("tasks", py + ["task_eval.py", "--repeat", "3", "--output", str(run_dir / "tasks.json")])
    run("plans", py + ["plan_eval.py", "--repeat", "3", "--output", str(run_dir / "plans.json")])
    run("replanning", py + ["scripts/compare_replanning.py", "--repeat", "10",
                             "--output", str(run_dir / "replanning.json")])
    run("voucher-replay", py + ["scripts/replay_voucher_policy.py", "--output-dir", str(run_dir / "voucher-replay")])
    run("mixed-cases", py + ["mixed_eval.py", "--output", str(run_dir / "mixed-cases.json")])
    # 用新建的干净索引做权重实验；测试写入的上传样本不能改变评测语料。
    run("refresh-index", py + ["-c", "import rag; rag.build_index()"])
    run("rrf", py + ["tune_rrf.py", "--holdout", "--output", str(run_dir / "rrf.json")])
    if args.with_model:
        run("model-plans", py + ["plan_eval.py", "--with-model", "--output", str(run_dir / "model-plans.json")])
        run("evidence-live", py + ["scripts/probe_evidence.py", "--output", str(run_dir / "evidence-live.json")])
        run("model-actions", py + ["action_eval.py", "--with-model", "--output", str(run_dir / "model-actions.json")])
        run("acceptance", py + ["scripts/acceptance_e2e.py", "--output", str(run_dir / "acceptance.json")])
    docker = shutil.which("docker")
    if docker:
        try:
            proc = subprocess.run([docker, "info", "--format", "{{.ServerVersion}}"],
                                  capture_output=True, text=True, timeout=20)
            report["docker"] = dict(daemon_available=proc.returncode == 0, container_acceptance="not_run")
        except (OSError, subprocess.TimeoutExpired):
            report["docker"] = dict(daemon_available=False, container_acceptance="not_run")
    else:
        report["docker"] = dict(daemon_available=False, container_acceptance="not_run")
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["source_unchanged"] = manifest == source_manifest()
    save()
    print(f"REPORT {run_dir / 'summary.json'}", flush=True)
    return int(not report["source_unchanged"] or any(
        s["exit_code"] != 0 or not s["source_unchanged"] for s in report["stages"]))


if __name__ == "__main__":
    raise SystemExit(main())
