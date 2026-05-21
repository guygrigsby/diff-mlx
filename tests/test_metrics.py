import json
from metrics import MetricsLogger


def test_logger_writes_jsonl(tmp_path):
    log_path = tmp_path / "metrics.jsonl"
    logger = MetricsLogger(log_path)
    logger.log(step=10, train_loss=4.2, lr=1e-4)
    logger.log(step=20, train_loss=4.0, lr=1.5e-4)
    logger.close()
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["step"] == 10
    assert json.loads(lines[1])["lr"] == 1.5e-4


def test_logger_appends_existing_file(tmp_path):
    log_path = tmp_path / "metrics.jsonl"
    l1 = MetricsLogger(log_path)
    l1.log(step=1)
    l1.close()
    l2 = MetricsLogger(log_path)
    l2.log(step=2)
    l2.close()
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["step"] == 1
    assert json.loads(lines[1])["step"] == 2
