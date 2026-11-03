from threading import Thread

import db


def test_keeper_created_by_lifespan_thread_can_be_closed_by_caller(tmp_path):
    path = tmp_path / "keeper.sqlite"
    db.enable_wal(path)
    worker = Thread(target=db.hold, args=(path,))
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive()
    db.release_all()
    # On Windows this fails if release_all silently discarded a keeper that
    # couldn't be closed because it belonged to the TestClient thread.
    path.unlink()
    assert not path.exists()
