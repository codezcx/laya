import pytest

from scripts.prefetch_physionet import edf_size, publish


def edf(path):
    header = bytearray(b" " * 512)
    header[:8] = b"0       "
    header[184:192] = b"512     "
    header[236:244] = b"2       "
    header[252:256] = b"1   "
    header[472:480] = b"4       "
    path.write_bytes(header + b"\0" * 16)


def test_publish_complete_edf(tmp_path):
    staging, target = tmp_path / "part", tmp_path / "data.edf"
    edf(staging)
    assert edf_size(staging) == 528
    result = publish(staging, target)
    assert result["created"] and result["bytes"] == 528
    assert target.stat().st_size == 528 and not staging.exists()


def test_partial_never_visible(tmp_path):
    staging, target = tmp_path / "part", tmp_path / "data.edf"
    edf(staging)
    staging.write_bytes(staging.read_bytes()[:-2])
    with pytest.raises(ValueError, match="Incomplete"):
        publish(staging, target)
    assert not target.exists() and staging.exists()


def test_existing_data_not_overwritten(tmp_path):
    staging, target = tmp_path / "part", tmp_path / "data.edf"
    edf(staging)
    target.write_bytes(b"preserve original")
    result = publish(staging, target)
    assert not result["created"]
    assert target.read_bytes() == b"preserve original"


def test_rejects_non_edf(tmp_path):
    staging = tmp_path / "part"
    staging.write_bytes(b"<html>error</html>" * 40)
    with pytest.raises(ValueError, match="Not an EDF"):
        edf_size(staging)
