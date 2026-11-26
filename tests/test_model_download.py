import json
import pytest
import torch
from safetensors.torch import save_file
from jev_train import runtime


def files(tmp_path, monkeypatch):
    monkeypatch.setitem(runtime.CONFIG, 'backbone_path', str(tmp_path))
    shard = tmp_path / 'model-00001.safetensors'
    save_file({'weight': torch.ones(4)}, shard)
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': shard.name}}))
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
        (tmp_path / name).write_text('{}')
    return shard


def test_refuses_partial_download_without_aria_marker(tmp_path, monkeypatch):
    shard = files(tmp_path, monkeypatch)
    value = shard.read_bytes()
    shard.write_bytes(value[:-4])
    with pytest.raises(FileNotFoundError, match='Download incomplete'):
        runtime.backbone_files()


def test_refuses_aria_marker_even_when_preallocated(tmp_path, monkeypatch):
    shard = files(tmp_path, monkeypatch)
    (tmp_path / (shard.name + '.aria2')).touch()
    with pytest.raises(FileNotFoundError, match='in progress'):
        runtime.backbone_files()


def test_complete_shard_and_metadata(tmp_path, monkeypatch):
    shard = files(tmp_path, monkeypatch)
    assert shard in runtime.backbone_files()
