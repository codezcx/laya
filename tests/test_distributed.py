import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from eeg_align.distributed import average_gradients, atomic_save


def worker(rank, rendezvous, output):
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    torch.manual_seed(17)
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=.01)
    x = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 10
    y = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 20
    for _ in range(3):
        opt.zero_grad()
        for part in (slice(rank*4,rank*4+2), slice(rank*4+2,rank*4+4)):
            (((model(x[part])-y[part])**2).mean()/2).backward()
        average_gradients(model.parameters(), 2)
        opt.step()
    torch.save(model.state_dict(), output + str(rank))
    dist.destroy_process_group()


def test_two_rank_matches_global_batch(tmp_path):
    mp.spawn(worker, args=(str(tmp_path/'rendezvous'),str(tmp_path/'rank')), nprocs=2)
    torch.manual_seed(17)
    model=torch.nn.Linear(3,2)
    opt=torch.optim.AdamW(model.parameters(),lr=.01)
    x=torch.arange(24,dtype=torch.float32).reshape(8,3)/10
    y=torch.arange(16,dtype=torch.float32).reshape(8,2)/20
    for _ in range(3):
        opt.zero_grad(); ((model(x)-y)**2).mean().backward();opt.step()
    for rank in range(2):
        actual=torch.load(tmp_path/f'rank{rank}',weights_only=True)
        for name,p in model.state_dict().items():torch.testing.assert_close(actual[name],p)


def test_atomic_failure_preserves_checkpoint(tmp_path):
    path=tmp_path/'last.pt';atomic_save({'step':1},path)
    with patch('torch.save',side_effect=OSError('simulated write failure')):
        try:atomic_save({'step':2},path)
        except OSError:pass
        else:raise AssertionError('failure missing')
    assert torch.load(path,weights_only=True)=={'step':1}
    assert list(tmp_path.iterdir())==[path]
