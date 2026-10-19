import torch
from eeg_align.linear_probe import spatial_pool

def test_spatial_pool_preserves_laterality_and_all_temporal_tokens():
    # Three channels, two temporal tokens each; group order is L/mid/R.
    z=torch.tensor([1.,3.,10.,14.,20.,24.])[:,None].expand(-1,512)
    pos=torch.tensor([[-1.,0.,0.],[.00001,1.,0.],[1.,0.,0.]])
    x=spatial_pool(z,pos,['C3','Cz','C4']).reshape(3,512)
    torch.testing.assert_close(x[:,0],torch.tensor([2.,12.,22.]))
    swapped=spatial_pool(z.reshape(3,2,512)[[2,1,0]].flatten(0,1),pos,['C3','Cz','C4'])
    assert not torch.equal(x.flatten(),swapped)

def test_missing_group_is_zero_not_nan():
    x=spatial_pool(torch.ones(4,512),torch.tensor([[-1.,0.,0.],[1.,0.,0.]]),['C3','C4']).reshape(3,512)
    assert torch.isfinite(x).all() and x[1].eq(0).all()

def test_resume_step_skips_only_checkpointed_batches():
    steps_per_epoch=769;resume=200
    executed=[epoch*steps_per_epoch+b+1 for epoch in range(3) for b in range(steps_per_epoch) if epoch*steps_per_epoch+b>=resume]
    assert executed==list(range(201,2308))


def test_whole_dataset_holdout_preserves_source_subject_splits():
    from eeg_align.linear_probe import partition_source_rows
    train=[dict(id='a',dataset='2a',subject='1'),dict(id='b',dataset='other',subject='2')]
    val=[dict(id='c',dataset='2a',subject='3'),dict(id='d',dataset='other',subject='4')]
    p=partition_source_rows(train,val,'2a')
    assert [r['id'] for r in p['train']]==['b']
    assert [r['id'] for r in p['validation']]==['d']
    assert [r['id'] for r in p['heldout']]==['a','c']
