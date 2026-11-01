"""Distance diagnostics must not equate tightness with class collapse."""
import numpy as np
import pytest
from scripts.plot_representations import geometry, normalize, pair_separation


def test_well_separated_classes():
    x=np.array([[1.,0.],[1.,.01],[0.,1.],[.01,1.]])
    g=geometry(x,np.array([0,0,1,1]),np.array(['a','b','a','b']))
    assert g['class_gap'] > .99
    assert g['class_macro_silhouette'] > .99


def test_collapsed_representation_not_separable():
    x=np.tile([1.,2.,3.],(4,1))
    g=geometry(x,np.array([0,0,1,1]),np.array(['a']*4))
    assert g['class_gap'] is None
    assert g['participation_rank'] == 0


def test_scale_invariance_and_no_self_pairs():
    x=np.array([[1.,0.],[1.,1.],[0.,1.],[1.,2.]])
    y=np.array([0,0,1,1]); subjects=np.array(['s']*4)
    a=geometry(x,y,subjects); b=geometry(x*17,y,subjects)
    assert a['class_gap'] == pytest.approx(b['class_gap'])
    dist=np.array([[0.,4.,8.,8.],[4.,0.,8.,8.],[8.,8.,0.,4.],[8.,8.,4.,0.]])
    assert pair_separation(dist,y) == dict(within=4.,between=8.,gap=.5)


def test_missing_within_subject_contrast_not_zero():
    x=np.array([[1.,0.],[1.,.2],[0.,1.],[.2,1.]])
    g=geometry(x,np.array([0,0,1,1]),np.array(['a','a','b','b']))
    assert g['subject_controlled_gap'] is None
    assert g['eligible_subjects'] == 0


def test_zero_vectors_rejected():
    with pytest.raises(ValueError):
        normalize(np.zeros((3,2)))
