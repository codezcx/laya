"""Frozen, provenance-bound MI readouts. Outputs are scores for fixed A=left/B=right."""
import json
from pathlib import Path
import numpy as np
from .cache import digest

WIDTHS={'states':5120,'reve':3072,'combined':8192}


def pool_reve(tokens):
    if tokens.ndim!=3 or tuple(tokens.shape[1:])!=(18,512):
        raise ValueError('Expected paired MI tokens [B,18,512]')
    return tokens.float().reshape(len(tokens),2,3,3,512).mean(3).reshape(len(tokens),3072)


def describe_head(path,feature,protocol,alpha,n_labels):
    path=Path(path);out=path.with_suffix('.json')
    assert feature in WIDTHS and not out.exists()
    out.write_text(json.dumps(dict(schema=1,feature=feature,weights=path.name,weights_sha256=digest(path),
        checkpoint_sha256=protocol['checkpoint_sha256'],token_audit_sha256=protocol['source_cache_sha256'],
        source_data_sha256=protocol['source_data_sha256'],alpha=alpha,n_labels=n_labels,
        combined_feature_order=['states','reve'],score_kind='ridge_regression_scores',
        mapping='A=left hand, B=right hand; fitted on fixed choice order; other prompt orders are stress tests'),indent=2))
    return out


class FrozenReadout:
    def __init__(self,path,checkpoint_sha256,token_audit_sha256,source_data_sha256):
        path=Path(path);self.metadata_sha256=digest(path);self.metadata=json.loads(path.read_text())
        m=self.metadata
        assert m['schema']==1 and m['combined_feature_order']==['states','reve']
        for k,v in [('checkpoint_sha256',checkpoint_sha256),('token_audit_sha256',token_audit_sha256),('source_data_sha256',source_data_sha256)]:
            if m[k]!=v:raise ValueError(f'Readout provenance mismatch: {k}')
        weights=path.parent/m['weights'];assert digest(weights)==m['weights_sha256']
        a=np.load(weights,allow_pickle=False);self.weight=a['weight'];self.bias=a['bias'];self.feature=m['feature']
        assert self.weight.shape==(2,WIDTHS[self.feature]) and self.bias.shape==(2,)
        assert np.isfinite(self.weight).all() and np.isfinite(self.bias).all()

    def score(self,state,reve):
        parts=[state,reve] if self.feature=='combined' else [state if self.feature=='states' else reve]
        if any(p is None for p in parts):raise ValueError('Missing readout features')
        x=np.concatenate(parts,axis=1).astype(np.float64)
        if x.shape[1]!=self.weight.shape[1] or not np.isfinite(x).all():raise ValueError('Invalid readout features')
        return x@self.weight.T+self.bias
