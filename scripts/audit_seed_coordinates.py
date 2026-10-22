"""Audit EEGLAB polar conversion and approximate registration to REVE; no training mutation."""
import json
import hashlib
from pathlib import Path
import numpy as np
import mne
from safetensors.numpy import load_file

BASE=Path('/inspire/hdd/global_user/zhangchenxi-253108310322')
BANK=BASE/'BSR-VideoMAE/reve-positions'
FILES={'SEED':BASE/'mne_data/SEED/SEED/channel_62_pos.locs',
       'SEED-VII':BASE/'mne_data/SEED-VII/channel_62_pos.locs'}

def fit(x,y,kind):
    if kind=='affine':
        a=np.linalg.lstsq(np.c_[x,np.ones(len(x))],y,rcond=None)[0]
        return lambda z:np.c_[z,np.ones(len(z))]@a
    xc=x-x.mean(0);yc=y-y.mean(0)
    u,s,v=np.linalg.svd(xc.T@yc);d=np.ones(3);d[-1]=np.linalg.det(u@v)
    r=u@np.diag(d)@v;scale=(s*d).sum()/(xc**2).sum()
    return lambda z:scale*(z-x.mean(0))@r+y.mean(0)

def main():
    names=json.loads((BANK/'config.json').read_text())['position_names']
    stored=load_file(str(BANK/'model.safetensors'))['embedding']
    bank={n.upper():v for n,v in zip(names,stored)}
    js=json.loads((BANK/'positions.json').read_text())
    bank_json_max=float(max(np.max(np.abs(bank[n.upper()]-v)) for n,v in js.items() if n.upper() in bank))
    standard=mne.channels.make_standard_montage('standard_1005').get_positions()['ch_pos']
    standard_errors=[np.linalg.norm(bank[n.upper()]-v) for n,v in standard.items() if n.upper() in bank]
    result={'mne_version':mne.__version__,'bank_sha256':hashlib.sha256((BANK/'model.safetensors').read_bytes()).hexdigest(),'bank_json_max_absolute_difference_m':bank_json_max,
            'bank_vs_mne_standard1005':{'n':len(standard_errors),'max_error_m':float(max(standard_errors))},'datasets':{}}
    for ds,path in FILES.items():
        montage=mne.channels.read_custom_montage(path,head_size=1.)
        pos=montage.get_positions()['ch_pos'];names=list(pos);xyz=np.array(list(pos.values()))
        polar=np.loadtxt(path,usecols=(1,2));theta=np.deg2rad(polar[:,0]);angle=np.pi*polar[:,1]
        manual=np.c_[np.sin(angle)*np.sin(theta),np.sin(angle)*np.cos(theta),np.cos(angle)]
        np.testing.assert_allclose(xyz,manual,atol=1e-12)
        common=[i for i,n in enumerate(names) if n.upper() in bank];missing=[i for i in range(len(names)) if i not in common]
        x=xyz[common];y=np.array([bank[names[i].upper()] for i in common]);fits={}
        for kind in ['similarity','affine']:
            f=fit(x,y,kind);res=np.linalg.norm(f(x)-y,axis=1)*1000
            loo=[]
            for i in range(len(x)):
                k=np.arange(len(x))!=i;loo.append(float(np.linalg.norm(fit(x[k],y[k],kind)(x[i:i+1])[0]-y[i])*1000))
            fits[kind]={'training_rms_mm':float(np.sqrt(np.mean(res**2))),'loo_median_mm':float(np.median(loo)),
                        'loo_rms_mm':float(np.sqrt(np.mean(np.array(loo)**2))),'loo_max_mm':max(loo),
                        'estimated_missing_m':{names[i]:f(xyz[i:i+1])[0].tolist() for i in missing},
                        'loo_by_electrode_mm':dict(zip([names[i] for i in common],loo))}
        result['datasets'][ds]={'path':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'channels':len(names),'common':len(common),'missing':[names[i] for i in missing],
             'formula_matches_mne':True,'registrations':fits}
    result['interpretation']='Polar conversion is exact for the supplied spherical template; registration to REVE and missing electrode XYZ are estimates, not measured anatomy. LOO errors on known electrodes do not validate true CB1/CB2 positions.'
    out=Path('artifacts/seed_coordinate_audit');out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
