"""Run the MI/MAT pipeline exclusively on physical GPUs 2 and 3."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
PYTHON=str(ROOT/'laya/.venv-eeg/bin/python')
OUT=ROOT/'artifacts/mi_mat'
OUT.mkdir(exist_ok=True,parents=True)
parser=argparse.ArgumentParser()
parser.add_argument('--jev-only',action='store_true')
parser.add_argument('--encoder-checkpoint',type=Path,help='Explicit REVE checkpoint for JEV-only training')
parser.add_argument('--contrast-only',action='store_true')
parser.add_argument('--mi-loss-only',action='store_true')
parser.add_argument('--mat-loss-only',action='store_true')
parser.add_argument('--split-contrast-heads',action='store_true')
parser.add_argument('--contrast-output',default='contrast')
parser.add_argument('--contrast-smoke-output',default='contrast_smoke')
parser.add_argument('--jev-output',default='jev')
parser.add_argument('--jev-smoke-output',default='jev_smoke')
args=parser.parse_args()
assert not (args.jev_only and args.contrast_only)
assert not (args.mi_loss_only and args.mat_loss_only)
assert not (args.mi_loss_only or args.mat_loss_only) or args.contrast_only
assert not args.split_contrast_heads or args.contrast_only
assert args.encoder_checkpoint is None or args.jev_only
jev_checkpoint=args.encoder_checkpoint or OUT/args.contrast_output/'last.pt'
status={}
def write(**values):
    status.update(values);temp=OUT/'pipeline.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(OUT/'pipeline.json')
def run(stage,name,checkpoint=None,smoke=False):
    dest=OUT/name
    if dest.exists():raise FileExistsError(dest)
    cmd=[PYTHON,'-u','-m','torch.distributed.run','--nproc-per-node=2','--master-port=29683','-m','eeg_align.train_joint','--stage',stage,'--out',str(dest)]
    if checkpoint:cmd+=['--encoder-checkpoint',str(checkpoint)]
    if smoke:cmd+=['--smoke']
    if stage=='contrast' and args.mi_loss_only:cmd+=['--mi-loss-only']
    if stage=='contrast' and args.mat_loss_only:cmd+=['--mat-loss-only']
    if stage=='contrast' and args.split_contrast_heads:cmd+=['--split-contrast-heads']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',OMP_NUM_THREADS='3',TOKENIZERS_PARALLELISM='false',PYTHONPATH=str(ROOT))
    with (OUT/f'{name}.log').open('x') as log:
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        write(stage=name,pid=p.pid,gpus=[2,3],started=time.time(),status='running')
        code=p.wait()
    if code:raise RuntimeError(f'{name} exited with code {code}')
    assert (dest/'complete.json').exists()
try:
    write(status='waiting_for_data',controller_pid=os.getpid(),gpus=[2,3])
    deadline=time.monotonic()+3600
    while not (ROOT/'datasets/mi_mat/complete.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Dataset preparation did not finish')
        time.sleep(5)
    checks=[('contrast_smoke','contrast',None),(args.jev_smoke_output,'jev',OUT/'contrast_smoke/best.pt')]
    if args.contrast_only:checks=[(args.contrast_smoke_output,'contrast',None)]
    if args.jev_only:
        assert jev_checkpoint.is_file()
        checks=[(args.jev_smoke_output,'jev',jev_checkpoint)]
    for name,stage,checkpoint in checks:
        if (OUT/name/'complete.json').exists():
            assert json.loads((OUT/name/'complete.json').read_text())['smoke']
            cfg=json.loads((OUT/name/'config.json').read_text())
            assert cfg['stage']==stage and cfg['smoke']
            if stage=='contrast':
                assert cfg.get('mi_loss_only',False)==args.mi_loss_only
                assert cfg.get('mat_loss_only',False)==args.mat_loss_only
                assert cfg.get('split_contrast_heads',False)==args.split_contrast_heads
            if stage=='jev':
                import hashlib
                assert cfg['encoder_sha256']==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                current=json.loads((ROOT/'configs/mi_emotion_train.json').read_text())
                for key in ['jev_microbatch','jev_eval_microbatch']:assert cfg[key]==current[key]
        else:run(stage,name,checkpoint,smoke=True)
    if not args.jev_only:run('contrast',args.contrast_output)
    if not args.contrast_only:run('jev',args.jev_output,jev_checkpoint)
    write(status='complete',finished=time.time())
except Exception as exc:
    write(status='failed',error=repr(exc),finished=time.time());raise
