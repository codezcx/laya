from pathlib import Path
from scripts.run_mi_covariance_v23 import main

if __name__ == '__main__':
    main(root=Path('artifacts/mi_reve_scaling_v24_controller'),
         feature_root=Path('artifacts/mi_reve_scaling_v24'),
         module='eeg_tasks.mi_reve_scaling_fit_v24')
