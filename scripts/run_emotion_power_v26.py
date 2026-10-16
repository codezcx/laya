from pathlib import Path
from scripts.run_mi_covariance_v23 import main

if __name__ == '__main__':
    main(root=Path('artifacts/emotion_power_v26_controller'),
         feature_root=Path('artifacts/emotion_power_v26/features'),
         module='eeg_tasks.emotion_power_fit_v26')
