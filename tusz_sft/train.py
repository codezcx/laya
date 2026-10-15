"""Primary entry point: local Qwen3-VL EEG RL checkpoint + LoRA + decision head."""
from .train_decision import main

if __name__ == "__main__":
    main()
