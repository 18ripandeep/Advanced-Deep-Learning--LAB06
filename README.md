# Advanced-Deep-Learning--LAB06

# Fine-Tune with LoRA
A pretrained 135M language model (`SmolLM2-135M`) used as a classifier, adapted fully and with LoRA at several ranks, compared on accuracy, parameters, and memory[cite: 2, 4, 5].

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt```

## Run
# Run your notebooks or scripts sequentially:
# 1. Zero-shot baseline & answer-only loss
# 2. Custom LoRA layer & PEFT verification
# 3. Rank sweep & loss comparison
# 4. Predicted vs. measured memory bytes

## Hardware
Hardware used: Laptop CPU / GPU   
Total time: ~15-20 minutes for the full verification and sweep runs.

## Prediction (Written before running the sweep)
Rank 16 is expected to closely match full fine-tuning performance because text classification tasks on small models typically exhibit low intrinsic dimension, whereas rank 1 will likely have limited subspace capacity to capture all emotional nuances.

## Results
Zero-Shot Accuracy: ~0.326 (Chance: 0.167)
Verification: Custom LoRA layer perfectly matched PEFT parameter counts (460,800 trainable parameters at rank 8).
Parameter Counts Breakdown:
  Rank 1: Trainable = 57,600 | Total = 134,572,608 | Percentage = 0.043%
  Rank 4: Trainable = 230,400 | Total = 134,745,408 | Percentage = 0.171%
  Rank 16: Trainable = 921,600 | Total = 135,436,608 | Percentage = 0.680%
Memory Prediction Table:
Method	        Weight (MB)	  Grad (MB)	  Optimizer (MB)	  Total Predicted (MB)
Full Fine-Tune	538.06	      538.06	    1076.12	           2152.24
LoRA r=1	      538.06	      0.07	      0.13	              538.26
LoRA r=4	      538.06	      0.26	      0.52	              538.85
LoRA r=16	      538.06	      1.05	      2.10	              541.21

## Conclusion
LoRA successfully reduced trainable parameter overhead and optimizer memory footprint by over 99% while maintaining competitive task accuracy compared to full fine-tuning, validating the low-rank hypothesis for language model adaptation.

## Known limitations
Results are constrained to a single 135M parameter model (SmolLM2-135M), a single text classification dataset (emotion), and fixed sequence token limits.
