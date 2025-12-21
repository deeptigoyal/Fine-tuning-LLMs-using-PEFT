# Fine-tuning-LLMs-using-PEFT
This code supervised-fine-tunes LLaMA-2-7B-Chat using QLoRA on a small instruction dataset.

Model loading (QLoRA):
The base LLaMA-2 model is loaded in 4-bit NF4 quantization using bitsandbytes to reduce GPU memory. Computation happens in float16.

Parameter-efficient tuning (LoRA):
Instead of updating all weights, LoRA adapters (rank=64, alpha=16) are attached to attention layers. Only these small matrices are trained, making fine-tuning feasible on limited hardware.

Dataset & tokenizer:
Instruction data (guanaco-llama2-1k) is tokenized using the LLaMA-2 chat tokenizer format.

Training (SFT):
SFTTrainer performs Supervised Fine-Tuning: the model learns to predict the next token given instruction–response pairs using standard cross-entropy loss.

Saving & inference:
The fine-tuned LoRA model is saved and used in a text-generation pipeline with LLaMA-2’s chat prompt template.

SFT vs RLHF (in this code):

SFT (used here):
Trains on labeled instruction–response data. Simple, stable, and cheap.

RLHF (not used):
Requires a reward model and preference data; optimizes behavior via reinforcement learning (e.g., PPO). More complex and costly.

👉 This code only uses SFT + QLoRA, no reward model or human preference optimization.
