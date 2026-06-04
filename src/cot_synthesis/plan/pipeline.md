The CoT sequence pipeline
The stage takes your already-cleaned, filtered, and decontaminated math problems (the JSONL output of the data pipeline, in {"soal": ..., "jawaban": ...} format) and turns them into verified reasoning traces ready for fine-tuning. It runs in seven steps:

Clean data in — the filtered JSONL from the data pipeline is the input.
Prompt wrapping — every problem is injected into a fixed prompt template via simple string formatting. The template instructs the model to solve step by step and put the final answer inside \boxed{}, then appends the problem text. The result is a list of prompts to send to the teacher.
Generate 8 CoT solutions per problem — DeepSeek-R1-Distill-Qwen-7B acts as the teacher model and produces 8 distinct solution paths per problem. Sampling multiple diverse traces is the part flagged as "referensi dari paper."
Validate output completeness — check each generation actually contains \boxed{...}. If it doesn't, the model either ran out of budget mid-reasoning or produced a malformed format, so it's discarded.
Validate correctness — extract the predicted answer from \boxed{} and compare it against the ground-truth jawaban from the dataset. This is rejection sampling: only traces that arrive at the right answer survive.
Keep all correct CoTs — every passing trace is retained (not just one per problem), so a single problem can contribute several correct reasoning paths.
Format to ChatML — each surviving trace becomes a conversations pair: the user turn holds the prompt template plus problem, the assistant turn holds the full CoT solution ending in \boxed{answer}. This is what feeds the training pipeline (tokenization → loss masking on assistant tokens only → QLoRA SFT).

The core logic is distillation by rejection sampling: a strong reasoning teacher generates many candidate chains, and only answer-correct, well-formatted ones become training data for the smaller Qwen2.5-0.5B/1.5B student.
The paper it references
This is NVIDIA's OpenMathReasoning work, published as the AIMO-2 winning-solution paper, "AIMO-2 Winning Solution: Building State-of-the-Art Mathematical Reasoning Models with the OpenMathReasoning Dataset" (arXiv:2504.16891).
The relevant points your flowchart borrows from it:

It's a large-scale math reasoning dataset — 540K unique problems sourced from AoPS forums, with 3.2M long chain-of-thought solutions, 1.7M tool-integrated reasoning solutions, and 566K GenSelect samples. Hugging Face
They used Qwen2.5-32B-Instruct to preprocess problems, and DeepSeek-R1 and QwQ-32B to generate solutions. Your pipeline mirrors this two-model split: a Qwen2.5 model for the preprocessing/validity-check stage and a DeepSeek-R1-distilled model as the solution generator. Hugging Face
The dataset was the foundation of their winning AIMO-2 Kaggle submission. arXiv

So your adaptation keeps the paper's recipe — generate many CoT traces per problem with a distilled-R1 teacher, then filter to correct ones — but scales it down for an Indonesian-language student model, with the multi-solution-per-problem generation and the boxed-answer verification being the two pieces directly lifted from the paper.
One thing worth flagging: the paper generates solutions with full DeepSeek-R1 and QwQ-32B, whereas you're using the 7B distilled variant as teacher. That's a reasonable compute trade-off, but it does mean your teacher's trace quality (and the fraction passing the correctness filter) will be lower than the paper's, so you may want to track your acceptance rate per problem as a sanity check on whether 8 generations is enough.