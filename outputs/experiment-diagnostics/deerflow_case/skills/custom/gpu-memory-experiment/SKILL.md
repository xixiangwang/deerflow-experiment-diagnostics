---
name: gpu-memory-experiment
description: Inspect a controlled CUDA training failure, select a lower micro-batch and verify with real execution.
---

Call inspect_gpu_oom first. Diagnose from the exception, failure stage, configured allocator cap
and current batch. A per-process cap can be exceeded even when the GPU has free memory.
If the tool observed a training CUDA OOM, select a smaller allowed micro-batch and call
verify_gpu_oom_repair. Explain the choice using the observed evidence. Preserve the effective
batch with the accumulation handled by the worker. Prioritize completing the verification under
the fixed cap. If the baseline is close to that cap and there is no evidence that an intermediate
batch fits, prefer the smallest permitted batch. If a larger repair choice still hits CUDA OOM,
use the remaining attempt on the smallest batch, or report the failure if no further run is possible.
Treat any library suggestion to change the allocator as a suggestion, not a measured diagnosis.

Return a single JSON object without Markdown. Use status (verified_repair or unresolved),
observed_fault, rationale, selected_micro_batch_size (integer or null), validation_steps
(integer), and limitations (array). Use verified_repair only when the tool reports verified=true.
Scope the result to the explicit allocator limit, synthetic input and short LoRA validation.
