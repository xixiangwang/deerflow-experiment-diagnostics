---
name: pretrained-experiment
description: Diagnose and validate the fixed local 0.5B pretrained-model LoRA target-module failure on GPU.
---

# Existing pretrained model trial

Inspect the actual fixed trial with inspect_pretrained_trial. If skipped due to resources,
state that validation was not performed. Never bypass the VRAM guard or request another model.
Read actual errors and available module names, retrieve official evidence when helpful, and
call verify_pretrained_repair with observed names. Do not equate a proposed change with success.
Success requires passed, finite gradients, changed adapter parameters and frozen base parameters.
The tool uses two authored short texts for two GPU steps, not a meaningful fine-tuning dataset.

Output one JSON object with status (verified_repair or unresolved), observed_fault, proposed_fix,
validation, evidence and limitations. State actual resource use and scope, and distinguish
document versions from the installed environment. Do not claim task quality, convergence,
generalization, QLoRA execution, model download or saved checkpoint creation.
