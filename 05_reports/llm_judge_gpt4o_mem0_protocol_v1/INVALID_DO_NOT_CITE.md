# INVALID - DO NOT CITE

This first-pass judge run incorrectly treated the LoCoMo Cat5
`adversarial_answer` field as the gold answer and passed Setting B `(a)/(b)`
labels to the judge without resolving them to option text. Cat1-Cat4 judgments
remain reusable as an exact cache, but the v1 aggregate scores are invalid.

Use `05_reports/llm_judge_gpt4o_mem0_protocol_v2_cat5_corrected/` instead.
