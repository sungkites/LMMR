# Result files

`strong_baselines.json` contains the complete training-matched and learned-compression comparisons on 22,398 official validation questions. Each method reports F1, EM, evidence length and answer-in-context rate. Paired differences use the same question IDs.

CSV names follow the manuscript's **source-table identifiers**, not the typeset table numbers. `table_20.csv` is the primary strong-baseline table; `table_01.csv` reports support diagnostics; `table_02.csv` contains ablations; `table_06.csv` lists settings; `table_14.csv` contains the expanded 21-comparison statistics. `table_19.csv` contains input templates. The remaining files preserve the other measured results and published-reference tables.

The IRCoT Flan-T5-XL result for 2Wiki is 55.10 F1, verified against Figure 9 of the original ACL 2023 paper. Published-reference rows retain the original evaluation settings.
