# S4 manifest: fine-tuning 14 MiniLM encoders under the PUBLISHED protocol (duplicates removed), 2026-11-16

Training data: released train/val minus every record whose conversation is a verbatim copy of a record in either 600-record test file
(6,912 train / 1,657 val ids, `grouped_ids.json` in each account's `aegis-s4-ids` dataset; built by `aegis/scripts/make_s4_inputs.py`).
`--es-holdout 0.15` (early stopping on a held-out slice of train; released val untouched). Exports vectors for test, authors' aegis.jsonl, Who&When, val, train.
Kernels are built by `aegis/scripts/make_s4_kernel.py`; status and durable pulls by `aegis/scripts/s4_status_pull.py`.

| Seed | Account | Kernel | Wave |
|---|---|---|---|
| 0 | shubhamy90 | aegis-s4-ft-s0 | pilot (queued 15:37; v1 failed on my argument-header bug, log kept in `s4_s0_try1`) |
| 1 | shubhamy90 | aegis-s4-ft-s1 | 1 |
| 3, 4 | aryanmahajan665 | aegis-s4-ft-s3, -s4 | 1 |
| 6, 7 | aryanmahajan7 | aegis-s4-ft-s6, -s7 | 1 |
| 9, 10 | amahaj55 | aegis-s4-ft-s9, -s10 | 1 |
| 12, 13 | aryanomgh65 | aegis-s4-ft-s12, -s13 | 1 |
| 2, 5, 8, 11 | shubhamy90, aryanmahajan665, aryanmahajan7, amahaj55 | | 2 (after a wave-1 kernel on that account finishes) |

Known difference: the older accounts' `aegis-raw-data/whowhen.jsonl` is 184 bytes larger than the copy from the authors' repo used elsewhere (likely newline handling, one per record); texts are keyed by SHA-1 of content so embeddings are unaffected, but this is not verified.
