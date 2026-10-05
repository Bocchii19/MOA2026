# MOA2026 ? Gemma-4-12B tr?n Furiosa RNGD

M? ngu?n **final c?a kernel round**, l?y t? `final.zip`, c?ng t?i li?u ki?n tr?c v? dataflow RNGD.

## M? ngu?n final

- [Crate](Furiosa-opt-gemma-12B/): to?n b? 52 file trong `final.zip` ???c gi? nguy?n t?ng byte; c?y `src/` kh?p ch?nh x?c v?i ZIP.
- [B?n giao kernel round](Furiosa-opt-gemma-12B/kinhnghiem.md): b?n final, c?c t?i ?u v? kinh nghi?m th?c nghi?m.
- [B?ng ch?ng k?m g?i](Furiosa-opt-gemma-12B/evidence/): hai b?n ghi tr?ng th?i submission.
- [Toolchain](Furiosa-opt-gemma-12B/rust-toolchain.toml) v? [Cargo.toml](Furiosa-opt-gemma-12B/Cargo.toml): c?u h?nh build c?a g?i.

| Submission | ?i?m trong b?n ghi | QKV cycles | SAO cycles | FFN cycles |
|---|---:|---:|---:|---:|
| [5100367f](Furiosa-opt-gemma-12B/evidence/5100367f.status) | 9.4574 | 64,859 | 35,136 | 194,746 |
| [59eb350a](Furiosa-opt-gemma-12B/evidence/59eb350a.status) | 9.3608 | 65,366 | 35,974 | 194,638 |

Theo `kinhnghiem.md` trong ZIP, `59eb350a` l? l?n n?p cu?i v? hai l?n n?p d?ng c?ng c?y code `k1v` (`40645d1`). ??y l? th?ng tin v? k?t qu? l?u trong g?i; l?n c?p nh?t Git n?y kh?ng ch?y l?i build, Arena hay official evaluation, v? kh?ng x?c minh leaderboard hi?n t?i.

## C?u tr?c

```text
.
??? Furiosa-opt-gemma-12B/
?   ??? src/                 # M? ngu?n final
?   ??? evidence/            # B?n ghi ?i k?m final.zip
?   ??? kinhnghiem.md        # B?n giao final
?   ??? ref/                # Fixture t? checkpoint 9.2
?   ??? reports/            # B?o c?o l?ch s? checkpoint 9.2
?   ??? scripts/            # C?ng c? t? checkpoint 9.2
?   ??? Cargo.toml
?   ??? Cargo.lock
?   ??? ...
??? docs/
??? README.md
```

`ref/`, `reports/`, `scripts/` v? c?c t?i li?u c? kh?ng c? trong ZIP ???c gi? l?i t? checkpoint 9.2. C?c s? li?u trong `WORK_LOG.md`, `RESULTS.md`, `OPTIMIZATION_RESULTS.csv`, b?o c?o v? `docs/` thu?c m?c ?o ri?ng c?a ch?ng. README b?n trong crate c?ng ???c gi? nguy?n t? ZIP v? c?n ph?n gi?i thi?u c?; d?ng trang n?y c?ng `kinhnghiem.md` ?? x?c ??nh b?n final.

## Baseline v? t?i li?u

- [Baseline c?a ban t? ch?c](https://github.com/micro2026-moa/furiosa-opt-gemma4-12B): repo ri?ng, d?ng ?? ??i chi?u skeleton.
- [M?c l?c ki?n tr?c RNGD](docs/README.md).
- [Trang tr?c quan ki?n tr?c v? dataflow](docs/rngd_kien_truc_dataflow.html): t?i repo r?i m? b?ng tr?nh duy?t.
- [Ngu?n t?o trang HTML](docs/html_src/).

M?t s? ???ng d?n `campaign/` trong b?o c?o tham chi?u workspace nghi?n c?u g?c.
