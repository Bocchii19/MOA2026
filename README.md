# MOA2026 — Gemma-4-12B trên Furiosa RNGD

Mã nguồn **final của kernel round**, lấy từ `final.zip`, cùng tài liệu kiến trúc và dataflow RNGD.

## Mã nguồn final

- [Crate](Furiosa-opt-gemma-12B/): toàn bộ 52 file trong `final.zip` được giữ nguyên từng byte; cây `src/` khớp chính xác với ZIP.
- [Bàn giao kernel round](Furiosa-opt-gemma-12B/kinhnghiem.md): bản final, các tối ưu và kinh nghiệm thực nghiệm.
- [Bằng chứng kèm gói](Furiosa-opt-gemma-12B/evidence/): hai bản ghi trạng thái submission.
- [Toolchain](Furiosa-opt-gemma-12B/rust-toolchain.toml) và [Cargo.toml](Furiosa-opt-gemma-12B/Cargo.toml): cấu hình build của gói.

| Submission | Điểm trong bản ghi | QKV cycles | SAO cycles | FFN cycles |
|---|---:|---:|---:|---:|
| [5100367f](Furiosa-opt-gemma-12B/evidence/5100367f.status) | 9.4574 | 64,859 | 35,136 | 194,746 |
| [59eb350a](Furiosa-opt-gemma-12B/evidence/59eb350a.status) | 9.3608 | 65,366 | 35,974 | 194,638 |

Theo `kinhnghiem.md` trong ZIP, `59eb350a` là lần nộp cuối và hai lần nộp dùng cùng cây code `k1v` (`40645d1`). Đây là thông tin và kết quả lưu trong gói; lần cập nhật Git này không chạy lại build, Arena hay official evaluation, và không xác minh leaderboard hiện tại.

## Cấu trúc

```text
.
├── Furiosa-opt-gemma-12B/
│   ├── src/                 # Mã nguồn final
│   ├── evidence/            # Bản ghi đi kèm final.zip
│   ├── kinhnghiem.md        # Bàn giao final
│   ├── ref/                 # Fixture từ checkpoint 9.2
│   ├── reports/             # Báo cáo lịch sử checkpoint 9.2
│   ├── scripts/             # Công cụ từ checkpoint 9.2
│   ├── Cargo.toml
│   ├── Cargo.lock
│   └── ...
├── docs/
└── README.md
```

`ref/`, `reports/`, `scripts/` và các tài liệu cũ không có trong ZIP được giữ lại từ checkpoint 9.2. Các số liệu trong `WORK_LOG.md`, `RESULTS.md`, `OPTIMIZATION_RESULTS.csv`, báo cáo và `docs/` thuộc mốc đo riêng của chúng. README bên trong crate cũng được giữ nguyên từ ZIP và còn phần giới thiệu cũ; dùng trang này cùng `kinhnghiem.md` để xác định bản final.

## Baseline và tài liệu

- [Baseline của ban tổ chức](https://github.com/micro2026-moa/furiosa-opt-gemma4-12B): repo riêng, dùng để đối chiếu skeleton.
- [Mục lục kiến trúc RNGD](docs/README.md).
- [Trang trực quan kiến trúc và dataflow](docs/rngd_kien_truc_dataflow.html): tải repo rồi mở bằng trình duyệt.
- [Nguồn tạo trang HTML](docs/html_src/).

Một số đường dẫn `campaign/` trong báo cáo tham chiếu workspace nghiên cứu gốc.
