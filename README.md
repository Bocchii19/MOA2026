# MOA2026 — Gemma-4-12B trên Furiosa RNGD

Mã nguồn gói **9.2** và tài liệu kiến trúc, dataflow RNGD.

```text
.
├── Furiosa-opt-gemma-12B/
│   ├── ref/
│   ├── reports/
│   ├── scripts/
│   ├── src/
│   ├── Cargo.toml
│   ├── Cargo.lock
│   └── ...
├── docs/
│   ├── rngd_kien_truc_dataflow.html
│   ├── html_src/
│   ├── data/
│   └── ...
└── README.md
```

## Mã nguồn và kết quả

- [Crate 9.2](Furiosa-opt-gemma-12B/README.md): hướng dẫn build, chạy và hợp đồng của skeleton.
- [Nhật ký tối ưu](Furiosa-opt-gemma-12B/WORK_LOG.md): diễn biến và kết quả từng thí nghiệm.
- [Báo cáo](Furiosa-opt-gemma-12B/reports/): dữ liệu và phân tích kèm theo gói.

`9.2` là tên checkpoint của gói `furiosa-opt-gemma4-12B_9_2.zip`.
Theo WORK_LOG, Round 13, source `849283b` có kết quả official `0ebbd92e`:
**9.1699×**, QKV **69,315**, SAO **35,669**, FFN **196,918** cycles.
Đây là kết quả được ghi trong gói; lần xuất repo này không chạy đánh giá mới.
Các báo cáo cũ trong gói được giữ nguyên theo mốc thời gian của chúng.

## Tài liệu

- [Mục lục kiến trúc RNGD](docs/README.md).
- [Trang trực quan kiến trúc và dataflow](docs/rngd_kien_truc_dataflow.html): tải repo rồi mở file bằng trình duyệt.
- [Nguồn tạo trang HTML](docs/html_src/).

Tài liệu `docs/` được lấy từ workspace hiện tại và có các mốc đo riêng;
không phải mọi số liệu trong tài liệu đều là số đo của checkpoint 9.2.
Một số đường dẫn `campaign/` trong báo cáo tham chiếu workspace nghiên cứu gốc.

## Nguồn gói

Nội dung crate được giữ nguyên từng byte từ ZIP, bỏ lịch sử `.git`, cache build và Python cache.
Tên thư mục `scripts/` được giữ để các lệnh trong crate tiếp tục dùng đúng đường dẫn.
Toolchain được cố định trong [rust-toolchain.toml](Furiosa-opt-gemma-12B/rust-toolchain.toml).

Giữ repo ở chế độ **private** theo quy định chia sẻ source của workspace trước hạn thi.
