# RGND_arch — Kiến trúc Furiosa RNGD và bản đồ điểm nghẽn của 3 kernel Stage 1

Lập 2026-09-25. Dựa trên:
- trace phần cứng **job 93413**: image bản best `b162deec` (official 8,4643), cả 3 kernel, 5/5 PASS;
- luồng lệnh compiler phát ra và descriptor DMA của cùng source;
- tài liệu Furiosa và forum #469/#435;
- các báo cáo `campaign/`;
- số đo phần cứng trong `packages/score9x_zip_20260925` (nguồn gốc chưa xác nhận, chỉ dùng số đo phần cứng).

## Mục lục

| File | Nội dung |
|---|---|
| [01_kien_truc_phan_cung.md](01_kien_truc_phan_cung.md) | Chip, cluster, PE, DMN, slice; HBM/DM/TRF/VRF; DMA engine; Tensor Unit; clock 2 GHz và 1 GHz; byte bắt buộc và sàn |
| [02_mo_hinh_thuc_thi.md](02_mo_hinh_thuc_thi.md) | Luồng lệnh phát ra: Wait{Dma}, ClusterSync, PageTableUpdate, FIFO DMA; vì sao có head/khe/tail; lịch tĩnh so với phần cứng |
| [03_chi_phi_bo_nho_va_DMA.md](03_chi_phi_bo_nho_va_DMA.md) | Bảng số đo: stream, lệnh nhỏ, căn 256, scale, DM→DM, sync, Tensor Unit |
| [04_phuong_phap_profiling.md](04_phuong_phap_profiling.md) | Cách chạy trace, ghép nhãn, đọc descriptor, đo A/B đúng cách, công cụ |
| [05_diem_nghen_QKV.md](05_diem_nghen_QKV.md) | Timeline + điểm nghẽn QKV |
| [06_diem_nghen_SAO.md](06_diem_nghen_SAO.md) | Timeline + điểm nghẽn SAO |
| [07_diem_nghen_FFN.md](07_diem_nghen_FFN.md) | Timeline + điểm nghẽn FFN |
| [08_quy_tac_toi_uu.md](08_quy_tac_toi_uu.md) | Nên / không nên, thứ tự ưu tiên |
| [09_can_bang_2_cluster.md](09_can_bang_2_cluster.md) | Hai cluster có cùng hiệu năng không: phần cứng ngang nhau, nhưng C0 gánh thêm việc nên C1 để không 1,3–2,4K mỗi kernel |
| [rngd_kien_truc_dataflow.html](rngd_kien_truc_dataflow.html) | Trang HTML: mặt phẳng chip theo tầng, tác dụng từng khối, dataflow hệ thống và dataflow từng kernel (timeline thật, mở bằng trình duyệt). Nguồn sinh: [html_src/](html_src/) |
| [data/](data/) | Timeline đầy đủ có nhãn source (3 kernel × 2 cluster); audit descriptor DMA |

## Tóm tắt trong 1 phút

1. **Băng thông stream không phải điểm nghẽn.**
   - Weight lớn chạy ~300–310 B/cycle mỗi cluster (~615 cả chip, ≈ 82% đỉnh HBM và ~95% trần đo được).
   - Sàn theo byte bắt buộc ở tốc độ này: QKV 51K / SAO 25,5K / FFN 162K (12,1×).
   - Bản best: 78,3K / 36,7K / 215,6K (8,46×).
2. **Toàn bộ khoảng cách là overhead**, gồm bốn loại:
   - **head**: lệnh DMA nhỏ + `Wait` + setup, và ở FFN thêm một `ClusterSync` ~3K;
   - **khe giữa weight**: DMA nhỏ nối tiếp; ở FFN là scale lệch 256 tốn ~10K mỗi ma trận;
   - **tail**: compute sau byte weight cuối, trao đổi xuyên cluster + sync, store;
   - **lệch khởi động giữa hai cluster**.
3. **Mô hình thực thi**:
   - chương trình là một luồng lệnh tuần tự;
   - `Wait{Dma}` đợi **toàn bộ** hàng đợi DMA;
   - DMA chạy FIFO;
   - scheduler tự quyết thứ tự, và thứ tự trong source không có tác dụng.
4. **Lịch tĩnh ×2 ≈ phần cứng** cho code ổn định, nhưng **thêm lệnh DMA hoặc lookup thì phần cứng tệ hơn dự đoán 3–9 lần**.

### Điểm nghẽn lớn nhất theo kernel (trace 93413, cluster 0)

| Kernel | Task | Điểm nghẽn 1 | Điểm nghẽn 2 | Điểm nghẽn 3 |
|---|---:|---|---|---|
| QKV | 85,5K | Tail 10,9K: **contraction V 6,1K** (tĩnh 1,2K), chuẩn hoá V 3,1K, scatter 1,65K | Head 8,9K: 2,6K prologue trống + bitmap 32 B + `x`/`rms` 240 B lệch | Khe Q→K 6,8K: rope đi vòng HBM + 5 cặp "DMA nhỏ → pass Sub" |
| SAO | 37,6K | Tail 9,9K: **projection 3,4K**, chuyển y + sync 2,5K, RMS 2,2K | Operand epilogue: 3 lệnh 1 engine, 240 B lệch (~1,6–1,8K mỗi lệnh; −0,8K nếu dời) | Store 240 B RMW + sync cuối 1,7K |
| FFN | 224,3K | Tail 38,8K: **stage1 down 22,8K ‖ chuỗi scale down 22,8K** → stage2 → DmaStos → sync 3,8K → post-norm | Hai khe 28,7K: **scale up/gate lệch 256 (packet 800 B) ~10K mỗi ma trận** + bảng FP4 | Head 11,2K: **`ClusterSync` ~3K** chặn up weight |

### Đòn bẩy có số đo (từ file 05–07)

| Đòn bẩy | Ước lời | Bằng chứng |
|---|---:|---|
| SAO: G3 (strip, packet 64 B, inter-slice reduce) + seed x 1/4 | −0,8…−1,5K | P9 −0,88K |
| SAO: operand epilogue dời vào pass cuối (B) | −0,8K | P9 −0,83K |
| SAO: bớt việc cho cluster muộn | −0,5…−0,8K | P9 probe −0,78K |
| QKV: contraction V packet 64 B | −1…−3K | P9 V64 |
| QKV: strip geometry | −2,2K | P9 |
| FFN: block scale căn 256 + InterTranspose (up/gate), mở rộng cho down | −4,7K (+ −2…−3,5K) | P9 Round 2b |
| FFN: bỏ `ClusterSync` ở head / DmaStos thẳng vào epilogue | −1…−3K | P9 B2 −1,4K |

Tổng hợp các đòn bẩy này dẫn tới cùng khoảng mà `campaign/PLAN_9X_20260925.md` đặt ra (~9,2–9,4×).
Các đòn bẩy lấy từ P9 phụ thuộc vào việc xác nhận nguồn gốc gói đó.
