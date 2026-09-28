# 05 — Điểm nghẽn `sliding_project_qkv` (QKV)

## Dữ liệu

- Trace job 93413, image bản best `b162deec` (code QKV giống hệt các bản từ 8,4451), seed 0, chế độ trace.
- Cluster 0: Task **85.522** cycle device. Cluster 1: 85.718.
- Official của cùng source: 76,7–79,7K, median 3 lần chạy.
- Timeline đầy đủ có nhãn source: [data/timeline_sliding_project_qkv.md](data/timeline_sliding_project_qkv.md).

| Chỉ số (cluster 0) | Giá trị |
|---|---|
| DMA busy union | 73.839 (86,3%) |
| TU busy union | 58.618 (68,5%) |
| Byte bắt buộc | 31,47 MB → sàn 51,1K @616 B/c |
| Overhead so với sàn | ~34K trace, ~27K official |

## 1. Timeline theo đoạn (cluster 0)

| Đoạn | Từ → Đến | Độ dài | Nội dung chính |
|---|---:|---:|---|
| **Head** | 0 → 8.877 | **8,9K** | DMA `x` 99–1.253 (1,15K). **Khoảng trống 1.253 → 3.871 (2,6K)** gồm 2 LoadSfr, 37 lệnh Core, `PageTableUpdate{1:4}` và `Wait{Dma}`. DMA `rms_weight` 3.877–5.029 (1,15K; **2 engine, packet 240 B lệch**). DMA bitmap `CustomBroadcast` 7.420–8.877 (1,46K; **packet 32 B**). Q weight phát ở 7.434 nhưng xếp hàng sau bitmap |
| Q weight | 7.434 → 35.201 | 27,8K | 7,86 MB/cluster; ~299 B/cycle/cluster tính từ khi bitmap xong (strip geometry của P9 đạt ~311) |
| **Khe Q→K** | 35.201 → 41.983 | **6,8K** | Chuỗi nối tiếp "DMA nhỏ → pass Sub dùng nó": `sq` 1,32K → gather cos 1,81K → pack → gather sin 1,79K → pack → `q_rms` 1,52K → store `rows0` 0,75K. Contraction Q (8,8K) chạy song song |
| K weight | 41.983 → 56.927 | 14,9K | ~277 B/cycle/cluster |
| **Khe K→V** | 56.927 → 60.791 | **3,9K** | Nạp lại rope 1,30K, `sk` 2,18K (có xếp hàng), `k_rms` 1,18K, và các pass TU của K đứng trước lệnh V trong luồng lệnh |
| V weight | 60.791 → 74.587 | 13,8K | ~285 B/cycle/cluster |
| **Tail** | 74.587 → 85.522 | **10,9K** | **Contraction V 6.133** (tĩnh chỉ 1.225, tức 5×) → 5 pass chuẩn hoá V ~3,1K → **scatter V 1,65K** ở cuối cùng. Store q, `sv` và scatter K chạy song song |

Đường găng xấp xỉ: head 8,9K + Q 26,3K + khe 6,8K + K 14,9K + khe 3,9K + V 13,8K + tail 10,9K ≈ 85,5K.

## 2. Điểm nghẽn xếp theo độ lớn

| # | Điểm nghẽn | Cycle (trace) | Bằng chứng | Hướng khắc phục (đã có số đo) |
|---|---|---:|---|---|
| Q1 | **Tail: contraction V chậm** | 6,1K | span 74.589–80.722; packet 32 B, `contract_lane Interleaved`, 8 hàng × 60 bước Outer | Packet 64 B + hi/lo broadcast theo Time bên TRF: P9 V64 đưa 3,2K → 2,15K, mức lời K1 −1,3K (khi ghép với A). Ước −2…−3K ở bản mình |
| Q2 | **Head: prologue + bitmap + DMA nhỏ lệch** | 8,9K | 2,6K trống trước pass TU đầu; bitmap 32 B xếp trước Q | Nạp `x`/`rms_weight` replicate 8 lần + `Broadcast01{4,8,30}`; nạp rms weight trước bằng data anchor "cộng 0" (P9). Gather 240/128 thường của mình từng thua (+0,5…+5K) nên cẩn thận. Ước −1…−2K |
| Q3 | **Khe Q→K: rope đi vòng HBM + 5 cặp DMA–TU** | 6,8K | chuỗi `sq`/cos/sin/`q_rms`/store | P9 `rope_h4` (`stage_rope_rows_r17`, `apply_rope_r21`): rope rows gom theo head trên 4 PE; `normalize_key_scaled` stream scale k theo K đã gather (5 pass thay vì 6). **Không dùng DM→DM** (+8K) |
| Q4 | **Tốc độ stream QKV thấp hơn trần 5–8%** | ~2,5–3,2K | 277–299 B/c/cluster so với 307+ của SAO/FFN | Strip geometry (mỗi slice giữ strip 256 cột của H, Inter-Slice Reducer cộng strip; engine đi hết một hàng 3.840 B nên bit 8 đổi mỗi 2 yêu cầu): P9 `qkv_s` **−2,2K** |
| Q5 | **Khe K→V** | 3,9K | pass TU của K-chain đứng trước lệnh V | Rule của P9: K-chain phải xong trước khi stream Q kết thúc ≥0,9K thì chuyển Q→V chỉ ~1,3K, không thì 3,9–5,7K |
| Q6 | **Scatter V đứng cuối + lệch cluster** | 1,65K + ~0–2,5K | span `Cluster` cuối ở cluster sớm ≈ độ lệch khởi động | Không bỏ được sync cuối (driver bắt buộc). Chỉ giảm việc của cluster muộn |

## 3. Đã đo là thua (đừng lặp lại)

| Hướng | Kết quả phần cứng | Nguồn |
|---|---|---|
| Rope DM→DM thay vòng HBM | +8,0K | [Camp] QKV_ROPE_DM2DM |
| Tách V thành 2 lệnh | +5,4K (row tile +0,7…1,0K) | [Camp] QKV_VSPLIT2, [P9] |
| Gather norm 128/240 (bỏ bitmap) | +0,5…+5K | [Camp] |
| Ép DMA nhỏ lên trước Q bằng anchor | +1,4K (khe K→Q giãn 4,15K) | [P9] |
| hi/lo trong lane TRF cho Q/K/V | −90 (nhiễu) | [P9] |

## 4. Ước lượng trần thực tế

- Tail tối thiểu ≈ contraction V (~2K khi dùng packet 64 B) + chuẩn hoá V (~2K) + scatter (1,65K) ≈ 5,5K.
- Head tối thiểu ≈ 2–3K.
- Hai khe tối thiểu ≈ 5K.

Cộng với stream ~52K, **QKV ~65–67K** là mức thực tế cho kiến trúc 3 lần stream này (P9 đang ở ~69,8K).
