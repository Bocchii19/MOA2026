# 03 — Chi phí bộ nhớ và DMA: bảng số đo

Mọi số liệu tính bằng **cycle device** (clock ~2 GHz, cùng đơn vị với leaderboard), trừ chỗ ghi "tĩnh".
Nguồn:
- [T93413]: trace của team, bản best.
- [Desc]: audit descriptor, xem [data/dma_descriptors.md](data/dma_descriptors.md).
- [P9]: số đo phần cứng của nhánh khác (chưa xác nhận nguồn gốc).
- [Book]: tài liệu Furiosa.

## 1. Stream weight lớn

| Trường hợp | Tốc độ đo | Nguồn |
|---|---|---|
| SAO O weight 7,86 MB/cluster, 8 engine, packet 256 B căn | 307 B/cycle/cluster (25,6K) | [T93413] |
| FFN up / gate / down 14,75 MB/cluster | 308 / 306 / 297 B/c/cluster | [T93413] |
| QKV Q / K / V (layout nguyên hàng) | ~299 / 277 / 285 B/c/cluster | [T93413] |
| QKV layout strip (bit 8 đổi mỗi 2 yêu cầu) | ~311 B/c/cluster | [P9] |
| Trần đo được cả chip | 617–633 B/cycle (≈ 1,1K/lệnh + byte / ~647) | [P9] |
| Một cluster stream một mình | 359–382 B/cycle | [P9] |
| Đỉnh HBM lý thuyết | 750 B/cycle device (1,5 TB/s) | [Book], [F469] |

**Stream đã ở ~95% trần thực tế.** Cách duy nhất còn lời từ stream là đưa các layout chậm (QKV nguyên hàng) về dạng mà bit 8 đổi liên tục.

## 2. Lệnh DMA nhỏ

| Lệnh | Payload | Engine × packet | Thời gian đo | Nguồn |
|---|---:|---|---:|---|
| `x` / `rms_weight` (QKV, FFN) | 15.360 B | 2 × 240 B lệch | 1,15K | [Desc], [T93413] |
| Bitmap `CustomBroadcast{256}` | 16.384 B | 8 × **32 B** | 1,46–1,59K | [Desc], [T93413] |
| Bảng FP4 (compiler sinh cho mỗi pass lookup) | 4.096 B | 8 × **8 B** | 1,96–3,21K (kèm chờ) | [Desc], [T93413] |
| Global scale FFN | 2.048 B | 8 × 32 B | ~1,1K | [Desc], [T93413] |
| Scale weight QKV (`sq`/`sk`/`sv`) | 4–8 KB | 8 × 256 B | 1,2–2,2K | [T93413] |
| Operand epilogue SAO (scale, norm, residual) | 7.680 B | **1 × 240 B lệch** | 1,14–1,78K | [Desc], [T93413] |
| Gather cos/sin (QKV) | 2 × 512 B | gather | 1,8K mỗi lệnh | [T93413] |
| Store output SAO/FFN | 7.680 B | **1 × 240 B lệch → RMW** | 0,86–0,87K | [Desc], [T93413] |
| Scatter K/V cache | per-head | scatter | 1,65K mỗi lệnh | [T93413] |

Luật rút ra:
- Mỗi lệnh nhỏ tốn **~0,76–0,89K khi đứng sau lệnh khác** trong hàng đợi, và **1,1–1,8K** khi hàng đợi đang rảnh hoặc khi đứng sau một `Wait` [P9], [T93413].
  Tổng thời gian được bảo toàn dù đặt ở đâu (FIFO). Chỉ phần rảnh và lần khởi động đầu là lấy lại được.
- Nạp replicate nhỏ ra 256 slice/cluster đắt hơn **~8,6K** so với nạp 1 slice/DMN rồi `Broadcast1` từ một nguồn [P9].
- Thời gian lệnh gần như không đổi theo số byte ở mức vài KB. **Bớt số lệnh** mới quan trọng, không phải bớt byte.

## 3. Block scale FFN (lệch 256 B)

| Scale | Packet | Thời gian đo | Tốc độ | Khi căn 256 (P9 Round 2b) |
|---|---|---:|---:|---:|
| Up 3,69 MB | 800 B lệch | 10,28K | ~180 B/c/cluster | 7,3K |
| Gate 3,69 MB | 800 B lệch | 10,19K | ~181 | 7,3K |
| Down 3,69 MB | 480 B lệch | 10,68K | ~173 | chưa làm |

Nguồn: [Desc], [T93413], [P9].
Đọc lệch 256 B tốn 2 yêu cầu HBM [Book]. Khối 16 hàng × 480 B = 7.680 B = 30 × 256 là căn được.

## 4. DM→DM và xuyên cluster

| Thao tác | Chi phí đo | Nguồn |
|---|---|---|
| DmaStos gom y (SAO, 7,68 KB) | 1,15K + sync 1,39K | [T93413] |
| DmaStos rows (FFN) | 2,17K + sync 3,78K | [T93413] |
| DM→DM có replicate (rope QKV) | +8K so với vòng HBM | [Camp] |
| DM→DM broadcast GeGLU vào 64 slice | ~76 B/cycle, +23,8K | [P9] |
| DM→DM cùng cluster | đọc và ghi tuần tự hoá | [Book] |
| `cluster_swap` | cùng DMA + sync như gather | [P9 census] |

## 5. Sync

| Loại | Chi phí | Ghi chú |
|---|---:|---|
| `ClusterSync` đặt trước weight đầu (FFN) | ~3,0K | chặn cả luồng lệnh |
| `ExplicitSync` sau DmaStos | 1,4–3,8K | đợi cluster kia |
| `DramReuse` (có lệnh ghi DRAM) | release sau lệnh ghi, acquire cuối kernel | driver bắt buộc cho mọi output |
| Sync cuối kernel | 0,85–1,0K (+ lệch khởi động giữa cluster 2–4K) | không bỏ được |
| Span trace | **phóng đại** chi phí sync | cùng binary SAO: mode info 36,8K, mode trace 38,1K [P9] |

## 6. Tensor Unit

| Thao tác | Chi phí đo | Nguồn |
|---|---|---|
| Contraction | ~1,4–1,7K cố định + ~1,9 cycle/bước Outer; packet 64 B giảm một nửa số bước | [P9] |
| Contraction V QKV (packet 32 B, 8 hàng) | 6,1K (tĩnh 1,2K) | [T93413] |
| Contraction lookup f4→f8 (FFN stage1) | ~22,5–23,3K cho 14,75 MB/cluster (~2,5 B/c/slice) | [T93413] |
| `StoTab` (nạp bảng FP4 vào Fetch Unit) | 7,3–8,9K (chạy chồng DMA) | [T93413] |
| Pass Vector nhỏ (RMS, sqrt, scale) | 0,4–0,7K mỗi pass | [T93413] |
| Pass VRF trên Sub chạy chồng DMA vào cùng DMN | ~1,4K thay vì 0,45–0,7K | [P9] |
| `memset` | ~7,5 B/c/slice | [P9] |
| Lệnh Core (tile/index) | ~30–90 cycle mỗi lệnh | [Camp] |
