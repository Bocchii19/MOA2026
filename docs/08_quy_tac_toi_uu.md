# 08 — Quy tắc tối ưu rút ra từ phần cứng (nên / không nên)

## Nên

1. **Bớt số lệnh DMA nhỏ trên hàng đợi.** Mỗi lệnh ~0,8–1,8K bất kể byte, chạy FIFO, tổng được bảo toàn.
   Gộp các tensor nhỏ nếu layout HBM cho phép; bỏ lệnh thừa như vòng HBM.
2. **Mọi lệnh DMA đi qua HBM phải căn 256 B, packet ≥ 256 B, bit 8 đổi trong vòng lặp trong cùng.**
   Lệch 256 thì đọc ×2, ghi RMW ×50. Các ứng viên đang lệch:
   - operand epilogue SAO: 240 B, 1 engine;
   - store output SAO/FFN: 240 B;
   - `x` / `rms_weight`: 240 B, 2 engine;
   - scale FFN: 800 / 480 B;
   - bitmap: 32 B; bảng FP4: 8 B.
3. **Nạp nhỏ vào 1 slice mỗi DMN, rồi phát bằng `Broadcast1` trên Switch**, thay vì replicate DMA ra 256 slice (rẻ hơn ~8,6K).
   Bài học từ SAO họ B (B2: official 36.674–36.687).
4. **Trao tay trên chip bằng `InterTranspose`/`Broadcast1` trên Main**, không dùng DM→DM (tuần tự hoá, ~76 B/c).
5. **Contraction dùng packet 64 B** (2 flit) để giảm một nửa bước Outer. Quan trọng nhất cho contraction nằm trên tail (V của QKV, projection SAO).
6. **Giảm việc cho cluster khởi động muộn** (lệch 2–4K), vì kernel kết thúc theo cluster muộn.
7. **Kiểm cấp phát SRAM sau mỗi thay đổi buffer** (`resourcelir.lifetime.json`).
   Buffer rơi vào slot của bảng FP4 đầu (địa chỉ 131.072) sinh `raw.wait main_context` và giữ chậm lệnh phía sau [P9].
8. **Đo bằng A/B xen kẽ trong cùng một job** (hoặc ABBA tuần tự); quyết định theo median nhiều lần chạy.
9. **Giữ nguyên ngữ nghĩa**: full RMSNorm + EPS, global scale, điểm round BF16, offset động, scatter cache.
   Dời điểm round chỉ khi không có phép toán nào nằm giữa.

## Không nên (đã đo là thua)

| Hướng | Vì sao thua | Mức thua |
|---|---|---:|
| Thêm lệnh DMA bằng cách tách weight thành tile/lệnh | mỗi lệnh thêm ~1,1K; scheduler chèn thêm lệnh nhỏ vào giữa | +0,8…+5,4K |
| Thêm pass `fetch_table_lookup` (tile lookup, giải mã sẵn) | mỗi pass thêm bảng DMA + StoTab ~7–9K | +13,5…+36,8K |
| DM→DM có replicate / broadcast | ~76 B/c, tuần tự hoá | +8…+24K |
| Gộp nhiều lệnh thành một lệnh lớn (weight+scale, scale 15 hàng) | consumer đầu phải đợi cả lệnh lớn | +4…+9K |
| Layout có dead slot hoặc run chẵn lần 256 (bit 8 không đổi) | stream chậm 6% hoặc hơn | +9…+17K |
| Ép thứ tự bằng anchor để DMA nhỏ lên trước | làm giãn khe khác | +1,4K |
| Tin lịch tĩnh để chấp nhận ý tưởng | FFN: giảm tĩnh nhưng tăng phần cứng 5–11K | — |
| A/B hai job song song | lệch có hệ thống | ±1K giả |
| Dùng biến môi trường scheduler của compiler | server official không áp dụng | — |

## Thứ tự ưu tiên khi tìm lời (giá trị trên mỗi cycle)

| Kernel | 1.000 cycle = | Overhead hiện tại (trace) | Nơi lời lớn nhất |
|---|---:|---:|---|
| SAO | +0,9% | ~12K | tail 9,9K: projection 3,4K, trao đổi + sync 2,5K, operand/store lệch |
| QKV | +0,43% | ~34K | head 8,9K, tail 10,9K (contraction V 6,1K), khe Q→K 6,8K |
| FFN | +0,15% | ~79K | tail 38,8K (stage1 ‖ chuỗi scale), 2 khe 28,7K (scale lệch 256), head 11,2K (`ClusterSync` 3K) |
