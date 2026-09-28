# 06 — Điểm nghẽn `sliding_attention_output` (SAO)

## Dữ liệu

- Trace job 93413, image bản best `b162deec`: SAO B2 (activation dùng chung theo cặp nhóm hàng) + tail 8,4451.
- Cluster 0: Task **37.561**. Cluster 1: 37.037.
- Official cùng source: 36.674 / 36.686.
- Timeline: [data/timeline_sliding_attention_output.md](data/timeline_sliding_attention_output.md).

| Chỉ số | Cluster 0 | Cluster 1 |
|---|---:|---:|
| DMA busy union | 33.131 (88,2%) | 28.863 (77,9%) |
| TU busy union | 10.970 (29,2%) | 11.488 (31,0%) |
| Byte bắt buộc | 15,74 MB → sàn 25,5K @616 B/c | |

SAO là kernel có **giá trị cao nhất cho mỗi cycle** (1.000 cycle ≈ +0,9% score).

## 1. Timeline theo đoạn (cluster 0)

| Đoạn | Từ → Đến | Độ dài | Nội dung |
|---|---:|---:|---|
| Head | 0 → 2.011 | 2,0K | DMA `xq` 101–1.519 (128 KB, 8 engine, packet 256 B căn) → encode Main 2.004–2.500 → phát weight O |
| Weight O | 2.011 → 27.657 | **25,6K** | 7,86 MB/cluster ≈ 307 B/cycle/cluster, sát trần. Trong lúc đó encode lo/hi, StoTrf (1,76K) và `ExplicitSync` của lần trao đổi DmaStos (rơi sau weight nên không tốn) |
| **Projection** | 27.659 → 31.074 | **3,4K** | contraction + inter-slice reduce 16 chunk; tĩnh 1.374 (×2,5) |
| DMA operand epilogue | 27.666 → 32.885 | song song | `o_weight_scale` 1,60K, `post_attn_rms_weight` 1,78K, `residual` 1,14K. **Cả ba: 1 engine, packet 240 B lệch 256, chỉ cluster 0** |
| **Chuyển `y` C1→C0 (DmaStos)** | 31.076 → 32.221 | 1,15K | gom 3.840 hàng về 32 slice của C0 |
| **Sync sau chuyển** | 32.223 → 33.612 | **1,39K** | `ExplicitSync` đợi phần của C1 |
| RMS + epilogue | 33.613 → 35.834 | 2,2K | 4 pass Main: partial RMS 0,56 → regular32 0,71 → sqrt 0,41 → normalize×w+res 0,53. Pass residual trên Sub 1,13K chạy song song |
| **Store output** | 35.836 → 36.707 | 0,87K | **1 engine, packet 240 B → ghi lệch 256 (RMW)** |
| **Sync cuối** | 36.709 → 37.558 | 0,85K | kết thúc kernel |

**Tail sau byte weight cuối = 9,9K = 3,4 + 1,15 + 1,39 + 2,2 + 0,87 + 0,85.** Head 2,0K. Tổng overhead ≈ 12K so với stream 25,6K.

## 2. Điểm nghẽn xếp theo độ lớn

| # | Điểm nghẽn | Cycle | Hướng khắc phục (có số đo) | Ước lời |
|---|---|---:|---|---:|
| S1 | **Projection sau weight** | 3,4K | Hình học G3 của P9: 16 nhóm hàng × 16 strip/cluster; mỗi slice đọc 120 mảnh đúng 256 B; x chia hi/lo vào 2 lane TRF; **contraction packet 64 B**; inter-slice reduce 16 strip để mỗi nhóm hàng ra đúng 1 slice epilogue; tile 92+28 với chuỗi delay. P9 đo được **−0,88K** so với ver16 (kèm seed x 1/4) | −0,8…−1,5K |
| S2 | **Chuyển y + sync** | 2,5K | Probe P9 (bỏ hẳn trao đổi, numerics sai) chỉ lời +52 trong chế độ chấm, nên phần này phần lớn đã chạy chồng. Trong image của mình nó nằm trên đường găng (1,15 + 1,39). Probe P9 bớt 6/30 hàng cho cluster muộn: **−779** | −0,5…−1K |
| S3 | **DMA operand epilogue lệch và 1 engine** | ~1,6–1,8K mỗi lệnh | Probe P9 bỏ norm/residual: **−827**, nên chúng nằm trên đường găng. Thay đổi B của P9 (norm weight stream bởi pass epilogue cuối; y giữ trong VRF; pass residual phát ngay sau gather): **−0,8K** | −0,8K |
| S4 | **Store output lệch + RMW** | 0,87K | Commit pass cuối vào layout 30 × 256 B hoặc ≥4 slice/cluster mảnh ×256; tương tự K3 V2b của P9 (sto_d 1,98K → 1,23K) | −0,3…−0,6K |
| S5 | **Head** | 2,0K | Seed x chỉ sống 1/4 + ring-16 `Broadcast1{16,16}` (P9 C2): −0,8K khi kết hợp G3. J1 của P9: buffer đích của gather do pass Sub tạo, nên sync sẵn sàng rời khỏi đầu kernel và tile bắt đầu ~1,5K thay vì 1,9–2,0K | −0,3…−0,5K |
| S6 | **Chuỗi RMS 4 pass** | 2,2K | Rút pass (EPS trong Clip, sqrt gộp vào pass cuối): T2 của P9 thua +332 trên epilogue ver16. Chỉ thử sau khi đổi layout | 0…−0,4K |

## 3. Đã đo là thua (đừng lặp lại)

| Hướng | Kết quả | Nguồn |
|---|---|---|
| Tách weight theo hàng (tile60, split60) | static +1,3…4,6K; Arena +790 | [Camp] |
| Chia K làm 2 lệnh weight | chặn trên ≈ 0 (tốn ~1,1K mỗi lệnh thêm) | [Camp] |
| Nạp operand epilogue sớm lên đầu kernel | +1,4…+2,3K | [P9] |
| Cắt chuỗi delay 24 → 8/4 | +0,68…+1,3K (chuỗi delay dùng để giữ thứ tự phát lệnh) | [P9] |
| Các biến thể tile G3 khác 92+28; biến thể tail/epilogue C | +0,3…+3,8K | [P9] |
| Rows2 FP32 tail + B16 (họ B) | không lời (official 37.081) | [Camp] |

## 4. Trần thực tế

- Stream 25,5K + head ~1,3K + tail ~5K (projection 1,5 + trao đổi/sync 1,5 + RMS 1,5 + store/sync 0,5) ≈ **32–33K**.
- Hạng 1 đạt 32,7K, tức họ đã ở gần trần này.
