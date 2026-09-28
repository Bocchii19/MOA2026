# 07 — Điểm nghẽn `decoder_feedforward` (FFN)

## Dữ liệu

- Trace job 93413, image bản best `b162deec`: FFN Ktiles, down d15 + SRAM, Commit Adapter.
- Cluster 0: Task **224.274**. Cluster 1: 222.361.
- Official cùng code FFN: 214,7–217,5K.
- Timeline: [data/timeline_decoder_feedforward.md](data/timeline_decoder_feedforward.md).

| Chỉ số (cluster 0) | Giá trị |
|---|---|
| DMA busy union | 197.904 (88,2%) |
| TU busy union | 158.766 (70,8%) |
| Byte bắt buộc | 99,53 MB (88,5 MB weight NVFP4 + 11,06 MB block scale) → sàn 161,6K @616 B/c |
| 3 stream weight | 47,8K + 48,2K + 49,6K = **145,6K** (~304 B/cycle/cluster, sát trần) |

## 1. Timeline theo đoạn (cluster 0)

| Đoạn | Từ → Đến | Độ dài | Nội dung |
|---|---:|---:|---|
| **Head** | 0 → 11.226 | **11,2K** | DMA `x` 96–1.248 → **`ClusterSync` 2.347 → ~5.361 (~3,0K, chặn luồng lệnh)** → DMA `rms` 1,16K → bitmap 1,47K → bảng FP4 2,29K → **up weight phát ở 11.226** |
| Up weight | 11.226 → 59.018 | 47,8K | encode hi/lo, StoTab (7,3K) chạy song song |
| **Khe up→gate** | 59.018 → 72.638 | **13,6K** | **Scale up 10,28K** (gói 7,2 KB/slice lệch 256 → ~184 B/c) + global scale + **bảng FP4 1,96K** → gate weight |
| Stage1 up | 59.024 → 81.476 | 22,5K | contraction lookup f4→f8, bị giới hạn bởi tốc độ lookup; che dưới khe và stream gate |
| Gate weight | 72.638 → 120.788 | 48,2K | stage2 up (5 tile × Sub 0,97K + Main 2,17K ≈ 14,4K) chạy song song |
| **Khe gate→down** | 120.788 → 135.875 | **15,1K** | **Scale gate 10,19K** + global scale + bitmap 1,59K + **bảng FP4 3,21K** → down weight |
| Stage1 gate | 120.790 → 144.048 | 23,3K | che dưới khe và stream down |
| Down weight | 135.875 → 185.428 | 49,6K | stage2 gate (~14,4K) → GeGLU (~2,9K) → encode down: max 2,77K + **all-gather lanes 9,06K** + TRF 4,15K |
| **Tail** | 185.428 → 224.274 | **38,8K** | xem mục 2 |

### Chi tiết tail (38,8K)

```
185,4K  down weight xong
  ├─ stage1 down (lookup)      185.432 → 208.211   22,8K   ┐ hai nhánh dài bằng nhau
  └─ DMA scale down 10,7K → decode (Sub) 7,0K → StoTrf 5,1K   ┘ (185,4 → 208,2K)
208,2K  stage2 down            208.212 → 211.197   3,0K
211,2K  DmaStos rows C1→C0     211.199 → 213.364   2,2K
213,4K  ExplicitSync           213.366 → 217.141   3,8K   ← đợi phần chuyển của cluster kia
217,1K  post-FFN RMS/residual/scale (~6 pass)       ~4,3K
222,4K  store output           222.446 → 223.308   0,86K
223,3K  sync cuối              223.310 → 224.271   0,96K
```

## 2. Điểm nghẽn xếp theo độ lớn

| # | Điểm nghẽn | Cycle | Cơ chế | Hướng khắc phục (có số đo) | Ước lời |
|---|---|---:|---|---|---:|
| F1 | **Tail: stage1 down ‖ chuỗi scale down** | 22,8K | lookup f4→f8 bị giới hạn tốc độ (~2,5 B/c/slice); weight cuối luôn làm lộ pass lookup của nó | Không tile được lookup: ltile3 +35K, predecode +13,5K. Chỉ đường "Round 2" có hiệu quả: **block scale căn 256 B trao tay bằng một InterTranspose{16,1} dùng chung**. P9 đạt **−4,7K** cho up/gate (Round 2b); với down chưa làm | −2…−3,5K (scale down) |
| F2 | **Scale up/gate lệch 256 → ~184 B/c** | 2 × 10,2K trên hàng đợi | gói 7,2 KB/slice lệch 256 nên mỗi lần đọc 2 yêu cầu | Round 2b của P9: 30 hàng/slice, cả 256 slice sống, gán lại hàng (slice 8k+c giữ cặp hàng 240k+16g+2c), scale nạp theo khối 16 hàng căn 3.840 B, một InterTranspose dùng chung, D1 cho gate. Scale service 9,8–10,7K → 7,3K; **−4,7K** | −4…−5K |
| F3 | **Head: `ClusterSync` rơi trước up weight** | ~3,0K | lệnh reserve của DmaStos (rows về C0) được xếp ASAP, nằm trước cả TU đầu tiên | J1 của P9 (áp cho SAO): buffer đích của trao đổi do một pass Sub tạo, nên sync rời khỏi đầu kernel (−0,2K ở SAO). Với FFN có thể lời lớn hơn vì sync đang chặn cả up weight. Hoặc B2 của P9: kết quả down DM→DM thẳng vào slice epilogue của C0, đo được **−1,4K** | −1…−3K |
| F4 | **Bảng FP4 và StoTab** | DMA bảng 1,5–3,2K × 3 trên hàng đợi; StoTab 7,3–8,9K | mỗi pass lookup nạp lại bảng; không có API dùng chung | Không giảm được số pass. Chỉ tránh đặt buffer ở slot SRAM của bảng đầu (địa chỉ 131.072) vì sẽ sinh `raw.wait main_context` (P9 trap) | 0 |
| F5 | **Encode activation cho down** | all-gather 9,1K + TRF 4,2K | `CustomBroadcast{256}` + bitmap | V2b của P9 (GeGLU hi/lo gom về 4 slice/cluster, mỗi slice 3.840 B nguyên 256 B, store 4 engine): −0,5…−0,9K. V1b (up global scale rời 64 slice GeGLU): −0,7…−1,9K | −1…−2K |
| F6 | **Trao đổi rows C1→C0 + sync** | 2,2 + 3,8K | DmaStos + đợi cluster kia | B2 của P9 −1,4K (như F3) | xem F3 |
| F7 | **Tranh chấp DM trong lúc stream** | stream gate chậm hơn up ~8% | contraction đọc DM cùng lúc stream nên stream chậm ~20% trong vùng chồng | Rút ngắn stage1 up (packet 64 B) để thu hẹp vùng chồng: **chưa ai đo** | −1…−3K |

## 3. Đã đo là thua (đừng lặp lại)

| Hướng | Kết quả | Nguồn |
|---|---|---|
| Tile down theo L hoặc theo hàng (thêm pass lookup) | +13,6…+36,8K | [Camp], [P9] |
| Giải mã sẵn down (5 hàng / toàn bộ) | +14,6K / +13,5K | [Camp] |
| Gộp lệnh: weight + scale 15 hàng; scale 4+8+3 → 15 | +4,3…+8,8K | [P9] |
| Up/Gate 32 hàng hoặc 30 sống/32 slot (DMN) | +8,9…+17,6K (bit 8 không đổi, dead slot) | [Camp], [P9] |
| DM→DM broadcast GeGLU vào 64 slice | +23,8K (~76 B/c) | [P9] |
| Gộp Up+Gate qua half-view | ≈ +110K (DMA nửa tốc) | [P9] |
| Bỏ bớt rounding BF16 ở tail post-FFN | vi phạm luật (§8 CLAUDE.md) | — |

## 4. Trần thực tế

- Hàng đợi DMA đã bão hoà: ~143K weight + ~50K non-weight. Chỉ **bớt việc DMA** mới có lợi, đổi thứ tự thì không.
- Theo ablation của P9, xoá hẳn một lần nạp scale 3,69 MB chỉ lời 2,7K (0,55 cycle thực cho mỗi cycle DMA tĩnh).
- Sàn thực tế của họ kiến trúc này khoảng **195–200K**, và chỉ đạt được khi:
  1. scale cả 3 ma trận căn 256 B;
  2. không có vòng HBM cho GeGLU;
  3. bớt ~6K việc Main ở pha down.

  Cả ba phải cùng xảy ra. Hạng 1 đạt 198K.
