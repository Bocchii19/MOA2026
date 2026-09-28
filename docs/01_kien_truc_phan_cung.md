# 01 — Kiến trúc phần cứng Furiosa RNGD (cập nhật cho SDK 0.8.1)

Tài liệu này hệ thống lại phần cứng theo góc nhìn **"cái gì giới hạn tốc độ kernel"**. Số liệu được ghi nguồn như sau:

| Ký hiệu | Nguồn |
|---|---|
| [Book] | *Programming Tensor Contraction Processors*, developer.furiosa.ai/furiosa-opt/book (các chương Kernel Design, Moving Tensors/DMA Engine, Memory Performance, Computing Tensors) |
| [SDK] | source `furiosa-opt-std` / `furiosa-opt-lower-types` 0.8.1 |
| [F469] | forum Furiosa #469: miền clock của profile so với lịch compiler |
| [F435] | forum Furiosa #435: giải thích chính thức của Furiosa về chi phí TDMA |
| [T93413] | trace của team, job 93413, image bản best `b162deec` (B2 + Ktiles), 2026-09-25 |
| [Camp] | báo cáo `campaign/*.md` |
| [P9] | `packages/score9x_zip_20260925/.../WORK_LOG.md`, một nhánh khác. **Nguồn gốc chưa xác nhận**, chỉ dùng số đo phần cứng của nó |

---

## 1. Phân cấp không gian

```
Chip (1 dùng trong Stage 1)
├── HBM3 48 GB, 32 kênh × 48 GB/s = 1,5 TB/s
├── 8 DMA engine (mỗi engine phục vụ 1 cặp DMN)
├── Cluster 0 ─┬─ 4 PE × 64 slice = 256 slice
│              │    PE = 2 DMN × 32 slice
│              │    slice = 1 Tensor Unit + 512 KB DM + 8 lane TRF + VRF 8 KB
└── Cluster 1 ─┴─ giống hệt
```

| Cấp | Số lượng | Ghi chú tối ưu |
|---|---|---|
| Cluster | 2 | Hai cluster **chạy cùng chương trình (SPMD)**; clock độc lập; đồng bộ bằng `ClusterSync` (tốn 1,4–3K cycle device) |
| PE | 4 / cluster | Mỗi PE (64 slice) gắn với **một DMA engine** cho lệnh nạp HBM→DM (engine i → cluster i/4, PE i%4) [Camp: descriptor audit] |
| DMN | 2 / PE, 8 / cluster | Mỗi DMN 128 B/cycle; phải **xen kẽ 2 DMN** mới đạt 256 B/cycle mỗi engine [Book] |
| Slice | 256 / cluster, 32 / DMN | Mỗi slice có hàng lệnh DM 2 mục; trải yêu cầu ra nhiều slice để giảm tranh chấp [Book] |
| Lane | 8 / slice | Hàng MAC của Contraction Engine; TRF 8 KB mỗi lane |

**Thay đổi so với thời 0.6.0:** stage 1 nay dùng **cả hai cluster**, mỗi kernel chia việc theo `Cluster = m![… / 2]`.
Ghi chú cũ trong `docs/RNGD_ARCHITECTURE.md` §14.1 ("một nửa chip không làm gì") không còn đúng với bản best.

## 2. Phân cấp bộ nhớ

| Tầng | Dung lượng | Băng thông lý thuyết | Băng thông **thực đo** |
|---|---|---|---|
| HBM3 | 48 GB | 1,5 TB/s = **750 B/cycle device** [F469] | Stream weight lớn ~**600–633 B/cycle** cả chip (≈ 80–84% đỉnh) [T93413, P9] |
| DM (SRAM) | 512 KB/slice, 256 MB/chip | 128 B/cycle/DMN (ở clock 1 GHz) | Không phải điểm nghẽn cho stream weight |
| TRF | 8 KB/lane, API tối đa 64 KB/slice | đọc 1 packet/cycle | Double-buffer (FirstHalf/SecondHalf) |
| VRF | 8 KB/slice | — | Một toán hạng `to_vrf` ≤ 8.192 B |

### 2.1 Ánh xạ địa chỉ HBM (quyết định tốc độ stream)

[Book, Memory Performance]:
- Bit 0–7 là byte; **bit 8 là stack**; bit 9–12 là channel (XOR với bit 13–28); bank bit 17–18, 19; row 21–33.
- Đọc cắt ngang biên 256 B → **2 lần truyền (phạt 2×)**.
- Ghi không căn → **read-modify-write (phạt ~50×)**.
- Đổi row trong cùng bank → chậm 30–40×.

Luật thực đo [P9 stream lab, ~20 biến thể phần cứng]:

| Điều kiện | Ảnh hưởng tốc độ stream |
|---|---|
| Vòng lặp trong cùng của engine giữ **bit 8 cố định** | −4% (−18% nếu cố định cả bit 8–10) |
| Mỗi engine chỉ ghi vào 1 DMN | −23% |
| Một cluster chỉ đọc 1 stack | −37% |
| Packet lệch 256 B hoặc packet 800–1.024 B | chỉ ~350–390 B/cycle |
| Có contraction đọc DM cùng lúc với stream | ~−20% trong vùng chồng |
| Chỉ một cluster stream | 359–382 B/cycle; hai cluster cùng stream: ~615–633 |

Nhận xét [Camp + P9]:
- Stream weight lớn của cả ba kernel trong bản best đã ở **~300–314 B/cycle mỗi cluster** (SAO 7,86 MB/cluster trong 25,6K; FFN 14,75 MB trong 47–49K), rất gần trần thực tế.
- Riêng QKV bản best chạy 284 / 270 / 281 B/cycle/cluster, tức 5–8% dưới trần, do layout nguyên hàng. Bản strip của P9 đạt 621.

### 2.2 DM và DMN

- DM mỗi slice: 16 bank × 4.096 row × 8 B. Địa chỉ liên tiếp rải qua 16 bank. [Book]
- **Bank starvation:** 64 lần truy cập liên tiếp vào cùng bank từ Fetch/Commit có thể làm DMA chết đói (DMA có ưu tiên thấp nhất). Vượt 4.096 cycle thì NoC timeout và reset cluster. [Book]
- DM→DM cùng cluster: đọc và ghi bị **tuần tự hoá** (cùng tranh bank) [Book]. Replicate DM→DM đo được chỉ ~76 B/cycle [P9].
  Bản QKV thử nghiệm rope DM→DM thua +8K [Camp].

## 3. DMA engine

| Thông số | Giá trị | Nguồn |
|---|---|---|
| Số engine | 8/chip, 1 engine cho mỗi cặp DMN | [Book] |
| Băng thông/engine | tới 256 B/cycle (lý thuyết); thực tế ~77 B/cycle device mỗi engine khi cả 8 cùng stream | [Book], [T93413] |
| Khởi động | ~500 cycle (clock compiler) ≈ ~1K cycle device mỗi lệnh | [Book], [P9: ~1,08K/lệnh] |
| Packet | tối đa 4.096 B (AXI); chia thành yêu cầu 256 B; HBM↔DM cần căn 8 B | [Book] |
| Chọn engine | Mặc định là engine cục bộ của DM (nguồn hoặc đích) | [Book] |

**Hành vi hàng đợi (đo thực tế):**
- Các lệnh DMA dùng chung engine chạy theo **FIFO**, không chồng nhau về dữ liệu. Chia một stream thành 2 hoặc 4 lệnh chỉ cộng thêm ~1,1K mỗi lệnh [P9].
- Trên trace, span "chồng" nhau chỉ là lệnh sau đã được phát ra khi lệnh trước còn chạy [Camp EMITTED_PROGRAM_MODEL].
- Lệnh nhỏ tốn ~0,76–0,89K khi đứng sau lệnh khác trong hàng đợi, và 1,4–1,5K nếu đứng sau DMA rảnh [P9]. Trace của team: 1,1–1,8K mỗi lệnh 7,7 KB [T93413].
- Lệnh chỉ ghi vào 1 PE thì chỉ dùng 1–2 engine. Ví dụ `x` và `rms_weight` của QKV: 2 engine, packet 240 B lệch 256 B [T93413 descriptor audit].

## 4. Tensor Unit

```
DM → Fetch → Fetch Adapter → Switch → Collect ─┬→ Contraction ─┐
                (lookup/cast)   (ring)  (flit 32B) ├→ Vector ─────┤→ Cast → Transpose → Commit Adapter → Commit → DM
                                                   ├→ TRF (Sub: StoTrf)
                                                   └→ VRF (StoVrf)
```

| Engine | Điều cần nhớ khi tối ưu |
|---|---|
| Fetch | Packet căn 8 B; đọc DM tranh chấp với DMA đang ghi (−20% stream khi chồng) |
| Fetch Adapter | `fetch_table_lookup` (f4→f8, f8→bf16) **chỉ có ở Main**. Bảng nằm trong **một thanh ghi của Fetch Unit**: mỗi pass lookup nạp lại bảng bằng một DMA 4 KB + một `StoTab` (~5,4–9K cycle device) [Camp, P9]. Không có API dùng chung bảng |
| Switch | Ring 256 router. `Broadcast1{slice1,slice0}` / `Broadcast01` / `InterTranspose` do compiler tự suy ring. `CustomBroadcast{ring}` cần **DMA nạp bitmap 16 KB** (packet 32 B) |
| Contraction | Outer → Packet reducer → Time reducer → Lane folder. Packet 32 B hoặc **64 B**. Thời gian thực ≈ **1,4–1,7K cố định + ~1,9 cycle mỗi bước Outer** [P9]; dùng packet 64 B thì số bước Outer giảm một nửa |
| Vector | Phần FP chạy **Way4** (4 phần tử/cycle), bằng nửa phần fixed. Inter-slice reducer gộp 256 slice |
| Cast | `f32 → bf16` là phép hạ độ chính xác duy nhất |
| Commit Adapter | `commit_cast::<bf16>()` hạ bf16 ở đầu ra mà không chiếm Cast Engine |

## 5. Context thực thi

| Context | Vai trò | Ghi chú đo được |
|---|---|---|
| Main | Toàn bộ pipeline TU (có contraction, lookup) | Phần lớn việc tính toán |
| Sub | Pipeline con (không có contraction, không có lookup) | Dùng để nạp TRF/VRF song song với Main. Pass VRF trên Sub chạy chồng DMA vào cùng DMN mất ~1,4K thay vì 0,45–0,7K [P9] |
| Tensor DMA (`tdma`) | Mọi lệnh HBM↔DM, DM↔DM | Một hàng đợi DMA chung cho chương trình |
| PCIe DMA (`pdma`) | Host↔HBM, HBM↔HBM | Không dùng được để nạp vào DM (`to_dm` chỉ nhận `Dma::Tensor`) |

**Thực tế quan trọng:** chương trình phát ra là **một luồng lệnh tuần tự** đan xen DMA, TU, Wait và Sync.
Xem [02_mo_hinh_thuc_thi.md](02_mo_hinh_thuc_thi.md).

## 6. Clock và đơn vị cycle

| Thứ được đo | Clock | Nguồn |
|---|---|---|
| Cycle leaderboard / harness / span trace | **~2,0 GHz** ("device cycle") | [F469] |
| Lịch tĩnh của compiler (`--dump-schedule`) | 1,0 GHz | [F469] |
| Quy đổi | device ≈ 2 × tĩnh. Bản best: QKV 1,95; FFN 1,99; SAO 1,74 | [Camp] |

Hệ quả:
- HBM 1,5 TB/s = **750 B/cycle device**.
- Trần tuyệt đối theo số byte bắt buộc: QKV 42,0K / SAO 21,0K / FFN 132,7K, tức **14,76×**.
- Ở tốc độ stream thực ~616 B/cycle: 51,1K / 25,5K / 161,6K, tức **12,1×** nếu overhead bằng 0.

## 7. Byte bắt buộc của 3 kernel

| Kernel | Weight + scale bắt buộc | Sàn @616 B/c | Bản best (official b162deec) | Overhead so với sàn |
|---|---:|---:|---:|---:|
| QKV | 31.473.664 B | 51,1K | 78,3K | 27,2K (35%) |
| SAO | 15.736.320 B | 25,5K | 36,7K | 11,2K (30%) |
| FFN | 99.532.800 B | 161,6K | 215,6K | 54,0K (25%) |

**Toàn bộ khoảng cách tới trần nằm ở overhead:** prologue, khe giữa các stream, tail, lệnh DMA nhỏ và sync.
Băng thông stream đã gần trần. Phân tích từng kernel ở các file 05–07.
