# PLAN — MOA 2026 Stage 1: FFN ≤ 205k (mục tiêu 199k) · QKV ≤ 77k · SAO ≤ 36,3k → score ~8,7×

Tài liệu này là **chỉ dẫn bắt buộc** cho mọi agent (Codex, Claude, …) làm tiếp campaign.
Đọc hết mục 0–4 và mục của kernel mình phụ trách (5 FFN · 6 QKV · 7 SAO) trước khi sửa bất kỳ file nào. Bối cảnh lịch sử nằm trong
[kinhnghiem.md](kinhnghiem.md); tài liệu này không lặp lại, chỉ nêu việc cần làm.

- Điểm xuất phát: [best_run_8.4451/](best_run_8.4451/) — official **8,4451**
  (QKV 78.092 · SAO 37.044 · FFN 215.459 cycle, median 3 seed).
- Baseline chấm điểm: [furiosa-opt-gemma4-12B/](furiosa-opt-gemma4-12B/).
  Score = trung bình nhân (geometric mean) của speedup ba kernel so với baseline.
- SDK bắt buộc: **furiosa-opt-std 0.8.1 / cargo-furiosa-opt 0.8.1**,
  Rust **nightly-2026-05-01**, giữ nguyên `Cargo.lock`.
- Hạn nộp: **30/09/2026 23:59 AoE**. Hôm nay: 24/09/2026.

---

## 0. Mục tiêu và phép tính score

Score mới = `8,4451 × (78092/Q × 37044/S × 215459/F)^(1/3)`.

| Mục tiêu | Tích tỉ lệ cần | Nếu chỉ FFN cải thiện |
|---|---:|---:|
| 8,6 | 1,056 | FFN ≤ 204.025 |
| 8,8 | 1,131 | FFN ≤ 190.428 |
| 9,0 | 1,210 | FFN ≤ 178.013 |

- **FFN ≤ 199k** (một đội khác đã đạt mức này; chỉ riêng FFN 199k → ~8,67).
- **QKV ≤ 77k, SAO ≤ 36,3k** (mục 6–7). Q1 (RoPE) đã thử 3 biến thể và thất bại (xem mục 6).
- Ước tính cụ thể theo kernel và score: mục 8. Nhắm thực tế **~8,7**; 8,9 là trường hợp lạc quan.

---

## 1. Luật cứng (vi phạm = hủy kết quả)

1. **Không đổi tên, tham số, kiểu trả về của hàm `#[device]`.** Chỉ `src/device/**`
   và thân hàm trong `src/ops.rs` được chấm. Mọi thứ khác (`src/bin`, `src/host`,
   `src/axes.rs`, …) bị server bỏ qua — axis mới phải khai báo bằng `axes![]` bên trong `src/device/`.
2. **Giữ nguyên ngữ nghĩa số học:** full RMSNorm (mean-square + EPS + sqrt), mọi global
   scale, ranh giới làm tròn BF16, offset động (RoPE/KV), GeGLU đúng thứ tự.
   Không special-case giá trị fixture. Không đọc vùng padding như dữ liệu thật.
3. **FFN không còn biên sai số:** max|Δ| hiện 0,0078 so với atol 0,01. **Không được bỏ lane `lo`**
   của hi/lo, không hạ độ chính xác partial dưới BF16.
4. **Không sửa [best_run_8.4451/](best_run_8.4451/) tại chỗ.** Mỗi thí nghiệm làm trên bản copy riêng (mục 3).
5. **Không nộp official (`moa-submitter submit`) khi chưa có người duyệt.** Custom Arena thì được tự chạy.
6. Code `device/shared/*` có thể dùng chung với kernel khác. Sau khi sửa, `cargo furiosa-opt build`
   toàn crate phải qua.

---

## 2. Setup công cụ (làm một lần)

Máy hiện có **cargo-furiosa-opt 0.6.0** global — **sai phiên bản**. Cài 0.8.1 vào root riêng,
không ghi đè bản global:

```sh
cargo +nightly-2026-05-01 binstall -y --root $HOME/.local/furiosa-0.8.1 cargo-furiosa-opt@0.8.1
export PATH=$HOME/.local/furiosa-0.8.1/bin:$PATH
cargo furiosa-opt --version    # phải in 0.8.1
```

Script phân tích schedule (đã có sẵn trong repo):

- [tools/sched_dma.py](tools/sched_dma.py) `<schedule.json>` — liệt kê mọi DMA, util, tổng DMA busy, makespan.
- [tools/sched_timeline.py](tools/sched_timeline.py) `<schedule.json>` — toàn bộ timeline sắp theo begin.

Dump schedule tĩnh (~45 giây/kernel):

```sh
cargo furiosa-opt compile ops::decoder_feedforward --exact \
    --dump-schedule target/schedules/ffn.json
python3 ../tools/sched_dma.py target/schedules/ffn.json
```

Kernel names: `ops::sliding_project_qkv`, `ops::sliding_attention_output`, `ops::decoder_feedforward`.
Luôn dùng `--exact`.

---

## 3. Quy trình mỗi thí nghiệm (bắt buộc)

```text
1. cp -r best_run_8.4451 campaign2/<ID>_<slug>        # ví dụ campaign2/F1_scale240
2. Sửa code trong bản copy.
3. Static: compile + dump schedule kernel bị ảnh hưởng, chạy sched_dma.py.
   Ghi makespan, DMA busy, và tail sau DMA cuối.
4. Cổng static: chỉ lên Arena nếu makespan giảm, HOẶC thí nghiệm được đánh dấu
   "REAL-ONLY" trong mục 5–7.
5. Arena: `RNGD_SUBMIT_TIMEOUT=70 ./scripts/rngd_test.sh --no-build` (3 seed × 3 kernel, median).
   Server giới hạn timeout ≤ 70 s — mặc định 120 của script sẽ bị từ chối.
   Cần ref/fixtures.safetensors và thư mục scripts/ (copy từ furiosa-opt-gemma4-12B/scripts).
   Build trước bằng `cargo furiosa-opt build --release --bin test_kernels` (bản 0.8.1).
   Chạy CONTROL (best_run nguyên bản) trong cùng khoảng thời gian.
6. Chấp nhận khi: mọi output PASS, VÀ median của kernel mục tiêu thấp hơn control ≥ 1%
   (FFN ≥ 2k, QKV ≥ 0,8k, SAO ≥ 0,4k cycle), VÀ hai kernel còn lại không chậm hơn quá 1%.
   Nhiễu giữa các run official đã thấy tới ~6%, nên một cặp A/B là chưa đủ để kết luận.
7. Freeze: campaign2/<ID>/RESULT.md + diff + schedule json + Arena log.
```

Mẫu `RESULT.md`:

```md
# <ID> <slug>
- Parent: best_run_8.4451 (hoặc <ID cha>)
- Thay đổi: <1–3 dòng>
- Static: QKV <x> (parent 39.958) · SAO <x> (parent 21.337) · FFN <x> (parent 108.457); DMA busy <x>
- Arena job: <id>  PASS/FAIL  QKV <x>/<control>  SAO <x>/<control>  FFN <x>/<control>
- max|Δ|: QKV <x> (atol 0,04) · SAO <x> (atol 0,05) · FFN <x> (atol 0,01)
- Kết luận: GIỮ / BỎ — lý do
```

Hệ số quy đổi đã hiệu chuẩn (0.8.1, bản best): **real ≈ 1,74–2,0 × static**
(QKV 39.958 → 78.092; SAO 21.337 → 37.044; FFN 108.457 → 215.459).
Chỉ dùng để ước lượng, không có hệ số cố định (xem kinhnghiem §6).
Bằng chứng cũ: Time Reducer static −3,7k → real −10k (2,7×); Commit Adapter static 0 → real −1,35k.

---

## 4. Chẩn đoán FFN (đo bằng SDK 0.8.1, bản best, tĩnh)

**Makespan tĩnh 108.457 · DMA busy 99.574 (92%).** Mọi DMA chạy **tuần tự trên một hàng đợi**.
Main/sub context rảnh phần lớn thời gian. FFN đang bị chặn bởi **chuỗi DMA**, không phải tính toán.

| Nhóm DMA | Cycle tĩnh | Util | Ghi chú |
|---|---:|---:|---|
| Weight up / gate / down (NVFP4) | 24.612 / 24.612 / 24.853 | 0,87–0,88 | **Đã ở trần** — mọi layout thử đều 0,86–0,88 |
| Block scale up / gate / down (f8) | 4.880 / 4.880 / 4.961 | **0,55** | Chunk mỗi slice nhỏ (7.200 B; down 15 × 480 B có stride) |
| DMA nhỏ do code (x, pre-rms, up_gs, gate_gs, down_gs, post-rms, residual, layer_scalar) | 616–742 mỗi cái, ≈ 5,4k | ~0 | Gần như toàn overhead lệnh DMA |
| Trao đổi C1→C0 (`rows.to_dm`) + store | 626 + 782 | ~0 | Nằm trong tail |
| **Do compiler chèn**: 3 × 838 (4 KB) trước mỗi lookup NVFP4, 2 × 719 (16 KB) | 3.952 | ~0 | 4 KB ≈ nạp LUT f4→f8 (LUT nằm trong register Fetch Unit của main). 16 KB chưa rõ |

Main context: stage 1 (lookup + contract) **9.263 cycle mỗi ma trận**, stage 2 ≈ 1.170, prologue ≈ 15k (được che bởi DMA của up).
Tail sau DMA nạp cuối (100.712 → 108.457) ≈ **7,7k**: stage2 down → exchange C1→C0 + đồng bộ (~2,3k) →
cộng partial → post-norm → residual → layer gate → store.

### Kết quả probe DMA (tĩnh, chỉ `to_dm`)

| Tensor / layout | Byte/slice | Cycle | Util |
|---|---:|---:|---:|
| scale up/gate, 256 slice × 30 hàng (hiện tại) | 7.200 | 4.880 | 0,556 |
| scale, 128 slice × 60 hàng | 14.400 | 4.416 | 0,614 |
| scale, 64 slice × 120 hàng | 28.800 | 4.159 | 0,652 |
| **scale, 32 slice × 240 hàng** `m![L / 240 % 32, 1 # 8]` | 57.600 | **3.557** | **0,762** |
| scale down, chia L (mảnh 480 B), mọi layout | — | 4.961–4.978 | 0,546 |
| scale down, chia H, 60 hàng đủ 960 B | 57.600 | 3.557 | 0,762 |
| weight, mọi layout (30/60/240 hàng, d15, chia H) | — | 24.6k–24.9k | 0,87–0,88 |

- Mapping 240 slice **không phân rã được** (slice = 4 PE × 2 DMN × 32) → dùng 256/128/64/32.
- `cluster_tile` trên DmTensorView **chưa được lower** trong 0.8.1 → dùng `unsafe { view.reshape() }`.
- TRF = 64 KB/slice (một nửa = 32 KB). VRF = 8 KB/slice.

**Bài học từ prototype F3:** bỏ DMA residual trùng làm DMA busy giảm 616 nhưng makespan chỉ
**−186**, vì nó nằm ở tail, nơi giới hạn là tính toán. Bỏ một DMA **giữa chuỗi**
(trước khi DMA down_w xong) mới cho lợi gần trọn giá trị của nó.

### Quy luật chung (đúng cho cả ba kernel, đã kiểm bằng static 0.8.1)

1. **Thứ tự `to_dm` trong source KHÔNG quyết định thứ tự DMA.** Đã thử dời ba lệnh nạp scale
   của QKV lên trước weight → schedule giống hệt từng cycle. Scheduler tự đặt DMA nhỏ ngay trước
   nơi dùng. Muốn đổi thứ tự thì phải đổi dependency, không phải đổi dòng code.
2. **Mỗi DMA nhỏ tốn cố định ~550–930 cycle tĩnh** (≈ 1–2k thật), gần như không phụ thuộc số byte.
   Đòn bẩy là **bỏ hẳn DMA**, hoặc gộp hai việc vào một DMA.
3. **Weight của cả ba kernel đã ở trần util** (0,82–0,88 với mọi layout đã probe). Đừng tốn thời
   gian đổi layout weight.
4. `SwitchConfig::CustomBroadcast` **nhiều khả năng sinh thêm một DMA 16 KB (719 cycle)** — nghi là
   bitmap routing. Bằng chứng: SAO không dùng CustomBroadcast và không có DMA này; QKV dùng 1 lần
   (`hidden::replicate`) và có 1 DMA; FFN dùng thêm 2 lần ở down và có 2 DMA. Chưa chứng minh trực tiếp.
5. **Nhiễu đo giữa hai job control cùng code:** QKV 80.115 / 79.319 (~1%), SAO 37.410 / 37.447,
   FFN 215.569 / 215.195. Chênh dưới ~1% không phải là cải thiện.
6. **`dma_gather_scaled` vào đích replicate compile được nhưng SAI trên HW.** Đích 2 cluster × 256 slice
   lower OK với chi phí 934 cycle, nhưng Arena job 85413 cho Q/K ra rác (~1e14). Comment cũ trong `qkv.rs`
   ("a gather cannot replicate its destination") **vẫn đúng**. Bài học chung: **compile/lower OK không
   chứng minh ngữ nghĩa đúng** — mọi mapping lạ phải qua Arena trước khi xây tiếp lên nó.

---

## 5. Hàng đợi thí nghiệm FFN (làm theo thứ tự)

Ngân sách: FFN 199k thật ≈ **≤ ~100k tĩnh (−8,5k)**. Các F-item dưới đây cộng lại ước ~−4 đến −6k tĩnh.
Phần còn lại phải đến từ hiệu ứng chỉ thấy trên HW thật (F5, F6) → **phải đo Arena sớm**.

### F0 — Đo thực tế trước khi tối ưu (ngày 1, bắt buộc)
- Chạy control trên Arena, lưu log.
- Thử profiling chi tiết: runtime 0.8.1 báo cycle cho mọi span mà image có đặt tên khi
  `FURIOSA_OPT_PROFILE ≥ info`, và có request mức `debug`. Tạo bản copy entrypoint đặt
  `FURIOSA_OPT_PROFILE=debug`, sửa `Collector` trong `src/bin/test_kernels.rs` để in **từng span**
  (tên, cluster, begin, end) thay vì gộp. File `src/bin` không được chấm nên sửa thoải mái.
  Nếu có span theo từng phase → dùng để quyết F5/F6.
- Probe chỉ-DMA thật: biến thể FFN chỉ `to_dm` 6 tensor weight/scale rồi store một tensor nhỏ.
  Output sẽ FAIL, nhưng harness vẫn in cycle. Cho biết sàn DMA thật và real/static của riêng DMA.

### F1 — Block scale up/gate nạp theo layout 32 slice × 240 hàng  ⭐ ưu tiên cao
- Ý tưởng: nạp `up_weight_scale` / `gate_weight_scale` vào
  `DmTensor<f8e4m3, Chip, Cluster2, UpGateRowsPaired, m![L % 240, H / 16]>` (util 0,762, −1,32k mỗi ma trận).
  DMA của hai scale này nằm **giữa chuỗi** → lợi gần trọn **~−2,6k tĩnh**.
- Hai cách đưa scale tới chỗ stage 2 cần:
  - **(a)** Phân phối lại 32 → 256 slice trên **sub context** bằng `switch` (switch generic cho cả
    main lẫn sub; xem `SwitchConfig::InterTranspose` / `Broadcast1`), rồi giữ stage 2 như cũ.
    Không để việc này chiếm main.
  - **(b)** Pool partial BF16 của stage 1 (30 hàng × 240) 8 → 1 slice và chạy stage 2 trên layout
    paired. Output rơi thẳng vào layout GeGLU cần, có thể bỏ `pool_rows`.
    Stage 2 trên mỗi slice dài ~8× → kiểm tra gate stage 2 + GeGLU + encode down có còn nằm gọn
    trong cửa sổ DMA down_w (~24,8k tĩnh) không.
- Không áp cho down scale: chia L luôn cho 0,546, và chia H thì 1920 hàng không chia đều 256 slice.
- Gate static: makespan ≤ 106,5k.

### F2 — Giảm DMA do compiler chèn (LUT / 16 KB)
- Xác định các lệnh 719 cycle / 16 KB (resourcelir `O61`, `O216` trong schedule JSON, có
  `description` rỗng). Nghi là bảng hằng cho `Sqrt` / `Erf`, hoặc spill VRF. Dùng `--dump-visa`
  và thử bỏ lần lượt từng op nghi ngờ trong một bản nháp chỉ để định danh.
- LUT 4 KB được nạp lại trước mỗi lookup (3 lần). Thử:
  - **(a)** gộp stage 1 của up và gate thành **một** op lookup trên `[Dummy2, L % 30, H]`
    (nạp hai weight vào hai tile của cùng một DmTensor bằng `to_dm_view` / `.tile`) → −838 và bớt setup;
  - **(b)** sắp lại để giữa hai op lookup không có op main nào khác.
- Rủi ro (a): gộp làm up không bắt đầu sớm được. Tổng việc main có thể vượt cửa sổ DMA down → tail lộ ra.
  Quyết bằng static.
- Lưu ý: kernel dùng GELU `Erf` trong khi reference dùng `gelu tanh`. Hiện vẫn qua tolerance —
  **không đổi** trừ khi F2 chứng minh bảng Erf là DMA 16 KB và phương án thay vẫn qua tolerance
  trên cả 3 seed.

### F3 — Dùng lại residual đã nạp ở đầu kernel  (đã có patch)
- Patch: [patches/F3_residual_reuse.diff](patches/F3_residual_reuse.diff)
  (`normalize_replicated_keep` + `add_residual_chunks_dm`, view reshape từ Cluster2 sang `Cluster`).
- Static: 108.457 → **108.271** (−186), DMA busy −616.
- Arena job 85413 (gộp chung với Q1): FFN **PASS**, 215.949 so với control 215.569 → **không đo được lợi**,
  nằm trong mức nhiễu. Không ưu tiên; chỉ gộp nếu sau này cần ghép với thay đổi khác cùng khu vực.

### F4 — Dời DMA nhỏ ra khỏi phần tail  (ưu tiên thấp)
- `down_gs`, `post_ff_rms_weight`, `layer_scalar` hiện được xếp **sau** `down_weight_scale` (tĩnh 96–100k).
- **Chỉ dời lệnh `to_dm` lên trên trong source sẽ không có tác dụng** (quy luật 1, mục 4).
  Chỉ thử nếu tạo được dependency thật, ví dụ dùng VRF của tail vào một phép sớm hơn mà vẫn giữ đúng ngữ nghĩa.
  Nếu không tạo được thì bỏ qua F4.

### F5 — Pipeline down theo tile hàng  (REAL-ONLY)
- `down_w` hiện là một DMA; stage 1 down (9,3k tĩnh) chạy sau khi byte cuối về tới.
  Chia 15 hàng/slice thành 3 tile × 5 hàng (mỗi tile liên tục trên HBM, không cần thêm reduce),
  nạp `down_weight_scale` **trước** `down_w`, chuẩn bị scale xong trong lúc down_w đang chạy.
  Dùng `HbmTensorView::tile` + `to_dm_view`, vòng `#[unroll]` của 0.8.1 nếu tiện.
- Static dự báo **trung tính hoặc xấu hơn nhẹ** (stage 1 hiện đã chồng lên down_s + DMA nhỏ).
  Chỉ làm nếu F0 cho thấy trên HW thật stage 1 down bị lộ sau DMA cuối.
- Cảnh báo từ kinhnghiem: SAO chia 2 tile đã thua vì mỗi lệnh DMA tốn ~1–2k cycle thật.
  Tối đa 3 tile.

### F6 — Rút ngắn tail trao đổi cluster  (REAL-ONLY, sau F1–F4)
- Trong tail: `rows.to_dm` (626) + ~1,7k khoảng đồng bộ + `finish_chunks` + post-norm (~1,5k) + store.
- Các hướng: bắt đầu nạp VRF của post-norm (weight, residual) trước khi exchange xong;
  gộp `add partial + × down_gs + round BF16` với bước mean-square của post-norm
  nếu giữ được đúng ranh giới BF16.
- **Không** quay lại hướng trao đổi statistic giữa hai cluster: đã thua (8,02–8,12).

### Không làm (đã chứng minh thua hoặc vô ích)
- Tăng util weight bằng đổi layout (đã ở trần 0,88 trong mọi layout).
- Down chia H (1920 hàng không chia đều 256 slice → mỗi slice gấp đôi việc).
- Mở rộng Time Reducer cho up/gate (ee91a460: 8,28).
- Nạp full scale down lên cả hai cluster (gấp đôi byte, chậm hơn).
- Bỏ lane `lo`, bỏ EPS/global scale, dời ranh giới BF16.

---

## 6. QKV — chẩn đoán và hàng đợi thí nghiệm

Không cần chờ FFN xong. Q1 đã có patch, chạy Arena song song được với F-item.
Code: [best_run_8.4451/src/device/sliding/qkv.rs](best_run_8.4451/src/device/sliding/qkv.rs).

### Chẩn đoán (static 0.8.1)

**Makespan 39.958 · DMA busy 37.343 (93%) · thật 78.092 (×1,95).**

| Đoạn | Cycle tĩnh | Ghi chú |
|---|---:|---|
| Mở đầu: x (690), input-rms weight (690), 16 KB compiler (719) | 1.003 → 3.102 | 719 nghi là bitmap của `CustomBroadcast` trong `replicate` |
| Weight Q | 13.383 (util 0,864) | Ở trần |
| Chen giữa Q và K: `sq` 561, gather cos 934, gather sin 934, `q_rms` 561, **store RoPE lên HBM 343** | ≈ 3,3k | Vòng HBM: gather → pack → store → nạp lại |
| Weight K | 6.991 (util 0,827) | Ở trần; probe 4/8/16/32 hàng/slice đều 0,823–0,83 |
| Chen giữa K và V: nạp lại RoPE 572, `sk` 555, `k_rms` 555 | ≈ 1,7k | |
| Weight V | 6.991 | |
| Tail sau V (35.561 → 39.958) | **4,4k** | `sv` 555 · store q 451 · scatter k 929 · V: project 1.225 → switch 327 → scale 345 → rms ~630 → normalize 345 → scatter v 929 → 600 |

Ngoài ra còn một khối `PeCore` dài 3,5k ngay sau store RoPE (19.897 → 23.404), nghi là C0 chờ C1.
Theo comment trong code và kinhnghiem, một store HBM giữa kernel buộc C0 chờ C1; lần trước bỏ
một store như vậy đo được **90,0k → 84,8k thật**.

### Q1 — Bỏ vòng HBM của RoPE  ❌ ĐÃ ĐÓNG (3 biến thể, không bản nào thắng)

| Biến thể | Cách làm | Static QKV | Arena | QKV thật |
|---|---|---:|---|---:|
| Q1 (job 85413) | gather thẳng vào `m![Dummy2]` × `m![Dummy256]` | 39.585 | **FAIL** q,k | (77.080, output sai) |
| Q1c (job 85461) | gather vào slice 0 của `m![Dummy2]`, pack, DM→DM replicate trong cluster | 39.450 | **FAIL** q,k | (78.751, output sai) |
| Q1b (job 85462) | gather 1 slice (như gốc), pack, **1 DMA DM→DM replicate xuyên cluster** | 42.542 | PASS | **84.881** (+5k) |
| Control (85415 / 85466) | gather → store HBM → nạp lại có replicate | 39.958 | PASS | 80.115 / 79.319 |

Kết luận đã kiểm chứng trên HW:
- `dma_gather_scaled` chỉ đúng khi đích là **một cluster, một slice**. Đích nhiều cluster hoặc nhiều slice
  compile được nhưng ghi sai.
- DM→DM replicate **đúng số học**, nhưng xuyên cluster tốn ~2,5k tĩnh / ~5k thật — đắt hơn store + reload qua HBM.
- Con số "−3k" của Q1 bản đầu đến từ code tính sai, không dùng làm tín hiệu.
- Vòng HBM hiện tại của RoPE là phương án tốt nhất đã biết. Không làm tiếp Q1 trừ khi có primitive mới.

### Q2 — Rút ngắn tail của V (4,4k)
- **(a)** Nhân row-scale `sv` và `unscale` ngay trong epilogue vector của `project_kv`
  (như bản QKV cũ dùng `s_vrf` sau `contract_lane`), bỏ pass `scale_head` riêng cho V (−345).
  **Phải giữ đúng thứ tự** `dot × row_scale × unscale` rồi mới làm tròn BF16 một lần —
  so với `scale_head` hiện tại, đây chỉ là dời vị trí tính, không đổi phép.
- **(b)** Tính mean-square của V head ngay trên layout projection (8 hàng/slice) bằng
  `vector_inter_slice_reduce` thay vì regroup bằng switch trước (−327 switch, có thể −300 rms).
  Rủi ro cao hơn, làm sau (a).
- Ước −0,3…−0,8k tĩnh. Áp tương tự cho K nếu (a) thắng.

### Q3 — Bỏ DMA 16 KB của `CustomBroadcast` trong `hidden::replicate`  (ưu tiên thấp)
- Đã thử `Broadcast01 { slice1: 32, slice0: 8, time0: 1 }` từ layout `LiveRows`: compiler đòi time
  `(H/120, 1 # 8)` và collect `# 64` → switch chở thêm 8× padding, nhiều khả năng đắt hơn 719 cycle tiết kiệm được.
- Chỉ làm nếu tìm được topology thường (Broadcast1 / Broadcast01 / Transpose) cho all-gather
  32 chunk → 256 slice **không** kèm padding thời gian.
  Hàm này dùng chung với FFN (`normalize_replicated`), nên lợi được hai lần.
- Không thay bằng DMA replicate từ HBM: tốn ~2 MB/cluster trên DMA, chậm hơn nhiều.

### Không làm với QKV
- Dời lệnh `to_dm` trong source (quy luật 1 — đã thử, schedule không đổi).
- Đổi layout K/V hoặc Q16 (79104064: 8,27; probe util không đổi).
- Đảo thứ tự weight để Q hoặc K về cuối (tail của Q/K dài hơn V vì có RoPE).
- Bỏ EPS của head-norm, hoặc bỏ unscale bằng lập luận "RMS triệt tiêu scale" — EPS làm nó không triệt tiêu đúng.

---

## 7. SAO — chẩn đoán và hàng đợi thí nghiệm

Code: [best_run_8.4451/src/device/sliding/output.rs](best_run_8.4451/src/device/sliding/output.rs).

### Chẩn đoán (static 0.8.1)

**Makespan 21.337 · DMA busy 17.392 (82%) · thật 37.044 (×1,74).**

| Đoạn | Cycle tĩnh | Ghi chú |
|---|---:|---|
| Nạp activation `xq.to_dm` | 1.003 → 1.936 (**933**, util 0,21) | Mỗi chunk 256 phần tử được replicate cho 16 nhóm hàng → weight phải chờ |
| Weight O | 13.383 (util 0,864) | Ở trần |
| Tail sau weight (15.319 → 21.337) | **6,0k** | projection 1.374 → `y.to_dm` sang C0 446 → PeCore 600 → nạp `residual` 616 → **PeCore 1.184 (chưa rõ)** → tính tail ~1,2k → store 782 → 600 |

`o_weight_scale` và `post_attn_rms_weight` (616 mỗi cái) chạy chồng với projection, không nằm trên đường găng.

### S1 — Nạp activation rẻ hơn  ⭐ ưu tiên cao, đơn giản
- Hiện `xq.to_dm` ghi `Qs % 256` vào cả 256 slice (16 nhóm hàng × 16 chunk) → util 0,21, 933 cycle,
  và weight O phải xếp hàng sau lệnh này.
- Đổi thành: nạp 4.096 phần tử vào **16 slice** (một bản mỗi chunk, không replicate), rồi replicate
  cho 16 nhóm hàng bằng `switch` (Broadcast1) trên **sub**, hoặc gộp vào bước encode hi/lo trên main.
  Encode (~1,4k) vẫn được che bởi 13,4k DMA weight.
- Mục tiêu: DMA x còn ~600 → weight bắt đầu sớm hơn **~300 cycle**, và toàn kernel sớm hơn tương ứng.
- Kiểm tra: trong schedule mới, weight O phải bắt đầu trước 1.700.

### S2 — Làm rõ và cắt 1.184 cycle `PeCore` trong tail
- Dùng `tools/sched_timeline.py` xem node PeCore 17.755 → 18.939 phụ thuộc vào gì
  (input/output tensor trong schedule JSON). Nghi là C0 chờ bản `y` của C1 tới nơi (đồng bộ cluster)
  hoặc chờ báo hoàn tất của DMA DM→DM.
- Nếu là đồng bộ: thử cho C0 bắt đầu phần tail không cần dữ liệu C1 trước khi C1 tới — ví dụ tính
  partial mean-square cho 16 slice của C0 ngay trên dữ liệu local, rồi chỉ chờ C1 ở bước regular32.
  Giữ **đúng một** EPS, đúng 32 partial.
- Nếu là độ trễ DMA: thử chỉ chuyển nửa của C1 (nửa của C0 đưa về slice liên tiếp bằng switch,
  không qua DMA).
- Ước −0,3…−1,2k tĩnh. REAL-ONLY nếu static không đổi.

### S3 — Residual nạp muộn
- `residual_hbm.to_dm` (616) đang nằm sau `y.to_dm` trên hàng đợi DMA, sát đường găng.
  Chỉ đổi thứ tự source sẽ không có tác dụng (quy luật 1). Thử tạo dependency để scheduler nạp nó
  cùng lúc với `o_weight_scale`, ví dụ nạp residual và rms weight trong **cùng một** DmTensor
  (hai tile, hai `to_dm_view`) mà `s_vrf` cũng đọc. Chỉ làm sau S1/S2.

### Không làm với SAO
- Nhân `o_weight_scale` trong epilogue projection **trước** khi làm tròn BF16 (đổi ranh giới làm tròn —
  kinhnghiem yêu cầu round BF16 sau projection, trước row scale).
- Chia weight 2 tile (050943fa: 38.110), tail16/stride16 (3a78a86c, da38678a), full-Q,
  Time64 (e2570c20), split-Q rows16 (28e2c1ba), trao đổi statistic giữa hai cluster.
- Bỏ lane `lo` để "tiết kiệm" — SAO bị giới hạn bởi DMA nên không được gì.

---

## 8. Ước tính cycle và score

Số dưới đây là ước lượng từ static × hệ số real/static, **chưa có Arena**. Không cộng thẳng các con số được,
vì chuỗi DMA dùng chung hàng đợi.

| Kernel | Hiện tại | Bảo thủ | Khả năng cao | Lạc quan | Nguồn chính |
|---|---:|---:|---:|---:|---|
| QKV | 78.092 | 78.000 | 77.000 | 75.500 | Q2 (Q1 đã đóng, xem mục 6) |
| SAO | 37.044 | 36.800 | 36.300 | 35.500 | S1, S2 |
| FFN | 215.459 | 210.000 | 205.000 | 199.000 | F1, F2, F5/F6 |
| **Score** | **8,445** | **8,54** | **8,69** | **8,90** | |

Cập nhật 24/09 sau khi Q1 thất bại: nhắm **~8,7**. FFN (F1) giờ là đòn bẩy chính.
Mức 9,0 cần thêm một hướng mới cho QKV, ngoài những gì plan đang có.

---

## 9. Lịch

Có thể chia việc song song theo kernel: một agent làm FFN (F*), một agent làm QKV + SAO (Q*, S*).
Mỗi agent làm trên copy riêng. Ghép lại vào ngày 28/9.

| Ngày | Agent FFN | Agent QKV/SAO | Đầu ra |
|---|---|---|---|
| 24–25/9 | Setup 0.8.1, F0, Arena cho F3 | Arena cho Q1 (patch có sẵn), làm S1 | `campaign2/F0_*`, `F3_*`, `Q1_*`, `S1_*` |
| 25–26/9 | F1 (a) rồi (b) | Q2 (a), S2 | candidate FFN #1, QKV #1, SAO #1 |
| 26–27/9 | F2; F5/F6 nếu F0 cho phép | Q2 (b), S3; Q3 nếu có thời gian | candidate #2 |
| 28/9 | **Ghép** các nhánh GIỮ vào một crate, chạy static cả ba kernel + Arena | | candidate full |
| 29/9 | A/B 2 lần với control, freeze, **người duyệt → nộp official** | | official #1 |
| 30/9 | Chỉnh lần cuối hoặc nộp lại bản tốt nhất; dừng trước 23:59 AoE | | — |

Khi ghép: F3 và Q3 cùng sửa `shared/hidden.rs`, nên phải merge tay và dump lại cả FFN lẫn QKV.

---

## 10. Báo cáo khi kết thúc phiên

Mỗi agent kết thúc phiên phải cập nhật `campaign2/LOG.md`, mỗi dòng một thí nghiệm:

```text
<ID> | parent | thay đổi | static (FFN/QKV/SAO) | Arena job | PASS? | cycle thật | GIỮ/BỎ
```

Không ghi đè kết quả cũ. Không tuyên bố cải thiện khi chưa có log Arena PASS kèm control.
