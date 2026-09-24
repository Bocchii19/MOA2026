# Kinh nghiệm tối ưu MOA 2026

Tài liệu này giải thích **các cải tiến thực sự có trong source 8,4451**.
Các submission khác chỉ được tóm tắt cách tiếp cận và kết quả để tránh lặp lại thử nghiệm.

**Bản chốt:** `9d65fb99`, evaluation `80179`, SDK **0.8.1**,
Rust **nightly-2026-05-01**. Campaign đang tạm dừng theo yêu cầu;
mục tiêu ≥9,0× chưa đạt.

## 1. Kết quả và source bàn giao

Tất cả output của ba seed official **PASS**. Server lấy median từng kernel:

| Kernel | Cycle official | Ba run trong log | Bit/cycle hữu ích |
|---|---:|---|---:|
| QKV | **78.092** | 81.399 / 78.092 / 76.871 | 3.224,27 |
| SAO | **37.044** | 37.569 / 36.791 / 37.044 | 3.398,41 |
| FFN | **215.459** | 220.933 / 215.459 / 215.459 | 3.695,66 |

## 2. QKV: giảm nạp lặp và bỏ staging trung gian

Code: [sliding/qkv.rs][qkv] và [shared/hidden.rs][hidden].

### Chia weight theo hàng và dùng chung activation

- Hai cluster xử lý các hàng khác nhau. Mỗi cluster dùng **Q32**: 64 slice,
  stride 4, 32 hàng/slice; **K8/V8**: 128 slice, stride 2, 8 hàng/slice.
- Mỗi slice đọc nguyên hàng weight FP8 dài **H=3.840**, căn theo 256 byte.
  Không mở rộng toàn bộ weight sang BF16 trước contraction.
- Input và norm weight được xử lý theo **32 chunk ×120**: tính đầy đủ
  RMSNorm, nhân norm weight, round BF16 rồi all-gather.
- Activation được encode **một lần**, dùng chung TRF cho Q/K/V. Ba projection
  vẫn tính riêng vì weight khác nhau.

**Quantization thực tế:** QKV dùng **một lane FP8 với scale thích nghi**,
không phải fixed64 hi/lo. Scale giải lượng tử:
`unscale = sqrt(max(x²) + 1e-30) / 256`; encode `FP8(x / unscale)`,
sau contraction áp lại unscale. Không suy từ comment hi/lo cũ trong file.

### Head norm và RoPE đi trực tiếp vào VRF

- Giữ mean-square trong DM và giữ EPS; kết quả sqrt của head RMS Q/K/V
  đi trực tiếp vào VRF, bỏ một cặp ghi DM → đọc lại.
- Bốn tích norm-weight/RoPE dùng cho Q/K cũng đi trực tiếp VRF.
- Cos/sin tại vị trí động được pack/load dùng chung; giữ offset động và
  scatter K/V cache. Không loại bỏ input RMSNorm hay thay ranh giới BF16.

**Tác dụng:** giảm các bước chuẩn bị dữ liệu và số lần ghi/đọc trung gian.
Riêng nhánh head-direct-VRF có static QKV **40.742 → 39.958**, instruction
**194 → 187**; cặp custom cùng seed ghi nhận **83.083 → 81.800 cycle**.
Đây là bằng chứng của cải tiến kế thừa, không phải một gain mới cộng thêm
vào kết quả 8,4451. [Báo cáo QKV](campaign/QKV_ROWS_VRF_20260923.md).

## 3. SAO: chia projection, gom tail về C0 và dùng regular32

Code: [sliding/output.rs][sao].
SAO ở đây là **O-projection + post-attention RMSNorm + residual**,
không phải toàn bộ QKᵀ/softmax/AV.

### Projection trên hai cluster

- Chia H=3.840 thành **1.920 hàng/cluster**.
- Mỗi cluster: **16 nhóm hàng ×120**, chiều Qs=4.096 chia **16 chunk ×256**.
  Các slice tính partial rồi reduce 16 chunk.
- Activation dùng **FP8 hi/lo fixed64**, giữ cả phần chính và phần dư;
  weight FP8 đi thẳng vào contraction.
- Giữ rounding **BF16 sau projection**, trước row scale và normalization.

### Thay dataflow của tail

```text
Projection BF16 ở hai cluster
→ gom một lần đủ 3.840 hàng về C0, 32 slice liên tiếp ×120
→ row scale → partial RMS
→ regular Broadcast1(32,1) → full-H RMS + EPS → sqrt vào VRF
→ normalize × norm weight + residual → store
```

Tail cũ giữ output phân tán, phải trao đổi statistic giữa hai cluster rồi
đưa RMS tới các consumer. Bản chốt **di chuyển output một lần sớm hơn**,
để toàn bộ tail chạy trên layout liên tiếp ở C0.

Regular32 thay custom ring256 cho bước gather statistic: không cần bitmap
routing của custom switch, vẫn dùng đủ 32 partial và tổng đúng một EPS
(`EPS/32` mỗi partial). Không bỏ trao đổi liên cluster; đổi dữ liệu và
thời điểm trao đổi để giảm phần việc nằm cuối đường găng.

**Lưu ý DMA:** descriptor của layout regular32 đã cho thấy weight được chia
đều cho tám DMA engine, **1.966.080 byte/engine**. Đây là coverage theo mapping,
không chứng minh cả tám engine luôn bão hòa đồng thời. Tối ưu không nằm ở
việc “bật thêm engine”, mà ở layout, staging và dependency.
[Báo cáo DMA](campaign/SAO_DMA_PIPELINE_20260923.md).

Bản 8,4451 **giữ SAO regular32**, không chứa SAO Time64/tail-unscale,
tail16/tail8 hoặc các nhánh full-Q trọn hàng.

## 4. FFN: NVFP4 trực tiếp, down d15, Time Reducer và partial BF16

Code: [shared/mlp.rs][mlp] và [shared/hidden.rs][hidden].
Entrypoint FFN gọi `shared::mlp::feedforward`, không phải helper cùng tên
trong `device/ffn.rs`.

### 4.1 Up/gate dùng chung encode, weight NVFP4 không materialize BF16

- Hai cluster chia L=15.360 thành **7.680/cluster**; up/gate dùng
  **256 slice ×30 hàng** mỗi cluster.
- Sau full pre-FFN RMSNorm, activation hi/lo fixed64 được encode một lần
  và dùng chung cho up/gate, nhưng hai ma trận vẫn contraction riêng.
- Weight NVFP4 `f4e2m1` được lookup trực tiếp thành FP8 trên đường fetch.
- Stage 1 tạo partial theo **block 16**; stage 2 dùng partial BF16 và
  block scale BF16 để contraction tiếp. Giữ global scale và GeGLU đúng thứ tự.

### 4.2 Down d15 và trao đổi partial qua SRAM

- Đổi từ **240 slice ×16 hàng** sang **256 slice ×15 hàng** mỗi cluster.
  Mỗi cluster vẫn tính một nửa chiều L, không nhân đôi toàn bộ phép tính.
- Activation cho down dùng hi/lo với scale thích nghi, all-gather nội cluster.
- Partial down không còn store/reload qua HBM để ghép hai nửa.
  Dữ liệu chuyển qua SRAM tới layout C0 **32 ×120**.
- Giữ mỗi scalar FP32 trong **ô 8 byte**: một giá trị sống + padding,
  vì DMA stride 4 byte không được hỗ trợ ở đường chuyển này.
- Epilogue cộng hai partial, áp down global scale, round BF16 và pack tại TU.
  Padding không được đưa vào phép tính.

Tức là giảm HBM trung gian nhưng vẫn giữ layout hợp lệ; “pack nhỏ nhất”
không luôn nhanh hoặc được SDK hỗ trợ.
[Báo cáo d15/SRAM](campaign/FFN_D15_SRAM_20260923.md).

### 4.3 Down hi/lo được cộng bằng Time Reducer

Bố trí hi/lo trong chiều Time của contraction thay vì cần một pass Vector
riêng để cộng lane. Vẫn dùng cả hi và lo, không giảm độ chính xác bằng cách
bỏ phần dư.

Thay đổi này bỏ pass lane-sum riêng và giữ Vector/Cast Engine cho sub-context
chuẩn bị block scale song song. Static FFN của nhánh này giảm
**112.167 → 108.451**; cặp custom ghi nhận **229.935 → 219.976 cycle**.
Không áp cùng thay đổi cho up/gate trong bản chốt vì nhánh all-Time đã thua.
[Báo cáo Time Reducer](campaign/FFN_TIME_REDUCTION_20260923.md).

### 4.4 Commit Adapter: thay đổi mới tạo bản 8,4451

**So với source 8,3881, chỉ `shared/mlp.rs` thay đổi.**
QKV, SAO và shared normalization giữ nguyên.

```text
Trước: stage1 FP32 → ghi DM FP32 → stage2 fetch, round BF16 → nhân block scale
Sau:   stage1 FP32 → commit_cast BF16 → ghi DM BF16 → stage2 đọc → nhân block scale
```

Không có phép toán giữa vị trí rounding cũ và mới. Đây là chuyển vị trí
rounding vốn có, không thêm lần quantize hoặc thay toàn bộ thuật toán.

- Lưu partial của up/gate/down bằng BF16 giảm logical SRAM writes
  **44.236.800 → 22.118.400 byte**; reads tương ứng cũng giảm một nửa.
  **Byte weight HBM không đổi**; đây không phải số đo bus counter.
- Bốn partial BF16 tạo packet **8 byte**, giữ alignment tối thiểu.
- Dùng `.commit_trim(...).commit_cast::<bf16>()`, không dùng
  `.cast::<bf16>()` thông thường để thực hiện thay đổi này.
- Cast Engine thông thường làm mất overlap chuẩn bị scale: static
  **112.382**. Commit Adapter giữ overlap: **108.457**, gần như bằng parent
  **108.451**. Vì thế chọn engine chuyển kiểu quan trọng hơn chỉ giảm byte.
- Stage 2 vẫn **8 phần tử sống + padding tới 16**; thử nghiệm packet16
  đầy đủ không thuộc bản chốt.

Custom seed 0: candidate **80127** / control **80128**, cả hai PASS 5/5.
FFN **219.629 / 220.984**, giảm **1.355 cycle (~0,61%)**. Seed 1/2 ở các
job riêng **80156/80157** cũng PASS. Đã freeze provisional khi thấy cải thiện.

Official FFN giảm 218 cycle so với 8,3881; QKV/SAO cũng thay đổi dù code
không đổi. **Không quy toàn bộ tăng score 8,3881 → 8,4451 cho Commit Adapter**,
và chưa coi một cặp A/B là gain ổn định.
[Báo cáo FFN partial BF16](campaign/FFN_BF16_PARTIAL_STORAGE_20260923.md).

### 4.5 Shared RMS và post-FFN RMS

Pre-normalization vẫn tính đầy đủ mean-square + EPS; chỉ bỏ store/reload
kết quả sqrt bằng đường trực tiếp VRF. Post-FFN RMS dùng regular32 trên
32 slice liên tiếp, giữ mean trong DM và sqrt trực tiếp VRF.
Sau đó vẫn đủ norm weight, residual và layer scalar theo entrypoint.

## 5. Các submission khác — cách tiếp cận và kết quả

Các dòng dưới là **những lượt official riêng**, không phải chuỗi ablation
đồng điều kiện. Không cộng gain giữa các dòng hoặc ghép cycle tốt nhất
để suy ra một bản code chưa đo. ID dẫn tới log gốc.

### Các mốc cải tiến

| Submission | Cách tiếp cận chính | Score official / kết quả |
|---|---|---|
| [9866de51](campaign/leaderboard/9866de51.log) | SAO replicate statistic trước trao đổi peer, peer-pair add/sqrt; post-FFN RMS full-ring | **8,0215**; SAO 39.627, FFN 230.257 |
| [6e1decfb](campaign/leaderboard/6e1decfb.log) | Gom full SAO tail về C0, dùng custom ring256 | **8,1197**; SAO 38.763 |
| [d96499b7](campaign/leaderboard/d96499b7.log) | SAO tail C0 dùng regular32 thay custom ring | **8,2063**; SAO 37.136 |
| [bc340e78](campaign/leaderboard/bc340e78.log) | Post-FFN regular32, pre-norm sqrt trực tiếp VRF | **8,2338**; FFN 228.352 |
| [a2e1f3dc](campaign/leaderboard/a2e1f3dc.log) | Down d15, ô FP32 8 byte và trao đổi SRAM | **8,2873**; FFN 222.961 |
| [c423d313](campaign/leaderboard/c423d313.log) | Tinh chỉnh down epilogue/staging, giữ d15 | **8,3057**; FFN 223.309 — score tăng không đồng nghĩa FFN nhanh hơn |
| [f5bc28aa](campaign/leaderboard/f5bc28aa.log) | Down hi/lo Time Reducer | **8,3067**; FFN 216.240 |
| [e2570c20](campaign/leaderboard/e2570c20.log) | SAO Time64, dời unscale tới tail | **8,3400**; SAO 37.277; không đưa SAO này vào bản chốt |
| [187d5107](campaign/leaderboard/187d5107.log) | QKV head RMS/RoPE trực tiếp VRF, giữ SAO regular32 | **8,3881**; QKV 78.975; parent trực tiếp của 8,4451 |

### Các hướng khác không chọn vào bản chốt

| Submission | Cách tiếp cận | Score official / kết quả |
|---|---|---|
| [ad71f1b2](campaign/leaderboard/ad71f1b2.log) | Ghép QKV direct-VRF và SAO Time64 | **8,3667**; không vượt 8,3881 |
| [79104064](campaign/leaderboard/79104064.log) | QKV Q16 thay Q32 | **8,2668**; QKV 81.996 |
| [3a78a86c](campaign/leaderboard/3a78a86c.log) | SAO tail16 ×240 | **8,3336**; SAO 37.973 |
| [da38678a](campaign/leaderboard/da38678a.log) | Trải SAO tail16 stride16 để dùng bốn DMA engine C0 | **8,2462**; SAO 39.324 — tăng coverage không đủ giảm cycle |
| [16aebcec](campaign/leaderboard/16aebcec.log) | Encode activation một lần/chunk rồi broadcast cho các nhóm hàng | **8,3144**; SAO 38.830 — giảm nạp lặp nhưng thêm switch/staging |
| [050943fa](campaign/leaderboard/050943fa.log) | SAO chia hai tile 60 hàng, overlap load/compute | **8,1251**; SAO 38.110 — có overlap nhưng tăng setup/sync |
| [28e2c1ba](campaign/leaderboard/28e2c1ba.log) | SAO split-Q rows16, chia sẻ input và pool output | **7,5556**; SAO 46.706 |
| [ee91a460](campaign/leaderboard/ee91a460.log) | Mở rộng Time Reducer hi/lo sang cả up/gate | **8,2800**; FFN 221.840 — không chọn all-Time |

Nhật ký và các thử nghiệm local khác nằm tại [campaign](campaign/UNLIMITED_20260923.md).
Nhánh SAO full-Q và FFN Commit Adapter + packet16 thử sau đó **không thuộc
ZIP 8,4451**; không coi chúng là bản PASS hoặc cải tiến đã được chốt.

## 6. Bài học và quy tắc giữ lại

- **Đổi dataflow/dependency, không chỉ đảo dòng code.** Giảm store/reload
  hoặc dùng chung tensor chỉ có ích khi rút ngắn tổng thời gian kernel.
- **Giữ overlap giữa main/sub/DMA.** Bỏ một pass nhưng chiếm Vector Engine
  khiến scale chờ vẫn có thể chậm hơn, như đối chứng Cast Engine của FFN.
- **Alignment và layout là điều kiện cứng.** Padding phục vụ chuyển dữ liệu,
  không phải phần tử thật của normalization. Không lách bằng đọc vùng padding.
- **Không có hệ số cố định từ static sang Arena.** Static dùng để hiểu lịch
  và sàng lọc; hiệu quả phải kiểm bằng job độc lập, cùng seed và control.
- **Không bỏ phép toán bắt buộc.** Giữ full RMSNorm, EPS, global scales,
  BF16 boundaries, offset động, reference và tolerance. TeaCache/token
  pruning/aggregation không có điểm reuse trực tiếp tương đương trong ba invocation này.
- **Protocol custom:** một job, một seed, **QKV ×1 → SAO ×1 → FFN ×1**,
  năm output PASS. Seed khác/A-B dùng job khác. Official tự chạy ba seed
  và lấy median; không nhầm score proxy với leaderboard.
- **Freeze khi PASS + cải thiện:** source, binary, hash, seed, log, cycle,
  throughput và sai số ở folder mới. Không ghi đè snapshot/ZIP cũ.

Khi tiếp quản, bắt đầu từ ZIP/source **8,4451**, giữ Cargo.lock và toolchain.
Tài liệu này chỉ cập nhật kinh nghiệm; không sửa kernel, repack ZIP hoặc
chạy thêm Arena. Campaign vẫn tạm dừng.

[freeze-84451]: freezes/20260923T135754Z_sdk081-ffn-commit-bf16-official-best-9d65fb99_job-80127-80156-80157_score-8p445_2d97f5dc/
[qkv]: campaign/submissions/sdk081_ffn_commit_bf16_partials/src/device/sliding/qkv.rs
[sao]: campaign/submissions/sdk081_ffn_commit_bf16_partials/src/device/sliding/output.rs
[mlp]: campaign/submissions/sdk081_ffn_commit_bf16_partials/src/device/shared/mlp.rs
[hidden]: campaign/submissions/sdk081_ffn_commit_bf16_partials/src/device/shared/hidden.rs
