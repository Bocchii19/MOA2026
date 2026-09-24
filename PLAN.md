# PLAN v2 — MOA 2026 Stage 1: chiến dịch official 9.x cho Claude

> Trạng thái chụp ngày 24/09/2026. Đây là tài liệu điều hành hiện hành, thay thế toàn bộ
> kế hoạch 8,7× trước đây. Claude phải cập nhật mục **Checkpoint sống** trước khi sửa code.

## 0. Kết quả cần đạt

Mục tiêu nghiệm thu là một submission official duy nhất đạt **ít nhất 9,0×**; mục tiêu vận hành
là **9,1×** để có biên chống nhiễu. Không được ghép cycle tốt nhất của các job khác nhau để
tuyên bố 9.x. Hạn Round 1 là **30/09/2026 23:59 AoE**; phải dành ngày 30/09 cho xác minh và
khôi phục submission tốt nhất, không dùng toàn bộ ngày cuối cho nhánh mới.

Checkpoint official đã xác minh gần nhất:

| Trường | Giá trị |
|---|---|
| Score | **8,4507×** |
| Submission | `55db7e7e` |
| Median official QKV / SAO / FFN | **77.826 / 37.072 / 215.604** cycle |
| Freeze bất biến | `../freezes/20260923T211438Z_sdk081-rows2-ffn-scalevrf-official-best-55db7e7e_job-82023-82024-82035-82037_score-8p451_e3326caf` |
| Source tree SHA-256 | `e3326cafcaeba762d8ba28aee46473071ccf24937ba204387fd7b8d26bccd348` |
| Custom binary SHA-256 | `0a3e67b09f7f7351b86a5d8425610966329cf5fe661f0cce63e79a411f68e784` |
| Static QKV / SAO / FFN | khoảng **39.958 / 21.346 / 108.457** cycle |

`best_run_8.4451/` trong thư mục này là mốc lịch sử, **không còn là parent mặc định**.
Parent của mọi thí nghiệm mới phải là source exact 8.4507 ở freeze trên, hoặc một champion
official mới hơn đã được freeze và xác minh.

Tại thời điểm viết, official `56375c32` của nhánh SAO CE/TU ones được ghi nhận đang build.
Custom A/B của nhánh đó chậm hơn control trung bình 700 cycle SAO, vì vậy không được coi là
candidate thắng. Claude phải thu đúng ID này trước; **không submit trùng**.

Các nguồn sự thật cần đọc theo thứ tự:

1. `../freezes/INDEX.md` — champion/freeze mới nhất.
2. `../campaign/SAO_CE_RMS_TU_ONES_20260924.md` — job official đang chờ ở checkpoint này.
3. `../campaign/CHAMPION_84507_PROFILE_20260924.md` — profile đúng image của champion.
4. `../campaign/EXPERIMENTS.md` và các báo cáo kernel liên quan — tránh lặp thí nghiệm đã đóng.
5. `../campaign/REPORT.md` — log Arena; các dòng proxy không phải official.

Nếu tài liệu khác mâu thuẫn với freeze/official log, freeze và official log thắng.

---

## 1. Ngân sách score: 9.x đòi hỏi thay đổi kiến trúc

Từ checkpoint 8,4507:

```text
score = 8,4507 × cbrt((77826 / Q) × (37072 / S) × (215604 / F))
```

Để đạt 9,0, tích speedup của ba kernel phải đạt **1,207951×**. Vì vậy vài cải tiến 0,3–1%
không đủ. Kế hoạch phải lấy phần lớn lợi ích từ SAO và thêm gain thật ở QKV/FFN.

| Gói | QKV | SAO | FFN | Score dự kiến | Vai trò |
|---|---:|---:|---:|---:|---|
| Ngưỡng mong manh | 76.000 | 32.000 | 212.000 | 8,9964 | Không đủ để chốt |
| **Gate 9,0** | **76.000** | **32.000** | **210.000** | **9,0248** | Mức tối thiểu |
| **Đích vận hành** | **75.000** | **31.500** | **210.000** | **9,1125** | Có buffer |
| Stretch | 74.000 | 31.000 | 205.000 | 9,2765 | Chỉ sau khi 9,0 đã có |

Nếu QKV và FFN đứng yên, SAO phải xuống **30.690** chỉ để chạm 9,0. Vì vậy:

- SAO là đường găng: cần khoảng **−5,1k cycle thật** để về 32k.
- QKV cần khoảng **−1,8k** để về 76k.
- FFN cần khoảng **−5,6k** để về 210k.

Sau mỗi kết quả, Claude phải tính lại frontier bằng số đo mới, không giữ target cũ một cách
máy móc:

```sh
python3 ../campaign/moa.py score <QKV> <SAO> <FFN>
```

---

## 2. Luật cứng

1. Giữ nguyên tên, chữ ký, tham số và kiểu trả về của các hàm `#[device]` được chấm.
2. Chỉ thay đổi source nằm trong phạm vi server chấp nhận. Mỗi candidate phải audit đủ 28
   device files và 19-file submission scope bằng tooling campaign hiện có.
3. Giữ đầy đủ RMSNorm, EPS, global scale, dynamic offset, RoPE/KV semantics, GeGLU, cả hai
   lane hi/lo và mọi ranh giới BF16 của parent, trừ khi thí nghiệm ghi rõ đây là thay đổi
   precision và vượt qua host adversarial screen trước khi lên NPU.
4. Không special-case fixture, seed, padding hay giá trị input. Padding không được tham gia math.
5. Không sửa freeze, ZIP hoặc champion tại chỗ. Mỗi giả thuyết dùng một copy/source tree và
   một target build riêng.
6. Compile PASS không chứng minh đúng số học; host PASS không chứng minh NPU PASS; static giảm
   không chứng minh cycle thật giảm; custom proxy không phải official.
7. Không gọi khoảng command là engine-active bandwidth. Không cộng B/c của hai clock cluster.
8. Chỉ một official được phép live. Job không rõ trạng thái phải `collect`/tra cứu đúng ID,
   tuyệt đối không submit lại.
9. Điểm cuối vòng là submission cuối hợp lệ trước hạn. Trước khi dừng campaign phải submit lại
   đúng source đã chọn và xác minh kết quả, không để một nhánh thử chậm hơn thành submission cuối.
10. Không tuyên bố đạt 9.x cho tới khi log official có score ≥9,0 và all outputs/all seeds PASS.

---

## 3. Bootstrap bắt buộc trước khi tối ưu

Claude thực hiện các bước này và ghi kết quả vào `campaign2/STATE.md`.

### 3.1. Đồng bộ checkpoint sống

- Kiểm tra `../freezes/INDEX.md`, các log leaderboard mới nhất và official đang live.
- Thu `56375c32` nếu còn pending; không tạo submission thay thế.
- Nếu có official mới >8,4507: xác minh all PASS, freeze/source hash/binary evidence rồi dùng nó
  làm parent. Nếu không, giữ exact 8.4507.
- Ghi rõ: champion official, source hash, binary/image hash, median ba kernel, static makespan,
  job đang live và timestamp UTC.

### 3.2. Tạo control và candidate độc lập

Chạy từ thư mục chứa file PLAN này. Không ghi đè nếu thư mục đã tồn tại:

```sh
mkdir -p campaign2
cp -a ../freezes/20260923T211438Z_sdk081-rows2-ffn-scalevrf-official-best-55db7e7e_job-82023-82024-82035-82037_score-8p451_e3326caf/source \
  campaign2/control_84507
cp -a campaign2/control_84507 campaign2/<ID>_<slug>
```

Trước khi dùng control, chạy source hash, build và schedule bằng runner hiện có:

```sh
cd campaign2/control_84507
python3 ../../../campaign/moa.py srchash
python3 ../../../campaign/moa.py schedule --tag control-84507-refresh
cargo furiosa-opt build --release --bin test_kernels
```

SDK bắt buộc: `furiosa-opt-std 0.8.1`, `cargo-furiosa-opt 0.8.1`,
`nightly-2026-05-01`; giữ nguyên `Cargo.lock`. Nếu rebuilt binary khác binary đã đo, đó là binary
mới và phải đo lại; không gán kết quả cũ cho nó.

### 3.3. Lập lower bound trước khi viết code

Cho từng kernel, từ schedule JSON và marker profile hãy lập bảng:

```text
critical path = transfer bắt buộc + compute bắt buộc + exposed sync/setup + final store
headroom = current makespan - lower bound
```

Chỉ mở một nhánh nếu gain cần thiết không lớn hơn headroom hợp lý, hoặc nhánh thay đổi dependency
graph để hạ lower bound. Việc cộng các duration đang overlap là sai.

---

## 4. Vòng lặp thực thi cho mọi giả thuyết

Mỗi candidate phải có `RESULT.md` ngay khi tạo, không đợi tới cuối.

### Gate A — Hypothesis card

Trước khi sửa code, ghi tối đa 15 dòng:

```text
ID / parent hash / kernel
Nút critical-path bị loại hoặc được overlap
Cycle tĩnh và cycle thật kỳ vọng
Dependency graph thay đổi thế nào
Byte/command/instruction thêm và bớt
Ranh giới số học phải giữ
Failure mode dự kiến
Điều kiện GO / KILL
```

Không được mở nhánh với lý do chung chung như “ít instruction hơn” hoặc “DMA nhanh hơn”.

### Gate B — Source và host correctness

- Diff chỉ chứa kernel/hàm dự kiến và test hỗ trợ.
- Thêm test layout/address/coverage; poison padding nếu có padding.
- Với thay đổi reduction order hoặc BF16 boundary, chạy fixture 3 seed và các ca zero, tiny,
  mixed-sign, extreme, sparse; checker/reference/tolerance phải giữ nguyên.
- Fail numerical gate thì đóng ngay, không nới tolerance.

### Gate C — Compiler/schedule

```sh
python3 ../../../campaign/moa.py schedule --tag <ID>
```

Ghi makespan, instruction count, DMA load/store count, thời điểm bắt đầu/kết thúc các node bị
ảnh hưởng và critical path mới. Source order không phải schedule order; chỉ dependency phát ra
mới là bằng chứng.

Quy tắc cấp Arena:

- Micro-change: static kernel phải giảm ít nhất **0,8%** hoặc loại một node đã được marker profile
  chứng minh là exposed.
- Architectural change: được một cặp hardware screen dù static trong khoảng **−0,5%…+2%**,
  nhưng `RESULT.md` phải giải thích điều mà static model không quan sát được.
- Static chậm >2% và không có thay đổi HW-only cụ thể: KILL, không Arena.

### Gate D — Hardware screen

Build một lần, khóa source hash, rồi chạy hai job độc lập cùng seed:

```sh
python3 ../../../campaign/moa.py build --tag <ID>
python3 ../../../campaign/moa.py submit --bin <candidate_sha> --seed-index 0 --label <ID>-cand-a --phase screening
python3 ../../../campaign/moa.py submit --bin <control_sha>   --seed-index 0 --label <ID>-ctrl-a --phase screening
```

Nếu candidate đúng và có tín hiệu, chạy cặp đảo thứ tự control→candidate. Không chạy đồng thời
hai submit trên ledger dùng chung. Mỗi job phải đúng ba launch: QKV×1, SAO×1, FFN×1.

### Gate E — Promotion

Một candidate chỉ được ghép khi:

- Tất cả output PASS, max error không xấu đi ngoài biên parent đã chấp nhận.
- Trung bình hai cặp A/B giảm kernel mục tiêu ít nhất **1,0%**; không cặp nào chậm >0,5%.
- Hai kernel không sửa không regression có hệ thống; cycle dao động của code không đổi không được
  gán thành gain của candidate.
- Seed 1 và 2 PASS cho chính binary/source đã khóa.
- Source/image hashes và submission scope khớp.

Gain nhỏ hơn 1% có thể lưu làm mảnh ghép nhưng không được dùng làm lý do dừng một workstream
kiến trúc. Mọi kiến trúc mới PASS custom phải tuân thủ chính sách official hiện hành của campaign;
chỉ submit khi không có official khác live.

---

## 5. Backlog ưu tiên để đạt 9.x

Thứ tự mặc định: **SAO architecture → FFN architecture → QKV tail → ghép**. Claude không được
dành hơn hai thí nghiệm liên tiếp cho micro-gain dưới 1% khi SAO vẫn >34k.

### S0 — Đo lại SAO exact 8.4507 và khóa budget

Mốc profile hiện tại: weight kết thúc khoảng 27.716 trace cycle; kernel kết thúc 37.608, tức
khoảng 9.892 cycle sau weight. Đoạn này gồm projection, row transport, RMS, epilogue, store và
sync — không phải 9.892 cycle idle.

Đầu ra bắt buộc:

- dependency DAG từ cuối weight tới final store;
- mỗi node ghi cluster/context, begin/end, input/output buffer;
- lower bound cho 32k;
- xác nhận node nào thật sự exposed trên cả static và trace.

### S1 — SAO producer-local fused scale + statistic, không full-row gather

Đây là nhánh có đòn bẩy lớn nhất. Mục tiêu là bỏ `y.to_dm` đưa toàn bộ 3.840 row FP32 về C0 và
không thay nó bằng một full-row copy khác.

Thiết kế cần thử theo thứ tự:

1. Giữ row trên hai cluster đã tạo chúng.
2. Trong epilogue projection hoặc pass đầu tiên ngay tại producer, áp `o_weight_scale` và tạo
   local sum-square từ chính giá trị sẽ được normalize.
3. Trao đổi **chỉ hai scalar/packet tổng hợp đã căn hàng**, dùng primitive đã chứng minh đúng;
   mỗi consumer nhận đúng tổng hai cluster, cộng EPS đúng một lần rồi sqrt.
4. Chạy norm-weight/residual/output local trên mỗi cluster; store đúng nửa H tương ứng.

Điểm mới bắt buộc so với các nhánh peer/local cũ: không compact 1.920 row bằng DMA trước khi tính,
không CustomBroadcast bitmap, không thêm một row-copy pass, không dùng padding làm statistic.
Nếu SDK buộc phải thêm full-row redistribution, đóng S1 thay vì lặp `local_dense16_peer`.

Gate S1: static SAO **≤19,5k** hoặc marker DAG loại/overlap được ít nhất 2,5k exposed cycle.
Mục tiêu hardware: **≤33,5k** ở vòng đầu, sau đó ≤32k.

### S2 — SAO fuse projection output với row-scale/RMS producer

Nếu S1 bị giới hạn bởi việc vừa lưu `y` vừa tạo statistic, thử một tensor hai plane/pair-output:

- plane 0 giữ row cần cho epilogue;
- plane 1 giữ partial `sum((y×scale)^2)` theo nhóm row;
- chỉ commit dữ liệu live, căn 8/32 byte hợp lệ;
- tránh pass đọc lại toàn bộ `y` chỉ để tính RMS.

Không lặp `packed_producer_stats`: bản cũ thêm row copy và chậm static. Candidate mới chỉ hợp lệ
nếu statistic được sinh trong pipeline đã đọc row, không tạo một full-vector pass phụ.

### S3 — SAO wavefront có điều kiện

Chỉ mở khi S0 chứng minh projection/tail chờ toàn bộ weight. Mục tiêu là cho nhóm row đầu bắt đầu
projection và tail trong khi phần weight sau còn được nạp.

Không lặp tile60/TRF4/TRF8 đã thua. Thiết kế mới phải thỏa cả ba:

- không tăng quá một large-DMA command;
- không materialize weight BF16/TRF toàn ma trận;
- schedule tổng giảm ít nhất 1,5k static trước Arena.

Nếu không đạt, đóng hướng wavefront và quay lại S1/S2.

### F1 — FFN paired up/gate stage1-stage2

Current FFN cần khoảng −5,6k cycle thật. Thử xử lý up và gate như hai plane của cùng một pipeline
để amortize LUT/setup, scale preparation và stage2 dispatch, nhưng vẫn giữ hai tensor weight và
hai global scale độc lập.

Yêu cầu:

- giữ BF16 block16 partial boundary của champion;
- không trì hoãn `down_weight` so với parent;
- không materialize full BF16 weight;
- báo riêng thời điểm gate weight, down weight, GeGLU và down stage1.

Đây không phải `wavefront_upgate` cũ (đổi thứ tự nhưng chậm), cũng không phải `upscale_geglu_fused`
(bỏ một BF16 boundary nhưng chậm). Gate: static FFN **≤106,0k** hoặc loại một node exposed được
profile chứng minh. Nếu fused pipeline làm down weight bắt đầu muộn, KILL.

### F2 — FFN direct two-cluster partial join + tail fusion

Sau down stage2, profile còn row SRAM và tail dài. Thử để mỗi output row nhận đúng hai partial
cluster trong một aligned pair, rồi trong một vector chain:

```text
add two partials → × down_global_scale → BF16 boundary → post-RMS statistic/output preparation
```

Mục tiêu là bỏ ít nhất một DM materialization/pass trong `finish_chunks`, không bỏ BF16 boundary,
EPS, residual hay layer scalar. Nếu vẫn cần cùng `rows.to_dm` cộng thêm pass mới thì không có
hypothesis gain và phải đóng.

### F3 — Chỉ xem lại down tiling khi loại được overhead cũ

Ba tile 5-row đã PASS nhưng chậm 3.335 cycle thật; two-tile từng vướng SDK view/store. Không mở
lại nếu vẫn có 25 DMA loads hoặc scale/lookup setup cho từng tile. Chỉ thử khi có thiết kế dùng
chung scale/LUT và giảm command count gần parent 19 loads; static phải ≤106k.

### Q1 — QKV fuse `scale_head` vào projection epilogue

Làm V trước vì tail V nằm cuối critical path. Đưa `row_scale × encoder_unscale` vào epilogue của
`project_kv`, với đúng thứ tự `dot × row_scale × unscale → BF16` một lần. Sau khi V thắng mới áp
cho K, rồi Q.

Phải kiểm tra việc preload `sv/sk/sq` có làm weight DMA bắt đầu muộn không. Source reordering đơn
thuần không đủ. Gate: giảm ≥400 static cycle hoặc loại pass exposed đã profile; target hardware
V-only ≥0,7%, cả K/V ≥1,5%.

### Q2 — QKV packed K/V common tail

K và V có cùng hình học `Ps×H`. Thử giữ raw K/V trong hai plane và dùng chung dispatch/setup cho
scale/unscale trước khi tách: K đi head RMS+RoPE+scatter, V đi head RMS+scatter. Không thay reduction
order và không ép K/V dùng cùng scale.

Đây phải là fusion thật trong emitted schedule, không chỉ macro/source cleanup. KILL nếu instruction
giảm nhưng makespan/weight gap không đổi.

---

## 6. Danh sách đóng — không lặp nguyên trạng

Claude phải search báo cáo tương ứng trước khi đề xuất họ gần giống. Các hướng sau đã có bằng chứng
compile hoặc hardware thua:

- QKV RoPE gather trực tiếp tới nhiều cluster/slice: compile được nhưng sai trên HW; DM→DM replicate
  đúng nhưng chậm khoảng 5k thật.
- QKV norm 30×128 và 16×240 gather: PASS nhưng chậm; rotate interleaved/sub-copy và one-pass absmax
  encoder cũng thua static.
- SAO tile60, TRF4/TRF8 weight staging, q-major, packed Q-reduce, CE-BF16/CE-TU ones, root multicast,
  local-dense peer, producer-stats có row copy: không mở lại nếu dependency graph không khác.
- FFN aligned scale reshard, 32-row DMN ownership, compact partial TRF, Rows6, upgate Time,
  wavefront-upgate, upscale/GeGLU fusion và tile5 down: đã thua hoặc không có gain ổn định.
- Đổi layout weight chỉ để tăng util khi weight đã khoảng 0,82–0,88; đổi thứ tự `to_dm` mà không
  đổi dependency; bỏ lane `lo`, EPS/global scale hoặc dời BF16 boundary không qua precision screen.

Một hướng đóng chỉ được mở lại nếu hypothesis card nêu chính xác bằng chứng mới và dependency/node
được loại khác bản cũ.

---

## 7. Ghép candidate và chốt 9.x

Không ghép sớm. Chỉ ghép các candidate đã qua Gate E độc lập.

Thứ tự ghép:

1. SAO winner lên exact champion.
2. FFN winner lên kết quả bước 1; kiểm tra image QKV/SAO không đổi ngoài dự kiến.
3. QKV winner lên kết quả bước 2.
4. Build sạch, schedule cả ba, host full suite, custom seeds 0/1/2.
5. Chạy hai cặp A/B seed0 với exact parent gần nhất; tính score trên từng **triplet cùng job**.
6. Audit accepted source, tạo official overlay và submit một lần.

Điều kiện chốt:

- QKV/SAO/FFN cùng một official có all PASS.
- Score official ≥9,0; ưu tiên tiếp tục tới ≥9,05 nếu còn thời gian để có buffer.
- Freeze chứa manifest, source hash, binary/image hashes nếu có, official log, custom A/B logs,
  schedule JSON, diff và `RESULT.md`.
- Tạo ZIP mới `best_run_<score>.zip`, không ghi đè ZIP cũ.
- Cập nhật `../kinhnghiem.md`, `../freezes/INDEX.md`, `../campaign/REPORT.md` và checkpoint đầu file này.

Lịch điều hành:

| Ngày | Trọng tâm | Exit criterion |
|---|---|---|
| 24/09 | Đồng bộ official, S0, lower bound | Control/hash/schedule/profile đã khóa |
| 25–26/09 | S1 rồi S2; song song logic F1 nhưng không dùng chung source | Có một kiến trúc giảm ≥1% hoặc bị KILL có bằng chứng |
| 27/09 | F2 và Q1; chỉ chạy Q2 nếu Q1 có tín hiệu | Các mảnh ghép đã qua Gate E |
| 28/09 | Ghép theo thứ tự SAO→FFN→QKV | Candidate full, host/schedule/Arena PASS |
| 29/09 | A/B xác nhận, audit, freeze, official | Official 9.x hoặc best fallback đã xác minh |
| 30/09 | Buffer/rollback | Submission cuối là source được chủ động chọn |

Nếu có nhiều agent, mỗi agent phải dùng source tree và target riêng; chỉ một coordinator được ghi
ledger và submit/collect job. Không để hai agent sửa cùng `shared/mlp.rs`, `REPORT.md` hoặc ledger.

Nếu tới 29/09 vẫn chưa có candidate package dự kiến ≥9,0, dừng micro-tuning và chỉ làm hai việc:

1. thử kiến trúc có headroom đủ lớn nhất còn lại;
2. bảo vệ submission official tốt nhất đã xác minh để không mất điểm cuối vòng.

---

## 8. Format báo cáo bắt buộc

`campaign2/STATE.md` luôn bắt đầu bằng:

```md
# Live state
- UTC:
- Official champion / score:
- Official medians QKV / SAO / FFN:
- Parent freeze / source hash:
- Active compiler/build/custom/official IDs:
- Current score frontier:
- Next single action:
```

Mỗi `campaign2/<ID>_<slug>/RESULT.md`:

```md
# <ID> — <slug>
- Parent source/image hash:
- Kernel và file thay đổi:
- Hypothesis / node critical-path:
- Numerical invariants:
- Host tests:
- Static parent → candidate; instruction; DMA load/store:
- Candidate binary/image hash:
- Arena jobs và thứ tự chạy:
- QKV / SAO / FFN từng job:
- Candidate/control ratio của kernel mục tiêu:
- Official submission/result (nếu có):
- Quyết định: PROMOTE / KEEP-AS-FRAGMENT / KILL
- Lý do và hướng kế tiếp:
```

Cuối mỗi phiên Claude phải:

1. cập nhật `STATE.md` và `RESULT.md` đang làm;
2. append một dòng vào `campaign2/LOG.md`, không sửa lịch sử;
3. liệt kê rõ process/job còn live và ID cần collect;
4. nêu champion official, không gọi proxy/Arena là official;
5. để lại đúng **một hành động kế tiếp có thể chạy ngay**.

Claude không được dừng ở việc đề xuất patch. Với mỗi item được chọn, phải đi tới một terminal state:
`KILL có bằng chứng`, `PASS nhưng chưa promote`, `PROMOTE`, hoặc `official terminal`.
