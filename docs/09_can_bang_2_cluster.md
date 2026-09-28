# 09 — Hai cluster có chạy cùng hiệu năng không?

**Câu trả lời ngắn:**
- **Phần cứng hai cluster chạy như nhau.** Cùng một lệnh TU lệch nhau vài cycle; stream weight lớn lệch ±2%.
- **Nhưng hai cluster không cân tải.** Cluster 0 (C0) được giao thêm việc mà chỉ C0 làm: operand epilogue, store output, gather rope, post-norm FFN.
  Vì vậy **C0 là đường găng ở cả 3 kernel**, còn C1 phải ngồi chờ **1,3–2,4K cycle ở sync cuối** mỗi kernel.
- Harness báo `max(window C0, window C1)`, nên phần chờ này nằm thẳng trong điểm số.

## 1. Cách đo

- 5 trace phần cứng, gồm image bản best (93413) và 4 image cũ hơn (82301, 80693, 78610, 86192).
- Hai cluster chạy **cùng một chương trình (SPMD)**, nên span cùng marker `bN:eM` là cùng một lệnh. So thời lượng từng cặp bằng `campaign/cluster_compare.py <log…>`.
- Bộ đếm cycle của mỗi cluster có mốc gốc riêng: lệch hằng số ~−317,78 triệu, gần như không đổi giữa các kernel trong một job.
  **Không so mốc tuyệt đối giữa hai cluster.** Chỉ so thời lượng và vị trí tính từ lúc Task của chính cluster đó bắt đầu.
- Harness (`src/bin/test_kernels.rs::window_cycles`) lấy union span của từng cluster rồi **lấy window dài nhất**.

## 2. Phần cứng: hai cluster ngang nhau

Trace 93413 (bản best), tỉ lệ tổng thời gian C1/C0 của các lệnh cùng marker:

| Loại lệnh | QKV | SAO | FFN | Ghi chú |
|---|---:|---:|---:|---|
| TU TuExec | 0,984 (lệch trung vị −4) | 1,103 (−4) | 1,006 (−4) | cùng tốc độ; lệch ở SAO do pass Sub của C1 tranh DMN với DmaStos (xem §4) |
| TU StoVrf / StoTrf | 0,98 / 1,00 | — / 1,49 | 0,98 / 0,98 | |
| TU StoTab | — | — | 0,94 | |
| Stream weight lớn | 1,002 / 0,983 / 1,054 | 1,005 | 1,004 / 0,988 / 0,993 | ±2%, trừ V của QKV +5% |
| Stream scale FFN | — | — | 1,018 / 1,033 / 1,034 | **C1 chậm hơn 2–3%** ở các stream không căn |
| Global scale FFN | — | — | 1,019 / 1,059 | |

4 trace còn lại cho cùng kết quả: TuExec C1/C0 = 0,98–1,02 cho QKV/FFN; stream lớn 0,98–1,02.
Không có dấu hiệu một cluster có clock hay đường HBM kém hơn. Riêng các stream nhỏ/lệch 256 B, C1 luôn chậm hơn vài phần trăm, phù hợp với ghi chú cũ trong `docs/LYTHUYET.md` rằng C0 được ưu tiên băng thông.

## 3. Khối lượng việc: C0 gánh nhiều hơn

Tổng thời gian **DMA nhỏ** (lệnh < 10K cycle), C1/C0:

| Trace | QKV | SAO | FFN |
|---|---:|---:|---:|
| 93413 (best) | 1,022 | **0,501** | **0,722** |
| 82301 | 0,998 | 0,540 | 0,632 |
| 80693 | 1,045 | 0,537 | 0,650 |
| 78610 | 0,995 | 0,542 | 0,596 |
| 86192 | 1,002 | 0,529 | 0,693 |

Nguồn gốc (tra theo dòng source trong [data/](data/)):

| Kernel | Việc chỉ C0 làm | Vì sao |
|---|---|---|
| SAO | nạp `o_weight_scale`, `post_attn_rms_weight`, `residual` (mỗi lệnh 1,1–1,8K) + store output 0,87K | tail đặt trên `LocalCl = m![1 # 2]` (chỉ cluster 0) sau khi gom y. Trên C1 các lệnh này chỉ ~160 cycle |
| FFN | nạp norm weight, residual, layer scalar (~1,2–3,9K) + store; nhận rows từ C1 (DmaStos) rồi làm post-norm | post-FFN tail trên `HiddenChunks` của C0 |
| QKV | gather cos/sin, pack, store `rows0` (rope trên `m![1 # 2]`) | stream K và V của C1 bắt đầu **sớm hơn 1,8–2,0K** so với C0 (tính từ Task của mỗi cluster), vì C1 bỏ qua chuỗi rope |

## 4. Ai chờ ai (tính từ Task của mỗi cluster)

| Trace | Kernel | Window C0 | Window C1 | Báo điểm | Chờ ở sync cuối | Chờ giữa kernel |
|---|---|---:|---:|---:|---|---|
| 93413 | QKV | 85.522 | **85.718** | C1 | **C1 chờ 2.177** | — |
| 93413 | SAO | **37.561** | 37.037 | C0 | **C1 chờ 1.461**, C0 chờ 849 | C0 chờ 1.389 (sau DmaStos) |
| 93413 | FFN | **224.274** | 222.361 | C0 | **C1 chờ 1.925**, C0 chờ 961 | **C0 chờ 3.775** (nhận rows từ C1) |
| 82301 | QKV / SAO / FFN | 84.680 / 37.608 / 221.106 | 84.984 / 36.476 / 219.481 | | C1 chờ 1.909 / 1.265 / 2.009 | FFN: C0 chờ 2.807 |
| 80693 | QKV / SAO / FFN | 85.168 / 37.020 / 220.814 | 86.142 / 37.159 / 219.952 | | C1 chờ 2.443 / 1.433 / 2.133 | FFN: C0 chờ 3.199 |
| 86192 | QKV | 86.201 | 82.483 | C0 | **C0 chờ 1.463** (lần này C1 khởi động muộn) | |
| 86192 | FFN | 224.191 | 222.629 | C0 | C1 chờ 2.401 | FFN: C0 chờ **6.073** |

Cách đọc:
- **Ở cuối kernel, C1 thường là bên về trước và chờ C0 khoảng 1,3–2,4K.** Đó là thời gian C1 để không.
- Ở QKV, window của C1 dài hơn vì phần chờ nằm trong window của nó. Thời gian thật của kernel vẫn do C0 quyết định.
- **Giữa kernel FFN thì ngược lại**: C0 chờ C1 2,8–6,1K ở `ExplicitSync` sau lần chuyển rows. C1 tới điểm này muộn hơn vì stream gate/down của C1 phát muộn hơn 1,0–1,3K (tính từ Task riêng) và stream scale chậm hơn 2–3%.
- **Lệch lúc khởi động là ngẫu nhiên**, cỡ 0–4K. Ở 86192 QKV, C1 khởi động muộn đến mức C0 phải chờ 1,46K.
  Khi C0 đang gánh thêm ~2K việc thì phần việc đó che bớt độ lệch; khi C1 khởi động muộn hơn phần việc đó thì chính độ lệch được tính vào điểm.

## 5. Hệ quả tối ưu

| Kernel | C1 để không ở cuối | Hướng cân tải | Ước lời | Giá trị score |
|---|---:|---|---:|---:|
| SAO | ~1,3–1,9K | Chia epilogue theo cluster: mỗi cluster nạp operand và store **nửa hàng của mình**, chỉ trao đổi statistic RMS thay vì gom y về C0. Hoặc giữ gom nhưng để C1 nạp trước operand của nửa kia. Probe P9 theo chiều ngược (bớt việc cluster muộn): −779 | −0,5…−1K | +0,45…0,9% |
| QKV | ~1,9–2,4K | Làm chuỗi rope (gather, pack, store) trên **cả hai cluster**, mỗi bên cho head của mình, hoặc chuyển sang C1 | −0,8…−1,5K | +0,35…0,65% |
| FFN | ~1,9–2,4K ở cuối; C0 chờ 2,8–6,1K giữa kernel | Epilogue post-FFN chia theo hàng cho cả hai cluster, thay vì gom rows về C0. P9 B2 (DM→DM thẳng vào slice epilogue): −1,4K | −1…−3K | +0,15…0,45% |

Lưu ý:
- Chia epilogue ra hai cluster sẽ cần trao đổi statistic RMS (một scalar/cluster) và thêm sync.
  Số đo cũ của campaign (họ "peer-pair") có lúc thua; trade-off phải đo bằng A/B trong một job.
- Vì độ lệch khởi động ngẫu nhiên, lời thực tế sẽ nhỏ hơn con số C1 để không. Hãy đo median nhiều lần chạy, không nhìn một lần.

## 6. Vì sao hay nghe nói "cluster 0 nhanh, cluster 1 chậm"

Câu này **đúng một phần**. Nó bắt nguồn từ số đo thời SDK 0.6.0 và từ cách đọc trace lúc đó.

| Nguồn gốc | Nội dung | Đánh giá lại với trace 0.8.1 (cả 2 cluster) |
|---|---|---|
| `docs/KINHNGHIEM.md` H5 (0.6.0) | "Cluster 1 luôn xong trễ", median ~10K (5–43K), ở mọi layout | Khi đó **trace chỉ hiện cluster 0**. Span `Cluster` trên C0 được hiểu là "đợi C1", nên mọi thời gian đồng bộ đều bị quy thành "C1 chậm". Nay có trace cả 2 cluster: sync là chi phí riêng (1,4–3,8K); C1 không chậm hơn ở phần việc giống nhau |
| H6 (0.6.0) | "C1 nạp chậm hơn khi TU của C0 rảnh"; thêm 9K TU cho C0 thì không làm kernel dài ra | Phù hợp với **ưu tiên tranh chấp** HBM/DMA nghiêng về C0 |
| `docs/LYTHUYET.md` §7.2 (a) | "C0 được ưu tiên băng thông" (giả thuyết, chưa chốt) | Stream lớn chỉ lệch ±2%, nhưng lệnh nhỏ / lệch 256 B của C1 **chậm hơn 2–6%** (scale FFN, global scale) |
| P9 WORK_LOG | "C1 khởi động muộn hơn 2–4K; C0 đợi C1 ở sync cuối" | Độ lệch khởi động là thật nhưng **ngẫu nhiên**. 86192 QKV: C0 đợi C1 1,46K. Các trace khác: C1 về trước. Không đo trực tiếp được vì hai bộ đếm có mốc riêng |

Cơ chế nghiêng về C1 có thể xác nhận bằng dữ liệu:
1. **Khởi động muộn**: runtime/firmware có vẻ khởi chạy C1 sau C0, lệch 0–4K mỗi lần.
2. **Thua khi tranh chấp tài nguyên chung**: lệnh DMA nhỏ, không căn và stream scale của C1 chậm hơn vài %.
   Ở FFN, stream gate/down của C1 phát muộn dần 1,0–1,3K so với C0 (tính từ Task riêng), đến mức C0 phải chờ rows của C1 **2,8–6,1K** ở giữa kernel.
3. **Chờ đồng bộ bị quy nhầm** thành "C1 chậm" khi chỉ nhìn trace của C0.

Furiosa không công bố thứ tự khởi chạy hay luật arbitration (forum #457: chỉ đo được bằng công cụ nội bộ).
Hai điểm đầu là suy luận từ số đo, chưa phải tài liệu chính thức.

**Nhưng trong code hiện tại, C0 mới là bên về đích muộn**, vì C0 gánh thêm tail/rope. Phần gánh thêm đó lớn hơn thiệt thòi của C1.
Cách cân đúng:
- chuyển bớt ~1K việc từ C0 sang C1, không chuyển hết;
- giữ lại một phần dư cho C0 để che độ lệch khởi động ngẫu nhiên của C1;
- đo median nhiều lần chạy.

## 7. Tái lập

```bash
cd campaign
python3 cluster_compare.py jobs/93413/log.txt jobs/82301/log.txt jobs/80693/log.txt jobs/78610/log.txt jobs/86192/log.txt
```
