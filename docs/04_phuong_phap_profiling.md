# 04 — Phương pháp profiling và đo đạc

## 1. Các lớp dữ liệu

| Lớp | Lấy bằng | Trả lời câu hỏi | Không trả lời được |
|---|---|---|---|
| Lịch tĩnh | `cargo furiosa-opt compile ops::<k> --exact --dump-schedule f.json` (qua `moa.py schedule`) | thứ tự compiler dự kiến, lifetime tĩnh, dòng source; byte cần đối chiếu descriptor | thời gian thật; chưa có hệ số quy đổi clock được xác nhận |
| Summary compiler | `--dump-summary DIR` (`campaign/dump_summary.sh`) | luồng lệnh thật (`fir/resourcelir.fir`), descriptor DMA (`text_form/resourcelir.desc.json`), cấp phát SRAM (`resourcelir.lifetime.json`) | thời gian thật |
| Trace phần cứng | job Arena `--kind harness --diagnostic-trace` trên biến thể `*_profile_ids` (runtime giữ marker ID) | span từng lệnh DMA/TU/Sync theo thời gian thật của mỗi cluster | byte HBM thực, engine nào đang bận |
| Đo hiệu năng | Arena mode info, nhiều lần chạy | cycle để so sánh | nguyên nhân |

## 2. Quy trình tìm điểm nghẽn (đã dùng cho file 05–07)

```bash
# 1. Biến thể profile = source cần đo + runtime giữ marker
cp -r campaign/variants/sdk081_84507_profile_ids <new>; thay src/device + src/ops.rs bằng source cần đo
# 2. Build + job trace (seed 0)
python3 moa.py build --tag <tag> --source-dir .
python3 moa.py submit --bin <sha> --seed-index 0 --label <l> --phase harness --kind harness --diagnostic-trace
# 3. Summary compiler cho cùng source (3 kernel)
bash campaign/dump_summary.sh <variant> <kernel> <tag>-<kernel>
cargo furiosa-opt compile ops::<kernel> --exact --dump-schedule sched/<tag>-<kernel>-summary/schedule.json
# 4. Ghép: span trace → lệnh phát ra → id ResourceLir → dòng source
py -3.12 campaign/trace_join.py campaign/jobs/<job>/log.txt <tag> <kernel> --md out.md
# 5. Descriptor DMA: engine, packet, căn 256
python3 campaign/dma_descriptors.py sched/<tag>-<kernel>-summary/text_form/resourcelir.desc.json --output d.json
# 6. Luồng lệnh (Wait/Sync/PTU nằm ở đâu)
py -3.12 campaign/fir_dump.py sched/<tag>-<kernel>-summary 0 <stop_pid>
```

Cách đọc kết quả:
1. **Mốc**: lúc weight lớn đầu tiên bắt đầu (head); các khe giữa weight; lúc byte weight cuối về (bắt đầu tail).
2. **Đoạn ngoài các span DMA đã thu** mà có TU hoặc Sync: đối chiếu FIR để tìm `Wait`/Sync, dependency hoặc compute sau weight cuối. Union span không phải bộ đếm engine busy, nên chưa đủ để định lượng mức sử dụng DMA vật lý.
3. **Lệnh nhỏ** trên hàng đợi: tra [data/dma_descriptors.md](data/dma_descriptors.md) xem engine và packet. Packet 240/32/8 B, hoặc 1 engine, là ứng viên.
4. **Đối chiếu thứ tự, dependency và cửa sổ từng cluster** giữa lịch tĩnh và trace. Giữ nguyên raw cycles; không dùng “schedule ×2” làm dự báo runtime. Chênh lệch có thể gồm clock domain, hàng đợi, dependency và contention; chưa hiệu chuẩn thì không quy hết về sai số compiler.

Lưu ý:
- Chưa xác nhận timestamp giữa hai cluster dùng chung mốc. Tính `max(end) - min(begin)` riêng từng cluster rồi lấy max; không trừ timestamp của cluster này cho cluster kia.
- Bật trace có thể thay đổi cửa sổ đo (P9: cùng binary SAO, mode trace 38,1K so với mode info 36,8K); mức thay đổi không phải hằng số và có thể khác dấu giữa kernel/job. Dùng trace để hiểu cấu trúc; quyết định bằng A/B ở mode info.
- Không mặc định toàn bộ span DMA là thời gian dữ liệu chạy trên HBM. Span có thể bao gồm thời gian chờ; cần đối chiếu descriptor và luồng Wait/Sync trước khi kết luận băng thông hoặc chồng lấp.

## 3. Đo hiệu năng để quyết định

Nhiễu đo được:
- Cùng binary, mỗi lần chạy lệch SAO ±300–600, QKV/FFN ±1K.
- Giữa các job: FFN có thể lệch tới ~7K (216,7K so với 223,8K) [P9].
- Official lệch ±0,05 điểm giữa các lần nộp cùng source (8,4247 so với 8,4643).

| Phương pháp | Đánh giá |
|---|---|
| Hai job A/B **chạy song song** | **Sai lệch có hệ thống**: candidate có SAO trùng code vẫn thấp hơn control 450–1.060 [Camp] |
| ABBA tuần tự (`campaign/ab_abba.sh`) | Không lệch, nhưng mỗi job chỉ 1 lần chạy/kernel nên vẫn nhiễu |
| **A/B xen kẽ trong một job** (P9 `scripts/opt/ab.sh`, `RUNS=21`, 3 vòng) | Tốt nhất: 42–63 mẫu mỗi bên. Ngưỡng tin: K1/K2 ≥ ~800 hoặc cùng dấu ≥5/6 vòng; A/A của K2 chỉ lệch +26 |
| Overlay 9 lần chạy (`measure.sh`, `harness_overlay.py`) | Để có baseline median/min/p25 |
| Official nhiều lần | Phép đo cuối cùng; chọn bài chốt theo median gộp từ các official |

Bẫy khi build:
- Cargo không relink khi nhiều worktree cùng tên package.
- Cache kernel furiosa-opt chỉ khoá theo tên kernel.

P9 đã sửa bằng `build.sh`: xoá fingerprint, mỗi worktree một thư mục out riêng, `flock`, và in md5 hai bên [P9].

## 4. Công cụ có sẵn trong repo

| Công cụ | Vị trí | Chức năng |
|---|---|---|
| `trace_join.py` | campaign/ | span trace + FIR + schedule → timeline có nhãn source (dùng cho file 05–07) |
| `fir_dump.py` | campaign/ | in luồng lệnh phát ra: Wait, Sync, PTU, LoadSfr, DMA, TU |
| `dma_descriptors.py` | campaign/ | engine, packet, căn 256, payload từng lệnh DMA |
| `dma_chain.py` | campaign/ | chuỗi DMA trong lịch tĩnh, cửa sổ rảnh, tail tĩnh |
| `marker_trace_audit.py` | campaign/ | nối marker với descriptor (SAO/FFN) |
| `ab_abba.sh`, `abba_multi.sh`, `seeds3.sh`, `build_abba.sh` | campaign/ | đo tuần tự |
| `scripts/opt/*` (P9) | packages/score9x…/scripts/opt | `ab.sh` A/B trong một job, `measure.sh`, `crit.py` (tách DMA-only / compute-only / idle), `clusters.py` (cửa sổ từng cluster và thời gian chờ), `desc.sh` |

## 5. Hiệu chuẩn chiến dịch 9_2 — chỉ custom Arena, 27/09/2026

Chiến dịch [top1_s92_469](../campaign/top1_s92_469/README.md) dùng parent `e209da1`,
kernel commit `849283b`, SDK 0.8.1. Không nộp leaderboard và không cập nhật champion official.
Số liệu dưới đây là **Arena proxy**, không xác nhận thứ hạng.

Control PASS 45 seed trong các job `112522`, `112526`, `112532`. Fixture giữ nguyên
bitwise ba seed cung cấp sẵn; 42 seed bổ sung dùng generator gốc. Việc sinh lại ba seed
đầu có một số sai khác làm tròn FFN trên host, nên không thay thế expected đã cung cấp.
Xem [fixture audit](../campaign/top1_s92_469/fixture_audit.json).

| Phép đo | Kết quả thực đo | Giới hạn diễn giải |
|---|---|---|
| Copy 2 MiB HBM→DM→HBM, job `112463` | Readback chính xác toàn bộ 524.288 i32; 20 mẫu ở mỗi warmup 2 và 8 | Đo cả hai chiều và chi phí lệnh; không phải phép đo băng thông đọc thuần |
| Copy, warmup 2 / 8 | Median cửa sổ max theo cluster: 12.268,5 / 11.756 cycle; median thời gian host: 16.140,5 / 15.660 ns | Hai median không tạo thành phép hiệu chuẩn GHz; giữ raw cycles |
| A/A, 3 vòng × 21 mẫu mỗi bên | Max chênh median tuyệt đối QKV/SAO/FFN: **178 / 136 / 317 cycle** | Ngưỡng nhiễu quan sát cho chiến dịch này; không phải cận tin cậy phổ quát |
| Info / trace, job `112435` | Median QKV: 69.589 / 69.614; SAO: 35.243 / 37.153; FFN: 197.358 / 196.173 | Ba mẫu mỗi mode chỉ dùng chẩn đoán; không suy ra hệ số chuyển đổi cố định |
| Full harness, job `112556`, 7 mẫu × 3 vòng | Arena proxy từng vòng: 9,22909 / 9,20995 / 9,19577; median **9,20995** | Mỗi proxy dùng đủ ba kernel trong cùng vòng; không ghép cycle tốt nhất giữa các vòng |

Chi tiết và raw job liên quan nằm trong [P0 results](../campaign/top1_s92_469/p0_results.json).
Lịch tĩnh control là 36.210 / 22.197 / 101.687 cycle; đó là số compiler, không nhân hai
để dự báo runtime hoặc suy ngược băng thông vật lý.

Runtime overlay chỉ thêm ID vào tên span; [cổng hash ảnh NPU](../campaign/top1_s92_469/marker_image_gate.json)
xác nhận cả ba kernel giống control và bản đo info. Trace job `112654` nối được toàn bộ
marker với FIR: 92 marker/cluster cho QKV, 55 cho SAO và 88 cho FFN trong mỗi run.
[Bảng timeline](../campaign/top1_s92_469/marker_trace_join.md) và JSON đi kèm giữ timestamp
gốc, cửa sổ riêng từng cluster, lệnh FIR và lifetime compiler.

Span DMA/VE có thể gồm chờ hàng đợi, dependency và sync. Tổng hợp union của span không
phải bộ đếm engine busy; hai span chồng nhau không đủ chứng minh phần cứng thực thi song song.
Descriptor cho biết packet, địa chỉ tương đối và payload; chưa xác nhận được toàn bộ địa chỉ
HBM vật lý/channel ở runtime. Kết luận nguyên nhân contention cần thêm bằng chứng, còn quyết
định giữ biến thể dựa trên correctness và A/B đảo chiều ở mode info.

## 6. QKV B1 RoPE — custom Arena, 28/09/2026

[Thử nghiệm B1](../campaign/top1_s92_469/block_research/qkv_rope_b1/RESULTS.md) dùng cùng parent
`9_2`, chỉ thay RoPE của Q/K. Thiết kế FMA với hai VRF operand không compile trên SDK 0.8.1.
Bản đo thực tế đưa tích sin từng nửa trực tiếp vào VRF, bỏ buffer DM trung gian và giữ phép
tính FP32 cùng các ranh giới BF16. Số Main pass RoPE tăng từ 6 lên 8.

Control và variant đều PASS 3 seed rồi 45 seed, cùng 7 case biên và kiểm vùng cache ngoài slot.
A/A mới có chênh lệch median `94 / −215 / −173 cycle`, lấy **215 cycle** làm mức nhiễu quan sát.
Sáu cặp AB/BA, mỗi bên 21 mẫu, cho `control − variant` lần lượt
`−16 / +112 / −273 / −182 / −195 / −332 cycle`. Median **−188,5 cycle (−0,273%)**,
chỉ **1/6 vòng** cùng dấu cải thiện; **không giữ B1, giữ control**. Không chạy gate 7×3 dành cho
ứng viên thắng. Không kết luận mức chậm nhỏ này là hồi quy đã vượt nhiễu.

Compiler giữ static makespan **36.210** và payload HBM descriptor **31.617.280 B**, trong khi
số instruction tăng **263 → 276**. Bỏ buffer DM không tự chứng minh giảm latency của cả QKV.
Raw samples và job IDs nằm trong [verdict](../campaign/top1_s92_469/block_research/qkv_rope_b1/verdict.json).
Đây là kiểm chứng **Arena-only**; không submit leaderboard, không tính score tổng ghép kernel.
