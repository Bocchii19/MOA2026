# Kiểm tra gói rngd-opt-profile-probe — 2026-09-25

Gói trên Drive là **mã nguồn một chương trình chẩn đoán profiling của Furiosa
RNGD**, không phải driver hệ điều hành, firmware hay bản tối ưu Gemma.
Repo đã dùng cùng cơ chế profiling của SDK để đo và chọn các tối ưu K1/K2/K3.
Chưa tìm thấy bằng chứng tích hợp hoặc chạy **chính gói này** trong mã nguồn và
lịch sử Git đã kiểm tra.

## Phạm vi và nguồn

- Nhánh làm việc: `audit/rngd-opt-profile-probe`, tạo từ `opt-next`.
- Commit được kiểm tra: `0aed7745cb1b655fa511356826efcb872d96203e`.
- [Tệp Drive do người dùng cung cấp](https://drive.google.com/file/d/1UiCeoYYTN-p2e418pG43FGP_jo0c2VhZ/view):
  `rngd-opt-profile-probe.tar.gz`, 20.003 byte, sửa lần cuối
  `2026-09-23T08:00:52.873Z` theo metadata Drive.
- SHA-256: `49c11b617a3e0d563a6093c9d95a1d08d4fedb6db04f7d255e73c2ae997c196f`.
- Bản giải nén phục vụ kiểm tra: `/tmp/rngd-probe-audit/rngd-opt-profile-probe/`.
  Đây là thư mục tạm; nguồn lâu dài là tệp Drive và checksum bên trên.
- Danh mục từng tệp, checksum, kết quả tính lại log: [evidence.json](evidence.json).

Đã đọc toàn bộ năm tệp Rust, README, manifest, toolchain và schedule; đối chiếu
SDK 0.8.1 cài tại máy, mã harness, scripts, work log, các commit liên quan và log
Arena có sẵn. Không thực hiện benchmark NPU mới trong lần kiểm tra này.

## Gói chứa gì?

Archive có 10 tệp và một mục thư mục `src/`; không chứa executable, module
kernel `.ko`, firmware, bộ cài hay kết quả benchmark thực tế.

| Thành phần trong archive | Chức năng đã xác nhận từ mã nguồn |
| :-- | :-- |
| `Cargo.toml`, `Cargo.lock` | Crate `rngd-opt-profile-probe`; pin `furiosa-opt-std = "=0.8.1"`, dùng Tokio và tracing |
| `rust-toolchain.toml` | Rust `nightly-2026-05-01`, rustfmt và clippy |
| `src/kernel.rs` | Một hàm `profile_probe`: HBM → DM → HBM; 524.288 số i32, tương đương 2 MiB; không có phép tính Gemma |
| `src/runtime.rs` | Tạo dữ liệu, khởi tạo thiết bị 1 chip/8 PE, upload, launch, đo thời gian; chế độ exact đọc kết quả và `assert_eq!` toàn bộ dữ liệu |
| `src/trace.rs` | Subscriber nhận target `span::npu`; thu `name`, `begin_cycle`, `end_cycle`, `tid`; xuất Chrome Trace JSON ra tệp hoặc stdout |
| `src/main.rs`, `src/lib.rs` | Cài subscriber và gọi bộ chạy; khai báo các module |
| `schedules/probe_schedule.json` | Schedule tĩnh: 5 instruction, makespan 6.047 cycle; DMA load 2.277 và store 2.167 cycle |
| `README.md` | Hướng dẫn build/chạy và mục tiêu kiểm tra clock domain, bandwidth, mức chi tiết của span |

Các con số schedule trên được tính lại từ JSON; chúng **không phải** thời gian
đo mới trên RNGD.

| Biến môi trường | Ý nghĩa |
| :-- | :-- |
| `PROBE_BENCH` | Mặc định tắt; giá trị khác `0` bật benchmark |
| `PROBE_WARMUP` | Mặc định 2 lần làm nóng |
| `PROBE_ITERS` | Mặc định 20 lần đo, tối thiểu 1 |
| `PROBE_TRACE` | Đường dẫn JSON; `stdout` hoặc `-` để xuất vào log; phải đặt để probe cài subscriber |
| `FURIOSA_OPT_PROFILE` | Cần `info`, `debug` hoặc `trace` để yêu cầu NPU profiling |

Benchmark bỏ readback kết quả và không kiểm tra lại dữ liệu mỗi lần; exact mode
mới kiểm tra bản sao. Nên chạy exact trước khi dùng số liệu benchmark.

## Repo đã dùng nó để tối ưu chưa?

**Chưa có bằng chứng dùng trực tiếp archive này.** Trước khi thêm báo cáo,
tìm `rngd-opt-profile-probe`, `rngd_opt_profile_probe`, `profile_probe`,
`PROBE_TRACE`, `PROBE_BENCH` trong các tệp mã/tài liệu liên quan không có kết quả.
`git log --all -G 'rngd[_-]opt[_-]profile[_-]probe|PROBE_TRACE|PROBE_BENCH' -- .`
cũng không có commit khớp. Điều này không chứng minh gói chưa từng được chạy
ngoài repo hoặc trong một lịch sử không còn truy cập được.

**Đã dùng cơ chế tương đương của SDK trong quy trình tối ưu:**

| Bằng chứng trong repo | Vai trò |
| :-- | :-- |
| [test_kernels.rs](../../src/bin/test_kernels.rs), dòng 673–787 | `Collector` nhận `span::npu`, đọc begin/end/cluster và bật subscriber khi profiling được yêu cầu |
| Cùng tệp, dòng 702–713, 810–860 | Tính cửa sổ riêng từng cluster rồi lấy cửa sổ dài nhất; đo các lần chạy và báo median |
| [harness_overlay.py](../../scripts/opt/harness_overlay.py), dòng 31–57 | Thêm tên span và xuất từng span để phân tích timeline |
| [measure.sh](../../scripts/opt/measure.sh), dòng 7–15 | 9 seeded runs; mặc định `info`, tùy chọn `--spans` dùng `trace` |
| [ab.sh](../../scripts/opt/ab.sh), dòng 22–26 | Chạy A/B luân phiên trong một job Arena, dùng `FURIOSA_OPT_PROFILE=info` |
| [crit.py](../../scripts/opt/crit.py) | Phân tích khoảng DMA và `Renegade::*` của cluster 0; nhãn compute ở đây là phân loại theo tên span, không phải bộ đếm utilization độc lập |
| [Trace lưu trong repo](../2026-09-22-codex/CODEX-control2-74421.spans.txt) | Có dữ liệu `Task`, `DMA`, `Renegade::TuExec`, `StoVrf`, `Core` thực tế |

SDK local `furiosa-opt-std-0.8.1/src/backend/npu/function.rs`, dòng 32–69,
xác nhận hai điều kiện: profile level từ `info` trở lên và subscriber chấp nhận
`span::npu`. SDK phát span ở mức tracing INFO kể cả khi yêu cầu mức profile
chi tiết hơn; bộ lọc INFO của probe vẫn nhận các span đó.

## Bằng chứng tối ưu đã áp dụng

Tính lại median trực tiếp từ log gốc, kiểm tra đủ 3 vòng A/B, mọi tiến trình
exit code 0 và không có dòng FAIL/panic. Hai log được sao chép nguyên byte vào
báo cáo để không phụ thuộc thư mục scratchpad.

| Tối ưu đã commit | Job có sẵn | Số mẫu mỗi nhánh | Median A → B | Giảm |
| :-- | --: | --: | --: | --: |
| K1: bố trí trọng số theo 15 strip, `0dd4d87` | 87684 | 63 | 71.973 → 69.741 | 2.232 cycle, 3,10% |
| K3: scale block 16 hàng căn chỉnh, chia sẻ InterTranspose, D1 trên Gate, `7ce2534` | 89090 | 21 | 213.187 → 208.437 | 4.750 cycle, 2,23% |

- [Log K1](ab_vk1s_b.log), [log K3](ab_vk3r_b.log).
- [WORK_LOG.md](../../WORK_LOG.md), mục “K1 strip geometry” và “K3 block scales,
  round 2”, ghi cả phân tích trace và quyết định áp dụng.
- `src/ops.rs` hiện gọi `sliding::qkv_s`; phần K3 đã có trong
  `src/device/shared/mlp.rs`.

Đây là số liệu các lần xác minh lịch sử với control riêng của từng thử nghiệm,
không phải benchmark mới của toàn bộ HEAD, điểm chấm chính thức mới hay mức
tăng hiệu năng do cài gói probe.

## Giá trị bổ sung và giới hạn

Probe hữu ích khi cần tái hiện vấn đề profiling bằng một kernel rất nhỏ,
kiểm tra collector, đối chiếu thời gian host với counter và xuất JSON để xem
timeline. Nó không cung cấp thuật toán Gemma, autotuning, driver mới hay cách
giảm cycle tự động. Không cần thay harness hiện tại bằng probe này.

Mục “Clock domains and the stream ceiling” trong `WORK_LOG.md` đã ghi nhận từ
ngày 2026-09-24 việc counter profile khoảng 2 GHz còn schedule theo clock PE
1 GHz (commit `cbc3ec8`). Đây là kết luận được repo ghi lại từ forum; archive
chỉ có mã thử và schedule, không có số đo hiệu chuẩn để tự chứng minh tỷ lệ đó.
Lần kiểm tra này không truy cập được bài forum `/t/469` qua web và không đo
lại clock, nên không coi đó là một kết quả xác minh độc lập mới.

Trace của probe giữ host theo microsecond và NPU theo cycle ở hai process
khác nhau, nhưng không chuyển đổi hoặc đồng bộ hai trục. Không đọc chiều dài
hai loại event như cùng đơn vị thời gian. Probe cũng chưa tự tính tần số
counter hoặc tổng hợp median NPU từ các lần launch.

Khi tiếp tục tối ưu, dùng trace để hiểu cấu trúc thực thi và dùng A/B cùng
profile level `info` để quyết định: `WORK_LOG.md` dòng 280–282 đã ghi cùng
binary K2 cho khoảng 36,8k ở `info` và 38,1k ở `trace`. Không so trực tiếp số
cycle ở hai mức profile rồi kết luận có speedup.

## Kiểm chứng lần audit này

- Tải đúng file ID, kiểm tra archive path/type trước khi giải nén, lưu checksum
  archive và 10 tệp con.
- Parse schedule và xác nhận 5 instruction, makespan 6.047, DMA 2.277/2.167.
- Đọc code và lịch sử Git; xác nhận các commit tối ưu nằm trong HEAD được audit.
- Tính lại hai log A/B, kiểm tra đủ vòng và exit code; lưu mẫu và checksum.
- Chỉ thêm báo cáo và bằng chứng; không thay kernel, dependency hoặc harness.
- Không build/chạy probe, không cài driver, không gửi job Arena hay submission
  mới trong lần audit này.
