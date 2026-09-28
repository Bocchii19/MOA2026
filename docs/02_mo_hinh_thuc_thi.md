# 02 — Mô hình thực thi: chương trình compiler phát ra chạy thế nào trên chip

Phần này giải thích **vì sao kernel có head, khe và tail**, dù băng thông stream đã gần trần. Đây là kiến thức nền cho mọi quyết định tối ưu.

## 1. Chuỗi biên dịch

```
Rust DSL (#[device]) → vISA → LIR → ResourceLIR (lập lịch + cấp phát SRAM) → chương trình phát ra (EDF/binary)
                                            │
                              --dump-schedule: lịch tĩnh (1 GHz, model của compiler)
                              --dump-summary : fir/resourcelir.fir (CBOR, 8 byte header) = luồng lệnh thật
                                               text_form/resourcelir.desc.json = descriptor DMA
```

- Đọc luồng lệnh: `python campaign/fir_dump.py sched/<tag>-summary [start] [stop_pid]` (cần Windows Python 3.12 + `cbor2`).
- Ghép với trace phần cứng: `campaign/trace_join.py`.

## 2. Luồng lệnh phát ra là một dòng tuần tự

Chương trình của mỗi cluster là **một danh sách lệnh chạy theo thứ tự** [Camp EMITTED_PROGRAM_MODEL]:

| Loại lệnh | Ý nghĩa | Chi phí trên phần cứng |
|---|---|---|
| `DmaCommand` | phát một lệnh DMA (bất đồng bộ) | phát ngay; dữ liệu chạy FIFO trên engine |
| `RenegadeCommand` (TuExec/StoTrf/StoVrf/StoTab) | lệnh Tensor Unit trên Main hoặc Sub | chạy trên TU; lệnh cần dữ liệu thì đợi |
| `LoadSfr` | nạp cấu hình (SFR) cho lệnh TU kế tiếp | nhỏ |
| `CoreImpCmd`, `IndexAccess`, `IndexWrite` | tính địa chỉ tile/view trên core vô hướng | ~30–90 cycle mỗi lệnh; ~19 lệnh setup ≈ 0,44K [P9] |
| `Wait{Dma:0}` | **đợi toàn bộ hàng đợi DMA xong** | chặn mọi lệnh phía sau |
| `Wait MainContext` / `Wait SubContext` | đợi context rảnh | chặn |
| `SyncRelease` + `ClusterSync` | bắt tay giữa hai cluster | **1,4–3,8K** trên phần cứng |
| `PageTableUpdate` | ánh xạ lại trang SRAM ảo → vật lý | đợi lệnh TU đang dùng trang cũ (quan sát thấy ~1,5K ở FFN) |

**Hệ quả 1 — `Wait{Dma:0}` quyết định thứ tự.** Lệnh TU nào đọc dữ liệu vừa nạp đều đứng sau một lệnh đợi *toàn bộ* hàng đợi DMA.
Vì vậy compiler buộc phải phát:

```
DMA nhỏ (x, rms, bảng…) → Wait → TU dùng chúng → … → DMA weight lớn
```

Nếu phát weight trước thì lệnh `Wait` sẽ đợi luôn cả weight. Đây là nguồn gốc của **head** trước weight đầu tiên,
và của các **khe** "DMA nhỏ → Wait → TU → DMA nhỏ" giữa các weight.

**Hệ quả 2 — lệnh TU phát trước DMA sẽ trì hoãn DMA đó.** Một pass TU đứng trước một DMA trong luồng lệnh làm DMA
phát muộn thêm ~450 cycle [P9]. Khi TU và DMA cùng bắt đầu ở một thời điểm tĩnh, compiler xếp pass TU lên trước.

**Hệ quả 3 — scheduler tự quyết thứ tự, source không quyết định.**
- DMA được đặt **muộn nhất có thể** so với consumer của nó.
- Weight lớn được phát khi contraction dùng weight trước đó bắt đầu (luật R1 [P9]).
- Đổi thứ tự câu lệnh, thứ tự store, thứ tự toán hạng hay `drop()` cho ra cùng một byte stream lệnh [P9 census].
- Công cụ ép thứ tự duy nhất:
  - *version chain*: ghi vào các tile của cùng một `DmTensor` thì chạy đúng thứ tự chương trình;
  - *data anchor*: tạo phụ thuộc dữ liệu giả;
  - vòng lặp cuộn: tạo barrier toàn phần, rất đắt.
- Biến môi trường của scheduler (`SCHEDULER_MANUAL_ORDERING_PATH`…) có tồn tại, nhưng server official build từ source nên **không dùng được**.

## 3. Đồng bộ giữa hai cluster

| Nguồn sinh sync | Vị trí trong luồng lệnh | Ví dụ trong bản best (trace 93413) |
|---|---|---|
| DM→DM xuyên cluster (DmaStos) | lệnh "reserve" + `ExplicitSync` được xếp ASAP. Nếu nó rơi trước weight đầu thì chặn luôn weight | FFN: ~3K ngay đầu kernel (2.347 → 5.361). SAO: rơi sau weight nên không tốn |
| Chờ phần chuyển DM→DM hoàn tất | ngay sau DmaStos | SAO 1,39K; FFN **3,78K** |
| Mọi lệnh ghi DRAM (store, scatter, `to_hbm`) | "release" ngay sau lệnh ghi, "acquire" ở cuối kernel | SAO: `DramReuse` 2,95K (chồng với tail) |
| Kết thúc kernel | luôn có | 0,85–1,0K |

- Hai cluster khởi động **lệch nhau 2–4K** (cluster 1 muộn hơn) [P9]. Sync cuối kernel để cluster sớm đợi cluster muộn,
  nên kernel = thời gian của cluster muộn + phần lệch còn lại.
- Probe của P9: bớt việc cho cluster muộn ở SAO đo được **−779**.

## 4. Lịch tĩnh so với phần cứng

| Tình huống | Quan hệ tĩnh → phần cứng |
|---|---|
| Bản best, tổng kernel | device ≈ 2 × tĩnh: QKV 1,95, FFN 1,99, SAO 1,74 (đơn vị clock khác nhau) |
| Thêm lệnh DMA / lệnh lookup / tách weight | phần cứng tệ hơn dự đoán ×2 từ **3 đến 9 lần**: rope DM2DM +8K (dự đoán +5,2K), V split +5,4K (+0,6K), predecode +14,6K (+3,6K), ltile3 +35K (+7,3K) |
| Contraction | phần cứng chậm hơn tĩnh ×2 khá nhiều: V contraction QKV tĩnh 1.225 → thực 6.133 (5×) |
| FFN, thay đổi nhỏ | lịch tĩnh có thể **ngược chiều** phần cứng: 4 thay đổi giảm tĩnh đều tăng 5–11K trên phần cứng [P9] |

**Quy tắc:** lịch tĩnh chỉ dùng để *loại* ý tưởng. Quyết định bằng A/B trên phần cứng
(xem [04_phuong_phap_profiling.md](04_phuong_phap_profiling.md)).

## 5. Tóm tắt mô hình thời gian một kernel

```
T_kernel ≈ head                    (DMA nhỏ + Wait + setup + [ClusterSync nếu rơi vào đầu])
         + Σ stream weight          (~bytes / 600–630 B/cycle cả chip)
         + Σ khe giữa weight        (DMA nhỏ ~0,8–1,8K mỗi lệnh, xếp nối tiếp + Wait/TU xen giữa)
         + tail                     (compute sau byte weight cuối + chuyển xuyên cluster + sync + store)
         + lệch cluster             (một phần)
```

Mục tiêu tối ưu thực tế:
1. Giảm **số lệnh DMA nhỏ** nằm trên hàng đợi (bỏ, gộp, căn 256, đặt vào chỗ DMA đang rảnh).
2. Rút **compute sau weight cuối**.
3. Bớt hoặc dời **sync**.

Tăng băng thông stream gần như không còn chỗ.
