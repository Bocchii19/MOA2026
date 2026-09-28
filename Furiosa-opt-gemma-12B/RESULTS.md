# ver16_8.4x_duc — QKV head chia 4 PE + AO/FFN tinh chỉnh tiếp — 8.4051×

Tất cả 3/3 kernel **PASS** (cả 3 lượt chạy của harness), `max|Δ|` và `mean|Δ|` giống hệt ver14 từng bit.
Toolchain **furiosa-opt 0.8.1**. Nguồn = ver14 (`ver14_8x_duc`) + 3 nhóm thay đổi (QKV, AO, FFN).
(Mẹo AO của ver15 chưa có trong bản này: trên epilogue 32 slice của ver16 thử tĩnh không lợi; đang đo trên phần cứng.)

| Kernel | ver14 (`a0af0cea`) | ver15 (`5f9689dd`) | **ver16** (`16b602fe`) | Speedup ver16 |
| :--- | ---: | ---: | ---: | :---: |
| `sliding_project_qkv` | 78,387 | 78,108 | **74,551** | **3.360×** |
| `sliding_attention_output` | 41,229 | 39,545 | **38,630** | **10.475×** |
| `decoder_feedforward` | 224,030 | 224,291 | **219,533** | **16.870×** |
| **Score (bảng chính thức)** | 8.0337× | 8.1527× | **8.4051×** ⭐ | |

Lần chấm thứ 2 cùng mã nguồn: `b1e5c347` = 75,162 / 39,144 / 217,513 → **8.3711×**.
Arena (median 3 lượt): job 79793 = 74,550 / 39,875 / 216,369 (8.357), job 79795 = 74,841 / 40,879 / 216,992 (8.269).
Baseline của bảng: 250,514 / 404,633 / 3,703,473.

## Thay đổi so với ver14

**QKV (−5.1%)** — file mới `src/device/sliding/qkv_h4.rs`, `qkv_h4_norm.rs`, `rope_h4.rs` (+ `mod.rs`, thân QKV trong `ops.rs`)
1. **Head chia ra 4 PE** (`KvHeadsAcrossSlices = [Ns % 4, 1 # 64]`), 3 lệnh gom head dùng `Broadcast1{64,1}`
   (ring 64) thay `TransposedBroadcast1` ring 256: 785 thay vì 2,427 cycle mỗi lần. Đuôi V 11.6k → 8.3k,
   hết kẹt ở ngắt điều khiển clock trước contraction V (0/16 lần, trước là 7/16).
2. **Nạp x và trọng số RMS đầu vào nhân bản 8 lần** (thay 32 lần) + all-gather `Broadcast01{4,8,30}`;
   trọng số RMS nạp trước + phụ thuộc "cộng 0" chính xác để weight K được phát ngay sau x.
3. **Gom Q và V thẳng theo thứ tự kênh** trong một pass (Transpose Engine đóng gói) — −0.4..−0.5k mỗi lần.
4. sqrt của V ghi VRF từ Main; một lần nạp TRF dùng cho cả K, Q, V (weight K/V reshape lên trục Q).

**AO (−2.9%)** — `src/device/sliding/projection.rs` (+ một chú thích kiểu trong `ops.rs`)
5. Weight O tile **24+6** (thay 26+4): tile đầu chừa chỗ trong trang DM 16 MiB đầu cho buffer scale → lệnh nạp
   scale chạy trong lúc contraction cuối.
6. **Chuỗi 24 pass copy đồng nhất** trước bước tách activation (không đổi giá trị) để bộ lập lịch xếp lệnh DMA
   tile 6 hàng lên trước → tile cuối về sớm hơn ~1k.
7. Epilogue trên **32 slice × 120 kênh** với all-gather `Broadcast1{32,1}`.

**FFN (−3.7%)** — `src/device/shared/mlp.rs` (+ một chú thích kiểu trong `ops.rs`)
8. Contraction các nhóm Down dùng **packet 64 B + `LaneMode::Sequential`** (4 tổng block mỗi flit xuất thay vì 1):
   các pass nhóm nhanh gấp 2 (5.7/6.5/8.0k → 2.8/3.6/3.6k).
9. Scale Down của các hàng 4..11 nạp bằng **một** lệnh DMA 8 hàng (−4.3k).
10. Epilogue trên **16 slice × 240 kênh** với `Broadcast1{16,1}` (−1.3k); sqrt ghi VRF từ Main.

Mọi switch config mới được kiểm bằng validator của compiler (`config_switch`) trên máy trước khi chạy phần cứng.

## Chạy thử
```bash
export PATH=$HOME/fo081/bin:$HOME/.cargo/bin:$PATH   # cargo-furiosa-opt 0.8.1
python3 scripts/generate_references.py               # tạo ref/fixtures.safetensors (cần torch)
RNGD_TIMEOUT=70 RNGD_SUBMIT_TIMEOUT=70 ./scripts/rngd_test.sh
```
