# Nguồn của `docs/rngd_kien_truc_dataflow.html`

Trang HTML tự chứa (không phụ thuộc thư viện; chỉ tải font IBM Plex từ Google Fonts nếu có mạng).
Sinh lại sau khi sửa `docs/data/timeline_*.md` hoặc các file ở đây:

```bash
python docs/html_src/build.py . docs/html_src     # chạy từ gốc repo
```

| File | Nội dung |
|---|---|
| `data_static.js` | khối phần cứng (`BLOCKS`), bảng số đo, thanh head/weight/khe/tail, bảng theo kernel (`KDOC`) viết tay từ `docs/01–09` |
| `imp.js` | **so với baseline ban tổ chức**: bảng cải tiến đánh số (`IMP`), phần baseline đã bị thay (`GHOST`), nút giữ nguyên (`EQN`); nguồn đối chiếu là `upstream/v0.8.1:src/` và comment thiết kế trong source best |
| `imp91.js` | **bản score_9_3 (official 9,1481)**: `IMP.*91` (thay đổi so với 8,4643 và baseline), sơ đồ `cfg*91`, `CLOSED91`, tab bản 9,1481. Nguồn: `packages/score_9_3` (`WORK_LOG.md`, `reports/`, comment trong `src/`). Timeline lấy từ `docs/data/s93_timeline_*.md` (trace job 104939) qua `tl91.py` |
| `tl91.py` | parse `docs/data/s93_timeline_*.md` (trace bản 9,1481) và gắn nhãn theo tên hàm + doc comment trong `packages/score_9_3/src` |
| `app1.js` | helper SVG, mặt phẳng chip, thẻ khối |
| `app2.js` | engine sơ đồ khối, dataflow hệ thống, chuỗi 3 kernel, luồng lệnh, hai cluster |
| `app3.js` | cấu hình dataflow QKV / SAO / FFN |
| `app4.js` | timeline swimlane thật, tab kernel, tab, giao diện tối/sáng |
| `build.py` | parse `docs/data/timeline_*.md` (span đo trên phần cứng, trace 93413), gắn nhãn theo source của bản best, ghép HTML |

Nhãn của từng span (`L` trong `build.py`) được suy ra từ dòng source trong
`campaign/submissions/sdk081_best_saob2_4451tail/`; đổi source best thì cập nhật bảng đó.
