"""Parse docs/data/s93_timeline_*.md (trace job 104939, score_9_3 build 0abcde6) into the TL structure."""
import re, pathlib

FILES91 = {"qkv91": "s93_timeline_sliding_project_qkv.md", "sao91": "s93_timeline_sliding_attention_output.md",
           "ffn91": "s93_timeline_decoder_feedforward.md"}

# fn -> (nhãn, mô tả ngắn viết tay; nếu rỗng thì lấy doc comment trong source)
FNL = {
 # K1
 "normalize_input_s": ("RMS x (strip)", "Input RMSNorm trên layout strip: mean square cộng 15 strip bằng Inter-Slice Reducer, phát lại 16 slot."),
 "spread_quarters": ("spread rms_weight (Broadcast1 ring 64)", "Hạt quarter-strip của rms weight phát bằng một pass `Broadcast1{4,16}`."),
 "spread_x_quarters": ("spread x (Broadcast1 ring 64)", "Hạt quarter-strip của x phát bằng một pass `Broadcast1{4,16}`; f32 qua switch để cộng `zero_vrf` (neo thứ tự)."),
 "split_s": ("x hi/lo (strip)", "Tách x thành hi = f8(16x), lo = f8(16x − hi) trên layout strip."),
 "q_strip_trf": ("nạp TRF strip", "MỘT lần nạp TRF strip dùng cho cả Q, K, V."),
 "load_q_strips": ("W_Q HBM→DM (15-strip)", "Weight Q FP8 theo hình học 15 strip × 256 B."),
 "load_weights_s": ("W HBM→DM (15-strip)", "Weight K/V FP8 theo hình học 15 strip × 256 B."),
 "contract_q_strips": ("Q contraction (64 B, 15 strip)", "Q = x·W_Q: packet 64 B, 1.024 bước Outer, Inter-Slice Reducer cộng 15 strip."),
 "contract_kv_strips": ("contraction", "K hoặc V trên cùng TRF strip với Q."),
 "gather_q_strips": ("gom head Q (ring 64)", "Bốn nhóm hàng của mỗi PE gom về slice đầu bằng `Broadcast1{4,16}`, theo thứ tự kênh."),
 "gather_kv_strips": ("gom head", "Gom head K hoặc V (ring 64)."),
 "project_s": ("scale sq HBM→DM", ""),
 "stage_rope_rows_r17": ("RoPE on-chip (r17)", "RoPE tính trên chip từ `rope_offset`: cos/sin(POS·f(i)); thay hai dma_gather và cặp store/load HBM."),
 "rope_ramp_double": ("RoPE · dải f (nhân đôi)", "Dựng dải f(i) bằng 8 lần nhân đôi, không dùng primitive index."),
 "sin_half": ("RoPE · sin", "Tính sin trên một nửa dải (range reduction)."),
 "apply_rope_r21": ("RoPE áp dụng (r21)", "`x·A + rotate_half(x)·B`; 2 pass mỗi tensor."),
 "normalize_query_scaled": ("norm Q (scale gộp)", "Mean square của s·q rồi w·q·s/rms → bf16."),
 "normalize_key_scaled": ("norm K (scale gộp)", "Scale K stream cùng K đã gom; 5 pass; sqrt ghi VRF."),
 "normalize_value_scaled_iv": ("norm V (scale gộp)", "V với scale gộp; sqrt ghi VRF từ Main."),
 # K2
 "load_input_g3": ("x seed (quarter-live)", "x nạp seed 128 B vào một phần tư slice, rồi Broadcast1{16,16} + compaction."),
 "copy_g3": ("bản sao x (delay)", "Pass copy đồng nhất để xếp DMA tile B trước contraction A (giá trị không đổi)."),
 "gather_buffer_g3": ("buffer gather (Sub)", "Buffer đích của gather do pass Sub tạo để sync rời đầu kernel (J1)."),
 "split_g3": ("x hi/lo (2 lane TRF)", "hi = f8(128x), lo = f8(128x − hi) vào hai lane TRF."),
 "g3_tile_fns": ("W_O tile", "Tile weight O: DMA hoặc contraction 64 B."),
 "operand_vrf": ("operand → VRF (Sub)", "Nạp operand epilogue vào VRF trên Sub."),
 "project_output_g3": ("epilogue G3", "Gather 32 × 480 B về C0, mean square, ring-32 all-gather reduce, sqrt, normalize, residual."),
 # K3
 "normalize_input": ("RMS x (pre-FFN)", "RMSNorm trước MLP, spread x bằng Broadcast1."),
 "load_ip_scales": ("scale Up/Gate", "Block scale căn 256 B; Up và Gate qua MỘT InterTranspose ring 16."),
 "load_ip_weight": ("W HBM→DM", "Weight NVFP4 30 hàng/slice, một lệnh DMA."),
 "ip_scale_to_bf": ("đổi scale FP8→BF16 (Main)", "Đổi scale FP8→BF16 trên Main."),
 "ip_scale_trf": ("scale → TRF (StoTrf)", "Nạp scale BF16 vào TRF."),
 "split_up_activation": ("x hi/lo (Up/Gate)", "Mã hoá x hi/lo một lần cho Up và Gate."),
 "project_up_and_gate": ("spread x → TRF", "Spread x và nạp TRF trên Main."),
 "up_gate_ip_bf": ("fused lookup + contraction", "Decode FP4 một lần rồi stream hai lần với x hi/lo; partial block-16 BF16."),
 "ip_scale_diag": ("scale khối: contraction chéo BF16", "Contraction đường chéo áp scale khối BF16, pass 6 hàng."),
 "gather_up_gate": ("regroup 120/slice (DM→DM)", "Hoán vị DM→DM trong cluster thành 120 phần tử mỗi slice."),
 "geglu_local": ("GeGLU trên chip", "GeGLU giữ trong DM; hand-over ring-8 replicate + stride-8 all-gather rồi nạp TRF trên Main."),
 "split_down_activation": ("activation Down hi/lo", "Mã hoá activation cho Down (hi/lo f8)."),
 "down_tile_fns": ("W_down tile", "Weight Down: tile 12 hàng (w0) và tile 3 hàng (w1)."),
 "down_scale_fn": ("scale Down (hàng)", "Block scale Down; hàng 4..11 một lệnh DMA."),
 "down_diag_blocks": ("Down: lookup fused + contraction", "12 hàng đầu: lookup fused → partial BF16."),
 "down_fused_group3": ("Down 3 hàng cuối (tile riêng)", "Tile 3 hàng cuối tách riêng để DMA weight chồng lên."),
 "down_diag_scale": ("Down: đổi scale + TRF (Main)", "Đổi scale FP8→BF16 và nạp TRF trên Main ngay sau decode."),
 "down_diag_finish": ("Down: contraction chéo", "Áp scale khối bằng contraction đường chéo."),
 "project_down": ("Down → cluster 0", "Partial cỡ H gom về C0, một intra-slice reduction."),
 "down_epilogue": ("epilogue hợp nhất", "Global scale, RMSNorm, residual, layer gate trên 8 slice × 480."),
 "feedforward": ("operand epilogue", ""),
}
ORD = {"contract_kv_strips": ["K", "V"], "gather_kv_strips": ["K", "V"], "up_gate_ip_bf": ["Up", "Gate"], "ip_scale_diag": ["Up", "Gate"],
       "ip_scale_to_bf": ["Up", "Gate"], "ip_scale_trf": ["Up", "Gate"], "g3_tile_fns": ["A", "B"], "operand_vrf": ["1", "2"],
       "rope_ramp_double": ["", ""], "load_weights_s": ["K", "V"]}
DMA_NAMES = {  # (file:line) -> (nhãn, loại, bytes text, bytes)
 "sliding/qkv_s.rs:174": ("rms_weight HBM→DM", "dma_load", "15.360 B (hạt quarter-strip)", 0),
 "sliding/qkv_s.rs:201": ("x HBM→DM", "dma_load", "15.360 B (hạt quarter-strip)", 0),
 "sliding/qkv_s.rs:39": ("W_Q HBM→DM (15-strip)", "dma_load", "15.728.640 B toàn chip", 15728640),
 "sliding/qkv_s.rs:396": ("W_K HBM→DM (15-strip)", "dma_load", "7.864.320 B toàn chip", 7864320),
 "sliding/qkv_s.rs:397": ("W_V HBM→DM (15-strip)", "dma_load", "7.864.320 B toàn chip", 7864320),
 "sliding/qkv_s.rs:383": ("sk / sv HBM→DM", "dma_load", "", 0),
 "sliding/qkv_s.rs:424": ("sq HBM→DM", "dma_load", "", 0),
 "sliding/rope_h4.rs:125": ("rope_offset HBM→DM (4 B)", "dma_load", "4 B (lệnh ~3,9K vì đứng sau stream K)", 0),
 "sliding/qkv_h4_norm.rs:153": ("k_rms_weight HBM→DM", "dma_load", "", 0),
 "sliding/qkv_h4_norm.rs:70": ("q_rms_weight HBM→DM", "dma_load", "", 0),
 "ops.rs:78": ("q → HBM", "dma_store", "", 0), "ops.rs:79": ("K scatter → cache", "dma_store", "", 0), "ops.rs:80": ("V scatter → cache", "dma_store", "", 0),
 "sliding/projection_g3.rs:42": ("x seed HBM→DM", "dma_load", "131.072 B: 128 B → 1/4 slice", 131072),
 "sliding/projection_g3.rs:127": ("W_O tile HBM→DM", "dma_load", "tile A 92 hàng · tile B 28 hàng", 0),
 "sliding/projection_g3.rs:256": ("o_weight_scale HBM→DM", "dma_load", "7.680 B", 0),
 "sliding/projection_g3.rs:257": ("norm_weight HBM→DM", "dma_load", "7.680 B", 0),
 "sliding/projection_g3.rs:258": ("residual HBM→DM", "dma_load", "7.680 B", 0),
 "sliding/projection_g3.rs:265": ("y C1→C0 (DmaStos)", "dma_dd", "32 mảnh × 480 B", 0),
 "ops.rs:150": ("out → HBM", "dma_store", "7.680 B", 0), "ops.rs:242": ("out → HBM", "dma_store", "7.680 B", 0),
 "shared/mlp.rs:1287": ("rms_weight HBM→DM", "dma_load", "15.360 B", 0), "shared/mlp.rs:1250": ("x HBM→DM", "dma_load", "15.360 B", 0),
 "shared/mlp.rs:384": ("scale Up HBM→DM (căn 256 B)", "dma_load", "", 0), "shared/mlp.rs:386": ("scale Gate HBM→DM (căn 256 B)", "dma_load", "", 0),
 "shared/mlp.rs:365": ("W HBM→DM (NVFP4, 30 hàng)", "dma_load", "29.491.200 B toàn chip", 29491200),
 "shared/mlp.rs:679": ("gate_global_scale HBM→DM", "dma_load", "", 0), "shared/mlp.rs:834": ("W_down tile HBM→DM", "dma_load", "K-half; w0 12 hàng · w1 3 hàng", 0),
 "shared/mlp.rs:860": ("scale Down HBM→DM", "dma_load", "hàng 4..11 một lệnh", 0),
 "shared/mlp.rs:804": ("down_global_scale HBM→DM", "dma_load", "", 0), "shared/mlp.rs:805": ("up_global_scale HBM→DM", "dma_load", "", 0),
 "shared/mlp.rs:806": ("layer_scalar HBM→DM", "dma_load", "", 0), "shared/mlp.rs:802": ("residual HBM→DM", "dma_load", "", 0),
 "shared/mlp.rs:803": ("post_ff_rms_weight HBM→DM", "dma_load", "", 0), "shared/mlp.rs:1101": ("down → C0 (DmaStos)", "dma_dd", "", 0),
}
ORD_DMA = {"shared/mlp.rs:365": ["Up", "Gate"], "shared/mlp.rs:834": ["w0", "w1"], "shared/mlp.rs:860": ["(1)", "(2)"], "sliding/projection_g3.rs:127": ["tile A", "tile B"],
           "sliding/qkv_s.rs:383": ["sk", "sv"]}


def parse91(key, root):
    data = pathlib.Path(root) / "docs" / "data"
    src = pathlib.Path(root) / "packages" / "score_9_3" / "src" / "device"
    cache = {}

    def lines(rel):
        if rel not in cache:
            p = src / rel
            cache[rel] = p.read_text(encoding="utf-8", errors="replace").splitlines() if p.exists() else []
        return cache[rel]

    def fn_of(rel, ln):
        s = lines(rel)
        for i in range(min(ln - 1, len(s) - 1), -1, -1):
            m = re.search(r"\bfn\s+(\w+)|macro_rules!\s+(\w+)", s[i])
            if m:
                return m.group(1) or m.group(2), i
        return "", -1

    def doc_of(rel, idx):
        s = lines(rel); out = []
        i = idx - 1
        while i >= 0 and (s[i].strip().startswith("///") or s[i].strip().startswith("#[") or s[i].strip() == ""):
            if s[i].strip().startswith("///"):
                out.append(s[i].strip()[3:].strip())
            i -= 1
        return " ".join(reversed(out))[:300]

    res = {}; cl = None; counters = {}
    for line in (data / FILES91[key]).read_text(encoding="utf-8").splitlines():
        m = re.match(r"### (\w+) — cluster (\d): Task ([\d,]+) cycles", line)
        if m:
            cl = int(m.group(2)); res[cl] = {"task": int(m.group(3).replace(",", "")), "spans": [], "dma": 0, "tu": 0}; counters = {}
            continue
        m = re.match(r"DMA busy union ([\d,]+) \(([\d.]+)%\), TU busy union ([\d,]+) \(([\d.]+)%\)", line)
        if m and cl is not None:
            res[cl]["dma"] = float(m.group(2)); res[cl]["tu"] = float(m.group(4)); continue
        if not line.startswith("| ") or not line[2].isdigit():
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|", 7)]
        b, e = int(c[0].replace(",", "")), int(c[1].replace(",", ""))
        kind, emit, op, life, srcs = c[3], c[4], c[5], c[6], (c[7] if len(c) > 7 else "")
        sp = {"b": b, "e": e, "em": None if emit == "None" else int(emit), "life": life}
        m2 = re.search(r"src/(?:device/)?([\w/]+\.rs):(\d+):\d+", srcs)
        rel, ln = (m2.group(1), int(m2.group(2))) if m2 else ("", 0)
        k2 = f"{rel}:{ln}"
        if kind == "Cluster":
            mm = re.search(r'"(ExplicitSync|DramReuse|ClusterSync|SyncRelease)"[^\d]*?"index": (\d+)', srcs)
            name, idx = (mm.group(1), mm.group(2)) if mm else ("Sync", "?")
            sp.update(k="sync", t="sync", u="", l=f"{name} #{idx}",
                      d=("`ExplicitSync`: đợi cluster kia." if name == "ExplicitSync" else "`DramReuse`: bắt buộc cho lệnh ghi DRAM; release sau lệnh ghi, acquire cuối kernel." if name == "DramReuse" else "Đồng bộ liên cluster."))
        elif kind == "DMA":
            sp["k"] = "dma"; sp["u"] = "d"
            if rel and k2 in DMA_NAMES:
                lab, typ, by, nb = DMA_NAMES[k2]
                if k2 in ORD_DMA:
                    n = counters.get(k2, 0); counters[k2] = n + 1
                    lab = lab.replace("HBM→DM", ORD_DMA[k2][min(n, len(ORD_DMA[k2]) - 1)] + " HBM→DM") if "HBM→DM" in lab else lab + " " + ORD_DMA[k2][min(n, len(ORD_DMA[k2]) - 1)]
                fn, fi = fn_of(rel, ln)
                sp.update(l=lab, t=typ, d=(FNL.get(fn, ("", ""))[1] or doc_of(rel, fi)), f=fn, s=f"src/device/{rel}:{ln}")
                if by:
                    sp["by"] = by
                if nb:
                    sp["bytes"] = nb
            elif rel:
                fn, fi = fn_of(rel, ln)
                sp.update(l=(FNL.get(fn, (fn, ""))[0] or fn) + " (DMA)", t="dma_load", d=doc_of(rel, fi), f=fn, s=f"src/device/{rel}:{ln}")
            else:
                lm = re.match(r"(\d+)-(\d+)", life); st = int(lm.group(2)) - int(lm.group(1)) if lm else 0
                if st == 719:
                    sp.update(l="bitmap CustomBroadcast", d="Bitmap 16.384 B (packet 32 B) cho `CustomBroadcast`; compiler sinh.", by="16.384 B · 8 engine · 32 B")
                else:
                    sp.update(l="bảng FP4 HBM→DM", d="Bảng giải mã FP4 4.096 B (packet 8 B), compiler sinh cho mỗi pass lookup; đi kèm `StoTab`.", by="4.096 B · 8 engine · 8 B")
                sp["t"] = "dma_load"
        else:
            main = "Main" in op
            sp["k"] = "tu"; sp["u"] = "m" if main else "s"
            if kind == "StoTab":
                sp.update(t="tab", l="StoTab · nạp bảng FP4", d="Nạp bảng FP4 vào thanh ghi Fetch Unit (chạy chồng DMA).", f="", s="")
            else:
                sp["t"] = "tu" if main else "vrf"
                if rel:
                    fn, fi = fn_of(rel, ln)
                    lab, desc = FNL.get(fn, (fn.replace("_", " "), ""))
                    if fn in ORD:
                        n = counters.get(k2, 0); counters[k2] = n + 1
                        suf = ORD[fn][min(n, len(ORD[fn]) - 1)]
                        lab = (lab + " " + suf).strip() if suf else lab
                    if kind == "StoTrf" and "TRF" not in lab:
                        lab += " (StoTrf)"
                    if kind == "StoVrf" and "VRF" not in lab:
                        lab += " (StoVrf)"
                    sp.update(l=lab, d=desc or doc_of(rel, fi), f=fn, s=f"src/device/{rel}:{ln}")
                else:
                    sp.update(l=kind, d="", f="", s="")
        res[cl]["spans"].append(sp)
    for c in res:
        res[c]["spans"].sort(key=lambda s: (s["b"], s["e"]))
    return {"c0": res[0], "c1": res[1]}
