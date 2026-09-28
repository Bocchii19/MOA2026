#!/usr/bin/env python3
"""Ghép docs/data/timeline_*.md + các phần JS/CSS thành docs/rngd_kien_truc_dataflow.html."""
import json, re, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tl91 import parse91

ROOT = pathlib.Path(sys.argv[1])
PARTS = pathlib.Path(sys.argv[2])
OUT = ROOT / "docs" / "rngd_kien_truc_dataflow.html"
DATA = ROOT / "docs" / "data"
SRC = ROOT / "campaign/submissions/sdk081_best_saob2_4451tail/src/device"

FILES = {"qkv": "timeline_sliding_project_qkv.md", "sao": "timeline_sliding_attention_output.md",
         "ffn": "timeline_decoder_feedforward.md"}

# (nhãn ngắn, mô tả) theo "file:dòng"
L = {
 "shared/hidden.rs:95": ("x HBM→DM", "Activation `x` (bf16 3.840 = 15.360 B) nạp vào DM. 2 engine, packet 240 B lệch 256."),
 "shared/hidden.rs:96": ("rms_weight HBM→DM", "Trọng số RMSNorm, 15.360 B, 2 engine, packet 240 B lệch 256."),
 "shared/hidden.rs:31": ("RMS · bình phương", "RMSNorm: bình phương từng phần tử."),
 "shared/hidden.rs:49": ("RMS · tổng từng phần", "Tổng Σx² từng phần rồi inter-slice reduce 256 slice; pass dài vì gồm reduce liên slice."),
 "shared/hidden.rs:66": ("RMS · sqrt → VRF", "sqrt của mean square, kết quả đi thẳng vào VRF (bỏ commit/reload)."),
 "shared/hidden.rs:99": ("rms_weight → VRF (Sub)", "Sub nạp trọng số norm vào VRF song song với Main."),
 "shared/hidden.rs:107": ("chuẩn hoá x·w", "x · (1/rms) · w."),
 "shared/hidden.rs:350": ("all-gather 1/2", "All-gather: mọi slice nhận đủ các chunk của x."),
 "shared/hidden.rs:361": ("all-gather 2/2", "Xếp chunk liền nhau thành `m![H]` compact."),
 "shared/hidden.rs:377": ("chunk rows → VRF (Sub)", "Đưa các hàng chunk vào VRF cho pass RMS cuối kernel."),
 "shared/hidden.rs:389": ("post-RMS · bình phương", "RMS sau MLP (fused): bình phương."),
 "shared/hidden.rs:411": ("post-RMS · tổng 32 số hạng", "Tổng đủ 32 số hạng, giữ EPS."),
 "shared/hidden.rs:431": ("post-RMS · sqrt", "Chỉ kết quả sqrt bỏ commit/reload."),
 "shared/hidden.rs:451": ("post-norm weight HBM→DM", "Trọng số RMSNorm sau MLP; chỉ C0 nạp."),
 "shared/hidden.rs:454": ("post-RMS · normalize", "Chuẩn hoá theo hàng chunk."),
 "shared/hidden.rs:476": ("scalar f32 HBM→DM", "Scalar f32 nạp vào DM (chỉ C0)."),
 "shared/hidden.rs:477": ("broadcast scalar → VRF", "Phát scalar ra các chunk qua VRF."),
 "shared/hidden.rs:490": ("out rows → HBM", "Store output cuối kernel, 1 engine, packet 240 B lệch → RMW."),
 "shared/hidden.rs:498": ("residual HBM→DM", "Residual, chỉ C0 nạp."),
 "shared/hidden.rs:501": ("cộng residual", "x + residual."),
 "shared/hidden.rs:520": ("× layer scalar", "Nhân layer scalar (0,375 trong fixture)."),
 "shared/hidden.rs:542": ("layer scalar HBM→DM", "Layer scalar bf16."),
 "shared/hidden.rs:543": ("layer scalar → VRF (Sub)", "Sub đưa scalar vào VRF, không chèn pass Main giữa các pass down."),
 "shared/mlp.rs:44": ("weight NVFP4 HBM→DM", "Weight NVFP4 đóng gói: 30 hàng/slice, 29.491.200 B toàn chip (14,75 MB/cluster), packet 256 B căn."),
 "shared/mlp.rs:46": ("block scale HBM→DM", "Block scale NVFP4 3.686.400 B; packet 800 B (up/gate) hoặc 480 B (down) lệch 256 → ~184 B/cycle."),
 "shared/mlp.rs:57": ("stage1 · lookup f4→f8", "Contraction có lookup f4→f8 trên Fetch Adapter; bị giới hạn ~2,5 B/cycle/slice nên dài ~22–23K, che dưới khe và stream sau."),
 "shared/mlp.rs:91": ("scale tile → VRF (Sub)", "Sub nạp tile block scale vào VRF."),
 "shared/mlp.rs:100": ("stage2 · partial × scale", "Nhân partial bf16 với block scale theo tile; 5 tile."),
 "shared/mlp.rs:119": ("cộng tile_sums", "Cộng các tổng tile."),
 "shared/mlp.rs:152": ("encode x hi/lo (1)", "Mã hoá x thành lane f8."),
 "shared/mlp.rs:182": ("encode hi (Sub)", "Sub tính phần hi của lane."),
 "shared/mlp.rs:192": ("encode x hi/lo (2)", "Mã hoá phần lo của lane."),
 "shared/mlp.rs:220": ("lanes → TRF (StoTrf)", "Nạp lane f8 vào TRF."),
 "shared/mlp.rs:232": ("pool 30 hàng", "Gộp 30 hàng của 8 slice liên tiếp (f32)."),
 "shared/mlp.rs:247": ("global scale HBM→DM", "Global scale 2.048 B, 8 engine × 32 B."),
 "shared/mlp.rs:248": ("global scale → VRF", "Đưa global scale vào VRF."),
 "shared/mlp.rs:255": ("× global scale → bf16", "Nhân global scale, làm tròn bf16."),
 "shared/mlp.rs:282": ("GeGLU · gate", "GeGLU: chuẩn bị gate."),
 "shared/mlp.rs:301": ("GeGLU · gelu (VRF)", "GeLU của gate."),
 "shared/mlp.rs:308": ("GeGLU · gelu × up", "Nhân GeLU(gate) với up."),
 "shared/mlp.rs:339": ("down · |x|", "Chuẩn bị tính unscale cho activation down."),
 "shared/mlp.rs:355": ("down · max theo slice", "max|x| trong từng slice."),
 "shared/mlp.rs:363": ("down · max toàn cụm", "max|x| qua cả cụm."),
 "shared/mlp.rs:376": ("down · unscale", "unscale = max|x| / 256."),
 "shared/mlp.rs:392": ("unscale → VRF", "Đưa unscale vào VRF."),
 "shared/mlp.rs:401": ("down · lane hi/lo", "Tách activation thành lane hi/lo f8."),
 "shared/mlp.rs:415": ("down · lanes (Sub)", "Sub tính lane."),
 "shared/mlp.rs:422": ("down · lanes (Main)", "Main tính lane."),
 "shared/mlp.rs:439": ("all-gather lanes", "`CustomBroadcast{256}` đưa lane hi/lo về mọi slice: 9,06K."),
 "shared/mlp.rs:449": ("gather → TRF (Sub)", "Nạp lane đã gom vào TRF: 4,15K."),
 "shared/mlp.rs:456": ("unscale rows → VRF", "unscale theo hàng."),
 "shared/mlp.rs:467": ("stage1 down · lookup", "Lookup f4→f8 của down. Weight cuối luôn làm lộ pass này trên tail: ~22,8K."),
 "shared/mlp.rs:484": ("giải mã scale down (Sub)", "Decode block scale down trên Sub: ~7,0K; chạy song song stage1."),
 "shared/mlp.rs:493": ("scale f32 → TRF", "StoTrf scale f32: ~5,1K."),
 "shared/mlp.rs:502": ("stage2 down", "Nhân partial với block scale down: ~3,0K."),
 "shared/mlp.rs:545": ("down weight HBM→DM", "Weight down 29.491.200 B (14,75 MB/cluster)."),
 "shared/mlp.rs:546": ("down scale HBM→DM", "Block scale down 3.686.400 B, packet 480 B lệch 256: ~10,7K."),
 "shared/mlp.rs:565": ("rows C1→C0 (DmaStos)", "DmaStos gom rows của C1 về C0 để làm post-norm; kèm ExplicitSync ~3,8K."),
 "shared/mlp.rs:568": ("rows của C1 (Sub)", "Sub xử lý phần hàng nhận từ C1."),
 "shared/mlp.rs:575": ("gộp rows C0+C1", "Main gộp hai nửa rows thành layout compact."),
 "sliding/qkv.rs:97": ("W_Q HBM→DM", "Weight Q FP8: 15.728.640 B toàn chip (7,86 MB/cluster); ~299 B/cycle/cluster."),
 "sliding/qkv.rs:98": ("W_K HBM→DM", "Weight K FP8: 7.864.320 B toàn chip; ~277 B/cycle/cluster."),
 "sliding/qkv.rs:99": ("W_V HBM→DM", "Weight V FP8: 7.864.320 B toàn chip; ~285 B/cycle/cluster."),
 "sliding/qkv.rs:107": ("rope HBM→DM", "Nạp lại bảng rope cho K: 1,30K."),
 "sliding/qkv.rs:129": ("encode · |x·w| max", "Bước tính encoder scale: max|x·w| trên cả vector 3.840."),
 "sliding/qkv.rs:145": ("encode · ms", "Chuẩn bị unscale."),
 "sliding/qkv.rs:160": ("unscale → VRF", "Đưa unscale vào VRF."),
 "sliding/qkv.rs:168": ("encode · x·w/unscale → f8", "Tạo lane f8: `x·w/unscale`."),
 "sliding/qkv.rs:185": ("lanes → TRF (StoTrf)", "Nạp lane vào TRF (shared_lanes)."),
 "sliding/qkv.rs:193": ("Q projection · contraction", "Q = x·W_Q, 8,8K."),
 "sliding/qkv.rs:218": ("Q: gom head về 1 slice", "Ring-local pool đưa cả head về slice đầu của nhóm (TU, không DMA)."),
 "sliding/qkv.rs:226": ("K: gom head", "Như Q, cho K."),
 "sliding/qkv.rs:234": ("V: gom head", "Như Q, cho V."),
 "sliding/qkv.rs:247": ("sq HBM→DM", "Scale theo hàng của Q."),
 "sliding/qkv.rs:248": ("sk HBM→DM", "Scale theo hàng của K (2,18K có xếp hàng)."),
 "sliding/qkv.rs:249": ("sv HBM→DM", "Scale theo hàng của V."),
 "sliding/qkv.rs:258": ("q → HBM", "Store q ra HBM."),
 "sliding/qkv.rs:261": ("K scatter → cache", "Scatter K vào ring cache theo `kv_offset`: ~1,65K."),
 "sliding/qkv.rs:264": ("V pass trước scatter", "Pass cuối trước scatter V."),
 "sliding/qkv.rs:279": ("V scatter → cache", "Scatter V vào ring cache, lệnh cuối kernel: ~1,65K."),
 "sliding/qkv.rs:289": ("K/V projection · contraction", "Contraction K (3,1K) hoặc V (6,1K, nằm trên tail)."),
 "sliding/qkv.rs:311": ("scale head → VRF (Sub)", "Sub nạp scale head."),
 "sliding/qkv.rs:314": ("unscale → VRF", "Sub đưa unscale vào VRF."),
 "sliding/qkv.rs:317": ("raw × scale", "Nhân scale vào kết quả projection thô."),
 "sliding/qkv.rs:342": ("rotate_half · sin", "Hạng tử xoay của RoPE."),
 "sliding/qkv.rs:357": ("→ VRF (Sub)", "Đưa hạng tử vào VRF."),
 "sliding/qkv.rs:364": ("x·A + rot·B (RoPE)", "Ghép RoPE."),
 "sliding/qkv.rs:384": ("head RMS · Σx²", "Tổng bình phương trong head."),
 "sliding/qkv.rs:401": ("head RMS · sqrt", "sqrt(mean(x²)+EPS) của head → VRF."),
 "sliding/qkv.rs:429": ("cos gather", "`cos.dma_gather_scaled(rope_offset)`: ~1,8K."),
 "sliding/qkv.rs:430": ("sin gather", "`sin.dma_gather_scaled(rope_offset)`: ~1,8K."),
 "sliding/qkv.rs:432": ("cos → pack (Sub)", "Đóng gói cos."),
 "sliding/qkv.rs:438": ("sin → pack (Sub)", "Đóng gói sin."),
 "sliding/qkv.rs:444": ("rows0 → HBM", "Store rows0 (q) ra HBM."),
 "sliding/qkv.rs:455": ("q/k rms_weight HBM→DM", "Trọng số RMSNorm theo head."),
 "sliding/qkv.rs:469": ("x·y → VRF (Sub)", "Nhân phần tử theo head, đưa vào VRF."),
 "sliding/qkv.rs:476": ("x·y (Main)", "Nhân phần tử theo head."),
 "sliding/qkv.rs:494": ("rotate_half (1)", "Hoán đổi hai nửa 128."),
 "sliding/qkv.rs:500": ("rotate_half (2)", "Hoán đổi hai nửa 128."),
 "sliding/output.rs:63": ("W_O HBM→DM", "Weight O FP8: 15.728.640 B toàn chip (7,86 MB/cluster), 8 engine, packet 256 B căn: ~307 B/cycle/cluster."),
 "sliding/output.rs:67": ("xq HBM→DM", "Đầu ra attention 131.072 B, 8 engine, packet 256 B căn."),
 "sliding/output.rs:69": ("encode xq (Main)", "Mã hoá xq thành lane f8 hi/lo (×2⁶)."),
 "sliding/output.rs:84": ("encode (Sub)", "Sub tính lane."),
 "sliding/output.rs:92": ("encode (Main 2)", "Mã hoá tiếp."),
 "sliding/output.rs:108": ("lanes → TRF (StoTrf)", "Nạp lane vào TRF: 1,76K (C1: 2,62K)."),
 "sliding/output.rs:120": ("O projection · contraction", "Chiếu O + inter-slice reduce 16 chunk: 3,4K (tĩnh 1.374, ×2,5). **Tail đầu tiên.**"),
 "sliding/output.rs:145": ("y C1→C0 (DmaStos)", "Gom y (3.840 hàng) về 32 slice của C0: 1,15K."),
 "sliding/output.rs:149": ("post_attn_rms_weight", "Norm weight 7.680 B, 1 engine, packet 240 B lệch 256, chỉ C0."),
 "sliding/output.rs:151": ("residual HBM→DM", "Residual 7.680 B, 1 engine, packet 240 B lệch, chỉ C0."),
 "sliding/output.rs:153": ("o_weight_scale", "Scale W_O 7.680 B, 1 engine, packet 240 B lệch, chỉ C0."),
 "sliding/output.rs:157": ("scale → VRF (Sub)", "Sub nạp scale."),
 "sliding/output.rs:165": ("RMS · partial", "RMS pass 1: bình phương từng phần."),
 "sliding/output.rs:175": ("RMS · regular32", "RMS pass 2: cộng 32 phần."),
 "sliding/output.rs:187": ("RMS · sqrt", "RMS pass 3: sqrt."),
 "sliding/output.rs:195": ("norm weight → VRF (Sub)", "Sub nạp trọng số norm."),
 "sliding/output.rs:198": ("residual → VRF (Sub)", "Sub nạp residual: 1,13K."),
 "sliding/output.rs:203": ("normalize×w + res", "RMS pass 4: normalize × w + residual."),
 "sliding/output.rs:217": ("out rows → HBM", "Store output 7.680 B, 1 engine, packet 240 B lệch → RMW: 0,87K."),
}
BYTES = {  # (nhãn payload, byte toàn chip nếu là stream weight)
 "sliding/qkv.rs:97": ("15.728.640 B toàn chip · packet 256 B", 15728640), "sliding/qkv.rs:98": ("7.864.320 B toàn chip", 7864320),
 "sliding/qkv.rs:99": ("7.864.320 B toàn chip", 7864320), "sliding/output.rs:63": ("15.728.640 B toàn chip · 8 engine · 256 B", 15728640),
 "sliding/output.rs:67": ("131.072 B · 8 engine · 256 B", 131072), "shared/mlp.rs:44": ("29.491.200 B toàn chip · 8 engine · 256 B", 29491200),
 "shared/mlp.rs:545": ("29.491.200 B toàn chip · 8 engine · 256 B", 29491200), "shared/mlp.rs:46": ("3.686.400 B toàn chip · packet 800 B lệch", 0),
 "shared/mlp.rs:546": ("3.686.400 B toàn chip · packet 480 B lệch", 0), "shared/hidden.rs:95": ("15.360 B · 2 engine · 240 B", 0),
 "shared/hidden.rs:96": ("15.360 B · 2 engine · 240 B", 0), "sliding/output.rs:149": ("7.680 B · 1 engine · 240 B", 0),
 "sliding/output.rs:151": ("7.680 B · 1 engine · 240 B", 0), "sliding/output.rs:153": ("7.680 B · 1 engine · 240 B", 0),
 "sliding/output.rs:217": ("7.680 B · 1 engine · 240 B", 0),
}
STORE = {"sliding/qkv.rs:444", "sliding/qkv.rs:258", "sliding/qkv.rs:261", "sliding/qkv.rs:279", "sliding/output.rs:217", "shared/hidden.rs:490"}
DD = {"sliding/output.rs:145", "shared/mlp.rs:565"}


def suffix_qkv(f, c):
    if c < 35000:
        return ""
    return " · Q" if c < 56900 else (" · K" if c < 74500 else " · V")


def suffix_ffn(c):
    if c < 110000:
        return " · up"
    if c < 170000:
        return " · gate"
    return ""


QKV_SUF = {311, 314, 317, 342, 357, 364, 384, 401, 432, 438, 469, 476, 494, 500, 455}
FFN_SUF = {57, 91, 100, 119, 232, 247, 248, 255, 44, 46}


def find_fn(rel, ln):
    p = SRC / rel
    if not p.exists():
        return ""
    src = p.read_text(encoding="utf-8", errors="replace").splitlines()
    for i in range(min(ln - 1, len(src) - 1), -1, -1):
        m = re.search(r"\bfn\s+(\w+)|macro_rules!\s+(\w+)", src[i])
        if m:
            return m.group(1) or m.group(2)
    return ""


def parse(key):
    out = {}
    cl = None
    info = {}
    for line in (DATA / FILES[key]).read_text(encoding="utf-8").splitlines():
        m = re.match(r"### (\w+) — cluster (\d): Task ([\d,]+) cycles", line)
        if m:
            cl = int(m.group(2))
            out[cl] = {"task": int(m.group(3).replace(",", "")), "spans": [], "dma": 0, "tu": 0}
            continue
        m = re.match(r"DMA busy union ([\d,]+) \(([\d.]+)%\), TU busy union ([\d,]+) \(([\d.]+)%\)", line)
        if m and cl is not None:
            out[cl]["dma"] = float(m.group(2)); out[cl]["tu"] = float(m.group(4))
            continue
        if not line.startswith("| ") or not line[2].isdigit():
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|", 7)]
        b, e = int(c[0].replace(",", "")), int(c[1].replace(",", ""))
        kind, emit, op, life, srcs = c[3], c[4], c[5], c[6], c[7] if len(c) > 7 else ""
        sp = {"b": b, "e": e, "em": None if emit == "None" else int(emit), "life": life}
        m2 = re.search(r"src/device/([\w/]+\.rs):(\d+):\d+", srcs)
        if kind == "Cluster":
            mm = re.search(r'"(ExplicitSync|DramReuse|ClusterSync|SyncRelease)"[^\d]*?"index": (\d+)', srcs)
            name, idx = (mm.group(1), mm.group(2)) if mm else ("Sync", "?")
            sp.update(k="sync", t="sync", u="", l=f"{name} #{idx}",
                      d=("`ExplicitSync`: đợi cluster kia (bắt tay hai cluster)." if name == "ExplicitSync" else
                         "`DramReuse`: driver bắt buộc cho lệnh ghi DRAM; release sau lệnh ghi, acquire cuối kernel." if name == "DramReuse" else
                         "Đồng bộ liên cluster."))
        elif kind == "DMA":
            sp["k"] = "dma"; sp["u"] = "d"
            if m2:
                rel, ln = m2.group(1), int(m2.group(2)); k2 = f"{rel}:{ln}"
                lab, desc = L.get(k2, (f"DMA {rel}:{ln}", ""))
                if int(ln) in FFN_SUF and rel == "shared/mlp.rs":
                    lab += suffix_ffn(b)
                if "Gather" in op:
                    sp["t"] = "dma_dd"
                elif "Scatter" in op or k2 in STORE:
                    sp["t"] = "dma_store"
                elif k2 in DD:
                    sp["t"] = "dma_dd"
                else:
                    sp["t"] = "dma_load"
                if rel == "sliding/qkv.rs" and ln == 455:
                    lab += suffix_qkv(ln, b)
                sp.update(l=lab, d=desc, f=find_fn(rel, ln), s=f"src/device/{rel}:{ln}")
                if k2 in BYTES:
                    sp["by"], sp["bytes"] = BYTES[k2][0], BYTES[k2][1]
            else:
                lm = re.match(r"(\d+)-(\d+)", life)
                span_static = int(lm.group(2)) - int(lm.group(1)) if lm else 0
                if span_static == 719 or (not lm and 1300 <= e - b <= 1700):
                    sp.update(l="bitmap CustomBroadcast", d="Bitmap 16.384 B (8 engine × packet **32 B**, lệch 256) cho `CustomBroadcast{256}`; compiler sinh, không có dòng source.", by="16.384 B · 8 engine · 32 B")
                else:
                    sp.update(l="bảng FP4 HBM→DM", d="Bảng giải mã FP4 4.096 B (8 engine × packet **8 B**), compiler sinh cho mỗi pass lookup; đi kèm `StoTab`.", by="4.096 B · 8 engine · 8 B")
                sp["t"] = "dma_load"
        else:  # TU
            main = "Main" in op
            sp["k"] = "tu"; sp["u"] = "m" if main else "s"
            if kind == "StoTab":
                sp.update(t="tab", l="StoTab · nạp bảng FP4", d="Nạp bảng FP4 vào thanh ghi của Fetch Unit (chạy chồng DMA): 7,3–8,9K.", f="", s="")
            else:
                sp["t"] = "tu" if main else "vrf"
                if m2:
                    rel, ln = m2.group(1), int(m2.group(2)); k2 = f"{rel}:{ln}"
                    lab, desc = L.get(k2, (f"{kind} {rel}:{ln}", ""))
                    if rel == "sliding/qkv.rs" and ln in QKV_SUF:
                        lab += suffix_qkv(ln, b)
                    if rel == "sliding/qkv.rs" and ln == 289:
                        lab = "K projection · contraction" if b < 65000 else "V projection · contraction ⚠"
                    if rel == "shared/mlp.rs" and ln in FFN_SUF:
                        lab += suffix_ffn(b)
                    if kind == "StoTrf" and "TRF" not in lab:
                        lab += " (StoTrf)"
                    if kind == "StoVrf" and "VRF" not in lab:
                        lab += " (StoVrf)"
                    sp.update(l=lab, d=desc, f=find_fn(rel, ln), s=f"src/device/{rel}:{ln}")
                else:
                    sp.update(l=kind, d="", f="", s="")
        out[cl]["spans"].append(sp)
    for cl in out:
        out[cl]["spans"].sort(key=lambda s: (s["b"], s["e"]))
    return {"c0": out[0], "c1": out[1]}


def main():
    TL = {k: parse(k) for k in FILES}
    for k in ('qkv91', 'sao91', 'ffn91'):
        TL[k] = parse91(k, ROOT)
    # gán số lượng để kiểm tra nhanh
    for k, v in TL.items():
        for c in ("c0", "c1"):
            print(k, c, v[c]["task"], len(v[c]["spans"]), "spans", "dma%", v[c]["dma"], "tu%", v[c]["tu"], file=sys.stderr)
    css = (PARTS / "style.css").read_text(encoding="utf-8")
    body = (PARTS / "body.html").read_text(encoding="utf-8")
    js = "".join((PARTS / f).read_text(encoding="utf-8") + "\n" for f in ("data_static.js", "imp.js", "imp91.js", "app1.js", "app2.js", "app3.js", "app4.js"))
    tl_js = "const TL=" + json.dumps(TL, ensure_ascii=False, separators=(",", ":")) + ";\n"
    html = ('<!DOCTYPE html>\n<html lang="vi">\n<head>\n<meta charset="UTF-8">\n<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            '<title>RNGD · phần cứng và dataflow</title>\n'
            '<link rel="preconnect" href="https://fonts.googleapis.com">\n<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap">\n'
            '<style>\n' + css + '</style>\n</head>\n<body>\n' + body + '\n<script>\n' + tl_js + js + '</script>\n</body>\n</html>\n')
    OUT.write_text(html, encoding="utf-8", newline="\n")
    print("wrote", OUT, len(html), "bytes", file=sys.stderr)


main()
