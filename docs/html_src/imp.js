// ===== So với baseline của ban tổ chức: cái gì đổi, đổi thế nào =====
// tag: imp = cải tiến cơ chế; new = mới ở 8,4643; (nhóm 'rem' là phần baseline làm nhưng best đã bỏ: nằm ở GHOST)
const BASE={ // cycle chấm điểm của baseline (hằng số trong công thức score) và best official b162deec
  qkv:{b:250514,o:78277,name:'QKV'}, sao:{b:404633,o:36674,name:'SAO'}, ffn:{b:3703473,o:215644,name:'FFN'}};
const TAGN={imp:'cải tiến',new:'mới 8,4643',eq:'giữ như baseline'};
const ORG={
 A:'nhánh 7,8382 (cơ chế đo ở SDK 0.6.0, code đã port sang 0.8.1)',
 B:'0.8.1, đã có official',
 C:'mới ở 8,4643 (so với 8,4451)'};
const IMP={
qkv:[
 {n:1,tag:'imp',t:'Hai cluster chia hàng khác nhau (Q32 / K8 / V8)',
  base:'`Cluster = m![1 # 2]`: chỉ cluster 0 có dữ liệu, cluster 1 không làm gì. Toàn bộ 31,5 MB weight đi qua một nửa chip.',
  best:'Cluster 0 lấy Q head 0–7 và K/V head 0–3, cluster 1 lấy nửa còn lại. Q: 64 slice × 32 hàng; K/V: 128 slice × 8 hàng, mỗi hàng bắt đầu đúng biên 256 B và trải đủ 8 engine.',
  how:'Mỗi cluster chỉ stream nửa byte weight (Q 7,86 MB/cluster). Stream weight lớn là điều kiện cần để đạt ~300 B/cycle/cluster (~615 cả chip).',
  ev:'Trace 93413: Q/K/V ~299 / 277 / 285 B/cycle/cluster. Layout 960 B lệch biên trước đó chỉ đạt 467 B/cycle so với 660 khi căn 256 B (×1,41, đo ở SDK 0.6.0, comment trong `qkv.rs`).',
  org:'A',nodes:['hq','hk','hv','dq','dk','dv','cq','ck','cv']},
 {n:2,tag:'imp',t:'Weight FP8 đi thẳng vào Tensor Unit; activation tách hi+lo f8',
  base:'Weight f8 → DM → `fetch_table_lookup::<bf16>` → **ghi weight BF16 vào DM** → đọc lại làm contraction BF16 × x BF16 → một pass nhân scale riêng.',
  best:'Contraction nhận f8 đúng như lưu. Activation encode **một lần** thành hai lane f8e4m3 (hi + phần dư lo) với `unscale = sqrt(max(x²)+1e-30)/256`, TRF dùng chung cho Q/K/V. Nhân `dot × row_scale × unscale` rồi round bf16 một lần.',
  how:'Lookup f8→bf16 chạy 2,6 B/cycle so với 32 B/cycle của `fetch` thường, và tốn thêm một lượt ghi + đọc DM. Bỏ hẳn pass giải mã; độ chính xác gần bf16 nhờ hi+lo.',
  ev:'Số 2,6 vs 32 B/cycle là comment trong `qkv.rs`. Official chỉ đo cả kernel (250,5K → 78,3K), không A/B riêng cơ chế này.',
  org:'A',nodes:['nenc','cq','ck','cv','dq','dk','dv']},
 {n:3,tag:'imp',t:'Activation tới mọi slice qua ring switch, không qua DMA / broadcast một nguồn',
  base:'`x` nằm trên một slice, RMSNorm rồi `broadcast_hidden` (`CustomBroadcast{256}`) từ một nguồn 240 flit. Replicate bằng DMA thì phải chuyển 1.966.080 B mỗi cluster trên chính engine mà weight cần.',
  best:'`x` và norm weight nạp thành 32 chunk × 120 (61.440 B/cluster), nhân norm weight tại chỗ, RMS reduce qua ring 32 slice rồi **all-gather một lượt ring**: mỗi slice chỉ inject 8 flit.',
  how:'All-gather ~2K cycle thay vì ~61K cycle của broadcast một nguồn; ring đang rảnh trong QKV nên gần như miễn phí. Nhân norm weight trước khi gather giữ nó ngoài VRF (8 KB không chứa nổi bản f32 của vector 3.840).',
  ev:'Hai con số 2K vs 61K là ước lượng trong comment `hidden.rs`, không phải A/B riêng. Giữ nguyên full RMSNorm (EPS, sqrt).',
  org:'A',nodes:['dsm','nrms','nenc']},
 {n:4,tag:'imp',t:'Norm theo head + RoPE gộp một chuỗi, đi thẳng vào VRF',
  base:'`normalize_query/key/value` là các hàm riêng (mean-square, sqrt, nhân), rồi `apply_rope`: cos/sin → DM → InterTranspose → `rotate_half` bằng nhiều pass q và k → nhân sin/cos → cộng → Broadcast1 trả về. Mỗi bước commit DM rồi nạp lại.',
  best:'Mỗi slice giữ một head. RoPE viết thành `(x·A + rotate_half(x)·B)/rms(x)` với `A = w·cos`, `B = rotate_half(w)·sin` là toán hạng VRF. Kết quả sqrt của head RMS và bốn tích norm-weight/RoPE đi thẳng vào VRF. Regroup theo head bằng switch pool trên TU (không DM→DM).',
  how:'Bỏ commit/nạp lại DM giữa các pass và bỏ các pass rotate_half riêng. DM→DM tuần tự hoá (~76 B/cycle) nên tránh hẳn.',
  ev:'Official 8,3881 khi thêm bước "head RMS/RoPE thẳng vào VRF" (leaderboard `187d5107`); rope DM→DM thay thế đã đo là thua +8,0K.',
  org:'B',nodes:['pq','pk','pv']},
 {n:5,tag:'imp',t:'Rope gather một lần, một store HBM; ít lệnh DMA giữa kernel',
  base:'Baseline gather cos/sin (`dma_gather_scaled`) rồi `to_dm` sang layout head-across-slices (DM→DM, ~76 B/cycle, tuần tự hoá). Bản trung gian của campaign dùng hai store HBM cho hai hàng RoPE.',
  best:'Gather cos/sin vào **một** slice, pack thành một tensor 2 hàng, **một store** ra HBM; mọi placement theo head nạp lại bằng DMA replicate thông thường. Regroup Q dùng `Broadcast1{8,4}`, K/V `Broadcast1{32,2}`.',
  how:'Tránh DM→DM; một store thay hai bớt một điểm đồng bộ liên cluster (store HBM giữa kernel bắt C0 đợi C1) và bớt một lệnh DMA nhỏ (~1–2K mỗi lệnh).',
  ev:'Đo so với bản trung gian hai store, không phải trực tiếp với baseline: 90,0K → 84,8K khi gộp store + regroup bằng switch (Arena job 32740/32761, SDK 0.6.0).',
  org:'A',nodes:['drp','hrope']}
],
sao:[
 {n:1,tag:'imp',t:'Hai cluster chia 3.840 hàng output (1.920 mỗi cluster)',
  base:'`Cluster = m![1 # 2]`: chỉ cluster 0 tính toàn bộ 15,7 MB weight O; cluster 1 rảnh.',
  best:'Mỗi cluster có 256 slice = 16 nhóm hàng × 120 và 16 chunk Qs × 256. Không tính trùng.',
  how:'Mỗi cluster chỉ stream 7,86 MB weight: ~307 B/cycle/cluster, ~95% trần thực tế.',
  ev:'Trace 93413: weight O 25,6K cho 7,86 MB/cluster. Baseline xếp toàn bộ byte vào một cluster.',
  org:'A',nodes:['hw','dw','proj']},
 {n:2,tag:'imp',t:'Bỏ broadcast 256 slice của `xq`: chia trục Qs thành 16 chunk + reduce liên slice',
  base:'`broadcast_sliding_heads`: replicate 4.096 phần tử vào mọi slice bằng `CustomBroadcast{256}` (bitmap 16 KB), rồi mỗi slice tính đủ 1.024 cột.',
  best:'Trục Qs chia 16 chunk × 256 trên các slice. Mỗi slice chỉ DMA chunk của mình (128 KB cả cluster, 8 engine, packet 256 B căn) và cộng partial bằng **một** `vector_inter_slice_reduce`.',
  how:'Bớt replicate ×256 và bớt bitmap 16 KB; phép reduce chạy trên Vector Engine thay vì 3 pass `add_partials`.',
  ev:'Trace: DMA `xq` 1,4K ở head. Không có A/B riêng.',
  org:'A',nodes:['hxq','dxq','proj']},
 {n:3,tag:'new',t:'B2: `xq` chỉ nạp vào 128 slice sản xuất, encode một lần, switch phát cho 2 nhóm hàng',
  base:'(bản 8,4451) `xq` nạp vào cả 256 slice, mỗi slice tự encode hi/lo rồi nạp TRF.',
  best:'`xq` nạp vào `XSl = [H%1920/240, 1#2, Qs/256]` (128 slice/cluster). Encode hi/lo một lần; `Broadcast1{slice1: 2, slice0: 16}` phát sang hai nhóm hàng khi nạp TRF (TRF giữ 2 slot vật lý, chỉ slot 0 là dữ liệu thật).',
  how:'DMA `xq` và pass encode chỉ còn một nửa; phép tính encode giữ nguyên nên cùng giá trị hi/lo.',
  ev:'Official: B2 36.687 / 36.686 / 36.674 (B4 36,9K, B8 36,9K, B16 36,7–36,9K); tail rows2 của 8,4507 ~37,2K (6 official). Chỉ official phân biệt được, Arena và ABBA không.',
  org:'C',nodes:['enc','dxq','hxq']},
 {n:4,tag:'imp',t:'Weight FP8 vào thẳng contraction, một stream liền thay cho 4 tile',
  base:'`output_partial` × 4 (CHUNK = 1024): mỗi lần DMA một tile weight, lookup f8→bf16 và **ghi weight BF16 vào DM**, contraction, rồi `add_partials` ba pass và một pass nhân channel scale.',
  best:'Một lệnh DMA O weight 15,7 MB (8 engine × 256 B) chạy liền; `contract_outer` nhận cặp f8 trực tiếp; activation ×2⁶ rồi tách hi+lo, contraction gộp cả hai lane trong một lượt.',
  how:'Không còn lookup f8→bf16 (2,6 B/cycle) trên đường weight, không còn 4 lần DMA + 4 lần contraction, và bớt 3 pass cộng partial.',
  ev:'Official chỉ đo cả kernel (404,6K → 36,7K). Không A/B riêng cơ chế này.',
  org:'A',nodes:['dw','enc','proj']},
 {n:5,tag:'imp',t:'Tail: gom một lần về C0, RMS regular32, sqrt vào VRF, store',
  base:'`to_dm` y → `shared::rmsnorm::normalize` (ReducingSlices + Broadcast1) → `residual::add` theo tile → `to_hbm`. Nhiều pass, mỗi pass commit DM rồi đọc lại.',
  best:'Projection BF16 trên hai cluster, gom đủ 3.840 hàng về C0 (32 slice × 120) một lần, row scale → partial RMS → `Broadcast1(32,1)` → full-H RMS + EPS → sqrt thẳng VRF → normalize × w + residual → store. Chỉ hai tổng norm một phần đi qua cluster.',
  how:'Đơn giản hoá trao đổi statistic và staging cuối kernel; giữ nguyên điểm round BF16 sau projection và full RMSNorm + EPS.',
  ev:'Official: tail C0 regular32 8,1197 → sqrt thẳng VRF 8,2338 (leaderboard `6e1decfb`, `bc340e78`); SAO peer-pair 8,0215.',
  org:'B',nodes:['gath','rms','epl','out']},
 {n:6,tag:'imp',t:'Operand epilogue dời khỏi hàng đợi trước weight O',
  base:'Baseline nạp `o_weight_scale` bằng một `to_dm` riêng trong `apply_output_channel_scale`; thứ tự các lệnh DMA do scheduler quyết định. Bản trung gian của campaign nạp scale trước weight O nên weight bắt đầu muộn.',
  best:'`o_weight_scale`, `post_attn_rms_weight`, `residual` nạp **sau** stream weight, ngay khi projection bắt đầu, nên weight O phát ở ~2K thay vì ~7K.',
  how:'Không bớt byte nào, chỉ đổi thứ tự trong FIFO để weight lớn khởi động sớm. Đổi lại ba lệnh này (1 engine, packet 240 B lệch 256) nằm trên tail.',
  ev:'Đo so với bản trung gian của campaign, không phải trực tiếp baseline: 51,7K → 46,3K (−10,5%) khi dời scale xuống tail (SDK 0.6.0). Thử nạp operand lên đầu kernel trên 0.8.1 thua +1,4…+2,3K.',
  org:'A',nodes:['dep','hs','hn','hr']}
],
ffn:[
 {n:1,tag:'imp',t:'Hai cluster chia trục L (7.680 mỗi cluster); up/gate 256 slice × 30 hàng',
  base:'`Cluster = m![1 # 2]`: chỉ cluster 0. Up/gate 60 hàng/slice, xử lý 4 hàng mỗi pass × 15 pass; down 120 hàng/slice × 30 pass.',
  best:'Up/gate mỗi cluster giữ 7.680 hàng: 256 slice × 30 hàng NVFP4 (packet 256 B căn). Down: 256 slice × 15 hàng (d15).',
  how:'Mỗi cluster chỉ stream nửa byte weight (14,75 MB/cluster mỗi ma trận, ~304–308 B/cycle/cluster) và số pass giảm một nửa.',
  ev:'Trace 93413: ba stream 47,8K + 48,2K + 49,6K = 145,6K.',
  org:'A',nodes:['hu','hg','hd','du','dg','dd']},
 {n:2,tag:'imp',t:'Lookup NVFP4 → FP8 trực tiếp vào contraction; encode x hi/lo một lần cho up và gate',
  base:'f4 → lookup f8 → commit; nạp scale vào VRF; **nhân scale để dựng weight BF16 trong DM** (4 hàng mỗi pass) rồi mới contraction. Mỗi hàng đi qua nhiều pass.',
  best:'Weight f4 lookup thành f8 ngay trên đường fetch và vào thẳng contraction với x dạng hi+lo f8; lane sum và reciprocal fixed scale gộp trong epilogue contraction. Bỏ hẳn pass "materialize weight BF16".',
  how:'Bỏ một lượt ghi/đọc DM mỗi hàng và ~15 pass nhỏ. Bù lại lookup vẫn giới hạn ~2,5 B/cycle/slice nên stage1 dài ~22–23K, nhưng che dưới stream weight tiếp theo.',
  ev:'Trace 93413: stage1 up 22,5K, gate 23,3K chạy song song stream gate/down. Official chỉ đo cả kernel (3,70M → 215,6K).',
  org:'A',nodes:['enc','s1u','s1g']},
 {n:3,tag:'new',t:'Stage2 up/gate: nhân block scale bằng VRF theo tile (Ktiles)',
  base:'(bản 8,4451) scale FP8 → cast BF16 ghi DM → StoTrf → contraction thứ hai với partial BF16.',
  best:'Scale FP8 → `fetch_cast` FP32 → VRF trên Sub (mỗi tile 30 hàng × 48 block = 5.760 B/slice). Main: partial BF16 → FP32 → nhân scale VRF → reduce theo block; 5 tile × 48 block, rồi gộp 5 tổng tile.',
  how:'Bỏ convert scale sang BF16 trong DM + StoTrf + contraction thứ hai. Chỉ cây reduce FP32 đổi nên không bit-exact với trước nhưng PASS.',
  ev:'Lịch tĩnh gần như không đổi (108.454 vs 108.457). Official FFN ~214,7–217,5K (8 lần); không tách được lợi ích riêng.',
  org:'C',nodes:['s2u','s2g']},
 {n:4,tag:'imp',t:'Down d15: NVFP4 direct + GeGLU hi/lo, trao đổi qua SRAM, Time Reducer, partial BF16',
  base:'Down 120 hàng/slice × 30 pass: decode tile weight thành BF16, `to_dm_view` DM→DM regroup cột, rồi contraction 4 hàng mỗi lượt; nhiều lần đọc/ghi DM.',
  best:'256 slice × 15 hàng nguyên (3.840 B liền căn), NVFP4 stream thẳng vào contraction với GeGLU dạng lane hi+lo; hai lane cộng bằng Time Reducer; partial lưu BF16 qua `commit_trim().commit_cast::<bf16>()`; rows chuyển qua SRAM, mỗi scalar FP32 một ô 8 byte.',
  how:'Không còn pass decode ngay sau down weight, bớt pass cộng lane riêng, giảm một nửa byte partial SRAM và để Cast Engine rảnh.',
  ev:'Official: d15 + SRAM → epilogue → Time Reducer 8,2873 → 8,3067 (`a2e1f3dc`, `f5bc28aa`); partial BF16 qua Commit Adapter 8,4451 (`9d65fb99`).',
  org:'B',nodes:['s1d','s2d','ag']},
 {n:5,tag:'imp',t:'GeGLU: pool 30 hàng, nhân global scale, GeLU trên VRF',
  base:'`geglu` dùng switch `Broadcast1{2,1}` cho up và gate riêng, nạp `up/gate_global_scale` bằng `to_dm` riêng và nhiều pass trung gian.',
  best:'`pool_rows` gộp 30 hàng của 8 slice liên tiếp (f32), `scale_pooled` nhân global scale và round bf16, GeGLU chạy trên VRF; kết quả encode hi/lo cho down.',
  how:'Bớt broadcast và bớt lệnh DMA/pass cho global scale (mỗi lệnh nhỏ tốn 1–2K).',
  ev:'Không có A/B riêng; nằm trong tổng 215,6K.',
  org:'A',nodes:['pu','pg','geglu']},
 {n:6,tag:'imp',t:'Pre/post RMSNorm, layer scalar và trao đổi rows trên layout 32 slice',
  base:'`normalize` × 2, `residual::add`, `scale_by_layer_gate` là các hàm riêng; x replicate bằng `to_dm`; kết quả quay về layout `Slice` rồi mới store.',
  best:'Norm trên 32 slice `HiddenChunks` (all-gather một lượt ring ~2K cycle thay vì ~61K của broadcast một nguồn); RMS fused: sqrt thẳng VRF, giữ EPS và đủ 32 số hạng; layer scalar bf16 nạp thẳng VRF trên Sub nên không có pass Main chen giữa các pass down; output rơi thẳng vào `HiddenChunks`.',
  how:'Bớt pass và bớt commit/nạp lại DM ở đầu và cuối kernel. Head còn 11,2K và tail 38,8K chủ yếu vì stage1 down và trao đổi rows.',
  ev:'Official: pre-norm sqrt thẳng VRF 8,2338 (`bc340e78`); post-FFN full-ring 8,0215 (`9866de51`).',
  org:'B',nodes:['pre','post','xfer','out','head']}
]};
// Phần baseline làm nhưng best đã bỏ (vẽ thành dải "ghost" dưới sơ đồ)
const GHOST={
qkv:[
 {t:'Chỉ cluster 0 làm việc',l:['`Cluster = m![1 # 2]`','cluster 1 rảnh; cả 31,5 MB','weight qua một nửa chip'],to:[1]},
 {t:'Dequant weight f8 → BF16 vào DM',l:['lookup f8→bf16 (2,6 B/cycle)','ghi weight BF16 vào DM, đọc lại','rồi contraction BF16 + pass scale'],to:[2]},
 {t:'Broadcast một nguồn 240 flit',l:['`x` trên 1 slice → `CustomBroadcast{256}`','~61K cycle (ước lượng comment)','hoặc replicate DMA 1,97 MB'],to:[3]},
 {t:'Norm head + RoPE nhiều hàm rời',l:['3 hàm norm riêng; rope: DM → InterTranspose','→ rotate_half nhiều pass → Broadcast1','mỗi bước commit DM rồi nạp lại'],to:[4,5]}],
sao:[
 {t:'Chỉ cluster 0; broadcast xq ×256',l:['`CustomBroadcast{256}` + bitmap 16 KB','mỗi slice tính đủ 1.024 cột'],to:[1,2,3]},
 {t:'4 tile weight × (DMA + lookup + contraction)',l:['CHUNK = 1024; ghi weight BF16 vào DM','`add_partials` 3 pass + 1 pass channel scale'],to:[4]},
 {t:'Tail nhiều hàm rời',l:['`to_dm` y → `normalize` (ReducingSlices)','→ `residual::add` theo tile → `to_hbm`'],to:[5]},
 {t:'Operand nhỏ nạp trước weight (bản trung gian)',l:['scale đứng trước weight O trong FIFO','weight O phát ~7K thay vì ~2K'],to:[6]}],
ffn:[
 {t:'Chỉ cluster 0, 60 hàng/slice, 4 hàng/pass',l:['15 pass up, 15 pass gate','120 hàng/slice × 30 pass cho down'],to:[1]},
 {t:'Dựng weight BF16 trong DM',l:['f4 → f8 → nhân scale thành BF16 vào DM','rồi mới contraction; mỗi hàng qua nhiều pass'],to:[2,3]},
 {t:'Down: decode + regroup DM→DM',l:['decode tile → BF16 → `to_dm_view`','contraction 4 hàng/lượt; partial FP32'],to:[4]},
 {t:'GeGLU, norm, residual, gate rời',l:['`geglu` Broadcast1{2,1} + global scale','`normalize` ×2, `residual::add`, `scale_by_layer_gate`'],to:[5,6]}]
};
function impByNode(key){const m={}; IMP[key].forEach(function(it){it.nodes.forEach(function(id){(m[id]=m[id]||[]).push(it);});}); return m;}
function markCfg(key,cfg){const m=impByNode(key); cfg.nodes.forEach(function(n){n.imps=m[n.id]||[];});
  cfg.key=key; cfg.ghosts=GHOST[key]; cfg.eqIds=(EQN[key]||[]); return cfg;}
// nút giữ như baseline (không đổi cơ chế)
const EQN={qkv:['hx','ok','ov','oq'],sao:[],ffn:['hx']};
function badgeRow(key){const b=BASE[key];return b.name+': baseline '+fmt(b.b)+' → best '+fmt(b.o)+' cycle (×'+(b.b/b.o).toFixed(2).replace('.',',')+')';}
function is91(key){return key.slice(-2)==='91';}
function tagName(key,it){return is91(key)?(it.tag==='imp'?'giữ từ 8,4643':'mới so với 8,4643'):TAGN[it.tag];}
function impTable(key){const v=is91(key);
  let h='<thead><tr><th>#</th><th>Loại</th><th>Phần</th><th>Baseline làm gì</th>'+(v?'<th>Bản 8,4643 làm gì</th>':'')+'<th>'+(v?'Bản 9,1481 làm cách nào':'Best làm cách nào')+'</th><th>Vì sao nhanh hơn</th><th>Bằng chứng</th><th>Xuất xứ</th></tr></thead><tbody>';
  IMP[key].forEach(function(it){h+='<tr class="rowlink" data-n="'+it.n+'"><td class="n"><span class="nb '+it.tag+'">'+it.n+'</span></td><td>'+esc(tagName(key,it))+'</td><td><b>'+md(it.t)+'</b></td><td>'+md(it.base)+'</td>'+(v?'<td>'+md(it.prev||'')+'</td>':'')+'<td>'+md(it.best)+'</td><td>'+md(it.how)+'</td><td>'+md(it.ev)+'</td><td>'+esc(ORG[it.org])+'</td></tr>';});
  return h+'</tbody>';}
function impPanelHtml(key,n){const its=n.imps||[]; if(!its.length&&!(DG_EQ(key,n.id))) return '';
  let h='<h4>So với baseline ban tổ chức</h4>';
  if(!its.length) return h+'<p>Giữ như baseline: cùng tensor và cùng phép tính.</p>';
  its.forEach(function(it){h+='<div class="chgbox"><span class="nb '+it.tag+'">'+it.n+'</span> <b>'+md(it.t)+'</b> <span>('+esc(tagName(key,it))+')</span>'+
    '<p><b>Baseline:</b> '+md(it.base)+'</p>'+(it.prev?'<p><b>Bản 8,4643:</b> '+md(it.prev)+'</p>':'')+'<p><b>'+(is91(key)?'Bản 9,1481':'Best')+':</b> '+md(it.best)+'</p><p><b>Cách cải thiện:</b> '+md(it.how)+'</p><p><b>Bằng chứng:</b> '+md(it.ev)+'</p><p class="site">Xuất xứ: '+esc(ORG[it.org])+'</p></div>';});
  return h;}
function DG_EQ(key,id){return (EQN[key]||[]).indexOf(id)>=0;}
