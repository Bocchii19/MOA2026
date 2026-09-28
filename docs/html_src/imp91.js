// ===== Bản score_9_3 (commit 0abcde6, official 9,1481) =====
// tag: new = thay đổi mới so với bản 8,4643; imp = cơ chế đã có ở 8,4643 (cải tiến so với baseline) và được giữ
BASE.qkv91={b:250514,o:68963,name:'QKV'}; BASE.sao91={b:404633,o:35920,name:'SAO'}; BASE.ffn91={b:3703473,o:197954,name:'FFN'};
const ORG91={
 W:'package `score_9_3` (đồng đội), đã paired A/B',
 O:'kế thừa từ bản 8,4643',
 R:'round 9–10 của score_9_3 (2026-09-26)'};
Object.assign(ORG,ORG91);
const K91={qkv:'qkv91',sao:'sao91',ffn:'ffn91'};
IMP.qkv91=[
 {n:1,tag:'new',t:'Weight Q/K/V theo hình học 15 strip × 256 B',
  base:'Mỗi slice giữ nguyên hàng weight 3.840 B (Q: 256 slice × 16 hàng).',
  prev:'Q32/K8/V8: mỗi slice giữ hàng nguyên, ~299 / 277 / 285 B/cycle/cluster (QKV chậm hơn SAO/FFN 5–8%).',
  best:'H = 3.840 = 15 strip × 256 cột. Slice (nhóm hàng, strip s) giữ 128 (Q) hoặc 64 (K/V) đoạn đúng 256 B; vòng lặp trong của mỗi engine đi hết một hàng 3.840 B qua 15 slice strip (stride 256, bit 8 của địa chỉ HBM đổi liên tục). Slot strip thứ 16 để trống.',
  how:'Đọc strip đạt ~630 B/cycle so với 604 / 579 / 583 B/cycle của layout hàng nguyên (số đo trên phần cứng trong `qkv_s.rs`). Inter-Slice Reducer cộng 15 strip.',
  ev:'Paired A/B −2.198 và verify −2.232 (commit `0dd4d87`).',
  org:'W',nodes:['hq','hk','hv','dq','dk','dv']},
 {n:2,tag:'new',t:'Prologue: x và rms_weight là "hạt quarter-strip", phát bằng Broadcast1 ring 64',
  base:'x đi qua `broadcast_hidden` một nguồn (`CustomBroadcast{256}` + bitmap 16 KB).',
  prev:'x + norm weight nạp thành 32 chunk × 120, all-gather `Broadcast01{4,8,30}` (cần bitmap 16 KB, 1,46K trên hàng đợi).',
  best:'rms weight rồi x nạp thành hạt: mỗi engine ghi 60 mảnh 128 B vào 15 đích; một pass `Broadcast1{4,16}` (ring 64) cho mỗi hạt. rms weight đi trước, neo `sum(0·w)` (VRF) bắt scheduler phát DMA weight K ngay sau x; x DMA phát 8 cycle sau khi w về.',
  how:'Bỏ bitmap CustomBroadcast và DMA bitmap 16 KB khỏi head; Broadcast1 thường ngắn hơn all-gather cũ; đường DMA weight lớn khởi động sớm hơn.',
  ev:'B−A −726 / −670, verify **−459** (9/9 vòng âm). Đặt x trước rms weight thì +404.',
  org:'R',nodes:['hx','dsm','nrms']},
 {n:3,tag:'new',t:'Contraction packet 64 B, MỘT lần nạp TRF strip cho cả Q, K, V',
  base:'Contraction packet 32 B; x bf16 trong TRF; weight BF16.',
  prev:'Contraction 32 B: 1.920 bước Outer; V contraction 6,1K nằm trên tail (tĩnh 1,2K).',
  best:'x tách hi/lo trên layout strip (`hi = f8(16x)`, `lo = f8(16x − hi)`); mỗi slice chỉ cần strip 512 B, nạp TRF một lần dùng cho cả ba contraction. Packet 64 B (hai flit weight, hi/lo là broadcast theo Time bên TRF): 1.024 bước Outer.',
  how:'Contraction ~1,4–1,7K cố định + ~1,9 cycle mỗi bước Outer: packet 64 B giảm một nửa số bước; bớt hai lần nạp TRF.',
  ev:'V contraction 3,2K → 2,15K trên phần cứng (K1 A+V64 −1.295, chạy lại −1.314, 6/6 vòng). Kết quả nhân 1/16 rồi round bf16.',
  org:'W',nodes:['nenc','cq','ck','cv']},
 {n:4,tag:'new',t:'Gom head bằng Broadcast1 ring 64 theo thứ tự kênh',
  base:'`InterTranspose` regroup + `Broadcast1{8,1}` trả về, nhiều pass.',
  prev:'`Broadcast1{32,2}` / `{8,4}` regroup trên ring lớn, kèm pass transpose K.',
  best:'Bốn nhóm hàng của mỗi PE (slot 0 của slice 64a + 16c) gom vào slice đầu của PE: mỗi nguồn gửi 128 hàng thành 8 packet 16 kênh, `Broadcast1{4,16}` (ring 64) nối `c` trong Time nên ra đúng thứ tự kênh `64e + 16c + p`.',
  how:'Ring 64 thay ring 256: một lần gom 785 cycle thay vì 2.427 (số đo ở ver16); bỏ pass transpose riêng nên chuỗi K xong sớm hơn.',
  ev:'Đuôi V 11,6K → 8,3K khi đổi ring (RESULTS ver16); không A/B riêng ở bản này.',
  org:'W',nodes:['pq','pk','pv']},
 {n:5,tag:'new',t:'Scale gộp vào norm theo head; K 5 pass; sqrt của V ghi thẳng VRF',
  base:'Pass nhân scale riêng rồi `normalize_*` là các hàm rời.',
  prev:'Scale nhân trong projection (sq/sk/sv riêng), head norm và RoPE vào VRF.',
  best:'`normalize_query/key_scaled`: mean square của `s·x`, rồi `w·x·s/rms → bf16` (bỏ pass scale riêng và một lần round bf16). Scale K stream cùng K đã gom (không nạp thẳng VRF): 5 pass thay vì 6. Sqrt của V ghi VRF từ Main.',
  how:'Bớt một pass và một điểm round; dời lệnh nạp `rope_offset` vào lúc stream K nên không mở rộng khe K→Q.',
  ev:'Paired A/B, 7/7 vòng: K1 −400…−700. V sqrt→VRF: −841 / −773, verify −903.',
  org:'W',nodes:['pq','pk','pv']},
 {n:6,tag:'new',t:'RoPE tính trên chip từ `rope_offset` (không gather cos/sin)',
  base:'`dma_gather_scaled` cos và sin, `to_dm`, InterTranspose, nhiều pass `rotate_half`.',
  prev:'Gather cos/sin vào một slice, pack, một store HBM, các placement nạp lại (khe Q→K 6,8K).',
  best:'Một load 4 B `rope_offset` rồi ~22 pass Main nhỏ trên 4 slice head, chạy khi Main rảnh dưới stream weight: tính `cos/sin(POS·f(i))` với `f(i) = 10000^(-(i mod 128)/128)`; dải f dựng bằng 8 lần nhân đôi, một pass Transpose đóng gói; `apply_rope_r21` dùng 2 pass mỗi tensor.',
  how:'Thay hai gather, cặp store/load HBM trung gian và sync ngẫu nhiên write→read (trace ban: 6,6K DMA + 1–20K sync, rồi 5K RoPE chờ ở đuôi).',
  ev:'Số trace nằm trong comment `rope_h4.rs`. ⚠ Kernel **không đọc tensor `cos`/`sin`** truyền vào, chỉ đúng khi bảng là bảng chuẩn θ = 10.000 như fixture; cần rà với quy tắc "không special-case input" trước khi chốt.',
  org:'W',nodes:['rp','rope']},
 {n:7,tag:'imp',t:'Giữ từ 8,4643: hai cluster chia head, weight FP8 vào thẳng contraction, activation hi+lo',
  base:'Chỉ cluster 0; weight f8 → bf16 vào DM qua lookup 2,6 B/cycle.',
  prev:'Cùng cơ chế.',
  best:'Cluster 0 lấy Q head 0–7, K/V head 0–3, cluster 1 nửa còn lại; weight f8 không dequant; K/V scatter và q store giữ nguyên.',
  how:'Xem bảng cải tiến của bản 8,4643.',
  ev:'Official 68.963 (K1) so với 78.277 của 8,4643.',
  org:'O',nodes:['cq','ck','cv','oq','ok','ov']}
];
IMP.sao91=[
 {n:1,tag:'new',t:'Hình học G3: 16 nhóm hàng × 16 strip, packet 64 B, hi/lo trong hai lane TRF',
  base:'Broadcast xq ×256, weight dequant BF16, 4 tile.',
  prev:'16 nhóm hàng × 120 và 16 chunk Qs × 256; contraction 32 B; reduce 16 chunk bằng `vector_inter_slice_reduce`; ra 1 stream weight duy nhất.',
  best:'Mỗi slice giữ 120 đoạn đúng 256 B (30.720 B); x strip 256 cột tách hi/lo vào hai lane TRF (`hi = f8(128x)`); contraction packet 64 B; Inter-Slice Reducer cộng 16 strip và để đúng một slice epilogue cho mỗi nhóm hàng, nên gather về C0 chỉ chuyển 32 mảnh × 480 B.',
  how:'Bước Outer giảm một nửa, không còn tách hàng ra nhiều slice sau reduce.',
  ev:'B−A −879, chạy lại −815 (6/6 vòng) khi kết hợp seed x (mục 2).',
  org:'W',nodes:['proj','hw']},
 {n:2,tag:'new',t:'x nạp "quarter-live seed" rồi Broadcast1{16,16} + compaction',
  base:'xq replicate vào mọi slice bằng CustomBroadcast{256}.',
  prev:'B2: xq nạp vào 128 slice sản xuất, encode một lần, `Broadcast1{2,16}` khi nạp TRF.',
  best:'DMA ghi 128 B (64 cột) vào một phần tư số slice, 16 đích mỗi engine; ring-16 `Broadcast1{16,16}` cho mỗi nhóm hàng đủ 4 mảnh 64 cột của strip; pass compaction bỏ padding.',
  how:'x về TRF ở ~1,33K thay vì ~1,73K khi replicate; không cần bitmap.',
  ev:'Nằm trong C1 + C2 (−879); replicate thuần chỉ −180 (không đủ).',
  org:'W',nodes:['hxq','dxq','enc']},
 {n:3,tag:'new',t:'Weight O chia hai tile 92 + 28 hàng, chuỗi 32 bản sao để xếp DMA',
  base:'4 tile weight × (DMA + lookup + contraction).',
  prev:'Một lệnh DMA 15,7 MB, contraction chỉ chạy được sau byte cuối (projection 3,4K trên tail).',
  best:'Tile A 92 hàng, tile B 28 hàng: contraction của tile cuối ngắn. Chuỗi 32 pass copy đồng nhất x (không đổi giá trị) làm x_trf sẵn sàng muộn để scheduler xếp DMA tile B trước contraction A. Cửa sổ 31–32 bản sao: 33 đặt chờ gather trước DMA residual, ≥ 34 xếp weight_scale trước contraction B.',
  how:'Tile cuối về sớm hơn và tail sau byte weight cuối ngắn hơn.',
  ev:'Tile 90+30 hoặc 31 bản sao: ±10…−31 (nhiễu); giữ 92+28, 32 bản sao. Một copy tốn ~0,35–1,3K.',
  org:'W',nodes:['dwa','dwb','delay']},
 {n:4,tag:'new',t:'Tile và mảnh x nhóm theo (Qs bit 11, Qs bit 8) để bit stack đổi trong vòng trong',
  base:'—',
  prev:'Stream W_O ~307 B/cycle/cluster (sát trần của họ layout đó).',
  best:'Vòng lặp trong của DMA tile luân phiên bit 8 (bit stack HBM) mỗi 2 yêu cầu.',
  how:'Tile 24 hàng: 22,7K → 20,8K (554 → 605 B/cycle).',
  ev:'B−A −1.497, verify −1.380.',
  org:'W',nodes:['dwa','dwb']},
 {n:5,tag:'new',t:'Epilogue: norm weight do pass cuối stream, y giữ trong VRF, residual VRF ngay sau gather',
  base:'`to_dm` y → normalize (ReducingSlices) → residual → store.',
  prev:'3 lệnh operand (1 engine, 240 B lệch) sau stream weight rồi 4 pass RMS + pass Sub residual 1,13K.',
  best:'Epilogue 32 slice × 120 kênh: mean square, all-gather reduce ring-32, sqrt, final. Trọng số norm được pass epilogue cuối stream trực tiếp; y giữ VRF; pass residual VRF phát ngay sau gather. DramReuse release ~34K thay vì ngay trước store.',
  how:'Pass VRF norm và chờ DMA của nó rời khỏi tail; sync DRAM sớm hơn.',
  ev:'B−A −830, verify −799; cộng D: −1.697 (38.527 → 36.830 ở ver16).',
  org:'W',nodes:['dep','epi','hs','hn','hr']},
 {n:6,tag:'new',t:'Buffer đích của gather do một pass Sub tạo (J1)',
  base:'—',
  prev:'DmaStos gom y C1→C0, `ExplicitSync` 1,39K trên đường găng, sync khởi động kernel ~1,3K.',
  best:'Buffer gather sinh từ pass Sub trên x nên sync sẵn sàng rời khỏi đầu kernel: tile A phát ở ~1,5K thay vì ~1,9–2,0K.',
  how:'Dời cluster_sync đầu kernel; chi phí trao đổi xuyên cluster còn ≤ ~0,8K trong chế độ chấm.',
  ev:'−202 / −178 (6/6 vòng). Bỏ hẳn trao đổi (probe, numerics sai) chỉ +52.',
  org:'W',nodes:['gath']},
 {n:7,tag:'imp',t:'Giữ từ 8,4643: hai cluster chia 3.840 hàng, weight FP8 vào thẳng contraction, activation hi+lo',
  base:'Chỉ cluster 0; weight dequant BF16; 4 tile.',
  prev:'Cùng cơ chế.',
  best:'1.920 hàng mỗi cluster; O weight f8 không dequant; x ×2⁶ tách hi/lo.',
  how:'Xem bảng cải tiến của bản 8,4643.',
  ev:'Official 35.920 (K2) so với 36.674.',
  org:'O',nodes:['out','hw']}
];
IMP.ffn91=[
 {n:1,tag:'new',t:'Block scale căn 256 B, Up và Gate qua MỘT InterTranspose ring 16',
  base:'Scale FP8 nạp qua VRF theo pass 4 hàng.',
  prev:'Scale up/gate packet 800 B lệch 256: ~10,2K mỗi ma trận (~184 B/cycle).',
  best:'Khối 16 hàng căn 3.840 B, mỗi slice có 30 hàng scale liền (7,2 KiB); Up và Gate cùng đi qua một `InterTranspose{16,1}` (interleave trên `Iug`), nên DMA scale Gate xếp trước weight Gate và hand-over chạy trong pha Up.',
  how:'Scale service 9,8–10,7K → 7,3K; DMA scale có consumer sớm hơn nên FIFO không bị nghẽn sau weight.',
  ev:'Paired A/B −4.659 / −6.988, verify −4.750 (commit `7ce2534`).',
  org:'W',nodes:['sc','du','dg','hu','hg']},
 {n:2,tag:'new',t:'Scale khối bằng contraction đường chéo BF16 (thay nhân scale FP32 theo tile)',
  base:'Weight f4 → f8 → nhân scale dựng weight BF16 trong DM.',
  prev:'Ktiles: scale FP8→FP32 vào VRF (Sub), Main nhân partial × scale theo 5 tile × 48 block (~14,4K mỗi ma trận).',
  best:'Contraction fused lookup FP4→FP8 cho 30 hàng; partial dot block-16 giữ BF16 (30 × 240 × 4 B = 28,8 KiB); sau đó một contraction đường chéo với scale BF16 gom theo pass 6 hàng (VRF chứa tối đa 8,5 hàng scale). Áp dụng cho Up, Gate và 12 hàng đầu của Down.',
  how:'Không còn nhân FP32 theo tile; DMA weight của Up/Gate là một lệnh, decode một lần.',
  ev:'Up diagonal gần trung tính trong K3 riêng (−119) nhưng −988 / −831 trong harness đủ (6/6 vòng âm).',
  org:'R',nodes:['s1u','s2u','s1g','s2g']},
 {n:3,tag:'new',t:'Down chia theo K giữa hai cluster; GeGLU ở lại trên chip',
  base:'Down 120 hàng/slice, 30 pass, decode + DM→DM regroup.',
  prev:'Down d15: 256 slice × 15 hàng mỗi cluster (chia theo L), GeGLU lưu HBM/DM rồi all-gather lane; DMA lane + bitmap.',
  best:'Mỗi cluster giữ một K-half 7.680 và Down tiêu thụ đúng phần GeGLU mà cluster đó vừa tính. GeGLU chạy trên 32 slice/cluster sau một hoán vị DM→DM trong cluster (120 phần tử mỗi slice, mọi packet nguyên). Chỉ partial cỡ H đi qua cluster ở epilogue.',
  how:'Bỏ store + reload HBM của GeGLU và sync đi kèm; DMA commands 28 → 23; lịch tĩnh 107.226 → 103.020.',
  ev:'K3 riêng 208.552 → 203.258 (−5.294; 31 seed, job 99368); harness đủ −4.923 (job 99310). Không kết hợp được với "Down chia theo L".',
  org:'R',nodes:['pu','pg','geglu','hd','dd']},
 {n:4,tag:'new',t:'Hand-over activation: ring-8 replicate + stride-8 all-gather, packet 24 B FP8, TRF load trên Main',
  base:'`CustomBroadcast{256}` replicate.',
  prev:'`CustomBroadcast{256}` + bitmap 16 KB: all-gather lane 9,06K + nạp TRF 4,15K.',
  best:'Hai pass Main `Broadcast1` (ring 32 stride-8 replicate, ring 8 all-gather) trên packet 24 B FP8 hi/lo, rồi nạp TRF; không cần DMA cấu hình switch.',
  how:'TRF load trên Main 1,3K so với 4,2K trên Sub; bỏ DMA bitmap.',
  ev:'Hand-over cục bộ trên Main −4.502 / đảo thứ tự −4.267; trên Sub +5.163.',
  org:'R',nodes:['ag','pre']},
 {n:5,tag:'new',t:'Epilogue hợp nhất trên 8 slice × 480 kênh, store thẳng HBM',
  base:'`normalize` ×2, `residual::add`, `scale_by_layer_gate` rời.',
  prev:'DmaStos rows C1→C0 2,2K + ExplicitSync 3,8K, rồi ~6 pass post-FFN ~4,3K.',
  best:'Down global scale, RMSNorm sau FFN, residual, layer gate trong một epilogue 8 slice × 480; kết quả rơi đúng layout để `to_hbm`. Partial H của hai cluster gom về C0 và cộng bằng một intra-slice reduction.',
  how:'Thay gather 1 slice, HBM round trip, cast, scale, normalize (relayout 8 slice + broadcast), residual 2 tile, layer gate: ~14K cycle tĩnh nối tiếp sau contraction cuối. Round: `bf16(y·gs)` và `bf16((n+r)·gate)`, lệch ≤ 1 ulp bf16.',
  ev:'Reduction một pass −922 (K3 riêng); tail sau w1 ~12K (g3 5,2K + push/sync + 5 pass + store).',
  org:'R',nodes:['xfer','epi','out']},
 {n:6,tag:'new',t:'Đặt việc đúng context: nạp TRF, đổi scale FP8→BF16 trên Main; spread x không cần config',
  base:'—',
  prev:'Nạp TRF và giải mã scale trên Sub (chậm hơn ~3,5×/flit).',
  best:'Nạp TRF GeGLU và đổi scale Down FP8→BF16 (+ nạp TRF scale) trên Main ngay sau decode; spread x bằng `Broadcast1` ring-32 stride-8 + ring-8, không cần bitmap.',
  how:'Khi đổi scale ở Sub còn dở thì lowering chèn `raw.wait sub_context` không nguồn trước DMA bảng g3, giữ w1 chậm 2,6K.',
  ev:'M5: −4.854 / −4.985, verify −4.932; K3 median 198.287 trên 45 seed PASS.',
  org:'R',nodes:['enc','s1d','ag']},
 {n:7,tag:'new',t:'Down: 12 hàng fused + 3 hàng cuối tile riêng; scale hàng 4..11 một lệnh DMA',
  base:'Decode tile 4 hàng/pass → BF16 → DM→DM.',
  prev:'Down d15 với scale packet 480 B lệch (~10,7K).',
  best:'12 hàng đầu: lookup fused → partial BF16 → contraction đường chéo; tile 3 hàng cuối tách để DMA weight của nó chạy chồng; scale hàng 4..11 nạp bằng một lệnh 8 hàng; contraction nhóm dùng packet 64 B + `LaneMode::Sequential`.',
  how:'Các pass nhóm nhanh gấp 2 (5,7/6,5/8,0K → 2,8/3,6/3,6K); bớt một lệnh DMA (−4,3K).',
  ev:'Gộp cả 15 hàng: +18.501 (hoặc +5.035 với hand-over cục bộ).',
  org:'W',nodes:['dd','s1d']},
 {n:8,tag:'imp',t:'Giữ từ 8,4643: hai cluster, NVFP4 lookup trực tiếp vào contraction, x hi/lo dùng chung Up/Gate',
  base:'Chỉ cluster 0, dựng weight BF16.',
  prev:'Cùng cơ chế.',
  best:'Up/Gate 256 slice × 30 hàng; lookup f4→f8 trên đường fetch; encode x hi/lo một lần.',
  how:'Xem bảng cải tiến của bản 8,4643.',
  ev:'Official 197.954 (K3) so với 215.644.',
  org:'O',nodes:['hu','hg']}
];
GHOST.qkv91=[
 {t:'Weight hàng nguyên Q32/K8/V8',l:['mỗi slice 1 hàng 3.840 B','~604 / 579 / 583 B/cycle'],to:[1]},
 {t:'Prologue: all-gather + bitmap',l:['x + norm weight 32 chunk × 120','Broadcast01 cần DMA bitmap 16 KB'],to:[2]},
 {t:'Contraction 32 B, TRF riêng',l:['1.920 bước Outer','V contraction 6,1K trên tail'],to:[3,5]},
 {t:'Regroup trên ring 256',l:['gom head ~2,4K mỗi lần','transpose K riêng'],to:[4]},
 {t:'RoPE: gather cos/sin + store HBM',l:['2 dma_gather ~1,8K + pack','khe Q→K 6,8K'],to:[6]}];
GHOST.sao91=[
 {t:'B2: xq 128 slice + Broadcast1{2,16}',l:['DMA xq 1,4K + encode Main/Sub','TRF 2 slot vật lý'],to:[1,2]},
 {t:'Một stream W_O, contraction sau byte cuối',l:['projection 3,4K trên tail','weight không nhóm theo bit stack'],to:[3,4]},
 {t:'Tail: operand sau weight, RMS 4 pass',l:['3 DMA 240 B lệch, 1 engine','+ pass Sub residual 1,13K'],to:[5,6]}];
GHOST.ffn91=[
 {t:'Scale lệch 256 B (800 / 480 B)',l:['scale up/gate ~10,2K mỗi ma trận','~184 B/cycle'],to:[1]},
 {t:'Ktiles: nhân scale FP32 theo tile',l:['5 tile × 48 block, Sub + Main','≈ 14,4K mỗi ma trận'],to:[2]},
 {t:'Down d15 chia theo L + GeGLU qua DMA',l:['all-gather lane 9,06K + bitmap','TRF load trên Sub 4,15K'],to:[3,4,6]},
 {t:'Epilogue nhiều pass sau Down',l:['DmaStos rows 2,2K + sync 3,8K','post-FFN ~6 pass 4,3K'],to:[5]}];
EQN.qkv91=['hq']; EQN.sao91=[]; EQN.ffn91=[];  // không nút nào giữ hệt (đều liên quan)
const CLOSED91={
qkv:[['Gom V trong hai pass norm (ring-64 + 32 flit vector)','+1.411','switch và vector cộng dồn, không chồng (~1,7K)'],
     ['V ring gather hợp nhất / R: scale trước V, VRF dựng dưới V','+4.125','K-chain xếp sau contraction Q làm chậm'],
     ['Chia V thành tile hàng (2+2, 3+1)','+723 / +1.027','thêm điểm nối DMA, tile nhỏ stream chậm'],
     ['x nạp trước rms weight','+404','seed thứ tự sai làm khe K→Q giãn'],
     ['Hạt quarter với thứ tự đầu cũ / f32 no-op','+854 / +783','chuỗi K norm/RoPE bị xếp sau Qc']],
sao:[['K2 epilogue theo cụm (strip-major)','+4.753','Inter-Slice Reducer phải reduce chữ số slice trong cùng; nhóm hàng ở stride 16'],
     ['ms + reducer r=32 + sqrt gộp','+4.308','inter-slice reduce 1 flit r=32 ~4,4–4,6K'],
     ['Tile A DMA trước, x sau','+2.424 / V5 +750','pass Broadcast1 cạnh stream weight chậm ~7× (~3,3K)'],
     ['Tile 64 B với tile 6 hàng','+384','scheduler chỉ gắn weight_scale với contraction cuối nếu ≥ 616 cycle tĩnh'],
     ['Bỏ hẳn trao đổi liên cluster (probe)','+52','trao đổi gần như miễn phí trong chế độ chấm']],
ffn:[['Down đủ 15 hàng cùng lúc','+18.501 (+5.035 với hand-over cục bộ)','—'],
     ['Up/Gate lookup gộp','+12.708','—'],
     ['Hand-over 120 giá trị theo shard','+4.380','—'],
     ['Broadcast01 đệm trực tiếp','+7.103','—'],
     ['Scale Down căn 256 trong layout K-split','+1.020','mảnh scale mỗi cluster là nửa K 480 B ở stride 960 B: chỉ hàng nguyên mới căn, gấp đôi byte'],
     ['Hand-over trên Sub','+5.163','Main nhanh hơn ~3,5×/flit']]};
// mốc đoạn của trace bản 9,1481 (cluster 0, job 104939, chế độ trace)
SEGBARS.push(
 {k:'QKV 9,1481',tot:76625,segs:[['hd','Head',6983,'rms_weight và x là hạt quarter-strip: DMA 97–1.469 và 4.389–5.757; spread ring 64; RMS strip; hi/lo; nạp TRF strip. Weight K phát ở 6.983'],['wt','K weight',13298,'K stream 15-strip, 3,93 MB/cluster (thứ tự stream: K, Q, V)'],['gp','Khe K→Q',1442,'rope_offset (4 B, ~3,9K vì xếp sau K), sk, k_rms_weight'],['wt','Q weight',27836,'Q stream 21.723 → 49.559'],['gp','Khe Q→V',4766,'Q contraction 6,3K chạy song song; sq và q_rms_weight'],['wt','V weight',13726,'V stream 54.325 → 68.051'],['tl','Tail',8574,'norm V, q store 1,1K, scatter K 1,6K, scatter V 1,6K, sync cuối 1,3K']]},
 {k:'SAO 9,1481',tot:46535,segs:[['hd','Head',2039,'x seed 128 B 96–1.718; spread + compaction 3,3K; tile A phát ở 2.039'],['wt','Tile A',19911,'92 hàng; DMA 2.039 → 21.950 (chồng 32 bản sao x)'],['gp','Khe A→B',4337,'DMA tile B xếp sau tile A (FIFO)'],['wt','Tile B',6895,'28 hàng; 26.287 → 33.182'],['tl','Tail',13353,'contraction B 4,7K, scale 2,1K, gather 32×480 B, DramReuse 5,1K, epilogue 32 slice, store 0,9K']]},
 {k:'FFN 9,1481',tot:204858,segs:[['hd','Head + scale',23829,'x, rms, scale Up 7,1K + Gate 7,4K, normalize x 5,5K, bảng FP4 2,9K + StoTab 5,6K; Up weight phát ở 23.829'],['wt','Up weight',47465,'NVFP4 30 hàng/slice'],['gp','Khe Up→Gate',3340,'bảng FP4 3,3K + gate_global_scale'],['wt','Gate weight',48012,'74.634 → 122.646'],['gp','Khe Gate→Down',2083,'bảng FP4 2,0K'],['wt','Down w0',40237,'12 hàng; 124.729 → 164.966'],['gp','Khe w0→w1',13666,'scale Down 8,7K + 11,1K, bảng g3 1,9K + StoTab 7,3K, hand-over ring-8 12,2K'],['wt','Down w1',11280,'3 hàng cuối; 178.632 → 189.912'],['tl','Tail',14946,'contraction 3 hàng 5,2K, operand epilogue (4 DMA nhỏ), partial → C0, sync, 5 pass epilogue, store 0,8K']]});
// cfg sơ đồ ---------------------------------------------------------------------------------
function cfgQKV91(){
  const Z=[{x:15,w:185,t:'① HBM'},{x:210,w:180,t:'② DMA → DM'},{x:400,w:280,t:'③ Prologue trên layout strip'},{x:690,w:225,t:'④ Chiếu 15 strip'},{x:925,w:310,t:'⑤ Gom head · norm · RoPE'},{x:1245,w:245,t:'⑥ Ghi ra HBM'}];
  const nodes=[
   N('hx',20,50,175,90,'--hbm',['x · rms_weight','15.360 B mỗi cái'],'Activation và trọng số RMSNorm. Nạp thành hạt quarter-strip: mỗi engine ghi 60 mảnh 128 B vào 15 đích.',[['Thứ tự','rms weight trước, x sau (đảo thứ tự +404)']]),
   N('hq',20,300,175,72,'--hbm',['W_Q FP8 + sq','7,86 MB/cluster'],'Weight Q: mỗi slice 128 đoạn đúng 256 B; vòng trong của engine đi hết một hàng 3.840 B qua 15 slice.',[['Đo','~630 B/cycle strip vs 604 hàng nguyên']]),
   N('hk',20,400,175,72,'--hbm',['W_K FP8 + sk','3,93 MB/cluster'],'K: mỗi slice 64 đoạn 256 B.',[['Đo','579 → ~630 B/cycle']]),
   N('hv',20,500,175,72,'--hbm',['W_V FP8 + sv','3,93 MB/cluster'],'V: như K.',[['Đo','583 → ~630 B/cycle']]),
   N('dsm',215,50,170,90,'--load',['DMA hạt x / rms','60 × 128 B / engine','15 đích · không bitmap'],'Hai lệnh DMA nhỏ; không còn DMA bitmap 16 KB vì dùng `Broadcast1` thường thay `CustomBroadcast`.',[['Neo','`sum(0·w)` trên hạt w (Sub) để x phát sau 8 cycle và DMA weight K phát trước khi x về']]),
   N('dq',215,300,170,72,'--load',['DMA stream Q','15-strip'],'Stream Q, mỗi engine đi qua 15 slice strip liên tiếp.',[]),
   N('dk',215,400,170,72,'--load',['DMA stream K','15-strip'],'Stream K.',[]),
   N('dv',215,500,170,72,'--load',['DMA stream V','15-strip'],'Stream V.',[]),
   N('nrms',410,50,270,90,'--vec',['RMSNorm x trên layout strip','Σx² 15 strip · Inter-Slice Reducer','phát lại 16 slot'],'Mean square cộng 15 strip bằng Inter-Slice Reducer (slot chết loại bằng valid-count), giữ EPS và sqrt đầy đủ.',[]),
   N('nenc',410,165,270,90,'--tu',['hi/lo mỗi strip → TRF','hi = f8(16x) · lo = f8(16x−hi)','MỘT lần nạp TRF cho Q/K/V'],'Mỗi slice chỉ cần strip 512 B; TRF dùng chung cho cả ba contraction.',[]),
   N('cq',700,300,205,72,'--tu',['Contraction Q · 64 B','1.024 bước Outer'],'Packet 64 B (hai flit weight, hi/lo là broadcast theo Time bên TRF); Inter-Slice Reducer cộng 15 strip; nhân 1/16, round bf16.',[]),
   N('ck',700,400,205,72,'--tu',['Contraction K · 64 B',''],'Như Q.',[]),
   N('cv',700,500,205,72,'--tu',['Contraction V · 64 B','3,2K → 2,15K'],'V contraction từ 3,2K xuống 2,15K khi dùng packet 64 B.',[]),
   N('rp',925,50,310,55,'--hbm',['rope_offset','4 B'],'Chỉ một load 4 B. Bảng cos/sin không được đọc trong bản này.',[]),
   N('rope',925,125,310,110,'--vec',['RoPE tính trên chip (r17)','1 load 4 B + ~22 pass Main nhỏ','trên 4 slice head'],'cos/sin(POS·f(i)) dựng từ `rope_offset` (Q24,7 → 4·POS), range reduction, dải f bằng 8 lần nhân đôi + một pass Transpose. Chạy khi Main rảnh dưới stream weight.',[['⚠','Không đọc tensor `cos`/`sin` đầu vào']]),
   N('pq',925,300,310,72,'--vec',['gom head (ring 64) → norm(s·q)','→ RoPE (r21, 2 pass)'],'`Broadcast1{4,16}` gom 4 nhóm hàng của PE về slice đầu theo thứ tự kênh; scale gộp vào norm; RoPE 2 pass mỗi tensor.',[]),
   N('pk',925,400,310,72,'--vec',['gom head → norm K (5 pass)','→ RoPE'],'Scale K stream cùng K đã gom; lệnh nạp rope_offset dời vào lúc stream K.',[]),
   N('pv',925,500,310,72,'--vec',['gom head → norm V','sqrt → VRF trực tiếp'],'Sqrt của V ghi VRF từ Main.',[]),
   N('oq',1255,300,225,72,'--store',['q → HBM',''],'Store q ra HBM.',[]),
   N('ok',1255,400,225,72,'--store',['K → scatter K cache',''],'Scatter K theo `kv_offset` (giữ nguyên).',[]),
   N('ov',1255,500,225,72,'--store',['V → scatter V cache','lệnh cuối kernel'],'Scatter V; sau đó chỉ còn sync cuối.',[])
  ];
  return {W:1500,H:640,zones:Z,nodes:nodes,ghostTitle:'BẢN 8,4643 làm gì (phần score_9_3 đã thay); số ở góc = mục thay thế nó',
   edges:[['hx','dsm'],['dsm','nrms'],['nrms','nenc','b','t'],['cq','pq'],['ck','pk'],['cv','pv'],['pq','oq'],['pk','ok'],['pv','ov'],['rp','rope','b','t']],
   lines:[
     {pts:[[385,336],[700,336]],col:'--load',lab:'weight stream',lx:540,ly:328},{pts:[[385,436],[700,436]],col:'--load'},{pts:[[385,536],[700,536]],col:'--load'},
     {pts:[[680,210],[692,210],[692,362],[700,362]],col:'--tu'},{pts:[[692,362],[692,462],[700,462]],col:'--tu'},{pts:[[692,462],[692,562],[700,562]],col:'--tu'},
     {pts:[[1080,235],[1080,300]],col:'--vec'}],
   notes:[{x:20,y:600,t:'Trong trace bản này stream đi theo thứ tự K → Q → V trên cùng hàng đợi DMA; chuỗi hậu xử lý của stream trước chạy chồng stream sau.',f:'--ink-2'},
     {x:20,y:620,t:'Số ở các khối: WORK_LOG và comment source; timeline thật ở phần dưới (job 104939).',f:'--ink-3'}]};
}
function cfgSAO91(){
  const Z=[{x:15,w:185,t:'① HBM'},{x:210,w:165,t:'② DMA → DM'},{x:385,w:225,t:'③ x seed + hi/lo'},{x:620,w:240,t:'④ Chiếu G3 (2 tile)'},{x:870,w:200,t:'⑤ Gom về C0'},{x:1080,w:220,t:'⑥ Epilogue 32 slice'},{x:1310,w:180,t:'⑦ Ghi ra'}];
  const nodes=[
   N('hxq',20,55,175,80,'--hbm',['xq (đầu ra attention)','128 KB · bf16'],'Đầu vào.',[]),
   N('dxq',215,55,155,80,'--load',['DMA x seed','128 B → 1/4 slice','16 đích / engine'],'DMA ghi 128 B (64 cột) vào một phần tư số slice.',[]),
   N('enc',390,45,215,100,'--tu',['ring-16 Broadcast1{16,16}','+ compaction','hi/lo 2 lane TRF'],'Mỗi nhóm hàng có đủ 4 mảnh 64 cột của strip; x tách `hi = f8(128x)`, `lo` vào hai lane TRF.',[['Đo','x về TRF ~1,33K thay vì ~1,73K']]),
   N('hw',20,215,175,80,'--hbm',['W_O FP8','15,73 MB','120 đoạn 256 B / slice'],'Mỗi slice giữ 30.720 B = 120 đoạn đúng 256 B.',[]),
   N('dwa',215,190,155,60,'--load',['DMA tile A · 92 hàng','nhóm theo (Qs b11, b8)','bit stack đổi · ~605 B/c'],'Vòng trong luân phiên bit 8: 554 → 605 B/cycle.',[['Đo','tile 24 hàng 22,7K → 20,8K']]),
   N('dwb',215,262,155,60,'--load',['DMA tile B · 28 hàng','contraction cuối ngắn'],'Tile cuối ngắn để contraction sau byte cuối nhỏ.',[]),
   N('delay',390,330,215,80,'--core',['Chuỗi 32 bản sao x','(delay, giá trị không đổi)','xếp DMA tile B trước'],'Làm x_trf sẵn sàng muộn để scheduler đặt DMA tile B trước contraction A. Cửa sổ 31–32.',[['Chi phí','~0,35–1,3K mỗi bản sao trên phần cứng']]),
   N('proj',625,205,230,100,'--tu',['Chiếu G3 · 64 B packet','16 strip · Inter-Slice Reducer','1 slice epilogue / nhóm hàng'],'Contraction hai tile; Inter-Slice Reducer cộng 16 strip, để 120 hàng f32 trên slice (g,0).',[]),
   N('gath',875,205,190,100,'--sync-ink',['Gom 32 × 480 B → C0','buffer do pass Sub tạo','+ ExplicitSync'],'Gather xuyên cluster; buffer đích sinh từ pass Sub nên sync sẵn sàng rời đầu kernel.',[['Đo','tile A phát ~1,5K thay vì ~1,9–2,0K']]),
   N('epi',1085,205,210,100,'--vec',['Epilogue 32 slice × 120','mean sq → ring-32 reduce','→ sqrt → norm·w + res'],'Trọng số norm do pass cuối stream; y giữ VRF; pass residual VRF ngay sau gather.',[]),
   N('out',1315,205,170,100,'--store',['Store output','+ sync cuối'],'Store rồi sync cuối kernel.',[]),
   N('hs',20,350,175,58,'--hbm',['o_weight_scale','7.680 B'],'',[]), N('hn',20,420,175,58,'--hbm',['post_attn_rms_weight','7.680 B'],'',[]), N('hr',20,490,175,58,'--hbm',['residual','7.680 B'],'',[]),
   N('dep',215,350,155,198,'--load',['operand epilogue','norm weight stream','bởi pass cuối','residual VRF ngay','sau gather'],'Ba operand nhỏ vẫn chỉ C0 nạp, nhưng norm weight không còn pass VRF và DMA chờ riêng trên tail.',[])
  ];
  return {W:1500,H:600,zones:Z,nodes:nodes,ghostTitle:'BẢN 8,4643 làm gì (phần score_9_3 đã thay); số ở góc = mục thay thế nó',
   edges:[['hxq','dxq'],['dxq','enc'],['enc','proj','r','l','',{xm:615}],['hw','dwa','r','l','',{}],['dwa','proj','r','l','',{xm:378}],['dwb','proj','r','l','',{xm:378}],['delay','proj','r','l','',{xm:615}],['proj','gath'],['gath','epi'],['epi','out'],['hs','dep','r','l','',{xm:205}],['hn','dep'],['hr','dep','r','l','',{xm:205}]],
   lines:[{pts:[[370,449],[1190,449],[1190,305]],col:'--load',dash:1,lab:'operand → VRF',lx:730,ly:441}],
   notes:[{x:20,y:585,t:'Số đo K2: 35.920 official (36.674 ở 8,4643); tail bị chặn bởi 3 sync xuyên cluster. Timeline thật ở phần dưới.',f:'--ink-2'}]};
}
function cfgFFN91(){
  const nodes=[
   N('hx',15,45,185,70,'--hbm',['x · rms_weight','15.360 B ×2'],'Activation và trọng số RMSNorm.',[]),
   N('pre',215,45,255,70,'--vec',['RMSNorm x + spread','Broadcast1 ring-32 stride-8','+ ring-8 all-gather'],'Spread x không cần cấu hình switch (không DMA bitmap).',[['Ghi chú','đặt việc trên Main, không trên Sub']]),
   N('enc',485,45,255,70,'--tu',['x hi/lo f8 → TRF (Main)','dùng cho Up và Gate'],'Nạp TRF trên Main 1,3K thay vì 4,2K trên Sub.',[]),
   N('sc',755,45,300,70,'--load',['Scale căn 256 B','Up + Gate qua MỘT InterTranspose ring 16'],'Khối 16 hàng căn 3.840 B; scale Gate xếp trước weight Gate nên hand-over chạy trong pha Up.',[['Đo','scale service 9,8–10,7K → 7,3K; −4,7K']]),
   N('hu',15,165,185,78,'--hbm',['W_up NVFP4 + scale','256 slice × 30 hàng liền','scale 7,2 KiB / slice'],'Weight liền 57,6 KiB mỗi slice; scale 30 hàng × 240 B liền.',[]),
   N('du',215,165,175,78,'--load',['DMA up','1 lệnh weight + scale'],'Một lệnh weight và một lệnh scale.',[]),
   N('s1u',410,165,255,78,'--tu',['fused lookup f4→f8 + contraction','30 hàng · partial block-16 BF16'],'Decode FP4 một lần rồi stream hai lần với x hi/lo; partial 30 × 240 × 4 B.',[]),
   N('s2u',680,165,240,78,'--tu',['scale khối: contraction chéo BF16','pass 6 hàng'],'VRF chứa tối đa 8,5 hàng scale nên tile xong theo pass 6 hàng, mỗi pass ghi cặp hàng 8 B.',[]),
   N('pu',935,165,240,78,'--vec',['regroup 120/slice','DM→DM trong cluster'],'Hoán vị thành 64 slice × 120 phần tử: mọi packet nguyên (30 bf16 = 60 B không chia hết 8 B).',[]),
   N('hg',15,285,185,78,'--hbm',['W_gate NVFP4 + scale','256 slice × 30 hàng'],'Giống Up.',[]),
   N('dg',215,285,175,78,'--load',['DMA gate','1 lệnh weight'],'Giống Up; scale đi trước weight.',[]),
   N('s1g',410,285,255,78,'--tu',['fused lookup + contraction','partial block-16 BF16'],'Giống Up.',[]),
   N('s2g',680,285,240,78,'--tu',['scale khối: contraction chéo BF16'],'Giống Up.',[]),
   N('pg',935,285,240,78,'--vec',['regroup 120/slice','DM→DM trong cluster'],'Giống Up.',[]),
   N('geglu',1195,165,290,198,'--vec',['GeGLU trên chip','32 slice / cluster','','giữ trong DM, không qua HBM','Down dùng K-half cùng cluster'],'GeGLU trên 32 slice mỗi cluster (gom 8 slice xen kẽ về slice đầu của khối 240 hàng). Kết quả ở lại DM rồi tới Down qua switch.',[]),
   N('ag',1195,395,290,98,'--tu',['ring-8 replicate + stride-8 all-gather','24 B FP8 hi/lo → TRF (Main 1,3K)'],'Hai pass Main Broadcast1 rồi nạp TRF; không DMA cấu hình.',[['Ablation','trên Sub +5.163 · trên Main −4.502']]),
   N('hd',15,525,185,78,'--hbm',['W_down NVFP4 + scale','K-split: mỗi cluster 1 K-half'],'Mảnh weight 3.840 B, mảnh scale 480 B.',[]),
   N('dd',215,525,175,78,'--load',['DMA down','w0 12 hàng · w1 3 hàng','scale 4..11 một lệnh'],'Tile 12 hàng và tile 3 hàng tách riêng để DMA w1 chạy chồng.',[]),
   N('s1d',410,525,255,78,'--tu',['12 hàng: lookup fused → BF16','→ contraction chéo (scale BF16)','3 hàng cuối tile riêng'],'Đổi scale FP8→BF16 và nạp TRF scale trên Main ngay sau decode; nếu để ở Sub, lowering chèn `raw.wait sub_context` giữ w1 chậm 2,6K.',[['Đo','M5 −4,9K (K3)']]),
   N('xfer',680,525,220,78,'--sync-ink',['partial H → C0','1 intra-slice reduction'],'Cả hai cluster gửi partial cỡ H về cluster 0; cộng một pass.',[['Đo','−922 khi gộp reduction một pass']]),
   N('epi',915,525,300,78,'--vec',['epilogue hợp nhất 8 slice × 480','gs · RMS · residual · gate'],'Thay gather 1 slice, HBM round trip, cast, scale, normalize, residual, gate (~14K cycle tĩnh).',[['Round','bf16(y·gs), bf16((n+r)·gate); ≤ 1 ulp']]),
   N('out',1230,525,255,78,'--store',['store thẳng HBM','+ sync cuối'],'Rơi đúng layout `EpilogueSlices` rồi `to_hbm`.',[])
  ];
  return {W:1500,H:700,zones:[],nodes:nodes,ghostTitle:'BẢN 8,4643 làm gì (phần score_9_3 đã thay); số ở góc = mục thay thế nó',
   edges:[['hx','pre'],['pre','enc'],['enc','sc'],['hu','du'],['du','s1u'],['s1u','s2u'],['s2u','pu'],['hg','dg'],['dg','s1g'],['s1g','s2g'],['s2g','pg'],['hd','dd'],['dd','s1d'],['s1d','xfer'],['xfer','epi'],['epi','out']],
   lines:[
     {pts:[[1175,204],[1195,204]],col:'--vec'},{pts:[[1175,324],[1195,324]],col:'--vec'},{pts:[[1340,363],[1340,395]],col:'--vec'},
     {pts:[[1195,444],[537,444],[537,525]],col:'--tu',lab:'activation hi/lo → Down',lx:880,ly:436},
     {pts:[[612,115],[612,140],[402,140],[402,182],[410,182]],col:'--tu',dash:1,lab:'x hi/lo → Up',lx:600,ly:134,la:'end'},{pts:[[402,182],[402,302],[410,302]],col:'--tu',dash:1}],
   notes:[{x:15,y:655,t:'DMA: 28 → 23 lệnh; lịch tĩnh 107.226 → 103.020. Hàng đợi liên tục từ ~0,1K tới khi w1 về (~186,4K); tail ~12K (g3 5,2K + push/sync + 5 pass + store).',f:'--ink-2'},
     {x:15,y:678,t:'Số lấy từ WORK_LOG, report K3 (job 99310, 99368) và trace job 104939.',f:'--ink-3'}]};
}
const KDOC91={
qkv:{chips:[['Official 0abcde6','68.963','giá trị 1.000 cycle ≈ +0,43%'],['So với 8,4643','−9.314','78.277 → 68.963'],['Sàn byte @600 B/c','52,4K','overhead ~16,6K']],
 intro:'Chiếu x đã chuẩn hoá thành Q, K, V với weight theo **hình học 15 strip** (mỗi engine đi hết một hàng qua 15 slice), prologue bằng hạt quarter-strip, contraction packet 64 B dùng một TRF strip cho cả ba, RoPE tính trên chip.'},
sao:{chips:[['Official 0abcde6','35.920','giá trị 1.000 cycle ≈ +0,9%'],['So với 8,4643','−754','36.674 → 35.920'],['Sàn byte @600 B/c','26,2K','overhead ~9,7K']],
 intro:'O projection theo **hình học G3** (16 nhóm hàng × 16 strip), hai tile 92 + 28 hàng, x nạp bằng seed quarter-live, epilogue 32 slice × 120 kênh. Lợi thế chủ yếu là tail ngắn hơn và weight stream đổi bit stack.'},
ffn:{chips:[['Official 0abcde6','197.954','giá trị 1.000 cycle ≈ +0,15%'],['So với 8,4643','−17.690','215.644 → 197.954'],['Sàn byte @600 B/c','165,9K','overhead ~32,1K']],
 intro:'Up/Gate 256 slice × 30 hàng với scale căn 256 B và **Down chia theo K giữa hai cluster** để GeGLU ở lại trên chip; scale khối bằng contraction đường chéo BF16; epilogue hợp nhất 8 slice × 480. Thay đổi lớn nhất so với 8,4643.'}};
// ---- dựng tab bản 9,1481 ----
function buildKernel91(key){const K=KDOC91[key], kk=K91[key], el=$('v91-'+key), name=BASE[kk].name;
  let h='<h2>'+name+' · bản 9,1481 <span class="tag">package score_9_3 · commit 0abcde6</span></h2><p class="sub">'+md(K.intro)+'</p><div class="chips">'+
    K.chips.map(function(c){return '<span class="chip">'+esc(c[0])+' <b>'+esc(c[1])+'</b>'+(c[2]?' · '+esc(c[2]):'')+'</span>';}).join('')+'</div>';
  h+='<div class="callout">Official <b>9,1481×</b> (submission 16fa33ae, K1 68.963 / K2 35.920 / K3 197.954). Trace phần cứng do tôi chạy trên build chẩn đoán của đúng source này (job 104939, PASS); các số paired A/B trong bảng lấy từ <code>WORK_LOG.md</code> và <code>reports/</code> của package.</div>';
  h+='<h2>Dataflow của kernel (bản 9,1481)</h2><p class="note">Khung đứt xanh + số = thay đổi <b>mới so với bản 8,4643</b>; khung tím = cơ chế đã có ở 8,4643 (cải tiến so với baseline) và được giữ. Bấm khối để xem baseline, bản 8,4643 và bản này làm gì.</p>'+
    '<div class="legend" id="lgdf-'+kk+'"></div><div class="layout"><div class="chart"><svg id="dfsvg-'+kk+'"></svg></div><aside id="panel-df-'+kk+'" class="closed"><p class="empty">Chưa chọn khối nào.</p></aside></div>';
  h+='<h2>Thay đổi so với bản 8,4643 và baseline</h2><div class="chips"><span class="spd">'+esc(badgeRow(kk).replace(/^[A-Z]+: /,''))+'</span><span class="chip">so với 8,4643: <b>'+esc(K.chips[1][1])+'</b> cycle</span></div>'+
    '<p class="note">Bằng chứng là paired A/B của đồng đội (9 seed × 3 vòng, chế độ <code>info</code>), chưa được tôi chạy lại. Official chỉ đo cả kernel nên không cộng dồn được từng dòng.</p><div class="tw"><table id="timp-'+kk+'"></table></div>';
  h+='<h2>Head, weight, khe, tail (cluster 0, trace)</h2><div id="segbar-'+kk+'"></div>';
  h+='<h2>Timeline đo trên phần cứng (trace job 104939, hai cluster)</h2><p class="note">Chế độ trace, seed 0, một lần chạy, build chẩn đoán (runtime giữ marker). <b>Trace phóng đại</b> chi phí sync và có nhiễu một lần chạy: QKV 76,6K, SAO 46,5K, FFN 204,9K so với official 68,9K / 35,9K / 198,0K. Dùng để hiểu cấu trúc, không dùng để so cycle.</p>'+
    '<div class="legend" id="lgtl-'+kk+'"></div><div class="ctl" id="ctl-'+kk+'"></div><div class="layout"><div class="chart"><svg id="tlsvg-'+kk+'"></svg></div><aside id="panel-tl-'+kk+'" class="closed"><p class="empty">Chưa chọn khối nào.</p></aside></div>';
  h+='<h2>Lệnh dài nhất trên cluster 0</h2><div class="tw"><table id="ttop-'+kk+'"></table></div>';
  h+='<h2>Đã thử và bị loại (bản 9,1481)</h2><div class="tw"><table id="tcl-'+kk+'"></table></div>';
  el.innerHTML=h;
  $('timp-'+kk).innerHTML=impTable(kk);
  $('timp-'+kk).onclick=function(ev){const r=ev.target.closest('tr.rowlink'); if(!r) return; const it=IMP[kk].find(function(q){return q.n===+r.dataset.n;}); CUR['panel-df-'+kk]=it.nodes[0]; panelNode('df-'+kk,it.nodes[0]); redrawDF(kk); $('dfsvg-'+kk).scrollIntoView({block:'center'});};
  tabl('tcl-'+kk,{h:['Hướng','Kết quả phần cứng','Lý do'],r:CLOSED91[key]});
  const T=TL[kk], top=T.c0.spans.map(function(s,i){return {s:s,i:i};}).filter(function(x){return x.s.k!=='sync';}).sort(function(a,b){return (b.s.e-b.s.b)-(a.s.e-a.s.b);}).slice(0,10);
  $('ttop-'+kk).innerHTML='<thead><tr><th>Lệnh</th><th>Đơn vị</th><th class="n">Bắt đầu</th><th class="n">Kết thúc</th><th class="n">Cycle</th><th class="n">% Task</th></tr></thead><tbody>'+top.map(function(x){const s=x.s;return '<tr class="rowlink" data-i="'+x.i+'"><td>'+esc(s.l)+'</td><td>'+esc(TNAME[s.t]||s.t)+'</td><td class="n">'+fmt(s.b)+'</td><td class="n">'+fmt(s.e)+'</td><td class="n">'+fmt(s.e-s.b)+'</td><td class="n">'+((s.e-s.b)/T.c0.task*100).toFixed(1)+'%</td></tr>';}).join('')+'</tbody>';
  $('ttop-'+kk).onclick=function(ev){const r=ev.target.closest('tr.rowlink'); if(r){tlDetail(kk,0,+r.dataset.i);$('tlsvg-'+kk).scrollIntoView({block:'center'});}};
  const S=SEGBARS[KIDX[kk]]; $('segbar-'+kk).innerHTML='<div class="bar">'+S.segs.map(function(s){return '<div class="'+s[0]+'" style="flex:'+s[2]+' 1 0" title="'+esc(s[1]+' · '+fmt(s[2])+' cycle · '+s[3])+'">'+(s[2]/S.tot>0.07?esc(s[1]):'')+'</div>';}).join('')+'</div><div class="barlab"><span>0</span><span>'+fmt(S.tot)+'</span></div>';
  legendChip('lgtl-'+kk,[['--load','DMA nạp HBM→DM'],['--store','DMA ghi DM→HBM'],['--hbm','DM→DM / gather / StoTab'],['--tu','Main (Tensor Unit)'],['--vrf','Sub (TRF/VRF)'],['--sync-ink','Sync']]);
  TLS[kk]={mode:'nen',sel:null,show:{d:true,m:true,s:true,y:true}}; tlControls(kk);
  $('ctl-'+kk).onclick=function(ev){const b=ev.target.closest('button'); if(!b) return; const st=TLS[kk]; if(b.dataset.m) st.mode=b.dataset.m; else if(b.dataset.k) st.show[b.dataset.k]=!st.show[b.dataset.k]; tlControls(kk); drawTL(kk);};
  $('tlsvg-'+kk).addEventListener('click',function(ev){const g=ev.target.closest('g.blk'); if(g) tlDetail(kk,+g.dataset.g,+g.dataset.i); else if(TLS[kk].sel){TLS[kk].sel=null;$('panel-tl-'+kk).classList.add('closed');drawTL(kk);}});
  legendChip('lgdf-'+kk,[['--hbm','HBM (tensor)'],['--load','DMA'],['--tu','Tensor Unit'],['--vec','Vector / RMS'],['--vrf','TRF / VRF'],['--store','ghi HBM'],['--sync-ink','đồng bộ']]);
  $('lgdf-'+kk).innerHTML+='<span><span class="nb new">1</span>mới so với 8,4643</span><span><span class="nb">7</span>giữ từ 8,4643</span><span><i style="background:none;border:1.5px dashed var(--store)"></i>bản 8,4643 làm (đã thay)</span>';
  $('dfsvg-'+kk).addEventListener('click',function(ev){const g=ev.target.closest('g.blk'); if(g){CUR['panel-df-'+kk]=g.dataset.id;panelNode('df-'+kk,g.dataset.id);redrawDF(kk);}});
}
