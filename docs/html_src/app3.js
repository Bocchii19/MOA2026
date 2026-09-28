// ===== dataflow từng kernel =====
function cfgQKV(){
  const Z=[{x:15,w:185,t:'① HBM'},{x:210,w:180,t:'② DMA → DM'},{x:400,w:280,t:'③ Chuẩn hoá x + encode'},{x:690,w:225,t:'④ Chiếu (Contraction)'},{x:925,w:310,t:'⑤ Hậu xử lý theo head'},{x:1245,w:245,t:'⑥ Ghi ra HBM'}];
  const nodes=[
   N('hx',20,50,175,90,'--hbm',['x · rms_weight','15.360 B mỗi cái','bf16 [3.840]'],'Activation vào và trọng số RMSNorm. Cả hai đi qua **2 engine, packet 240 B lệch 256** nên tốn ~1,15K mỗi lệnh.',[['Payload','15.360 B mỗi tensor'],['Trace','DMA `x` 99–1.253; `rms_weight` 3.877–5.029']]),
   N('hrope',20,160,175,60,'--hbm',['cos / sin (rope)','gather theo rope_offset'],'Bảng RoPE nạp bằng `dma_gather_scaled(rope_offset)`; mỗi lệnh ~1,8K.',[['Trace','gather cos 1,81K · sin 1,79K trong khe Q→K']]),
   N('hq',20,300,175,72,'--hbm',['W_Q FP8 + sq','7,86 MB/cluster'],'Weight Q (FP8) và scale theo hàng `sq`. Layout nguyên hàng: ~299 B/cycle/cluster, dưới trần 307+ của SAO/FFN.',[['Byte','15.728.640 B toàn chip (descriptor)']]),
   N('hk',20,400,175,72,'--hbm',['W_K FP8 + sk','3,93 MB/cluster'],'Weight K và scale `sk`: ~277 B/cycle/cluster.',[['Byte','7.864.320 B toàn chip']]),
   N('hv',20,500,175,72,'--hbm',['W_V FP8 + sv','3,93 MB/cluster'],'Weight V và scale `sv`: ~285 B/cycle/cluster (V của C1 chậm hơn +5%).',[['Byte','7.864.320 B toàn chip']]),
   N('dsm',215,50,170,90,'--load',['DMA nhỏ','x 1,15K · rms 1,15K','bitmap 16 KB · 1,46K'],'Ba lệnh DMA nhỏ nối tiếp ở head. Bitmap 16 KB (packet **32 B**) cho `CustomBroadcast{256}` xếp hàng **trước** weight Q, nên Q phát ở 7.434 chứ không sớm hơn.',[['Đoạn','head 0 → 8.877 (8,9K)'],['Khoảng trống','1.253 → 3.871: 2 LoadSfr, 37 lệnh Core, PageTableUpdate, Wait{Dma}']]),
   N('drp',215,160,170,60,'--load',['DMA gather rope','2 × ~1,8K'],'Gather cos/sin theo offset động, mỗi lệnh ~1,8K, xếp nối tiếp với pack → cặp DMA–TU trong khe Q→K.',[]),
   N('dq',215,300,170,72,'--load',['DMA stream Q','27,8K · ~299 B/c'],'Stream weight lớn nhất của QKV. Phát ở 7.434, kết thúc 35.201.',[['Đoạn','Q weight 7.434 → 35.201']]),
   N('dk',215,400,170,72,'--load',['DMA stream K','14,9K · ~277 B/c'],'Phát khi contraction Q bắt đầu dùng weight trước (luật R1).',[['Đoạn','41.983 → 56.927']]),
   N('dv',215,500,170,72,'--load',['DMA stream V','13,8K · ~285 B/c'],'Stream V, kết thúc ở 74.587; sau đó chỉ còn tail.',[['Đoạn','60.791 → 74.587']]),
   N('nrms',410,50,270,90,'--vec',['RMSNorm x (Main)','Σx² → inter-slice reduce','→ sqrt → x · w'],'RMSNorm đầy đủ (kể cả EPS) trên hidden `x`. Dùng chung `shared/hidden.rs`: bình phương, tổng từng phần (4,9K vì gồm reduce liên slice), sqrt→VRF, chuẩn hoá x·w.',[['Trace','TU: 557 + 4.853 + 411 + 1.126 cycle'],['Sau đó','all-gather 240 → mọi slice giữ đủ vector (2,1K + 1,3K)']]),
   N('nenc',410,165,270,90,'--tu',['Encode FP8 → TRF','unscale = max|x·w|/256','lane hi/lo → StoTrf'],'Mã hoá activation thành hai lane f8 (hi/lo) để contraction FP8 giữ độ chính xác; `unscale` tính trên cả vector 3.840. Lane nạp vào TRF bằng `StoTrf` trên Sub.',[['Trace','pass 5,2K + 0,4K + 2,3K; StoTrf 1,3K (14.217 → 23.868)']]),
   N('cq',700,300,205,72,'--tu',['Contraction Q','8,8K (Main)'],'Q = x · W_Q. Mỗi slice giữ nguyên hàng nên không cần trao đổi liên slice.',[['Trace','35.203 → 44.042 (8.839)']]),
   N('ck',700,400,205,72,'--tu',['Contraction K','3,1K'],'K = x · W_K.',[['Trace','56.929 → 59.996 (3.067)']]),
   N('cv',700,500,205,72,'--tu',['Contraction V ⚠','6,1K trên tail'],'V = x · W_V, **nằm trên đường găng cuối kernel**: packet 32 B, `contract_lane Interleaved`, 8 hàng × 60 bước Outer. Lịch tĩnh chỉ 1.225 nhưng thực tế 6.133 (×5).',[['Trace','74.589 → 80.722 (6.133)'],['Hướng','packet 64 B + hi/lo broadcast theo Time: −2…−3K']]),
   N('pq',925,300,310,72,'--vec',['×sq → head RMS (q_rms)','→ RoPE (cos, sin)'],'Nhân scale, RMSNorm theo head, rồi RoPE: `x·A + rotate_half(x)·B`. Cả chuỗi là 5 cặp DMA–TU trong khe Q→K (6,8K).',[['Khe Q→K','35.201 → 41.983 (6,8K)']]),
   N('pk',925,400,310,72,'--vec',['×sk → head RMS (k_rms)','→ RoPE (nạp lại rope 1,3K)'],'Cùng chuỗi cho K, cộng thêm nạp lại rope. Khe K→V 3,9K.',[['Khe K→V','56.927 → 60.791 (3,9K)']]),
   N('pv',925,500,310,72,'--vec',['×sv → chuẩn hoá V','5 pass ~3,1K'],'Chuẩn hoá V (không RoPE). Chạy sau contraction V, nên nằm trọn trên tail.',[['Trace','80.723 → 83.866']]),
   N('oq',1255,300,225,72,'--store',['q → HBM (rows0)','DMA 0,75K'],'Store q (`rows0`) song song với các lệnh sau.',[['Trace','store rows0 ~0,75K trong khe Q→K']]),
   N('ok',1255,400,225,72,'--store',['K → scatter K cache','1,65K'],'Scatter K vào ring cache theo `kv_offset`. 1,65K mỗi lệnh.',[['Ghi chú','scatter per-head']]),
   N('ov',1255,500,225,72,'--store',['V → scatter V cache','1,65K · cuối kernel'],'Lệnh cuối cùng của kernel; sau đó chỉ còn sync cuối (~0,85K). Cluster sớm chờ cluster muộn ở đây.',[['Trace','kết thúc 85.522 (C0) · 85.718 (C1)']])
  ];
  return {W:1500,H:640,zones:Z,nodes:nodes,
   edges:[['hx','dsm'],['hrope','drp'],['hq','dq'],['hk','dk'],['hv','dv'],['dsm','nrms'],['nrms','nenc','b','t'],['cq','pq'],['ck','pk'],['cv','pv'],['pq','oq'],['pk','ok'],['pv','ov']],
   lines:[
     {pts:[[385,330],[700,330]],col:'--load',lab:'weight stream',lx:540,ly:322},
     {pts:[[385,436],[700,436]],col:'--load'},
     {pts:[[385,536],[700,536]],col:'--load'},
     {pts:[[680,210],[692,210],[692,352],[700,352]],col:'--tu'},
     {pts:[[692,352],[692,452],[700,452]],col:'--tu'},
     {pts:[[692,452],[692,552],[700,552]],col:'--tu'},
     {pts:[[300,220],[300,282],[1080,282],[1080,300]],col:'--load',dash:1,lab:'cos/sin cho Q (K dùng lại)',lx:520,ly:275}],
   notes:[{x:20,y:600,t:'Q → K → V chạy tuần tự trên cùng hàng đợi DMA; chuỗi hậu xử lý của stream trước chạy chồng stream sau. Bấm khối để xem số đo.',f:'--ink-2'},
     {x:20,y:620,t:'Đường găng: head 8,9K + Q 26,3K + khe 6,8K + K 14,9K + khe 3,9K + V 13,8K + tail 10,9K ≈ 85,5K.',f:'--ink-3'}]};
}
function cfgSAO(){
  const Z=[{x:15,w:185,t:'① HBM'},{x:210,w:165,t:'② DMA → DM'},{x:385,w:225,t:'③ Encode activation'},{x:620,w:240,t:'④ Chiếu W_O'},{x:870,w:200,t:'⑤ Gom y về C0'},{x:1080,w:220,t:'⑥ RMS + epilogue'},{x:1310,w:180,t:'⑦ Ghi ra'}];
  const nodes=[
   N('hxq',20,55,175,80,'--hbm',['xq (đầu ra attention)','128 KB · bf16'],'Đầu vào của kernel: đầu ra attention đã gom theo chunk 256.',[['Payload','131.072 B, 8 engine, packet 256 B căn']]),
   N('dxq',215,55,155,80,'--load',['DMA xq','1,4K · 8 engine'],'Head của SAO: 101 → 1.519. Xong trước khi phát weight O.',[['Trace','DMA 101–1.519 (1.418)']]),
   N('enc',390,45,215,100,'--tu',['Encode xq (Main + Sub)','f8 hi/lo · ×2⁶','→ StoTrf (TRF) 1,76K'],'Mỗi slice mã hoá chunk 256 của xq thành hai lane f8 (hi/lo) và nạp vào TRF. Chạy **chồng** stream weight O nên không thêm cycle.',[['Trace','Main 496 + Sub 1.009 + Main 683 + StoTrf 1.758']]),
   N('hw',20,215,175,80,'--hbm',['W_O FP8','15,73 MB','7,86 MB/cluster'],'Weight O của toàn chip 15.728.640 B.',[['Byte bắt buộc','15,74 MB (kèm scale) → sàn 25,5K @616 B/c']]),
   N('dw',215,215,155,80,'--load',['DMA stream W_O','25,6K · ~307 B/c'],'Stream duy nhất của SAO: 2.011 → 27.657. Sát trần thực tế (~95%).',[['Đoạn','25,6K trên 37,6K tổng']]),
   N('proj',625,205,230,100,'--tu',['Chiếu O · Contraction','dot theo chunk','reduce 16 slice · 3,4K'],'**Tail #1.** Contraction chỉ chạy được sau khi byte weight cuối về: 27.659 → 31.074. Inter-slice reduce 16 chunk. Lịch tĩnh 1.374 (thực ×2,5).',[['Hướng','hình học G3 + packet 64 B: −0,8…−1,5K']]),
   N('gath',875,205,190,100,'--sync-ink',['Gom y C1 → C0','DmaStos 1,15K','+ ExplicitSync 1,39K'],'DmaStos gom 3.840 hàng y về 32 slice của C0, rồi `ExplicitSync` đợi phần của C1. **Nằm trên đường găng** (2,5K).',[['Trace','31.076 → 33.612'],['Probe','bớt 6/30 hàng cho cluster muộn: −779']]),
   N('rms',1085,205,210,100,'--vec',['RMS 4 pass (Main)','partial → regular32','→ sqrt → norm×w+res'],'RMSNorm sau attention, cộng residual. 0,56 + 0,71 + 0,41 + 0,53 = 2,2K.',[['Trace','33.613 → 35.834']]),
   N('out',1315,205,170,100,'--store',['Store out_rows','0,87K · RMW','+ sync cuối 0,85K'],'DMA ghi output **1 engine, packet 240 B lệch 256** → read-modify-write. Rồi sync cuối kết thúc kernel.',[['Trace','store 35.836 → 36.707; sync 36.709 → 37.558']]),
   N('hs',20,350,175,58,'--hbm',['o_weight_scale','7.680 B'],'Scale của W_O.',[]),
   N('hn',20,420,175,58,'--hbm',['post_attn_rms_weight','7.680 B'],'Trọng số RMSNorm sau attention.',[]),
   N('hr',20,490,175,58,'--hbm',['residual','7.680 B'],'Residual cộng vào cuối.',[]),
   N('dep',215,350,155,198,'--load',['3 DMA operand','epilogue (chỉ C0)','1 engine · 240 B lệch','1,60 + 1,78 + 1,14K'],'Ba lệnh DMA nhỏ, **1 engine, packet 240 B lệch 256, chỉ cluster 0**. Probe bỏ norm/residual: −827, nên nằm trên đường găng.',[['Hướng','dời vào pass cuối (B): −0,8K']]),
   N('epl',1085,350,210,100,'--vrf',['Pass Sub (VRF)','scale 0,76K · norm w 0,67K','residual 1,13K'],'Sub nạp scale, trọng số norm và residual vào VRF song song với các pass Main.',[['Trace','29.272 · 31.067 · 32.889']])
  ];
  return {W:1500,H:600,zones:Z,nodes:nodes,
   edges:[['hxq','dxq'],['dxq','enc'],['enc','proj','r','l','',{xm:615}],['hw','dw'],['dw','proj'],['proj','gath'],['gath','rms'],['rms','out'],['hs','dep','r','l','',{xm:205}],['hn','dep'],['hr','dep','r','l','',{xm:205}],['epl','rms','t','b']],
   lines:[{pts:[[370,449],[1085,449]],col:'--load',dash:1,lab:'operand → VRF',lx:730,ly:441}],
   notes:[{x:20,y:585,t:'Head 2,0K + weight O 25,6K + tail 9,9K (projection 3,4 + y/sync 2,5 + RMS 2,2 + store 0,87 + sync 0,85) = 37,5K. C0 làm epilogue và store; C1 chờ ở sync cuối.',f:'--ink-2'}]};
}
function cfgFFN(){
  const nodes=[
   N('hx',15,45,185,70,'--hbm',['x · rms_weight','15.360 B ×2 · 240 B lệch'],'Activation và trọng số RMSNorm. 2 engine, packet 240 B lệch.',[['Trace','x 96–1.248 · rms 5.367–6.526']]),
   N('pre',215,45,255,70,'--vec',['RMSNorm x + all-gather','Σx² → sqrt → x·w'],'RMSNorm trước MLP (`shared/hidden.rs`) rồi all-gather 240 → mọi slice giữ đủ vector.',[['Trace','TU 558 + 5.313 + 407 + 6.947; all-gather 2.113 + 1.277']]),
   N('enc',485,45,255,70,'--tu',['encode_half (up/gate)','x → f8 hi/lo → TRF'],'Mã hoá x thành hai lane f8 (hi/lo); StoTrf 2,3K. Dùng chung cho stage1 của up và gate.',[['Trace','pass 1.327 + 795 + 2.763 (×2) + StoTrf 2.255']]),
   N('head',755,45,300,70,'--sync-ink',['Head 11,2K','ClusterSync ~3K chặn luồng lệnh'],'`ClusterSync` (reserve của DmaStos xếp ASAP) rơi **trước** TU đầu và trước up weight: 2.347 → ~5.361. Sau đó rms → bitmap 1,47K → bảng FP4 2,29K → up weight phát ở 11.226.',[['Hướng','J1: buffer đích do pass Sub tạo, hoặc B2: −1,4K']]),
   N('hu',15,165,185,78,'--hbm',['W_up NVFP4 + scale','29,49 MB + 3,69 MB'],'Weight up: 29.491.200 B toàn chip (14,75 MB/cluster) + block scale 3.686.400 B, packet **800 B lệch 256**.',[]),
   N('du',215,165,175,78,'--load',['DMA stream up','47,8K · ~308 B/c','scale 10,28K · 184 B/c'],'Stream up 11.226 → 59.018. Scale up (khe up→gate) 10,28K vì lệch 256.',[['Hướng','Round 2b: scale căn 256, −4…−5K']]),
   N('s1u',410,165,255,78,'--tu',['stage1: lookup f4→f8','Contraction 22,5K','StoTab 7,3K ‖'],'Lookup f4→f8 giới hạn ~2,5 B/cycle/slice; chạy 59.024 → 81.476, che dưới khe và stream gate. Mỗi pass lookup cần DMA bảng FP4 + StoTab.',[['Trace','contraction 22.452; StoTab 7.347']]),
   N('s2u',680,165,240,78,'--tu',['stage2: partial × scale','5 tile: Sub 0,97 + Main 2,17K','≈ 14,4K'],'Nhân partial với block scale, 5 tile, chạy song song stream gate.',[['Trace','81.872 → ~97.9K']]),
   N('pu',935,165,240,78,'--vec',['pool 30 hàng','× global scale → up bf16'],'Gộp 30 hàng của 8 slice liên tiếp, nhân global scale, làm tròn bf16.',[['DMA','global scale 2.048 B · 8 engine × 32 B']]),
   N('hg',15,285,185,78,'--hbm',['W_gate NVFP4 + scale','29,49 MB + 3,69 MB'],'Giống up.',[]),
   N('dg',215,285,175,78,'--load',['DMA stream gate','48,2K · ~306 B/c','scale 10,19K'],'Stream gate 72.638 → 120.788; stream chậm hơn up ~8% vì contraction stage1 up đọc DM cùng lúc (tranh chấp ~20%).',[['Khe gate→down','120.788 → 135.875 (15,1K)']]),
   N('s1g',410,285,255,78,'--tu',['stage1: lookup f4→f8','Contraction 23,3K','StoTab 8,2K ‖'],'Giống up: 120.790 → 144.048.',[['Trace','contraction 23.258; StoTab 8.226']]),
   N('s2g',680,285,240,78,'--tu',['stage2: partial × scale','≈ 14,4K'],'Giống up.',[]),
   N('pg',935,285,240,78,'--vec',['pool 30 hàng','× global scale → gate bf16'],'Giống up.',[]),
   N('geglu',1195,165,290,198,'--vec',['GeGLU','gelu(gate) · up','~2,9K trên VRF','','rồi encode cho down:','unscale max|x|/256 · lane hi/lo'],'GeGLU (`geglu`): GeLU của gate nhân up. Sau đó mã hoá kết quả cho down: `unscale = max|x|/256` (max 2,77K + 0,43 + 0,39K), tách hi/lo f8.',[['Trace','160.712 → ~169.6K']]),
   N('ag',1195,395,290,98,'--tu',['all-gather lanes 9,06K','+ TRF 4,15K (Sub)'],'`CustomBroadcast{256}` + bitmap 16 KB đưa lane hi/lo về mọi slice. Lớn thứ hai sau lookup ở pha down.',[['Trace','169.590 → 178.647; TRF 178.648 → 182.799'],['Hướng','V2b: −0,5…−0,9K · V1b: −0,7…−1,9K']]),
   N('hd',15,525,185,78,'--hbm',['W_down NVFP4 + scale','29,49 MB + 3,69 MB'],'Weight down, scale packet **480 B lệch 256**.',[]),
   N('dd',215,525,175,78,'--load',['DMA stream down','49,6K · ~297 B/c','scale 10,7K'],'Stream down 135.875 → 185.428. Scale down 10,7K nằm trên tail.',[]),
   N('s1d',410,525,255,78,'--tu',['stage1 down: lookup','22,8K ‖ chuỗi scale 22,8K'],'**Tail #1.** Weight cuối luôn làm lộ pass lookup của nó: 185.432 → 208.211. Song song: DMA scale 10,7K → decode (Sub) 7,0K → StoTrf 5,1K.',[['Ràng buộc','không tile được lookup: ltile3 +35K, predecode +13,5K']]),
   N('s2d',680,525,220,78,'--tu',['stage2 down','3,0K'],'208.212 → 211.197.',[]),
   N('xfer',915,525,240,78,'--sync-ink',['rows C1→C0 (DmaStos)','2,2K + ExplicitSync 3,8K'],'DmaStos rows 211.199 → 213.364, rồi **ExplicitSync 3,78K** đợi phần chuyển của cluster kia.',[['Hướng','B2: DM→DM thẳng vào slice epilogue C0: −1,4K']]),
   N('post',1170,525,190,78,'--vec',['post-FFN','RMS · residual · ×scalar','~4,3K'],'~6 pass: RMS (bình phương, tổng, sqrt), normalize, cộng residual, nhân layer scalar. Vẫn giữ đủ điểm round bf16.',[['Trace','217.446 → 222.445']]),
   N('out',1375,525,110,78,'--store',['store','0,86K','+ sync 0,96K'],'Store 1 engine, packet 240 B lệch → RMW; sync cuối.',[['Trace','222.446 → 224.271']])
  ];
  return {W:1500,H:700,zones:[],nodes:nodes,
   edges:[['hx','pre'],['pre','enc'],['enc','head'],['hu','du'],['du','s1u'],['s1u','s2u'],['s2u','pu'],['hg','dg'],['dg','s1g'],['s1g','s2g'],['s2g','pg'],['hd','dd'],['dd','s1d'],['s1d','s2d'],['s2d','xfer'],['xfer','post'],['post','out']],
   lines:[
     {pts:[[1175,204],[1195,204]],col:'--vec'},{pts:[[1175,324],[1195,324]],col:'--vec'},
     {pts:[[1340,363],[1340,395]],col:'--vec'},
     {pts:[[1195,444],[537,444],[537,525]],col:'--tu',lab:'lane hi/lo → stage1 down',lx:880,ly:436},
     {pts:[[612,115],[612,140],[402,140],[402,182],[410,182]],col:'--tu',dash:1,lab:'lane hi/lo → up',lx:600,ly:134,la:'end'},
     {pts:[[402,182],[402,302],[410,302]],col:'--tu',dash:1}],
   notes:[
     {x:15,y:655,t:'DMA FIFO: up weight → scale up → gate weight → scale gate → down weight → scale down. Head 11,2K + up 47,8K + khe 13,6K + gate 48,2K + khe 15,1K + down 49,6K + tail 38,8K = 224,3K.',f:'--ink-2'},
     {x:15,y:678,t:'Ba lần lookup f4→f8 (stage1) đều cần DMA bảng FP4 4 KB + StoTab 7–9K; không có API dùng chung bảng.',f:'--ink-3'}]};
}
