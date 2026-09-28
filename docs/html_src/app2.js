// ===== engine sơ đồ khối dùng chung =====
const DG={};   // pid -> {cfg, nodes:{}}
function N(id,x,y,w,h,c,l,d,kv){return {id:id,x:x,y:y,w:w,h:h,c:c,l:l,d:d||'',kv:kv||[]};}
function anchor(n,s){return s==='r'?[n.x+n.w,n.y+n.h/2]:s==='l'?[n.x,n.y+n.h/2]:s==='t'?[n.x+n.w/2,n.y]:[n.x+n.w/2,n.y+n.h];}
function drawDiagram(svgId,pid,cfg){
  DG[pid]={cfg:cfg,nodes:{}}; cfg.nodes.forEach(function(n){DG[pid].nodes[n.id]=n;});
  const svg=$(svgId), sel=CUR['panel-'+pid]; let o='<rect width="'+cfg.W+'" height="'+cfg.H+'" fill="'+css('--panel')+'"/>';
  (cfg.zones||[]).forEach(function(z,i){o+='<rect x="'+z.x+'" y="24" width="'+z.w+'" height="'+(cfg.H-34)+'" fill="'+(i%2?css('--panel'):css('--panel-2'))+'" stroke="'+css('--rule-2')+'"/>'+svgText(z.x+z.w/2,17,z.t,{s:13,w:600,f:css('--ink-2')});});
  (cfg.lines||[]).forEach(function(l){o+=poly(l.pts,l.col?css(l.col):null,{dash:l.dash,lab:l.lab,lx:l.lx,ly:l.ly,la:l.la});});
  (cfg.edges||[]).forEach(function(e){const a=DG[pid].nodes[e[0]],b=DG[pid].nodes[e[1]],fs=e[2]||'r',ts=e[3]||'l',p1=anchor(a,fs),p2=anchor(b,ts),op=e[5]||{};
    const col=op.col?css(op.col):css('--ink-3');
    if(Math.abs(p1[1]-p2[1])<1&&(fs==='r'||fs==='l')||Math.abs(p1[0]-p2[0])<1&&(fs==='b'||fs==='t')) o+=arrow(p1[0],p1[1],p2[0],p2[1],col,e[4],op);
    else if(fs==='r'&&ts==='l'){const xm=op.xm||(p1[0]+p2[0])/2;o+=poly([p1,[xm,p1[1]],[xm,p2[1]],p2],col,{dash:op.dash,lab:e[4],lx:xm+6,ly:(p1[1]+p2[1])/2,la:'start'});}
    else if(fs==='b'&&ts==='t'){const ym=op.ym||(p1[1]+p2[1])/2;o+=poly([p1,[p1[0],ym],[p2[0],ym],p2],col,{dash:op.dash,lab:e[4],lx:p1[0]+6,ly:ym-4,la:'start'});}
    else o+=arrow(p1[0],p1[1],p2[0],p2[1],col,e[4],op);});
  cfg.nodes.forEach(function(n){o+=box(n.id,n.x,n.y,n.w,n.h,css(n.c),n.l,{sel:sel,fo:.14,lh:15.5,dash:n.dash});
    const im=n.imps||[];
    if(im.length){const col=css(im.some(function(i){return i.tag==='imp';})?'--chg':'--newc');
      o+='<rect pointer-events="none" x="'+(n.x-4)+'" y="'+(n.y-4)+'" width="'+(n.w+8)+'" height="'+(n.h+8)+'" rx="5" fill="none" stroke="'+col+'" stroke-width="2" stroke-dasharray="6 3"/>';
      im.forEach(function(i,k){const cx=n.x+n.w-6-k*24, cy=n.y-4, c2=css(i.tag==='new'?'--newc':'--chg');
        o+='<g pointer-events="none"><circle cx="'+cx+'" cy="'+cy+'" r="10.5" fill="'+c2+'"/>'+svgText(cx,cy+4.2,String(i.n),{s:12,w:700,f:css(i.tag==='new'?'--paper':'--chg-on'),m:1})+'</g>';});}
    else if((cfg.eqIds||[]).indexOf(n.id)>=0){o+='<g pointer-events="none"><circle cx="'+(n.x+n.w-6)+'" cy="'+(n.y-4)+'" r="9" fill="'+css('--ink-3')+'"/>'+svgText(n.x+n.w-6,n.y-0.2,'=',{s:13,w:700,f:css('--panel'),m:1})+'</g>';}});
  (cfg.notes||[]).forEach(function(t){o+=svgText(t.x,t.y,t.t,{a:t.a||'start',s:12.5,f:css(t.f||'--ink-3'),m:!!t.m});});
  let H=cfg.H;
  if(cfg.ghosts&&cfg.ghosts.length){const y0=cfg.H, n=cfg.ghosts.length, gw=(cfg.W-30-(n-1)*12)/n; H=cfg.H+128;
    o+='<rect x="0" y="'+y0+'" width="'+cfg.W+'" height="128" fill="'+css('--panel-2')+'"/><line x1="0" y1="'+y0+'" x2="'+cfg.W+'" y2="'+y0+'" stroke="'+css('--rule')+'"/>'+
       svgText(15,y0+20,cfg.ghostTitle||'BASELINE của ban tổ chức làm gì (phần best đã thay hoặc bỏ); số ở góc = cải tiến thay thế nó',{a:'start',s:13,w:600,f:css('--store')});
    cfg.ghosts.forEach(function(g,i){const x=15+i*(gw+12), y=y0+32, h=88;
      o+='<rect x="'+x+'" y="'+y+'" width="'+gw+'" height="'+h+'" rx="3" fill="'+css('--store')+'" fill-opacity=".06" stroke="'+css('--store')+'" stroke-width="1.4" stroke-dasharray="5 4"/>'+
         svgText(x+10,y+20,'⌫ '+fit(g.t,gw,7.4),{a:'start',s:13,w:600,f:css('--store')});
      g.l.forEach(function(t,k){o+=svgText(x+10,y+40+k*16,fit(t.split('`').join(''),gw,6.4),{a:'start',s:12,f:css('--ink-2')});});
      g.to.forEach(function(nn,k){const it=(IMP[cfg.key]||[]).find(function(q){return q.n===nn;}); const c2=css(it&&it.tag==='new'?'--newc':'--chg');
        const cx=x+gw-14-k*24, cy=y+h-14; o+='<circle cx="'+cx+'" cy="'+cy+'" r="10" fill="'+c2+'"/>'+svgText(cx,cy+4,String(nn),{s:12,w:700,f:css(it&&it.tag==='new'?'--paper':'--chg-on'),m:1});});
      o+=svgText(x+gw-14-g.to.length*24-2,y+h-10,'thay bằng',{a:'end',s:11.5,f:css('--ink-3'),m:1});});}
  svg.setAttribute('viewBox','0 0 '+cfg.W+' '+H); svg.innerHTML=o;}
function panelNode(pid,id){const n=DG[pid].nodes[id];
  let h='<h3>'+esc(n.l[0])+'</h3><p class="site">'+esc(n.l.slice(1).join(' · '))+'</p><p>'+md(n.d)+'</p>';
  if(n.kv.length) h+='<dl>'+n.kv.map(function(k){return '<dt>'+esc(k[0])+'</dt><dd style="font-family:var(--sans);font-weight:400">'+md(k[1])+'</dd>';}).join('')+'</dl>';
  const ky=pid.replace('df-',''); if(IMP[ky]) h+=impPanelHtml(ky,n);
  openPanel('panel-'+pid,h);}

// ===== A. đường đi dữ liệu =====
function cfgSys(){
  const nodes=[
    N('hbm',20,190,150,110,'--hbm',['HBM3','weight · activation','KV cache'],'Nguồn của mọi weight FP8/NVFP4, scale, activation và đích của output, KV cache.',[['Đỉnh','750 B/cycle device'],['Đo được','~616 B/cycle cả chip']]),
    N('dma',215,190,150,110,'--load',['DMA engine × 8','FIFO · ~1K/lệnh','≤ 256 B/c mỗi engine'],'Vận chuyển HBM→DM. Lệnh DMA phát bất đồng bộ và chạy FIFO; `Wait{Dma:0}` đợi cả hàng đợi.',[['Thực đo','~77 B/cycle mỗi engine khi 8 engine cùng stream']]),
    N('dm',410,190,150,110,'--dm',['DM · 512 KB/slice','16 bank','256 MB / chip'],'SRAM của slice, nơi weight và activation nằm chờ Tensor Unit đọc.',[['Băng thông','128 B/cycle/DMN']]),
    N('fetch',605,190,120,110,'--tu',['Fetch','+ Adapter','lookup f4→f8'],'Fetch đọc DM, Adapter giải mã NVFP4 (f4→f8) và f8→bf16. Lookup chỉ có ở Main.',[['Bảng FP4','DMA 4 KB + StoTab 5,4–9K mỗi pass']]),
    N('switch',770,190,130,110,'--tu',['Switch → Collect','ring 256 router','flit 32 B'],'Ring trong cluster: broadcast, chuyển vị, gom flit thành packet.',[['Bitmap','`CustomBroadcast` cần DMA bitmap 16 KB']]),
    N('contraction',945,170,190,50,'--tu',['Contraction (MAC)','~1,9 cycle/bước Outer'],'Nhân ma trận. Đọc toán hạng tĩnh từ TRF, weight chảy qua Fetch.',[['Cố định','~1,4–1,7K mỗi lệnh']]),
    N('vector',945,245,190,50,'--vec',['Vector','RMS · sqrt · GeLU · scale'],'Chuẩn hoá, scale, GeLU, residual; phần FP chạy Way4.',[['Pass nhỏ','0,4–0,7K mỗi pass']]),
    N('commit',1180,190,150,110,'--tu',['Cast · Transpose','Commit Adapter','Commit'],'Hạ f32→bf16, đổi layout, commit về DM (hoặc TRF/VRF qua StoTrf/StoVrf).',[['Điểm round','`commit_cast::<bf16>()` không chiếm Cast Engine']]),
    N('trf',995,60,90,60,'--vrf',['TRF','8 KB/lane'],'Toán hạng tĩnh (activation f8 hi/lo).',[]),
    N('vrf',995,325,90,40,'--vrf',['VRF · 8 KB'],'Scale, 1/rms, unscale.',[]),
    N('tab',580,60,170,60,'--hbm',['bảng FP4 (4 KB)','StoTab mỗi pass'],'Compiler sinh một DMA bảng cho mỗi pass lookup; nạp vào một thanh ghi của Fetch Unit.',[['FFN','3 lần: 1,5–3,2K DMA + 7,3–8,9K StoTab']]),
    N('bitmap',770,60,150,60,'--hbm',['bitmap (16 KB)','packet 32 B'],'Định tuyến `CustomBroadcast{256}`; DMA 8 engine × 32 B, 1,46–1,59K.',[]),
    N('dmo',410,400,150,90,'--dm',['DM · kết quả','output, K/V, y'],'Kết quả đã commit, chờ DMA đẩy đi.',[]),
    N('dmao',215,400,150,90,'--store',['DMA ghi','store · scatter'],'Ghi output, scatter K/V cache. 1 engine, packet 240 B lệch 256 gây read-modify-write (~×50).',[['Store output','0,86–0,87K']]),
    N('hbmo',20,400,150,90,'--hbm',['HBM','output · KV cache'],'Đích cuối.',[]),
    N('dmn',1180,400,150,90,'--sync-ink',['DmaStos','C1 → C0 (DM→DM)'],'Gom kết quả của cluster 1 về cluster 0 để làm epilogue. Kèm `ExplicitSync` 1,4–3,8K.',[['SAO','1,15K + sync 1,39K'],['FFN','2,17K + sync 3,78K']])
  ];
  const ids={};nodes.forEach(function(n){ids[n.id]=n;});
  return {W:1500,H:520,nodes:nodes,zones:[],
    edges:[['hbm','dma','r','l','',{}],['dma','dm','r','l','',{}],['dm','fetch','r','l','',{}],['fetch','switch','r','l','',{}],['switch','contraction','r','l','',{xm:922}],['switch','vector','r','l','',{xm:922}],['contraction','commit','r','l','',{xm:1158}],['vector','commit','r','l','',{xm:1158}],
      ['trf','contraction','b','t','',{}],['vrf','vector','t','b','',{}]],
    lines:[
      {pts:[[665,120],[665,190]],col:'--hbm'},
      {pts:[[845,120],[845,190]],col:'--hbm'},
      {pts:[[1255,300],[1255,385],[485,385],[485,400]],col:'--dm'},
      {pts:[[410,445],[365,445]],col:'--store'},
      {pts:[[215,445],[170,445]],col:'--store'},
      {pts:[[1255,385],[1255,400]],col:'--sync-ink',dash:1}],
    notes:[{x:95,y:178,t:'≤ 750 B/c',m:1,a:'middle'},{x:290,y:178,t:'~77 B/c/engine',m:1,a:'middle'},{x:485,y:178,t:'128 B/c/DMN',m:1,a:'middle'},{x:665,y:178,t:'8 B packet',m:1,a:'middle'},{x:835,y:178,t:'32 B flit',m:1,a:'middle'},
      {x:20,y:40,t:'Đường vào: HBM → DMA → DM → Fetch → Switch → Contraction/Vector',f:'--ink-2'},
      {x:20,y:372,t:'Đường ra: Commit → DM → DMA ghi → HBM',f:'--ink-2'},
      {x:1255,y:520-12,t:'Hai cluster chạy song song; kết quả epilogue gom về C0',a:'middle'},
      {x:20,y:510,t:'Cả chip ~616 B/cycle stream weight; overhead nằm ở khe, head, tail chứ không ở băng thông.',f:'--ink-2'}]};
}
// ===== B. chuỗi 3 kernel =====
function drawChain(){
  const svg=$('chainsvg'); let o='<rect width="1500" height="350" fill="'+css('--panel')+'"/>';
  const K=[['qkv','sliding_project_qkv','QKV','31,47 MB','78,3K','51,1K','+0,43%',70,'--tu'],['sao','sliding_attention_output','SAO','15,74 MB','36,7K','25,5K','+0,9%',770,'--store'],['ffn','decoder_feedforward','FFN','99,53 MB','215,6K','161,6K','+0,15%',1130,'--vec']];
  K.forEach(function(k){const x=k[7], w=(k[2]==='QKV'?330:(k[2]==='SAO'?300:330)), col=css(k[8]);
    o+='<g class="blk" data-go="'+k[0]+'"><rect class="b" x="'+x+'" y="70" width="'+w+'" height="215" rx="4" fill="'+col+'" fill-opacity=".12" stroke="'+col+'" stroke-width="1.6"/>'+
      svgText(x+w/2,98,k[2],{s:20,w:600,f:col})+svgText(x+w/2,118,k[1],{s:12,m:1,f:css('--ink-2')})+
      svgText(x+16,150,'Byte bắt buộc',{a:'start',s:13,f:css('--ink-3')})+svgText(x+w-16,150,k[3],{a:'end',s:13,w:600,m:1})+
      svgText(x+16,174,'Sàn @616 B/c',{a:'start',s:13,f:css('--ink-3')})+svgText(x+w-16,174,k[5],{a:'end',s:13,w:600,m:1})+
      svgText(x+16,198,'Official 8,4643',{a:'start',s:13,f:css('--ink-3')})+svgText(x+w-16,198,k[4],{a:'end',s:13,w:600,m:1})+
      svgText(x+16,222,'1.000 cycle ≈',{a:'start',s:13,f:css('--ink-3')})+svgText(x+w-16,222,k[6]+' điểm',{a:'end',s:13,w:600,m:1})+svgText(x+w/2,238,'baseline '+fk(BASE[k[0]].b)+' → 8,4643: '+fk(BASE[k[0]].o)+' (×'+(BASE[k[0]].b/BASE[k[0]].o).toFixed(1).replace('.',',')+')',{s:12,w:600,f:css('--chg'),m:1})+svgText(x+w/2,254,'score_9_3: '+fk(BASE[k[0]+'91'].o)+' (×'+(BASE[k[0]].b/BASE[k[0]+'91'].o).toFixed(1).replace('.',',')+')',{s:12,w:600,f:css('--newc'),m:1})+
      svgText(x+w/2,'270','bấm để xem dataflow chi tiết →',{s:12,f:css('--ink-3'),m:1})+'</g>';});
  o+='<g><rect x="450" y="90" width="270" height="150" rx="4" fill="'+css('--panel-2')+'" stroke="'+css('--ink-3')+'" stroke-dasharray="6 4"/>'+
    svgText(585,130,'Attention',{s:16,w:600,f:css('--ink-2')})+svgText(585,152,'score · softmax · ·V trên KV cache',{s:12.5,f:css('--ink-3')})+svgText(585,176,'nằm giữa QKV và SAO,',{s:12.5,f:css('--ink-3')})+svgText(585,194,'không thuộc 3 kernel chấm',{s:12.5,f:css('--ink-3')})+'</g>';
  o+=arrow(20,165,70,165,css('--ink-3'),'x bf16',{dx:0,dy:-8});
  o+=arrow(400,165,450,165,css('--ink-3'),'',{});
  o+=arrow(720,165,770,165,css('--ink-3'),'',{});
  o+=arrow(1070,165,1130,165,css('--ink-3'),'',{});
  o+=arrow(1460,165,1490,165,css('--ink-3'),'',{});
  o+=svgText(425,138,'q, K/V',{s:11.5,m:1,f:css('--ink-3')})+svgText(745,138,'xq',{s:11.5,m:1,f:css('--ink-3')})+svgText(1100,138,'y',{s:11.5,m:1,f:css('--ink-3')});
  o+=svgText(235,50,'HBM: weight FP8 Q/K/V + RoPE + KV cache',{s:12.5,f:css('--ink-3'),m:1})+svgText(920,50,'HBM: weight FP8 O + norm + residual',{s:12.5,f:css('--ink-3'),m:1})+svgText(1295,50,'HBM: weight NVFP4 + scale',{s:12.5,f:css('--ink-3'),m:1});
  o+=arrow(235,56,235,70,css('--hbm'));o+=arrow(920,56,920,70,css('--hbm'));o+=arrow(1295,56,1295,70,css('--hbm'));
  o+='<rect x="70" y="302" width="1390" height="34" fill="'+css('--panel-2')+'" stroke="'+css('--rule')+'"/>'+
     svgText(84,324,'Tổng 8,4643: 330,6K (78,3 + 36,7 + 215,6) · score_9_3: 302,8K (69,0 + 35,9 + 198,0) · sàn stream 238,2K. Phần còn lại (64,6K ở score_9_3) là overhead, không phải băng thông.',{a:'start',s:13.5,f:css('--ink-2')});
  svg.innerHTML=o;}
// ===== C. luồng lệnh (minh hoạ) =====
function drawExec(){
  const svg=$('execsvg'); let o='<rect width="1500" height="420" fill="'+css('--panel')+'"/>';
  const rows=[['Luồng lệnh (tuần tự)',60],['Hàng đợi DMA (FIFO)',130],['Tensor Unit · Main',200],['Tensor Unit · Sub',260],['Sync',320]];
  rows.forEach(function(r){o+=svgText(10,r[1]+24,r[0],{a:'start',s:12.5,w:600,f:css('--ink-2')})+'<line x1="190" y1="'+(r[1]+38)+'" x2="1490" y2="'+(r[1]+38)+'" stroke="'+css('--rule-2')+'"/>';});
  function B(x,y,w,t,c,tip){return '<g><rect x="'+x+'" y="'+y+'" width="'+w+'" height="30" rx="2" fill="'+css(c)+'" fill-opacity=".85"/><title>'+esc(tip||t)+'</title>'+svgText(x+w/2,y+20,fit(t,w,6.6),{s:12,f:'#fff',w:600})+'</g>';}
  o+=B(200,66,70,'DMA x','--load','DmaCommand x')+B(276,66,70,'DMA rms','--load')+B(352,66,70,'DMA bitmap','--load')+B(430,66,66,'Wait{Dma:0}','--sync-ink','đợi TOÀN BỘ hàng đợi DMA')+B(502,66,80,'TU: RMS','--tu')+B(588,66,110,'DmaCommand W1','--load','phát weight lớn khi TU dùng weight trước bắt đầu (luật R1)')+B(704,66,90,'Wait{Dma}','--sync-ink')+B(800,66,84,'TU: gap 1','--tu')+B(890,66,110,'DmaCommand W2','--load')+B(1006,66,90,'Wait{Dma}','--sync-ink')+B(1102,66,84,'TU: tail','--tu')+B(1192,66,110,'store + sync','--store');
  o+=B(200,136,70,'x','--load')+B(272,136,70,'rms','--load')+B(344,136,70,'bitmap','--load')+B(590,136,430,'W1: weight lớn (~300 B/cycle/cluster)','--load','stream weight')+B(1024,136,50,'nhỏ','--load')+B(1078,136,300,'W2 …','--load')+B(1380,136,70,'store','--store');
  o+=B(500,206,90,'RMS + encode','--tu')+B(700,206,330,'Contraction dùng W1 (chạy chồng stream W2)','--tu','contraction chồng DMA → stream chậm ~20% trong vùng chồng')+B(1100,206,190,'tail: compute','--tu');
  o+=B(430,266,220,'Sub: nạp TRF/VRF song song','--vrf')+B(1030,266,130,'Sub: scale','--vrf');
  o+=B(1130,326,100,'DmaStos','--sync-ink')+B(1240,326,100,'ExplicitSync','--sync-ink')+B(1400,326,80,'sync cuối','--sync-ink');
  o+='<rect x="200" y="46" width="300" height="4" fill="'+css('--store')+'"/>'+svgText(350,42,'HEAD',{s:12,w:600,f:css('--store'),m:1})+
     '<rect x="590" y="46" width="430" height="4" fill="'+css('--load')+'"/>'+svgText(805,42,'STREAM WEIGHT',{s:12,w:600,f:css('--load'),m:1})+
     '<rect x="704" y="364" width="320" height="4" fill="'+css('--tu')+'"/>'+svgText(864,384,'KHE giữa weight: DMA nhỏ → Wait → TU',{s:12,w:600,f:css('--tu'),m:1})+
     '<rect x="1100" y="364" width="380" height="4" fill="'+css('--sel')+'"/>'+svgText(1290,384,'TAIL: compute sau byte weight cuối + sync + store',{s:12,w:600,f:css('--sel'),m:1});
  o+=svgText(10,410,'Minh hoạ (không theo tỉ lệ): Wait{Dma:0} đợi cả hàng đợi nên compiler buộc phải phát DMA nhỏ trước, rồi Wait, rồi TU, rồi mới tới weight lớn.',{a:'start',s:12,f:css('--ink-3')});
  svg.innerHTML=o;}
// ===== D. hai cluster: dùng trace SAO thật =====
function drawClusters(){
  const T=TL.sao; if(!T) return; const svg=$('clsvg'); let o='<rect width="1500" height="400" fill="'+css('--panel')+'"/>';
  const X0=170, W=1290, mx=Math.max(T.c0.task,T.c1.task)*1.02, sx=W/mx;
  const rowsDef=[['C0 · DMA',0,'d',60],['C0 · Main',0,'m',96],['C0 · Sub',0,'s',132],['C1 · DMA',1,'d',200],['C1 · Main',1,'m',236],['C1 · Sub',1,'s',272]];
  rowsDef.forEach(function(r){o+=svgText(10,r[3]+18,r[0],{a:'start',s:12.5,w:600,f:css('--ink-2')})+'<line x1="'+X0+'" y1="'+(r[3]+30)+'" x2="1490" y2="'+(r[3]+30)+'" stroke="'+css('--rule-2')+'"/>';});
  [[0,'CLUSTER 0 · Task '+fmt(T.c0.task)],[1,'CLUSTER 1 · Task '+fmt(T.c1.task)]].forEach(function(c,i){o+=svgText(10,(i?184:44),c[1],{a:'start',s:13,w:600,f:css('--sel')});});
  const colOf={dma_load:'--load',dma_store:'--store',dma_dd:'--hbm',tu:'--tu',vrf:'--vrf',tab:'--hbm'};
  [0,1].forEach(function(ci){const C=ci?T.c1:T.c0;
    C.spans.forEach(function(s){ if(s.k==='sync'){const y0=ci?188:48,h=(ci?276+34:132+34)-y0+0, dur=s.e-s.b;
        if(dur<8000) o+='<rect x="'+(X0+s.b*sx)+'" y="'+y0+'" width="'+Math.max(2,dur*sx)+'" height="'+(h)+'" fill="'+css('--sync')+'" opacity=".5"><title>'+esc(s.l+' · '+fmt(dur)+' cycle')+'</title></rect>';
        return;}
      const row=rowsDef.find(function(r){return r[1]===ci&&r[2]===(s.k==='dma'?'d':(s.u==='m'?'m':'s'));}); if(!row) return;
      const c=css(colOf[s.t]||'--ink-3'), w=Math.max(1.5,(s.e-s.b)*sx);
      o+='<rect x="'+(X0+s.b*sx)+'" y="'+(row[3]+2)+'" width="'+w+'" height="26" rx="1.5" fill="'+c+'" fill-opacity=".8"><title>'+esc(s.l+' · '+fmt(s.b)+'→'+fmt(s.e)+' ('+fmt(s.e-s.b)+')')+'</title></rect>';});});
  // trục
  for(let c=0;c<=mx;c+=5000){o+='<line x1="'+(X0+c*sx)+'" y1="34" x2="'+(X0+c*sx)+'" y2="330" stroke="'+css('--rule-2')+'"/>'+svgText(X0+c*sx,346,fk(c),{s:11.5,f:css('--ink-3'),m:1});}
  // chú thích sync
  o+=svgText(X0,372,'Vùng đỏ nhạt = ExplicitSync / DramReuse ngắn hơn 8K. C1 chờ C0 sau DmaStos rồi chờ ở sync cuối; C0 chờ khi nhận phần y của C1. Mốc 0 của mỗi cluster là Task của chính nó.',{a:'start',s:12.5,f:css('--ink-3')});
  o+=svgText(X0,392,'Dữ liệu thật: trace job 93413, SAO. Rê chuột lên khối để đọc số.',{a:'start',s:12.5,f:css('--ink-3')});
  svg.innerHTML=o;}
// ===== E. thanh head/weight/khe/tail =====
function sysBars(){let h='';const mx=Math.max.apply(null,SEGBARS.map(function(b){return b.tot;}));
  SEGBARS.forEach(function(b){h+='<h3>'+b.k+' · '+fmt(b.tot)+' cycle</h3><div class="bar" style="width:'+(b.tot/mx*100).toFixed(1)+'%">';
    b.segs.forEach(function(s){h+='<div class="'+s[0]+'" style="flex:'+s[2]+' 1 0" title="'+esc(s[1]+' · '+fmt(s[2])+' cycle · '+s[3])+'">'+(s[2]/b.tot>0.06?esc(s[1]):'')+'</div>';});
    h+='</div><div class="barlab"><span>0</span><span>'+fk(b.tot)+' cycle</span></div>';});
  $('sysbars').innerHTML=h;}
