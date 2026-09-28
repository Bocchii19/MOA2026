// ===== timeline thật (swimlane) =====
const TLS={};
const KEYS=['qkv','sao','ffn'];
const KIDX={qkv:0,sao:1,ffn:2,qkv91:3,sao91:4,ffn91:5};
const TCOL={dma_load:'--load',dma_store:'--store',dma_dd:'--hbm',tu:'--tu',vrf:'--vrf',tab:'--hbm',sync:'--sync-ink'};
const TNAME={dma_load:'DMA nạp · HBM → DM',dma_store:'DMA ghi · DM → HBM',dma_dd:'DMA · DM → DM / gather',tu:'Tensor Unit · Main',vrf:'Tensor Unit · Sub',tab:'StoTab · nạp bảng FP4',sync:'Đồng bộ (Cluster)'};
function tlMap(key,mode){
  const T=TL[key], all=T.c0.spans.concat(T.c1.spans), mx=Math.max(T.c0.task,T.c1.task)+300;
  if(mode==='that'){const s=0.03;return {segs:[[0,mx,s]],mx:mx,H:mx*s};}
  let iv=[];
  all.forEach(function(sp){const d=sp.e-sp.b; if(sp.k==='sync'&&d>=8000) return;
    if(d<10000) iv.push([sp.b-250,sp.e+250]); else{iv.push([sp.b-250,sp.b+1000]);iv.push([sp.e-1000,sp.e+250]);}});
  iv.sort(function(a,b){return a[0]-b[0];});
  const mg=[]; iv.forEach(function(v){v=[Math.max(0,v[0]),Math.min(mx,v[1])]; if(mg.length&&v[0]<=mg[mg.length-1][1]+500) mg[mg.length-1][1]=Math.max(mg[mg.length-1][1],v[1]); else mg.push(v);});
  let busy=0; mg.forEach(function(v){busy+=v[1]-v[0];});
  const sd=Math.min(0.07,Math.max(0.018,2400/busy)), si=0.0045; const segs=[]; let cur=0;
  mg.forEach(function(v){if(v[0]>cur) segs.push([cur,v[0],si]); segs.push([v[0],v[1],sd]); cur=v[1];});
  if(cur<mx) segs.push([cur,mx,si]);
  let H=0; segs.forEach(function(s){H+=(s[1]-s[0])*s[2];});
  return {segs:segs,mx:mx,H:H,sd:sd};
}
function mapY(M,c){let y=0; for(const s of M.segs){ if(c<=s[0]) break; y+=(Math.min(c,s[1])-s[0])*s[2]; } return y;}
function packLanes(list){ // list: [{i,b,e}] -> gán lane cục bộ
  const a=list.slice().sort(function(x,y){return x.b-y.b||x.e-y.e;}), res={}; let grp=[],gEnd=-1,laneEnds=[];
  function flush(){const n=Math.max(1,laneEnds.length); grp.forEach(function(g){res[g.i]={l:g.lane,n:n};}); grp=[];laneEnds=[];}
  a.forEach(function(s){ if(grp.length&&s.b>=gEnd) {flush();gEnd=-1;}
    let l=laneEnds.findIndex(function(e){return e<=s.b;}); if(l<0){l=laneEnds.length;laneEnds.push(0);} laneEnds[l]=s.e; grp.push({i:s.i,lane:l}); gEnd=Math.max(gEnd,s.e);});
  flush(); return res;}
function segOf(key,c){const S=SEGBARS[KIDX[key]]; let t=0; for(const s of S.segs){ if(c<t+s[2]) return {n:s[1],b:t,e:t+s[2],x:s[3],cls:s[0]}; t+=s[2]; } const l=S.segs[S.segs.length-1]; return {n:l[1],b:t-l[2],e:t,x:l[3],cls:l[0]};}
function drawTL(key){
  const st=TLS[key], T=TL[key], svg=$('tlsvg-'+key), M=tlMap(key,st.mode), TOP=44, X=[142,786], CW={d:210,m:190,s:170,y:34}, GAP=4;
  const W=1428, H=TOP+M.H+34; let o='<rect width="'+W+'" height="'+H+'" fill="'+css('--panel')+'"/>';
  const colX=function(g){const x0=X[g]; return {d:x0,m:x0+CW.d+GAP,s:x0+CW.d+CW.m+2*GAP,y:x0+CW.d+CW.m+CW.s+3*GAP};};
  // nén
  M.segs.forEach(function(s){ if(s[2]<0.02){const y0=TOP+mapY(M,s[0]), h=mapY(M,s[1])-mapY(M,s[0]); o+='<rect x="0" y="'+y0+'" width="'+(W-6)+'" height="'+h+'" fill="'+css('--squeeze')+'"/>'; if(h>26) o+=svgText(6,y0+h/2+4,'nén',{a:'start',s:11,f:css('--ink-3'),m:1});}});
  // tiêu đề cột
  [0,1].forEach(function(g){const c=colX(g), tk=g?T.c1:T.c0;
    o+=svgText(X[g],16,'CLUSTER '+g+' · Task '+fmt(tk.task),{a:'start',s:13,w:600,f:css('--sel')});
    [['d','DMA'],['m','Main'],['s','Sub'],['y','Sync']].forEach(function(cc){o+=svgText(c[cc[0]]+CW[cc[0]]/2,36,cc[1],{s:12.5,w:600,f:css('--ink-2')})+'<rect x="'+c[cc[0]]+'" y="'+TOP+'" width="'+CW[cc[0]]+'" height="'+M.H+'" fill="none" stroke="'+css('--rule-2')+'"/>';});});
  // đoạn head/khe/tail (theo cluster 0)
  let t=0,lastY=-99; SEGBARS[KIDX[key]].segs.forEach(function(s){const y=TOP+mapY(M,t);
    o+='<line x1="0" y1="'+y+'" x2="'+(W-6)+'" y2="'+y+'" stroke="'+css('--ink-3')+'" stroke-dasharray="2 4" opacity=".55"/>';
    if(y-lastY>=13){o+=svgText(6,y+11,s[1],{a:'start',s:11,w:600,f:css(s[0]==='hd'?'--store':(s[0]==='wt'?'--load':(s[0]==='gp'?'--tu':'--sel'))),m:1});lastY=y;} t+=s[2];});
  // trục
  const step=st.mode==='that'?(TL[key].c0.task>100000?10000:2000):(TL[key].c0.task>100000?5000:2000); let lastT=-99;
  for(let c=0;c<=M.mx;c+=step){const y=TOP+mapY(M,c); if(y-lastT<15) continue; lastT=y; o+=svgText(126,y+4,fk(c),{a:'end',s:11,f:css('--ink-3'),m:1})+'<line x1="130" y1="'+y+'" x2="140" y2="'+y+'" stroke="'+css('--ink-3')+'"/>';}
  // băng sync ngắn
  [0,1].forEach(function(g){const c=colX(g); (g?T.c1:T.c0).spans.forEach(function(sp){ if(sp.k==='sync'&&sp.e-sp.b<8000&&st.show.y){const y0=TOP+mapY(M,sp.b),h=Math.max(3,mapY(M,sp.e)-mapY(M,sp.b));
      o+='<rect x="'+c.d+'" y="'+y0+'" width="'+(c.y+CW.y-c.d)+'" height="'+h+'" fill="'+css('--sync')+'" opacity=".8"/>';}});});
  // khối
  let bl='';
  [0,1].forEach(function(g){const c=colX(g), sp=(g?T.c1:T.c0).spans, cols={d:[],m:[],s:[],y:[]};
    sp.forEach(function(s,i){const k=s.k==='dma'?'d':(s.k==='sync'?'y':(s.u==='m'?'m':'s')); if(!st.show[k]) return; cols[k].push({i:i,b:s.b,e:s.e});});
    Object.keys(cols).forEach(function(k){const L=packLanes(cols[k]);
      cols[k].forEach(function(q){const s=sp[q.i], ln=L[q.i], w=CW[k]/ln.n, x=c[k]+ln.l*w, y=TOP+mapY(M,s.b), h=Math.max(2.6,mapY(M,s.e)-mapY(M,s.b)), col=css(TCOL[s.t]||'--ink-3'), on=(st.sel&&st.sel[0]===g&&st.sel[1]===q.i);
        bl+='<g class="blk" data-g="'+g+'" data-i="'+q.i+'"><rect class="b" x="'+(x+.5)+'" y="'+y.toFixed(1)+'" width="'+(w-1.5).toFixed(1)+'" height="'+h.toFixed(1)+'" rx="1.5" fill="'+col+'" fill-opacity="'+(s.k==='sync'?.55:.82)+'" stroke="'+(on?css('--ink'):col)+'" stroke-width="'+(on?2.5:.8)+'"><title>'+esc(s.l+' · '+fmt(s.b)+' → '+fmt(s.e)+' ('+fmt(s.e-s.b)+' cycle)')+'</title></rect>';
        if(h>=13&&w>=44) bl+=svgText(x+w/2,y+Math.min(h/2+4,15),fit(s.l,w,6.1),{s:11,f:'#fff',w:500});
        bl+='</g>';});});});
  svg.setAttribute('viewBox','0 0 '+W+' '+H); svg.innerHTML=o+bl;}
function tlDetail(key,g,i){
  const st=TLS[key], T=TL[key], C=g?T.c1:T.c0, s=C.spans[i], pid='panel-tl-'+key; st.sel=[g,i]; CUR[pid]=1;
  const sg=segOf(key,s.b), dur=s.e-s.b, task=C.task;
  let h='<h3>'+esc(s.l)+'</h3><p class="site">Cluster '+g+' · '+esc(TNAME[s.t]||s.t)+'</p>';
  if(s.d) h+='<p>'+md(s.d)+'</p>';
  h+='<dl><dt>Bắt đầu</dt><dd>'+fmt(s.b)+'</dd><dt>Kết thúc</dt><dd>'+fmt(s.e)+'</dd><dt>Thời lượng</dt><dd>'+fmt(dur)+' cycle ('+(dur/task*100).toFixed(1)+'% Task)</dd>';
  if(s.by){const r=s.bytes?(s.bytes/2/dur):0; h+='<dt>Payload</dt><dd>'+esc(s.by)+'</dd>'; if(r>50) h+='<dt>Tốc độ</dt><dd>~'+r.toFixed(0)+' B/cycle/cluster</dd>';}
  h+='<dt>Đoạn</dt><dd>'+esc(sg.n)+' ('+fk(sg.b)+' → '+fk(sg.e)+')</dd>';
  if(s.f) h+='<dt>Hàm</dt><dd>'+esc(s.f)+'</dd>';
  if(s.s) h+='<dt>Source</dt><dd>'+esc(s.s)+'</dd>';
  if(s.life) h+='<dt>Lifetime tĩnh</dt><dd>'+esc(s.life)+' (1 GHz)</dd>';
  if(s.em!=null) h+='<dt>emit#</dt><dd>'+esc(s.em)+'</dd>';
  const O=(g?T.c0:T.c1).spans.find(function(x){return s.em!=null&&x.em===s.em&&x.k===s.k&&x.u===s.u;});
  if(O) h+='<dt>Cluster '+(1-g)+'</dt><dd>'+fmt(O.b)+' → '+fmt(O.e)+' ('+fmt(O.e-O.b)+'; '+((O.e-O.b)/dur).toFixed(2)+'×)</dd>';
  h+='</dl>';
  if(sg.x) h+='<h4>Đoạn này trong tài liệu</h4><p>'+md(sg.x)+'</p>';
  openPanel(pid,h); drawTL(key);}
function tlControls(key){const st=TLS[key];
  $('ctl-'+key).innerHTML='<span class="l">Trục thời gian:</span>'+
   '<button class="tbtn" data-m="nen" aria-pressed="'+(st.mode==='nen')+'">Nén khoảng chỉ có stream</button>'+
   '<button class="tbtn" data-m="that" aria-pressed="'+(st.mode==='that')+'">Thang thật</button>'+
   '<span class="l" style="margin-left:14px">Hiện:</span>'+
   [['d','DMA'],['m','Main'],['s','Sub'],['y','Sync']].map(function(x){return '<button class="tbtn" data-k="'+x[0]+'" aria-pressed="'+st.show[x[0]]+'">'+x[1]+'</button>';}).join('');}

// ===== tab kernel =====
function segTable(K){return {h:['Đoạn','Từ → Đến (cycle)','Độ dài','Nội dung'],r:K.segs.map(function(s){return [s[0],s[1],s[2],s[3]];})};}
function buildKernel(key){const K=KDOC[key], el=$('t-'+key);
  let h='<h2>'+esc(K.short)+' · <code>'+esc(K.entry)+'</code></h2><p class="sub">'+md(K.intro)+'</p><div class="chips">'+
    K.chips.map(function(c){return '<span class="chip">'+esc(c[0])+' <b>'+esc(c[1])+'</b>'+(c[2]?' · '+esc(c[2]):'')+'</span>';}).join('')+'</div>';
  h+='<h2>Dataflow của kernel</h2><p class="note">Sơ đồ khối: tensor đi từ HBM qua DMA vào DM, qua các pass Tensor Unit rồi ra HBM. Bấm khối để xem số đo trên phần cứng.</p>'+
    '<div class="legend" id="lgdf-'+key+'"></div><div class="layout"><div class="chart"><svg id="dfsvg-'+key+'"></svg></div><aside id="panel-df-'+key+'" class="closed"><p class="empty">Chưa chọn khối nào.</p></aside></div>';
  h+='<h2>Cải tiến so với baseline của ban tổ chức</h2>'+'<div class="chips"><span class="spd">'+esc(badgeRow(key).replace(/^[A-Z]+: /,''))+'</span><span class="chip">điểm official <b>8,4643×</b> so với baseline 1,0×</span></div>'+'<p class="note">Số trong vòng tròn trên sơ đồ khớp với số ở bảng dưới. <b>Không có A/B riêng cho từng cải tiến</b>: official chỉ đo cả kernel nên các con số chỉ là bằng chứng gián tiếp; dòng "Bằng chứng" ghi rõ khi số đo thuộc bản trung gian hoặc SDK 0.6.0.</p>'+'<div class="tw"><table id="timp-'+key+'"></table></div>';
  h+='<h2>Head, weight, khe, tail (cluster 0)</h2><div id="segbar-'+key+'"></div>';
  h+='<h2>Timeline đo trên phần cứng (trace 93413, hai cluster)</h2><p class="note">Mỗi khối là một span thật của một lệnh DMA hoặc Tensor Unit; mỗi cluster có mốc 0 riêng (Task của chính nó). Span DMA bắt đầu lúc <b>phát lệnh</b>, nên span chồng nhau thường chỉ là lệnh sau xếp hàng sau lệnh trước. Đường đứt ngang là ranh giới head/weight/khe/tail lấy từ tài liệu.</p>'+
    '<div class="legend" id="lgtl-'+key+'"></div><div class="ctl" id="ctl-'+key+'"></div>'+
    '<div class="layout"><div class="chart"><svg id="tlsvg-'+key+'"></svg></div><aside id="panel-tl-'+key+'" class="closed"><p class="empty">Chưa chọn khối nào.</p></aside></div>';
  h+='<h2>Timeline theo đoạn</h2><div class="tw"><table id="tseg-'+key+'"></table></div>';
  h+='<h2>Lệnh dài nhất trên cluster 0</h2><div class="tw"><table id="ttop-'+key+'"></table></div>';
  h+='<h2>Điểm nghẽn xếp theo độ lớn</h2><div class="tw"><table id="tneck-'+key+'"></table></div>';
  h+='<div class="two"><div><h2>Đã đo là thua (đừng lặp lại)</h2><div class="tw"><table id="tclosed-'+key+'"></table></div></div><div><h2>Trần thực tế</h2><div class="callout">'+md(K.ceiling)+'</div><h3>Đường găng xấp xỉ</h3><div class="callout">'+md(K.path)+'</div></div></div>';
  el.innerHTML='<div class="ctl" id="vsw-'+key+'"><span class="l">Phiên bản kernel:</span><button class="tbtn" data-v="91">9,1481 · score_9_3 (mới)</button><button class="tbtn" data-v="84">8,4643 · b162deec (có trace thật)</button></div><div id="v91-'+key+'"></div><div id="v84-'+key+'">'+h+'</div>';
  $('vsw-'+key).onclick=function(ev){const b=ev.target.closest('button'); if(b) setVer(key,b.dataset.v);};
  $('timp-'+key).innerHTML=impTable(key);
  $('timp-'+key).onclick=function(ev){const r=ev.target.closest('tr.rowlink'); if(!r) return; const it=IMP[key].find(function(q){return q.n===+r.dataset.n;}); const nd=it.nodes[0];CUR['panel-df-'+key]=nd;panelNode('df-'+key,nd);redrawDF(key);$('dfsvg-'+key).scrollIntoView({block:'center'});};
  tabl('tseg-'+key,segTable(K),{num:[1,2],hot:function(r){return /^(Head|Khe|Tail|Sync|Store|Chuyển|Projection|RMS)/.test(r[0]);}});
  tabl('tneck-'+key,{h:['#','Điểm nghẽn','Cycle','Bằng chứng','Hướng khắc phục (có số đo)'],r:K.neck});
  tabl('tclosed-'+key,{h:['Hướng','Kết quả phần cứng','Nguồn'],r:K.closed});
  // top spans
  const T=TL[key], top=T.c0.spans.map(function(s,i){return {s:s,i:i};}).filter(function(x){return x.s.k!=='sync';}).sort(function(a,b){return (b.s.e-b.s.b)-(a.s.e-a.s.b);}).slice(0,10);
  $('ttop-'+key).innerHTML='<thead><tr><th>Lệnh</th><th>Đơn vị</th><th class="n">Bắt đầu</th><th class="n">Kết thúc</th><th class="n">Cycle</th><th class="n">% Task</th></tr></thead><tbody>'+top.map(function(x){const s=x.s;return '<tr class="rowlink" data-i="'+x.i+'"><td>'+esc(s.l)+'</td><td>'+esc(TNAME[s.t]||s.t)+'</td><td class="n">'+fmt(s.b)+'</td><td class="n">'+fmt(s.e)+'</td><td class="n">'+fmt(s.e-s.b)+'</td><td class="n">'+((s.e-s.b)/T.c0.task*100).toFixed(1)+'%</td></tr>';}).join('')+'</tbody>';
  $('ttop-'+key).onclick=function(ev){const r=ev.target.closest('tr.rowlink'); if(r){tlDetail(key,0,+r.dataset.i);$('tlsvg-'+key).scrollIntoView({block:'center'});}};
  // thanh đoạn
  const S=SEGBARS[KIDX[key]]; let b='<div class="bar">'+S.segs.map(function(s){return '<div class="'+s[0]+'" style="flex:'+s[2]+' 1 0" title="'+esc(s[1]+' · '+fmt(s[2])+' cycle · '+s[3])+'">'+(s[2]/S.tot>0.07?esc(s[1]):'')+'</div>';}).join('')+'</div><div class="barlab"><span>0</span><span>'+fmt(S.tot)+'</span></div>';
  $('segbar-'+key).innerHTML=b;
  legendChip('lgtl-'+key,[['--load','DMA nạp HBM→DM'],['--store','DMA ghi DM→HBM'],['--hbm','DM→DM / gather / StoTab'],['--tu','Main (Tensor Unit)'],['--vrf','Sub (TRF/VRF)'],['--sync-ink','Sync']]);
  legendChip('lgdf-'+key,[['--hbm','HBM (tensor)'],['--load','DMA'],['--tu','Tensor Unit'],['--vec','Vector / RMS / GeLU'],['--vrf','TRF / VRF'],['--store','ghi HBM'],['--sync-ink','đồng bộ']]);
  $('lgdf-'+key).innerHTML+='<span><span class="nb">1</span>cải tiến so với baseline (khung đứt)</span><span><span class="nb new">3</span>mới ở 8,4643</span><span><span class="nb eq">=</span>giữ như baseline</span><span><i style="background:none;border:1.5px dashed var(--store)"></i>baseline làm (đã thay)</span>';
  TLS[key]={mode:'nen',sel:null,show:{d:true,m:true,s:true,y:true}};
  tlControls(key);
  $('ctl-'+key).onclick=function(ev){const b=ev.target.closest('button'); if(!b) return; const st=TLS[key];
    if(b.dataset.m) st.mode=b.dataset.m; else if(b.dataset.k) st.show[b.dataset.k]=!st.show[b.dataset.k]; tlControls(key); drawTL(key);};
  $('tlsvg-'+key).addEventListener('click',function(ev){const g=ev.target.closest('g.blk'); if(g) tlDetail(key,+g.dataset.g,+g.dataset.i); else if(TLS[key].sel){TLS[key].sel=null;$('panel-tl-'+key).classList.add('closed');drawTL(key);}});
  $('dfsvg-'+key).addEventListener('click',function(ev){const g=ev.target.closest('g.blk'); if(g){CUR['panel-df-'+key]=g.dataset.id;panelNode('df-'+key,g.dataset.id);redrawDF(key);}});
}
const CFGS={qkv:function(){return markCfg('qkv',cfgQKV());},sao:function(){return markCfg('sao',cfgSAO());},ffn:function(){return markCfg('ffn',cfgFFN());},
  qkv91:function(){return markCfg('qkv91',cfgQKV91());},sao91:function(){return markCfg('sao91',cfgSAO91());},ffn91:function(){return markCfg('ffn91',cfgFFN91());}};
const VER={qkv:'91',sao:'91',ffn:'91'};
function setVer(key,v){VER[key]=v; $('v91-'+key).style.display=(v==='91')?'block':'none'; $('v84-'+key).style.display=(v==='84')?'block':'none';
  Array.prototype.forEach.call($('vsw-'+key).querySelectorAll('button'),function(b){b.setAttribute('aria-pressed',String(b.dataset.v===v));});}
function redrawDF(key){drawDiagram('dfsvg-'+key,'df-'+key,CFGS[key]());}

// ===== tab, theme, khởi tạo =====
const TABS=[['chip','① Mặt phẳng chip'],['blocks','② Tác dụng từng khối'],['sys','③ Dataflow hệ thống'],['qkv','④ QKV'],['sao','⑤ SAO'],['ffn','⑥ FFN']];
function goTab(id){TABS.forEach(function(t){$('t-'+t[0]).classList.toggle('on',t[0]===id);});
  Array.prototype.forEach.call($('pages').children,function(b){b.classList.toggle('on',b.dataset.t===id);});
  try{history.replaceState(null,'','#'+id);}catch(e){} window.scrollTo(0,0);}
function redrawAll(){
  drawChip(); drawDiagram('syssvg','sys',cfgSys()); drawChain(); drawExec(); drawClusters();
  KEYS.forEach(function(k){redrawDF(k); drawTL(k); redrawDF(k+'91'); drawTL(k+'91');});}
function init(){
  $('pages').innerHTML=TABS.map(function(t){return '<button data-t="'+t[0]+'">'+esc(t[1])+'</button>';}).join('');
  $('pages').onclick=function(ev){const b=ev.target.closest('button'); if(b) goTab(b.dataset.t);};
  legendChip('lg-chip',[['--host','Host'],['--hbm','HBM'],['--load','DMA / DMN'],['--sel','Cluster · PE · Slice'],['--tu','Tensor Unit'],['--dm','DM (SRAM)'],['--vec','Vector'],['--vrf','TRF / VRF / lane'],['--sync-ink','Sync liên cluster']]);
  legendChip('lg-sys',[['--hbm','HBM'],['--load','DMA'],['--dm','DM'],['--tu','Tensor Unit'],['--vec','Vector'],['--vrf','TRF/VRF'],['--store','ghi HBM'],['--sync-ink','xuyên cluster']]);
  tabl('memtab',MEMTAB); tabl('addrtab',ADDRTAB); tabl('clocktab',CLOCKTAB); tabl('bytetab',BYTETAB,{num:[1,2,3,4]});
  blockCards(); tabl('ctxtab',CTXTAB); tabl('stream-ops',STREAMOPS); tabl('synctab',SYNCTAB); tabl('balancetab',BALANCETAB); tabl('statictab',STATICTAB);
  sysBars();
  KEYS.forEach(function(k){buildKernel(k); buildKernel91(k); setVer(k,VER[k]);});
  $('chipsvg').addEventListener('click',function(ev){const g=ev.target.closest('g.blk'); if(g){CUR['panel-chip']=g.dataset.id;panelBlock(g.dataset.id,'panel-chip');drawChip();}else if(CUR['panel-chip']){CUR['panel-chip']=null;$('panel-chip').classList.add('closed');drawChip();}});
  $('syssvg').addEventListener('click',function(ev){const g=ev.target.closest('g.blk'); if(g){CUR['panel-sys']=g.dataset.id;panelNode('sys',g.dataset.id);drawDiagram('syssvg','sys',cfgSys());}});
  $('chainsvg').addEventListener('click',function(ev){const g=ev.target.closest('g[data-go]'); if(g) goTab(g.dataset.go);});
  $('themebtn').onclick=function(){const r=document.documentElement, dark=(r.dataset.theme==='dark')||(!r.dataset.theme&&matchMedia('(prefers-color-scheme: dark)').matches);
    r.dataset.theme=dark?'light':'dark'; $('themebtn').setAttribute('aria-pressed',String(!dark)); redrawAll();};
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change',redrawAll);
  document.addEventListener('keydown',function(ev){ if(ev.key==='Escape'){document.querySelectorAll('aside .close').forEach(function(b){b.click();});}});
  redrawAll();
  const h=(location.hash||'').slice(1); goTab(TABS.some(function(t){return t[0]===h;})?h:'chip');
}
if(document.fonts&&document.fonts.ready) document.fonts.ready.then(init); else init();
