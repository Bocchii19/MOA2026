// ===== helpers =====
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
function fmt(n){return Math.round(n).toLocaleString('vi-VN');}
function fk(n){return n>=1000?(n/1000).toFixed(n>=10000?1:2).replace('.',',')+'K':String(Math.round(n));}
function esc(s){return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function md(s){return esc(s).replace(/\*\*(.+?)\*\*/g,'<b>$1</b>').replace(/`(.+?)`/g,'<code>$1</code>');}
function fit(t,w,px){const n=Math.floor((w-8)/px);return t.length<=n?t:t.slice(0,Math.max(1,n-1))+'…';}
function $(id){return document.getElementById(id);}
function tabl(id,d,opt){opt=opt||{};let h='<thead><tr>'+d.h.map(function(x,i){return '<th'+(opt.num&&opt.num.indexOf(i)>=0?' class="n"':'')+'>'+esc(x)+'</th>';}).join('')+'</tr></thead><tbody>';
  d.r.forEach(function(r,ri){h+='<tr'+(opt.hot&&opt.hot(r,ri)?' class="hot"':'')+'>'+r.map(function(c,i){return '<td'+(opt.num&&opt.num.indexOf(i)>=0?' class="n"':'')+'>'+md(c)+'</td>';}).join('')+'</tr>';});
  $(id).innerHTML=h+'</tbody>';}
function tierCol(t){return css(TIERS[t].col);}
function svgText(x,y,t,o){o=o||{};return '<text x="'+x+'" y="'+y+'" text-anchor="'+(o.a||'middle')+'" font-size="'+(o.s||13)+'" font-weight="'+(o.w||400)+'" fill="'+(o.f||css('--ink'))+'" font-family="'+(o.m?css('--mono'):css('--sans'))+'"'+(o.x||'')+'>'+esc(t)+'</text>';}
function box(id,x,y,w,h,col,lines,o){o=o||{};const on=(o.sel===id);
  let s='<g class="blk" data-id="'+id+'"><rect class="b" x="'+x+'" y="'+y+'" width="'+w+'" height="'+h+'" rx="3" fill="'+col+'" fill-opacity="'+(o.fo||.14)+'" stroke="'+col+'" stroke-width="'+(on?3:1.4)+'"'+(o.dash?' stroke-dasharray="5 4"':'')+'/>';
  const n=lines.length, lh=o.lh||16, y0=y+h/2-(n-1)*lh/2+4;
  lines.forEach(function(t,i){const sub=i>0&&o.sub!==false;s+=svgText(x+w/2,y0+i*lh,fit(t,w,(sub?6.3:7)*(o.fs||1)),{s:(sub?12:(o.s||13.5)),w:(i===0?600:400),f:(i===0?css('--ink'):css('--ink-2'))});});
  return s+'</g>';}
function arrow(x1,y1,x2,y2,col,lab,o){o=o||{};col=col||css('--ink-3');const id='ah'+Math.abs((x1*7+y1*13+x2*17+y2*19)|0);
  const ang=Math.atan2(y2-y1,x2-x1), ah=8, bx=x2-Math.cos(ang)*1, by=y2-Math.sin(ang)*1;
  const p1=[bx-ah*Math.cos(ang-.45),by-ah*Math.sin(ang-.45)], p2=[bx-ah*Math.cos(ang+.45),by-ah*Math.sin(ang+.45)];
  let s='<line x1="'+x1+'" y1="'+y1+'" x2="'+(x2-Math.cos(ang)*4)+'" y2="'+(y2-Math.sin(ang)*4)+'" stroke="'+col+'" stroke-width="'+(o.w||1.6)+'"'+(o.dash?' stroke-dasharray="5 4"':'')+'/>'+
    '<polygon points="'+bx+','+by+' '+p1.join(',')+' '+p2.join(',')+'" fill="'+col+'"/>';
  if(o.both){const a2=ang+Math.PI, q1=[x1-ah*Math.cos(a2-.45)*-1,0]; const bx2=x1+Math.cos(ang)*1, by2=y1+Math.sin(ang)*1;
    const r1=[bx2+ah*Math.cos(ang-.45),by2+ah*Math.sin(ang-.45)], r2=[bx2+ah*Math.cos(ang+.45),by2+ah*Math.sin(ang+.45)];
    s+='<polygon points="'+bx2+','+by2+' '+r1.join(',')+' '+r2.join(',')+'" fill="'+col+'"/>';}
  if(lab) s+=svgText((x1+x2)/2+(o.dx||0),(y1+y2)/2+(o.dy||-6),lab,{s:12,f:css('--ink-3'),m:true});
  return s;}
function poly(pts,col,o){o=o||{};let s='<polyline points="'+pts.map(function(p){return p.join(',');}).join(' ')+'" fill="none" stroke="'+(col||css('--ink-3'))+'" stroke-width="'+(o.w||1.6)+'"'+(o.dash?' stroke-dasharray="5 4"':'')+'/>';
  const a=pts[pts.length-2], b=pts[pts.length-1]; const ang=Math.atan2(b[1]-a[1],b[0]-a[0]), ah=8;
  s+='<polygon points="'+b.join(',')+' '+(b[0]-ah*Math.cos(ang-.45))+','+(b[1]-ah*Math.sin(ang-.45))+' '+(b[0]-ah*Math.cos(ang+.45))+','+(b[1]-ah*Math.sin(ang+.45))+'" fill="'+(col||css('--ink-3'))+'"/>';
  if(o.lab) s+=svgText(o.lx,o.ly,o.lab,{s:12,f:css('--ink-3'),m:true,a:o.la||'middle'});
  return s;}

// ===== panel chung =====
function openPanel(id,html){const p=$(id);p.innerHTML='<button class="close" aria-label="Đóng">×</button>'+html;p.classList.remove('closed');
  p.querySelector('.close').onclick=function(){p.classList.add('closed');p.innerHTML='<p class="empty">Chưa chọn khối nào.</p>';CUR[id]=null;const m=id.match(/^panel-tl-(\w+)/);if(m&&TLS[m[1]])TLS[m[1]].sel=null;redrawAll();};}
const CUR={};
function panelBlock(id,pid){const b=BLOCKS[id], t=TIERS[b.t];
  let h='<h3>'+esc(b.n)+'</h3><p class="site"><span class="tier">T'+t.n+'</span>'+esc(t.name)+'</p>';
  h+='<dl><dt>Số lượng</dt><dd>'+esc(b.cnt)+'</dd>'+b.spec.map(function(s){return '<dt>'+esc(s[0])+'</dt><dd style="font-family:var(--sans);font-weight:400">'+md(s[1])+'</dd>';}).join('')+'</dl>';
  h+='<h4>Tác dụng</h4><p>'+md(b.role)+'</p>';
  if(b.tips.length) h+='<h4>Khi tối ưu</h4><ul>'+b.tips.map(function(x){return '<li>'+md(x)+'</li>';}).join('')+'</ul>';
  h+='<p class="site">Nguồn: '+esc(b.src)+'</p>';
  openPanel(pid,h);}

// ===== ① mặt phẳng chip =====
function drawChip(){
  const svg=$('chipsvg'), W=1500, H=1200, sel=CUR['panel-chip'];
  let o='<rect width="'+W+'" height="'+H+'" fill="'+css('--panel')+'"/>';
  const bands=[[0,110,'T0'],[110,240,'T1'],[240,360,'T2'],[360,838,'T3'],[838,1028,'T4'],[1028,1200,'T5']];
  bands.forEach(function(b,i){o+='<rect x="0" y="'+b[0]+'" width="'+W+'" height="'+(b[1]-b[0])+'" fill="'+(i%2?css('--panel'):css('--panel-2'))+'"/>'+
    '<line x1="0" y1="'+b[1]+'" x2="'+W+'" y2="'+b[1]+'" stroke="'+css('--rule')+'"/>';
    const t=TIERS[i]; o+='<rect x="10" y="'+(b[0]+10)+'" width="36" height="22" rx="2" fill="'+css(t.col)+'"/>'+svgText(28,b[0]+26,'T'+t.n,{s:13,w:600,f:'#fff',m:true});
    const words=t.name.split(' '); let ln=[],cur='';words.forEach(function(w){if((cur+' '+w).length>16&&cur){ln.push(cur);cur=w;}else cur=cur?cur+' '+w:w;});ln.push(cur);
    ln.forEach(function(l,k){o+=svgText(10,b[0]+50+k*15,l,{a:'start',s:12,f:css('--ink-2')});});});
  const S=function(id,x,y,w,h,t,l){return box(id,x,y,w,h,tierCol(BLOCKS[id].t),l,{sel:sel,fo:.15});};
  // T0 host + T1 hbm
  o+=S('host',560,26,380,58,0,['CPU Host · PCIe DMA (pdma)','khởi chạy kernel · đọc cycle']);
  o+=arrow(750,84,750,138,css('--host'),'PCIe: Host↔HBM',{dx:66,dy:-10});
  o+=S('hbm',250,126,420,96,1,['HBM3 · Stack 0','16 kênh × 48 GB/s']);
  o+=S('hbm',830,126,420,96,1,['HBM3 · Stack 1','16 kênh × 48 GB/s']);
  for(let s=0;s<2;s++)for(let k=0;k<16;k++){o+='<rect x="'+(266+s*580+k*24.6)+'" y="204" width="17" height="10" fill="'+css('--hbm')+'" fill-opacity=".45"/>';}
  o+=svgText(750,176,'48 GB · 750 B/cycle',{s:13,w:600,f:css('--hbm'),m:true})+svgText(750,194,'(đo ~616)',{s:12,f:css('--ink-3'),m:true});
  // T2 dma engines
  for(let e=0;e<8;e++){const x=190+e*152; o+=arrow(x+66,222,x+66,262,css('--load'));}
  for(let e=0;e<8;e++){const x=190+e*152; o+=box('dma',x,262,132,64,css('--load'),['DMA E'+e,e<4?'→ C0 · PE'+e:'→ C1 · PE'+(e-4)],{sel:sel,fo:.15});}
  [[330,'E0–E3 phục vụ Cluster 0'],[940,'E4–E7 phục vụ Cluster 1']].forEach(function(t){o+='<rect x="'+(t[0]-8)+'" y="336" width="210" height="18" fill="'+css('--panel-2')+'"/>'+svgText(t[0],349,t[1],{a:'start',s:12,f:css('--ink-3'),m:true});});
  // T3 clusters
  const cx=[150,850], cw=640, cy=372, chh=456;
  for(let c=0;c<2;c++){const x0=cx[c];
    o+='<g class="blk" data-id="cluster"><rect class="b" x="'+x0+'" y="'+cy+'" width="'+cw+'" height="'+chh+'" rx="5" fill="'+css('--sel')+'" fill-opacity=".05" stroke="'+css('--sel')+'" stroke-width="'+(sel==='cluster'?3:1.5)+'"/>'+
       svgText(x0+16,cy+24,'CLUSTER '+c+' · 4 PE × 64 slice = 256 slice',{a:'start',s:14,w:600,f:css('--sel')})+'</g>';
    for(let p=0;p<4;p++){const px=x0+14+(p%2)*318, py=cy+40+Math.floor(p/2)*206, pw=300, ph=190;
      o+='<g class="blk" data-id="pe"><rect class="b" x="'+px+'" y="'+py+'" width="'+pw+'" height="'+ph+'" rx="3" fill="'+css('--panel')+'" stroke="'+css('--sel')+'" stroke-opacity=".6" stroke-width="'+(sel==='pe'?3:1)+'"/>'+
         svgText(px+10,py+17,'PE '+p+' · DMA E'+(c*4+p),{a:'start',s:12.5,w:600,f:css('--ink-2')})+'</g>';
      for(let d=0;d<2;d++){const dy=py+26+d*66;
        o+='<g class="blk" data-id="dmn"><rect class="b" x="'+(px+10)+'" y="'+dy+'" width="'+(pw-20)+'" height="20" rx="2" fill="'+css('--load')+'" fill-opacity=".18" stroke="'+css('--load')+'" stroke-width="'+(sel==='dmn'?2.5:1)+'"/>'+
           svgText(px+pw/2,dy+14,'DMN '+(d?'b':'a')+' · 128 B/cycle',{s:11.5,f:css('--ink-2'),m:true})+'</g>';
        o+='<g class="blk" data-id="slice">';
        for(let k=0;k<32;k++){const hl=(c===0&&p===0&&d===0&&k===0);
          o+='<rect class="b" x="'+(px+10+k*8.75)+'" y="'+(dy+26)+'" width="7" height="16" fill="'+(hl?css('--tu'):css('--sel'))+'" fill-opacity="'+(hl?.95:.4)+'"/>';}
        o+='</g>';}
      o+='<g class="blk" data-id="ring"><line class="b" x1="'+(px+10)+'" y1="'+(py+142)+'" x2="'+(px+pw-10)+'" y2="'+(py+142)+'" stroke="'+css('--vec')+'" stroke-width="'+(sel==='ring'?3.5:2)+'" stroke-dasharray="5 3"/>'+
         svgText(px+pw-10,py+160,'ring switch · 256 router',{a:'end',s:11.5,f:css('--vec'),m:true})+'</g>';
      o+='<g class="blk" data-id="core"><rect class="b" x="'+(px+10)+'" y="'+(py+150)+'" width="96" height="30" rx="2" fill="'+css('--core')+'" fill-opacity=".2" stroke="'+css('--core')+'" stroke-width="'+(sel==='core'?2.5:1)+'"/>'+svgText(px+58,py+169,'PE Core',{s:12,f:css('--ink-2')})+'</g>';
      o+=svgText(px+pw-10,py+177,'64 slice: TU + DM + TRF/VRF',{a:'end',s:11.5,f:css('--ink-3')});
    }}
  o+='<g class="blk" data-id="sync"><line class="b" x1="'+(cx[0]+cw)+'" y1="596" x2="'+cx[1]+'" y2="596" stroke="'+css('--sync-ink')+'" stroke-width="'+(sel==='sync'?4:2.5)+'"/></g>'+
     arrow(cx[0]+cw+2,596,cx[1]-2,596,css('--sync-ink'),'',{both:true})+
     svgText(cx[0]+cw+29,574,'Cluster',{s:11,f:css('--sync-ink'),m:true})+svgText(cx[0]+cw+29,588,'Sync',{s:11,f:css('--sync-ink'),m:true})+
     svgText(cx[0]+cw+29,618,'DmaStos',{s:11,f:css('--sync-ink'),m:true})+svgText(cx[0]+cw+29,632,'swap/slice',{s:11,f:css('--sync-ink'),m:true});
  for(let e=0;e<8;e++){o+=arrow(256+e*152,326,256+e*152,372,css('--load'),'',{w:1.1,dash:1});}
  // zoom lines from highlighted slice
  o+=svgText(cx[0]+14+22,cy+40+26+34,'◂ ô tím: slice được phóng to ở T4',{a:'start',s:11,f:css('--tu'),m:1});
  // T4 zoom
  o+=svgText(420,848,'Phóng to một slice (ô tím ở PE 0, Cluster 0)',{a:'start',s:12,f:css('--tu'),m:true});
  const T4=function(id,x,y,w,h,l,col){return box(id,x,y,w,h,col||css('--tu'),l,{sel:sel,fo:.14,lh:15});};
  o+=T4('dm',170,852,210,130,['DM · SRAM','512 KB · 16 bank','4.096 row × 8 B'],css('--dm'));
  o+=arrow(275,832,275,850,css('--load'),'DMA ghi',{dx:44,dy:0});
  o+=T4('fetch',420,858,80,56,['Fetch','packet 8 B']);
  o+=T4('fadapter',514,858,100,56,['Fetch Adapter','lookup f4→f8']);
  o+=T4('switch',628,858,80,56,['Switch','ring 256']);
  o+=T4('collect',722,858,80,56,['Collect','flit 32 B']);
  o+=T4('contraction',830,858,110,56,['Contraction','MAC · 8 lane']);
  o+=T4('vector',956,858,110,56,['Vector','Way4 · RMS/GeLU'],css('--vec'));
  o+=T4('cast',1090,858,70,56,['Cast','f32→bf16']);
  o+=T4('transpose',1174,858,90,56,['Transpose','đổi layout']);
  o+=T4('cadapter',1274,858,110,56,['Commit Adapter','hạ bf16']);
  o+=T4('commit',1392,858,80,56,['Commit','→ DM']);
  o+=arrow(380,886,420,886,css('--ink-3'));o+=arrow(500,886,514,886,css('--ink-3'));o+=arrow(614,886,628,886,css('--ink-3'));
  o+=arrow(708,886,722,886,css('--ink-3'));o+=arrow(802,886,830,886,css('--ink-3'));
  o+=poly([[815,886],[815,842],[1011,842],[1011,858]],css('--ink-3'));
  o+=poly([[885,914],[885,934],[1125,934],[1125,914]],css('--ink-3'));
  o+=arrow(1066,886,1090,886,css('--ink-3'));o+=arrow(1160,886,1174,886,css('--ink-3'));o+=arrow(1264,886,1274,886,css('--ink-3'));o+=arrow(1384,886,1392,886,css('--ink-3'));
  o+=poly([[1432,914],[1432,1012],[275,1012],[275,984]],css('--dm'),{lab:'kết quả ghi lại DM · rồi DMA đẩy về HBM',lx:560,ly:1004});
  o+=svgText(1480,848,'Main: đủ pipeline · Sub: không Contraction, không lookup',{a:'end',s:11.5,f:css('--ink-3'),m:true});
  // T5
  const T5=function(id,x,y,w,h,l){return box(id,x,y,w,h,css('--vrf'),l,{sel:sel,fo:.15,lh:15});};
  o+=T5('trf',560,1090,200,66,['TRF · 8 KB / lane','toán hạng tĩnh (x hi/lo)']);
  o+=T5('lane',800,1090,170,66,['Lane MAC × 8','hàng MAC của Contraction']);
  o+=T5('vrf',1010,1090,190,66,['VRF · 8 KB / slice','scale · 1/rms · unscale']);
  o+=arrow(760,1123,800,1123,css('--vrf'));
  o+=poly([[885,1090],[885,916]],css('--vrf'));
  o+=poly([[1030,1090],[1030,916]],css('--vrf'));
  o+=svgText(170,1075,'StoTrf / StoVrf (Sub context) nạp TRF và VRF từ DM',{a:'start',s:12,f:css('--ink-3'),m:true});
  o+=svgText(170,1100,'TRF: đọc 1 packet/cycle · double-buffer',{a:'start',s:12,f:css('--ink-3'),m:true});
  o+=svgText(170,1120,'VRF: một toán hạng ≤ 8.192 B',{a:'start',s:12,f:css('--ink-3'),m:true});
  svg.innerHTML=o;}

function legendChip(el,items){$(el).innerHTML=items.map(function(i){return '<span><i style="background:var('+i[0]+')"></i>'+esc(i[1])+'</span>';}).join('');}

// ===== ② thẻ khối =====
function blockCards(){let h='';
  TIERS.forEach(function(t){
    const ids=Object.keys(BLOCKS).filter(function(k){return BLOCKS[k].t===t.n;});
    h+='<h3><span class="tier">T'+t.n+'</span>'+esc(t.name)+'</h3><div class="cards">';
    ids.forEach(function(k){const b=BLOCKS[k];
      h+='<div class="card" style="border-left-color:var('+t.col+')"><h4>'+esc(b.n)+'</h4><p class="k">'+esc(b.cnt)+' · '+esc(b.src)+'</p><p>'+md(b.role)+'</p>'+
        (b.spec.length?'<ul>'+b.spec.slice(0,3).map(function(s){return '<li><b>'+esc(s[0])+':</b> '+md(s[1])+'</li>';}).join('')+'</ul>':'')+
        (b.tips.length?'<ul style="margin-top:6px;color:var(--ink-2)">'+b.tips.slice(0,2).map(function(x){return '<li>'+md(x)+'</li>';}).join('')+'</ul>':'')+'</div>';});
    h+='</div>';});
  $('blockcards').innerHTML=h;}
