### sliding_attention_output — cluster 0: Task 37,561 cycles
DMA busy union 33,131 (88.2%), TU busy union 10,970 (29.2%), spans 24

| begin | end | dur | kind | emit# | op | static lifetime | source |
|---:|---:|---:|---|---:|---|---|---|
| 101 | 1,519 | 1,418 | DMA | 4 | DmaCommand | 1003-1742 | src/device/sliding/output.rs:67:60  xq.to_dm(&mut ctx.tdma) |
| 2,004 | 2,500 | 496 | TuExec | 16 | Main RenegadeCommand#O | 1742-2087 | src/device/sliding/output.rs:69:5  ctx.main .begin(xd.view()) .fetch:: |
| 2,011 | 27,657 | 25,646 | DMA | 17 | DmaCommand | 1742-15125 | src/device/sliding/output.rs:63:71  o_weight.to_dm(&mut ctx.tdma) |
| 4,155 | 5,164 | 1,009 | TuExec | 24 | Sub RenegadeCommand#O1 | 2166-2461 | src/device/sliding/output.rs:84:64  ctx .sub .begin(hl.view().tile::<m |
| 5,165 | 5,848 | 683 | TuExec | 30 | Main RenegadeCommand#O | 2461-2806 | src/device/sliding/output.rs:92:5  ctx.main .begin(xd.view()) .fetch:: |
| 5,849 | 7,607 | 1,758 | StoTrf | 36 | Sub RenegadeCommand#O1 | 2806-3607 | src/device/sliding/output.rs:108:85  ctx .sub .begin(hl.view()) // Mov |
| 5,881 | 31,075 | 25,194 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 1 |
| 27,659 | 31,074 | 3,415 | TuExec | 45 | Main RenegadeCommand#O | 15125-16499 | src/device/sliding/output.rs:120:81  ctx .main .begin(wd.view()) .fetc |
| 27,666 | 29,270 | 1,604 | DMA | 46 | DmaCommand | 15125-15741 | src/device/sliding/output.rs:153:9  o_weight_scale.to_dm(&mut ctx.tdma |
| 29,272 | 30,029 | 757 | TuExec | 51 | Sub RenegadeCommand#O3 | 15741-16034 | src/device/sliding/output.rs:157:67  ctx.sub .begin(sd.view()).fetch:: |
| 29,278 | 31,061 | 1,783 | DMA | 52 | DmaCommand | 15741-16357 | src/device/sliding/output.rs:149:9  post_attn_rms_weight.to_dm(&mut ct |
| 31,067 | 31,741 | 674 | TuExec | 59 | Sub RenegadeCommand#O3 | 16357-16650 | src/device/sliding/output.rs:195:67  ctx.sub .begin(wn.view()).fetch:: |
| 31,076 | 32,221 | 1,145 | DMA | 64 | DmaCommand | 16499-16945 | src/device/sliding/output.rs:145:9  y.to_dm(&mut ctx.tdma) |
| 31,742 | 32,885 | 1,143 | DMA | 68 | DmaCommand | 16945-17561 | src/device/sliding/output.rs:151:9  residual_hbm.to_dm(&mut ctx.tdma) |
| 32,223 | 33,612 | 1,389 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 2 |
| 32,887 | 35,835 | 2,948 | Cluster | None | Cluster |  | {"Synchronization": {"DramReuse": {"path": [], "index": 28}} |
| 32,889 | 34,016 | 1,127 | TuExec | 78 | Sub RenegadeCommand#O3 | 17561-17854 | src/device/sliding/output.rs:198:67  ctx.sub .begin(res.view()).fetch: |
| 33,613 | 34,176 | 563 | TuExec | 84 | Main RenegadeCommand#O | 18545-18856 | src/device/sliding/output.rs:165:72  ctx.main .begin(y_local).fetch::< |
| 34,177 | 34,886 | 709 | TuExec | 91 | Main RenegadeCommand#O | 18856-19169 | src/device/sliding/output.rs:175:81  ctx.main .begin(partial_ms.view() |
| 34,887 | 35,300 | 413 | TuExec | 97 | Main RenegadeCommand#O | 19169-19450 | src/device/sliding/output.rs:187:70  ctx.main .begin(norm_local).fetch |
| 35,301 | 35,834 | 533 | TuExec | 102 | Main RenegadeCommand#O | 19450-19761 | src/device/sliding/output.rs:203:65  ctx.main .begin(y_local).fetch::< |
| 35,836 | 36,707 | 871 | DMA | 108 | DmaCommand | 19761-20543 | src/device/sliding/output.rs:217:5  out_rows.to_hbm_view(&mut ctx.tdma |
| 36,709 | 37,558 | 849 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 4 |

### sliding_attention_output — cluster 1: Task 37,037 cycles
DMA busy union 28,863 (77.9%), TU busy union 11,488 (31.0%), spans 24

| begin | end | dur | kind | emit# | op | static lifetime | source |
|---:|---:|---:|---|---:|---|---|---|
| 101 | 1,507 | 1,406 | DMA | 4 | DmaCommand | 1003-1742 | src/device/sliding/output.rs:67:60  xq.to_dm(&mut ctx.tdma) |
| 2,084 | 2,576 | 492 | TuExec | 16 | Main RenegadeCommand#O | 1742-2087 | src/device/sliding/output.rs:69:5  ctx.main .begin(xd.view()) .fetch:: |
| 2,091 | 27,857 | 25,766 | DMA | 17 | DmaCommand | 1742-15125 | src/device/sliding/output.rs:63:71  o_weight.to_dm(&mut ctx.tdma) |
| 4,101 | 5,156 | 1,055 | TuExec | 24 | Sub RenegadeCommand#O1 | 2166-2461 | src/device/sliding/output.rs:84:64  ctx .sub .begin(hl.view().tile::<m |
| 5,157 | 5,836 | 679 | TuExec | 30 | Main RenegadeCommand#O | 2461-2806 | src/device/sliding/output.rs:92:5  ctx.main .begin(xd.view()) .fetch:: |
| 5,837 | 8,457 | 2,620 | StoTrf | 36 | Sub RenegadeCommand#O1 | 2806-3607 | src/device/sliding/output.rs:108:85  ctx .sub .begin(hl.view()) // Mov |
| 5,869 | 30,457 | 24,588 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 1 |
| 27,859 | 30,456 | 2,597 | TuExec | 45 | Main RenegadeCommand#O | 15125-16499 | src/device/sliding/output.rs:120:81  ctx .main .begin(wd.view()) .fetc |
| 27,866 | 28,029 | 163 | DMA | 46 | DmaCommand | 15125-15741 | src/device/sliding/output.rs:153:9  o_weight_scale.to_dm(&mut ctx.tdma |
| 28,276 | 28,961 | 685 | TuExec | 51 | Sub RenegadeCommand#O3 | 15741-16034 | src/device/sliding/output.rs:157:67  ctx.sub .begin(sd.view()).fetch:: |
| 28,283 | 28,445 | 162 | DMA | 52 | DmaCommand | 15741-16357 | src/device/sliding/output.rs:149:9  post_attn_rms_weight.to_dm(&mut ct |
| 28,963 | 30,763 | 1,800 | TuExec | 59 | Sub RenegadeCommand#O3 | 16357-16650 | src/device/sliding/output.rs:195:67  ctx.sub .begin(wn.view()).fetch:: |
| 30,458 | 31,657 | 1,199 | DMA | 64 | DmaCommand | 16499-16945 | src/device/sliding/output.rs:145:9  y.to_dm(&mut ctx.tdma) |
| 30,764 | 31,661 | 897 | DMA | 68 | DmaCommand | 16945-17561 | src/device/sliding/output.rs:151:9  residual_hbm.to_dm(&mut ctx.tdma) |
| 31,659 | 33,202 | 1,543 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 2 |
| 31,663 | 35,407 | 3,744 | Cluster | None | Cluster |  | {"Synchronization": {"DramReuse": {"path": [], "index": 28}} |
| 31,665 | 33,608 | 1,943 | TuExec | 78 | Sub RenegadeCommand#O3 | 17561-17854 | src/device/sliding/output.rs:198:67  ctx.sub .begin(res.view()).fetch: |
| 33,203 | 33,762 | 559 | TuExec | 84 | Main RenegadeCommand#O | 18545-18856 | src/device/sliding/output.rs:165:72  ctx.main .begin(y_local).fetch::< |
| 33,763 | 34,464 | 701 | TuExec | 91 | Main RenegadeCommand#O | 18856-19169 | src/device/sliding/output.rs:175:81  ctx.main .begin(partial_ms.view() |
| 34,465 | 34,874 | 409 | TuExec | 97 | Main RenegadeCommand#O | 19169-19450 | src/device/sliding/output.rs:187:70  ctx.main .begin(norm_local).fetch |
| 34,875 | 35,406 | 531 | TuExec | 102 | Main RenegadeCommand#O | 19450-19761 | src/device/sliding/output.rs:203:65  ctx.main .begin(y_local).fetch::< |
| 35,408 | 35,571 | 163 | DMA | 108 | DmaCommand | 19761-20543 | src/device/sliding/output.rs:217:5  out_rows.to_hbm_view(&mut ctx.tdma |
| 35,573 | 37,034 | 1,461 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 4 |
