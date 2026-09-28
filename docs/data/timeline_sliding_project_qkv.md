### sliding_project_qkv — cluster 0: Task 85,522 cycles
DMA busy union 73,839 (86.3%), TU busy union 58,618 (68.5%), spans 79

| begin | end | dur | kind | emit# | op | static lifetime | source |
|---:|---:|---:|---|---:|---|---|---|
| 99 | 1,253 | 1,154 | DMA | 17 | DmaCommand | 1003-1693 | src/device/shared/hidden.rs:95:65  x.to_dm(&mut ctx.tdma) |
| 3,871 | 4,428 | 557 | TuExec | 62 | Main RenegadeCommand#O | 1693-2004 | src/device/shared/hidden.rs:31:68  ctx .main .begin(x.view()) .fetch:: |
| 3,877 | 5,029 | 1,152 | DMA | 63 | DmaCommand | 1693-2383 | src/device/shared/hidden.rs:96:70  rms_weight.to_dm(&mut ctx.tdma) |
| 4,429 | 9,282 | 4,853 | TuExec | 68 | Main RenegadeCommand#O | 2004-4583 | src/device/shared/hidden.rs:49:78  ctx .main .begin(partial.view()) .f |
| 5,031 | 9,104 | 4,073 | TuExec | 71 | Sub RenegadeCommand#O9 | 2383-2676 | src/device/shared/hidden.rs:99:74  ctx .sub .begin(weight.view()) .fet |
| 7,420 | 8,877 | 1,457 | DMA | 72 | DmaCommand | 2383-3102 |  |
| 7,434 | 35,201 | 27,767 | DMA | 74 | DmaCommand | 3102-16485 | src/device/sliding/qkv.rs:97:64  q_weight.to_dm(&mut ctx.tdma) |
| 9,283 | 9,694 | 411 | TuExec | 82 | Main RenegadeCommand#O | 4583-4864 | src/device/shared/hidden.rs:66:5  ctx .main .begin(mean_square.view()) |
| 9,695 | 10,821 | 1,126 | TuExec | 87 | Main RenegadeCommand#O | 4864-5175 | src/device/shared/hidden.rs:107:74  ctx .main .begin(x.view()) .fetch: |
| 10,823 | 12,936 | 2,113 | TuExec | 94 | Main RenegadeCommand#O | 5175-8005 | src/device/shared/hidden.rs:350:75  ctx .main .begin(chunks.view()) .f |
| 12,937 | 14,216 | 1,279 | TuExec | 99 | Main RenegadeCommand#O | 8005-8749 | src/device/shared/hidden.rs:361:95  ctx .main .begin(gathered.view())  |
| 14,217 | 19,452 | 5,235 | TuExec | 107 | Main RenegadeCommand#O | 8749-9990 | src/device/sliding/qkv.rs:129:56  ctx .main .begin(on_q_slices(&xw)) . |
| 19,453 | 19,840 | 387 | TuExec | 112 | Main RenegadeCommand#O | 9990-10272 | src/device/sliding/qkv.rs:145:61  ctx .main .begin(ms.view()) .fetch:: |
| 19,841 | 20,278 | 437 | StoVrf | 119 | Sub RenegadeCommand#O9 | 10272-10539 | src/device/sliding/qkv.rs:160:66  ctx .sub .begin(unscale.view()) .fet |
| 20,279 | 22,572 | 2,293 | TuExec | 124 | Main RenegadeCommand#O | 10539-11780 | src/device/sliding/qkv.rs:168:58  ctx .main .begin(on_q_slices(&xw)) . |
| 20,285 | 20,632 | 347 | StoVrf | 126 | Sub RenegadeCommand#O1 | 10539-10806 | src/device/sliding/qkv.rs:314:55  ctx.sub .begin(unscale).fetch::<m![1 |
| 20,633 | 20,978 | 345 | StoVrf | 131 | Sub RenegadeCommand#O1 | 10806-11073 | src/device/sliding/qkv.rs:314:55  ctx.sub .begin(unscale).fetch::<m![1 |
| 20,979 | 21,324 | 345 | StoVrf | 135 | Sub RenegadeCommand#O1 | 11073-11340 | src/device/sliding/qkv.rs:314:55  ctx.sub .begin(unscale).fetch::<m![1 |
| 22,574 | 23,868 | 1,294 | StoTrf | 142 | Sub RenegadeCommand#O9 | 11780-12523 | src/device/sliding/qkv.rs:185:72  ctx .sub .begin(shared_lanes) .fetch |
| 35,203 | 44,042 | 8,839 | TuExec | 148 | Main RenegadeCommand#O | 16485-20590 | src/device/sliding/qkv.rs:193:72  ctx .main .begin(wq_work) .fetch::<m |
| 35,509 | 36,828 | 1,319 | DMA | 150 | DmaCommand | 16564-17125 | src/device/sliding/qkv.rs:247:56  sq_hbm.to_dm(&mut ctx.tdma) |
| 36,830 | 37,797 | 967 | TuExec | 154 | Sub RenegadeCommand#O1 | 17125-17452 | src/device/sliding/qkv.rs:311:52  ctx.sub .begin(scale.view()).fetch:: |
| 36,836 | 38,641 | 1,805 | DMA | 155 | DmaCommandGather | 17125-18059 | src/device/sliding/qkv.rs:429:73  cos.dma_gather_scaled(rope_offset) |
| 38,643 | 39,126 | 483 | TuExec | 163 | Sub RenegadeCommand#O2 | 18059-18390 | src/device/sliding/qkv.rs:432:5  ctx.sub .begin(cos_row.view()) .fetch |
| 38,650 | 40,441 | 1,791 | DMA | 164 | DmaCommandGather | 18059-18993 | src/device/sliding/qkv.rs:430:73  sin.dma_gather_scaled(rope_offset) |
| 40,443 | 41,217 | 774 | TuExec | 174 | Sub RenegadeCommand#O3 | 18993-19324 | src/device/sliding/qkv.rs:438:5  ctx.sub .begin(sin_row.view()) .fetch |
| 40,449 | 41,969 | 1,520 | DMA | 175 | DmaCommand | 18993-19554 | src/device/sliding/qkv.rs:455:52  rms_weight.to_dm(&mut ctx.tdma) |
| 41,971 | 42,444 | 473 | TuExec | 182 | Sub RenegadeCommand#O5 | 19554-19881 | src/device/sliding/qkv.rs:469:56  ctx .sub .begin(y) .fetch::<m![Ds /  |
| 41,977 | 42,729 | 752 | DMA | 183 | DmaCommand | 19554-19897 | src/device/sliding/qkv.rs:444:5  rows0.to_hbm(&mut ctx.tdma) |
| 41,983 | 56,927 | 14,944 | DMA | 186 | DmaCommand | 19897-26888 | src/device/sliding/qkv.rs:98:63  k_weight.to_dm(&mut ctx.tdma) |
| 42,731 | 56,936 | 14,205 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 9 |
| 44,043 | 44,668 | 625 | TuExec | 198 | Main RenegadeCommand#O | 20590-20981 | src/device/sliding/qkv.rs:218:54  ctx .main .begin(q_src.view()) .fetc |
| 44,669 | 45,350 | 681 | TuExec | 203 | Main RenegadeCommand#O | 20981-21326 | src/device/sliding/qkv.rs:317:5  ctx.main.begin(x.view()) .fetch::<m![ |
| 45,351 | 46,112 | 761 | TuExec | 208 | Main RenegadeCommand#O | 21326-21671 | src/device/sliding/qkv.rs:384:55  ctx .main .begin(x.view()) .fetch::< |
| 46,113 | 46,492 | 379 | TuExec | 215 | Main RenegadeCommand#O | 21671-21952 | src/device/sliding/qkv.rs:401:5  ctx .main .begin(ms.view()) .fetch::< |
| 46,493 | 46,826 | 333 | TuExec | 220 | Main RenegadeCommand#O | 21952-22224 | src/device/sliding/qkv.rs:494:5  ctx.main .begin(x.tile::<m![Ds], 128, |
| 46,983 | 47,316 | 333 | TuExec | 228 | Main RenegadeCommand#O | 22303-22575 | src/device/sliding/qkv.rs:500:5  ctx.main .begin(x.tile::<m![Ds], 128, |
| 47,317 | 47,650 | 333 | TuExec | 233 | Main RenegadeCommand#O | 22575-22847 | src/device/sliding/qkv.rs:494:5  ctx.main .begin(x.tile::<m![Ds], 128, |
| 47,323 | 49,071 | 1,748 | TuExec | 236 | Sub RenegadeCommand#O5 | 22575-22902 | src/device/sliding/qkv.rs:469:56  ctx .sub .begin(y) .fetch::<m![Ds /  |
| 48,599 | 49,074 | 475 | TuExec | 243 | Main RenegadeCommand#O | 22926-23198 | src/device/sliding/qkv.rs:500:5  ctx.main .begin(x.tile::<m![Ds], 128, |
| 56,929 | 59,996 | 3,067 | TuExec | 254 | Main RenegadeCommand#O | 26888-28113 | src/device/sliding/qkv.rs:289:67  ctx.main .begin(weight) .fetch::<m![ |
| 56,937 | 58,240 | 1,303 | DMA | 257 | DmaCommand | 26888-27460 | src/device/sliding/qkv.rs:107:69  rope_hbm.to_dm(&mut ctx.tdma) |
| 57,099 | 59,277 | 2,178 | DMA | 261 | DmaCommand | 27460-28015 | src/device/sliding/qkv.rs:248:57  sk_hbm.to_dm(&mut ctx.tdma) |
| 59,279 | 60,699 | 1,420 | TuExec | 269 | Sub RenegadeCommand#O1 | 28015-28342 | src/device/sliding/qkv.rs:311:52  ctx.sub .begin(scale.view()).fetch:: |
| 59,285 | 60,465 | 1,180 | DMA | 270 | DmaCommand | 28015-28570 | src/device/sliding/qkv.rs:455:52  rms_weight.to_dm(&mut ctx.tdma) |
| 59,997 | 60,784 | 787 | TuExec | 276 | Main RenegadeCommand#O | 28113-28440 | src/device/sliding/qkv.rs:226:55  ctx .main .begin(k_src.view()) .fetc |
| 60,785 | 61,458 | 673 | TuExec | 283 | Main RenegadeCommand#O | 28542-28886 | src/device/sliding/qkv.rs:476:5  ctx .main .begin(x) .fetch::<m![Ds /  |
| 60,791 | 74,587 | 13,796 | DMA | 284 | DmaCommand | 28570-35561 | src/device/sliding/qkv.rs:99:63  v_weight.to_dm(&mut ctx.tdma) |
| 61,459 | 62,144 | 685 | TuExec | 292 | Main RenegadeCommand#O | 28886-29231 | src/device/sliding/qkv.rs:342:51  ctx .main .begin(rot.view()) .fetch: |
| 62,145 | 62,826 | 681 | TuExec | 297 | Main RenegadeCommand#O | 29231-29576 | src/device/sliding/qkv.rs:317:5  ctx.main.begin(x.view()) .fetch::<m![ |
| 62,151 | 62,992 | 841 | StoVrf | 299 | Sub RenegadeCommand#O1 | 29231-29622 | src/device/sliding/qkv.rs:357:56  ctx .sub .begin(t.view()) .fetch::<m |
| 62,983 | 63,316 | 333 | TuExec | 305 | Main RenegadeCommand#O | 29655-29927 | src/device/sliding/qkv.rs:494:5  ctx.main .begin(x.tile::<m![Ds], 128, |
| 63,987 | 64,402 | 415 | TuExec | 315 | Main RenegadeCommand#O | 30006-30278 | src/device/sliding/qkv.rs:500:5  ctx.main .begin(x.tile::<m![Ds], 128, |
| 64,403 | 65,198 | 795 | TuExec | 320 | Main RenegadeCommand#O | 30278-30623 | src/device/sliding/qkv.rs:384:55  ctx .main .begin(x.view()) .fetch::< |
| 65,199 | 65,578 | 379 | TuExec | 326 | Main RenegadeCommand#O | 30623-30904 | src/device/sliding/qkv.rs:401:5  ctx .main .begin(ms.view()) .fetch::< |
| 65,579 | 65,914 | 335 | TuExec | 331 | Main RenegadeCommand#O | 30904-31176 | src/device/sliding/qkv.rs:494:5  ctx.main .begin(x.tile::<m![Ds], 128, |
| 66,543 | 67,252 | 709 | TuExec | 338 | Main RenegadeCommand#O | 31255-31527 | src/device/sliding/qkv.rs:500:5  ctx.main .begin(x.tile::<m![Ds], 128, |
| 67,254 | 68,220 | 966 | TuExec | 343 | Main RenegadeCommand#O | 31527-31871 | src/device/sliding/qkv.rs:476:5  ctx .main .begin(x) .fetch::<m![Ds /  |
| 68,221 | 69,361 | 1,140 | TuExec | 350 | Sub RenegadeCommand#O8 | 31950-32277 | src/device/sliding/qkv.rs:469:56  ctx .sub .begin(y) .fetch::<m![Ds /  |
| 69,362 | 70,036 | 674 | TuExec | 355 | Main RenegadeCommand#O | 32477-32821 | src/device/sliding/qkv.rs:476:5  ctx .main .begin(x) .fetch::<m![Ds /  |
| 70,038 | 70,791 | 753 | TuExec | 360 | Main RenegadeCommand#O | 32821-33166 | src/device/sliding/qkv.rs:342:51  ctx .main .begin(rot.view()) .fetch: |
| 70,331 | 70,788 | 457 | TuExec | 363 | Sub RenegadeCommand#O8 | 32900-33227 | src/device/sliding/qkv.rs:469:56  ctx .sub .begin(y) .fetch::<m![Ds /  |
| 70,792 | 71,488 | 696 | TuExec | 370 | Main RenegadeCommand#O | 33227-33572 | src/device/sliding/qkv.rs:364:5  ctx.main .begin(x.view()) .fetch::<m! |
| 70,798 | 72,234 | 1,436 | StoVrf | 372 | Sub RenegadeCommand#O1 | 33227-33618 | src/device/sliding/qkv.rs:357:56  ctx .sub .begin(t.view()) .fetch::<m |
| 72,235 | 72,908 | 673 | TuExec | 379 | Main RenegadeCommand#O | 33818-34162 | src/device/sliding/qkv.rs:476:5  ctx .main .begin(x) .fetch::<m![Ds /  |
| 72,909 | 73,896 | 987 | TuExec | 384 | Main RenegadeCommand#O | 34162-34507 | src/device/sliding/qkv.rs:364:5  ctx.main .begin(x.view()) .fetch::<m! |
| 74,589 | 80,722 | 6,133 | TuExec | 393 | Main RenegadeCommand#O | 35561-36786 | src/device/sliding/qkv.rs:289:67  ctx.main .begin(weight) .fetch::<m![ |
| 78,468 | 79,447 | 979 | DMA | 394 | DmaCommand | 35561-36012 | src/device/sliding/qkv.rs:258:5  q.view().to_hbm_view(&mut ctx.tdma, q |
| 78,629 | 80,221 | 1,592 | DMA | 396 | DmaCommand | 36012-36567 | src/device/sliding/qkv.rs:249:57  sv_hbm.to_dm(&mut ctx.tdma) |
| 80,223 | 81,124 | 901 | TuExec | 401 | Sub RenegadeCommand#O1 | 36567-36894 | src/device/sliding/qkv.rs:311:52  ctx.sub .begin(scale.view()).fetch:: |
| 80,229 | 81,879 | 1,650 | DMA | 402 | DmaCommandScatter | 36567-37496 | src/device/sliding/qkv.rs:261:5  k.dma_scatter::<m![1], _, _>(kv_offse |
| 80,723 | 81,510 | 787 | TuExec | 408 | Main RenegadeCommand#O | 36786-37113 | src/device/sliding/qkv.rs:234:55  ctx .main .begin(v_src.view()) .fetc |
| 81,511 | 82,192 | 681 | TuExec | 415 | Main RenegadeCommand#O | 37113-37458 | src/device/sliding/qkv.rs:317:5  ctx.main.begin(x.view()) .fetch::<m![ |
| 82,193 | 82,954 | 761 | TuExec | 420 | Main RenegadeCommand#O | 37458-37803 | src/device/sliding/qkv.rs:384:55  ctx .main .begin(x.view()) .fetch::< |
| 82,955 | 83,364 | 409 | TuExec | 428 | Main RenegadeCommand#O | 37803-38084 | src/device/sliding/qkv.rs:401:5  ctx .main .begin(ms.view()) .fetch::< |
| 83,365 | 83,866 | 501 | TuExec | 433 | Main RenegadeCommand#O | 38084-38429 | src/device/sliding/qkv.rs:264:56  ctx .main .begin(v.view()) .fetch::< |
| 83,867 | 85,515 | 1,648 | DMA | 437 | DmaCommandScatter | 38429-39358 | src/device/sliding/qkv.rs:279:5  v.dma_scatter::<m![1], _, _>(kv_offse |
| 85,517 | 85,519 | 2 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 1 |

### sliding_project_qkv — cluster 1: Task 85,718 cycles
DMA busy union 71,144 (83.0%), TU busy union 57,958 (67.6%), spans 77

| begin | end | dur | kind | emit# | op | static lifetime | source |
|---:|---:|---:|---|---:|---|---|---|
| 98 | 1,250 | 1,152 | DMA | 17 | DmaCommand | 1003-1693 | src/device/shared/hidden.rs:95:65  x.to_dm(&mut ctx.tdma) |
| 4,038 | 4,591 | 553 | TuExec | 62 | Main RenegadeCommand#O | 1693-2004 | src/device/shared/hidden.rs:31:68  ctx .main .begin(x.view()) .fetch:: |
| 4,044 | 5,818 | 1,774 | DMA | 63 | DmaCommand | 1693-2383 | src/device/shared/hidden.rs:96:70  rms_weight.to_dm(&mut ctx.tdma) |
| 4,592 | 9,451 | 4,859 | TuExec | 68 | Main RenegadeCommand#O | 2004-4583 | src/device/shared/hidden.rs:49:78  ctx .main .begin(partial.view()) .f |
| 5,820 | 9,449 | 3,629 | TuExec | 71 | Sub RenegadeCommand#O9 | 2383-2676 | src/device/shared/hidden.rs:99:74  ctx .sub .begin(weight.view()) .fet |
| 7,720 | 9,802 | 2,082 | DMA | 72 | DmaCommand | 2383-3102 |  |
| 7,734 | 35,560 | 27,826 | DMA | 74 | DmaCommand | 3102-16485 | src/device/sliding/qkv.rs:97:64  q_weight.to_dm(&mut ctx.tdma) |
| 9,452 | 9,863 | 411 | TuExec | 82 | Main RenegadeCommand#O | 4583-4864 | src/device/shared/hidden.rs:66:5  ctx .main .begin(mean_square.view()) |
| 9,864 | 11,256 | 1,392 | TuExec | 87 | Main RenegadeCommand#O | 4864-5175 | src/device/shared/hidden.rs:107:74  ctx .main .begin(x.view()) .fetch: |
| 11,258 | 13,363 | 2,105 | TuExec | 94 | Main RenegadeCommand#O | 5175-8005 | src/device/shared/hidden.rs:350:75  ctx .main .begin(chunks.view()) .f |
| 13,364 | 14,641 | 1,277 | TuExec | 99 | Main RenegadeCommand#O | 8005-8749 | src/device/shared/hidden.rs:361:95  ctx .main .begin(gathered.view())  |
| 14,642 | 19,873 | 5,231 | TuExec | 107 | Main RenegadeCommand#O | 8749-9990 | src/device/sliding/qkv.rs:129:56  ctx .main .begin(on_q_slices(&xw)) . |
| 19,874 | 20,257 | 383 | TuExec | 112 | Main RenegadeCommand#O | 9990-10272 | src/device/sliding/qkv.rs:145:61  ctx .main .begin(ms.view()) .fetch:: |
| 20,258 | 20,695 | 437 | StoVrf | 119 | Sub RenegadeCommand#O9 | 10272-10539 | src/device/sliding/qkv.rs:160:66  ctx .sub .begin(unscale.view()) .fet |
| 20,696 | 22,985 | 2,289 | TuExec | 124 | Main RenegadeCommand#O | 10539-11780 | src/device/sliding/qkv.rs:168:58  ctx .main .begin(on_q_slices(&xw)) . |
| 20,702 | 21,045 | 343 | StoVrf | 126 | Sub RenegadeCommand#O1 | 10539-10806 | src/device/sliding/qkv.rs:314:55  ctx.sub .begin(unscale).fetch::<m![1 |
| 21,046 | 21,391 | 345 | StoVrf | 131 | Sub RenegadeCommand#O1 | 10806-11073 | src/device/sliding/qkv.rs:314:55  ctx.sub .begin(unscale).fetch::<m![1 |
| 21,392 | 21,733 | 341 | StoVrf | 135 | Sub RenegadeCommand#O1 | 11073-11340 | src/device/sliding/qkv.rs:314:55  ctx.sub .begin(unscale).fetch::<m![1 |
| 22,987 | 24,277 | 1,290 | StoTrf | 142 | Sub RenegadeCommand#O9 | 11780-12523 | src/device/sliding/qkv.rs:185:72  ctx .sub .begin(shared_lanes) .fetch |
| 35,562 | 44,397 | 8,835 | TuExec | 148 | Main RenegadeCommand#O | 16485-20590 | src/device/sliding/qkv.rs:193:72  ctx .main .begin(wq_work) .fetch::<m |
| 35,866 | 37,155 | 1,289 | DMA | 150 | DmaCommand | 16564-17125 | src/device/sliding/qkv.rs:247:56  sq_hbm.to_dm(&mut ctx.tdma) |
| 37,157 | 38,094 | 937 | TuExec | 154 | Sub RenegadeCommand#O1 | 17125-17452 | src/device/sliding/qkv.rs:311:52  ctx.sub .begin(scale.view()).fetch:: |
| 38,095 | 38,579 | 484 | TuExec | 163 | Sub RenegadeCommand#O2 | 18059-18390 | src/device/sliding/qkv.rs:432:5  ctx.sub .begin(cos_row.view()) .fetch |
| 38,888 | 39,670 | 782 | TuExec | 174 | Sub RenegadeCommand#O3 | 18993-19324 | src/device/sliding/qkv.rs:438:5  ctx.sub .begin(sin_row.view()) .fetch |
| 38,900 | 40,182 | 1,282 | DMA | 175 | DmaCommand | 18993-19554 | src/device/sliding/qkv.rs:455:52  rms_weight.to_dm(&mut ctx.tdma) |
| 40,184 | 40,653 | 469 | TuExec | 182 | Sub RenegadeCommand#O5 | 19554-19881 | src/device/sliding/qkv.rs:469:56  ctx .sub .begin(y) .fetch::<m![Ds /  |
| 40,190 | 40,350 | 160 | DMA | 183 | DmaCommand | 19554-19897 | src/device/sliding/qkv.rs:444:5  rows0.to_hbm(&mut ctx.tdma) |
| 40,196 | 54,886 | 14,690 | DMA | 186 | DmaCommand | 19897-26888 | src/device/sliding/qkv.rs:98:63  k_weight.to_dm(&mut ctx.tdma) |
| 40,655 | 54,895 | 14,240 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 9 |
| 44,398 | 45,019 | 621 | TuExec | 198 | Main RenegadeCommand#O | 20590-20981 | src/device/sliding/qkv.rs:218:54  ctx .main .begin(q_src.view()) .fetc |
| 45,020 | 45,697 | 677 | TuExec | 203 | Main RenegadeCommand#O | 20981-21326 | src/device/sliding/qkv.rs:317:5  ctx.main.begin(x.view()) .fetch::<m![ |
| 45,698 | 46,455 | 757 | TuExec | 208 | Main RenegadeCommand#O | 21326-21671 | src/device/sliding/qkv.rs:384:55  ctx .main .begin(x.view()) .fetch::< |
| 46,456 | 46,831 | 375 | TuExec | 215 | Main RenegadeCommand#O | 21671-21952 | src/device/sliding/qkv.rs:401:5  ctx .main .begin(ms.view()) .fetch::< |
| 46,832 | 47,161 | 329 | TuExec | 220 | Main RenegadeCommand#O | 21952-22224 | src/device/sliding/qkv.rs:494:5  ctx.main .begin(x.tile::<m![Ds], 128, |
| 47,318 | 47,647 | 329 | TuExec | 228 | Main RenegadeCommand#O | 22303-22575 | src/device/sliding/qkv.rs:500:5  ctx.main .begin(x.tile::<m![Ds], 128, |
| 47,648 | 47,977 | 329 | TuExec | 233 | Main RenegadeCommand#O | 22575-22847 | src/device/sliding/qkv.rs:494:5  ctx.main .begin(x.tile::<m![Ds], 128, |
| 47,654 | 49,288 | 1,634 | TuExec | 236 | Sub RenegadeCommand#O5 | 22575-22902 | src/device/sliding/qkv.rs:469:56  ctx .sub .begin(y) .fetch::<m![Ds /  |
| 48,814 | 49,291 | 477 | TuExec | 243 | Main RenegadeCommand#O | 22926-23198 | src/device/sliding/qkv.rs:500:5  ctx.main .begin(x.tile::<m![Ds], 128, |
| 54,888 | 57,953 | 3,065 | TuExec | 254 | Main RenegadeCommand#O | 26888-28113 | src/device/sliding/qkv.rs:289:67  ctx.main .begin(weight) .fetch::<m![ |
| 54,896 | 56,211 | 1,315 | DMA | 257 | DmaCommand | 26888-27460 | src/device/sliding/qkv.rs:107:69  rope_hbm.to_dm(&mut ctx.tdma) |
| 55,058 | 57,236 | 2,178 | DMA | 261 | DmaCommand | 27460-28015 | src/device/sliding/qkv.rs:248:57  sk_hbm.to_dm(&mut ctx.tdma) |
| 57,238 | 58,656 | 1,418 | TuExec | 269 | Sub RenegadeCommand#O1 | 28015-28342 | src/device/sliding/qkv.rs:311:52  ctx.sub .begin(scale.view()).fetch:: |
| 57,244 | 58,424 | 1,180 | DMA | 270 | DmaCommand | 28015-28570 | src/device/sliding/qkv.rs:455:52  rms_weight.to_dm(&mut ctx.tdma) |
| 57,954 | 58,737 | 783 | TuExec | 276 | Main RenegadeCommand#O | 28113-28440 | src/device/sliding/qkv.rs:226:55  ctx .main .begin(k_src.view()) .fetc |
| 58,738 | 59,407 | 669 | TuExec | 283 | Main RenegadeCommand#O | 28542-28886 | src/device/sliding/qkv.rs:476:5  ctx .main .begin(x) .fetch::<m![Ds /  |
| 58,744 | 73,282 | 14,538 | DMA | 284 | DmaCommand | 28570-35561 | src/device/sliding/qkv.rs:99:63  v_weight.to_dm(&mut ctx.tdma) |
| 59,408 | 60,089 | 681 | TuExec | 292 | Main RenegadeCommand#O | 28886-29231 | src/device/sliding/qkv.rs:342:51  ctx .main .begin(rot.view()) .fetch: |
| 60,090 | 60,767 | 677 | TuExec | 297 | Main RenegadeCommand#O | 29231-29576 | src/device/sliding/qkv.rs:317:5  ctx.main.begin(x.view()) .fetch::<m![ |
| 60,096 | 60,935 | 839 | StoVrf | 299 | Sub RenegadeCommand#O1 | 29231-29622 | src/device/sliding/qkv.rs:357:56  ctx .sub .begin(t.view()) .fetch::<m |
| 60,926 | 61,255 | 329 | TuExec | 305 | Main RenegadeCommand#O | 29655-29927 | src/device/sliding/qkv.rs:494:5  ctx.main .begin(x.tile::<m![Ds], 128, |
| 61,832 | 62,243 | 411 | TuExec | 315 | Main RenegadeCommand#O | 30006-30278 | src/device/sliding/qkv.rs:500:5  ctx.main .begin(x.tile::<m![Ds], 128, |
| 62,244 | 63,065 | 821 | TuExec | 320 | Main RenegadeCommand#O | 30278-30623 | src/device/sliding/qkv.rs:384:55  ctx .main .begin(x.view()) .fetch::< |
| 63,066 | 63,441 | 375 | TuExec | 326 | Main RenegadeCommand#O | 30623-30904 | src/device/sliding/qkv.rs:401:5  ctx .main .begin(ms.view()) .fetch::< |
| 63,442 | 63,773 | 331 | TuExec | 331 | Main RenegadeCommand#O | 30904-31176 | src/device/sliding/qkv.rs:494:5  ctx.main .begin(x.tile::<m![Ds], 128, |
| 64,456 | 65,163 | 707 | TuExec | 338 | Main RenegadeCommand#O | 31255-31527 | src/device/sliding/qkv.rs:500:5  ctx.main .begin(x.tile::<m![Ds], 128, |
| 65,165 | 66,133 | 968 | TuExec | 343 | Main RenegadeCommand#O | 31527-31871 | src/device/sliding/qkv.rs:476:5  ctx .main .begin(x) .fetch::<m![Ds /  |
| 66,134 | 67,270 | 1,136 | TuExec | 350 | Sub RenegadeCommand#O8 | 31950-32277 | src/device/sliding/qkv.rs:469:56  ctx .sub .begin(y) .fetch::<m![Ds /  |
| 67,271 | 67,941 | 670 | TuExec | 355 | Main RenegadeCommand#O | 32477-32821 | src/device/sliding/qkv.rs:476:5  ctx .main .begin(x) .fetch::<m![Ds /  |
| 67,943 | 68,910 | 967 | TuExec | 360 | Main RenegadeCommand#O | 32821-33166 | src/device/sliding/qkv.rs:342:51  ctx .main .begin(rot.view()) .fetch: |
| 68,246 | 68,689 | 443 | TuExec | 363 | Sub RenegadeCommand#O8 | 32900-33227 | src/device/sliding/qkv.rs:469:56  ctx .sub .begin(y) .fetch::<m![Ds /  |
| 68,911 | 69,605 | 694 | TuExec | 370 | Main RenegadeCommand#O | 33227-33572 | src/device/sliding/qkv.rs:364:5  ctx.main .begin(x.view()) .fetch::<m! |
| 68,917 | 70,301 | 1,384 | StoVrf | 372 | Sub RenegadeCommand#O1 | 33227-33618 | src/device/sliding/qkv.rs:357:56  ctx .sub .begin(t.view()) .fetch::<m |
| 70,302 | 70,971 | 669 | TuExec | 379 | Main RenegadeCommand#O | 33818-34162 | src/device/sliding/qkv.rs:476:5  ctx .main .begin(x) .fetch::<m![Ds /  |
| 70,972 | 71,737 | 765 | TuExec | 384 | Main RenegadeCommand#O | 34162-34507 | src/device/sliding/qkv.rs:364:5  ctx.main .begin(x.view()) .fetch::<m! |
| 73,284 | 78,749 | 5,465 | TuExec | 393 | Main RenegadeCommand#O | 35561-36786 | src/device/sliding/qkv.rs:289:67  ctx.main .begin(weight) .fetch::<m![ |
| 76,499 | 77,476 | 977 | DMA | 394 | DmaCommand | 35561-36012 | src/device/sliding/qkv.rs:258:5  q.view().to_hbm_view(&mut ctx.tdma, q |
| 76,660 | 78,238 | 1,578 | DMA | 396 | DmaCommand | 36012-36567 | src/device/sliding/qkv.rs:249:57  sv_hbm.to_dm(&mut ctx.tdma) |
| 78,240 | 79,157 | 917 | TuExec | 401 | Sub RenegadeCommand#O1 | 36567-36894 | src/device/sliding/qkv.rs:311:52  ctx.sub .begin(scale.view()).fetch:: |
| 78,246 | 79,904 | 1,658 | DMA | 402 | DmaCommandScatter | 36567-37496 | src/device/sliding/qkv.rs:261:5  k.dma_scatter::<m![1], _, _>(kv_offse |
| 78,750 | 79,533 | 783 | TuExec | 408 | Main RenegadeCommand#O | 36786-37113 | src/device/sliding/qkv.rs:234:55  ctx .main .begin(v_src.view()) .fetc |
| 79,534 | 80,211 | 677 | TuExec | 415 | Main RenegadeCommand#O | 37113-37458 | src/device/sliding/qkv.rs:317:5  ctx.main.begin(x.view()) .fetch::<m![ |
| 80,212 | 80,969 | 757 | TuExec | 420 | Main RenegadeCommand#O | 37458-37803 | src/device/sliding/qkv.rs:384:55  ctx .main .begin(x.view()) .fetch::< |
| 80,970 | 81,381 | 411 | TuExec | 428 | Main RenegadeCommand#O | 37803-38084 | src/device/sliding/qkv.rs:401:5  ctx .main .begin(ms.view()) .fetch::< |
| 81,382 | 81,879 | 497 | TuExec | 433 | Main RenegadeCommand#O | 38084-38429 | src/device/sliding/qkv.rs:264:56  ctx .main .begin(v.view()) .fetch::< |
| 81,880 | 83,536 | 1,656 | DMA | 437 | DmaCommandScatter | 38429-39358 | src/device/sliding/qkv.rs:279:5  v.dma_scatter::<m![1], _, _>(kv_offse |
| 83,538 | 85,715 | 2,177 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 1 |
