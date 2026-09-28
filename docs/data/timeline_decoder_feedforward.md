### decoder_feedforward — cluster 0: Task 224,274 cycles
DMA busy union 197,904 (88.2%), TU busy union 158,766 (70.8%), spans 102

| begin | end | dur | kind | emit# | op | static lifetime | source |
|---:|---:|---:|---|---:|---|---|---|
| 96 | 1,248 | 1,152 | DMA | 8 | DmaCommand | 1003-1693 | src/device/shared/hidden.rs:95:65  x.to_dm(&mut ctx.tdma) |
| 2,347 | 211,198 | 208,851 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 7 |
| 5,361 | 5,919 | 558 | TuExec | 87 | Main RenegadeCommand#O | 1693-2004 | src/device/shared/hidden.rs:31:68  ctx .main .begin(x.view()) .fetch:: |
| 5,367 | 6,526 | 1,159 | DMA | 88 | DmaCommand | 1693-2383 | src/device/shared/hidden.rs:96:70  rms_weight.to_dm(&mut ctx.tdma) |
| 5,920 | 11,233 | 5,313 | TuExec | 93 | Main RenegadeCommand#O | 2004-4583 | src/device/shared/hidden.rs:49:78  ctx .main .begin(partial.view()) .f |
| 6,528 | 9,464 | 2,936 | TuExec | 96 | Sub RenegadeCommand#O6 | 2383-2676 | src/device/shared/hidden.rs:99:74  ctx .sub .begin(weight.view()) .fet |
| 8,764 | 10,236 | 1,472 | DMA | 97 | DmaCommand | 2383-3102 |  |
| 9,465 | 11,752 | 2,287 | DMA | 101 | DmaCommand | 3102-3940 |  |
| 11,226 | 59,018 | 47,792 | DMA | 103 | DmaCommand | 3940-28552 | src/device/shared/mlp.rs:44:13  $weight_packed.to_dm(&mut $ctx.tdma) |
| 11,234 | 11,641 | 407 | TuExec | 108 | Main RenegadeCommand#O | 4583-4864 | src/device/shared/hidden.rs:66:5  ctx .main .begin(mean_square.view()) |
| 11,642 | 18,589 | 6,947 | TuExec | 113 | Main RenegadeCommand#O | 4864-5175 | src/device/shared/hidden.rs:107:74  ctx .main .begin(x.view()) .fetch: |
| 11,754 | 19,101 | 7,347 | StoTab | 116 | Sub RenegadeCommand#O8 | 4864-6157 |  |
| 18,592 | 20,705 | 2,113 | TuExec | 124 | Main RenegadeCommand#O | 6239-9069 | src/device/shared/hidden.rs:350:75  ctx .main .begin(chunks.view()) .f |
| 20,706 | 21,983 | 1,277 | TuExec | 131 | Main RenegadeCommand#O | 9069-9813 | src/device/shared/hidden.rs:361:95  ctx .main .begin(gathered.view())  |
| 22,380 | 23,707 | 1,327 | TuExec | 139 | Main RenegadeCommand#O | 10013-10774 | src/device/shared/mlp.rs:152:9  ctx.main .begin(x_half) .fetch::<m![H  |
| 24,408 | 25,203 | 795 | TuExec | 154 | Sub RenegadeCommand#O3 | 11053-11556 | src/device/shared/mlp.rs:182:13  ctx .sub .begin(hi_half_raw) .fetch:: |
| 25,204 | 27,967 | 2,763 | TuExec | 158 | Main RenegadeCommand#O | 11556-12317 | src/device/shared/mlp.rs:192:9  ctx.main .begin(x_half) .fetch::<m![H  |
| 28,364 | 29,691 | 1,327 | TuExec | 167 | Main RenegadeCommand#O | 12517-13278 | src/device/shared/mlp.rs:152:9  ctx.main .begin(x_half) .fetch::<m![H  |
| 30,396 | 31,189 | 793 | TuExec | 179 | Sub RenegadeCommand#O5 | 13557-14060 | src/device/shared/mlp.rs:182:13  ctx .sub .begin(hi_half_raw) .fetch:: |
| 31,190 | 33,953 | 2,763 | TuExec | 183 | Main RenegadeCommand#O | 14060-14821 | src/device/shared/mlp.rs:192:9  ctx.main .begin(x_half) .fetch::<m![H  |
| 33,954 | 36,209 | 2,255 | StoTrf | 191 | Sub RenegadeCommand#O7 | 14821-16044 | src/device/shared/mlp.rs:220:5  ctx.sub .begin(lanes) .fetch::<m![Dumm |
| 59,024 | 81,476 | 22,452 | TuExec | 200 | Main RenegadeCommand#O | 28558-37821 | src/device/shared/mlp.rs:57:99  $ctx .main .begin($chunk.0.view()) .fe |
| 59,031 | 69,310 | 10,279 | DMA | 201 | DmaCommand | 28558-33438 | src/device/shared/mlp.rs:46:13  $weight_scale.to_dm(&mut $ctx.tdma) |
| 59,037 | 70,324 | 11,287 | DMA | 211 | DmaCommand | 33438-34098 | src/device/shared/mlp.rs:247:84  global_scale.to_dm(&mut ctx.tdma) |
| 69,612 | 70,589 | 977 | TuExec | 216 | Sub RenegadeCommand#O8 | 33517-33960 | src/device/shared/mlp.rs:91:107  $ctx.sub .begin(s_tile) .fetch::<m![L |
| 70,591 | 70,939 | 348 | StoVrf | 225 | Sub RenegadeCommand#O2 | 34098-34365 | src/device/shared/mlp.rs:248:89  ctx .sub .begin(global_scale.view())  |
| 70,622 | 72,578 | 1,956 | DMA | 226 | DmaCommand | 34098-34936 |  |
| 72,580 | 81,474 | 8,894 | StoTab | 231 | Sub RenegadeCommand#O1 | 34936-36229 |  |
| 72,638 | 120,788 | 48,150 | DMA | 233 | DmaCommand | 34942-59554 | src/device/shared/mlp.rs:44:13  $weight_packed.to_dm(&mut $ctx.tdma) |
| 81,872 | 84,037 | 2,165 | TuExec | 248 | Main RenegadeCommand#O | 38021-38661 | src/device/shared/mlp.rs:100:17  $ctx.main.begin(p_tile) .fetch::<m![L |
| 84,038 | 85,012 | 974 | TuExec | 256 | Sub RenegadeCommand#O1 | 38661-39104 | src/device/shared/mlp.rs:91:107  $ctx.sub .begin(s_tile) .fetch::<m![L |
| 85,013 | 87,180 | 2,167 | TuExec | 264 | Main RenegadeCommand#O | 39183-39823 | src/device/shared/mlp.rs:100:17  $ctx.main.begin(p_tile) .fetch::<m![L |
| 87,181 | 88,156 | 975 | TuExec | 268 | Sub RenegadeCommand#O1 | 39823-40266 | src/device/shared/mlp.rs:91:107  $ctx.sub .begin(s_tile) .fetch::<m![L |
| 88,157 | 90,324 | 2,167 | TuExec | 276 | Main RenegadeCommand#O | 40345-40985 | src/device/shared/mlp.rs:100:17  $ctx.main.begin(p_tile) .fetch::<m![L |
| 90,325 | 91,300 | 975 | TuExec | 280 | Sub RenegadeCommand#O1 | 40985-41428 | src/device/shared/mlp.rs:91:107  $ctx.sub .begin(s_tile) .fetch::<m![L |
| 91,301 | 93,468 | 2,167 | TuExec | 288 | Main RenegadeCommand#O | 41507-42147 | src/device/shared/mlp.rs:100:17  $ctx.main.begin(p_tile) .fetch::<m![L |
| 93,469 | 94,143 | 674 | TuExec | 292 | Sub RenegadeCommand#O1 | 42147-42590 | src/device/shared/mlp.rs:91:107  $ctx.sub .begin(s_tile) .fetch::<m![L |
| 94,144 | 96,309 | 2,165 | TuExec | 299 | Main RenegadeCommand#O | 42590-43230 | src/device/shared/mlp.rs:100:17  $ctx.main.begin(p_tile) .fetch::<m![L |
| 96,310 | 96,825 | 515 | TuExec | 305 | Main RenegadeCommand#O | 43230-43585 | src/device/shared/mlp.rs:119:9  $ctx.main.begin(tile_sums.view()) .fet |
| 96,826 | 97,425 | 599 | TuExec | 310 | Main RenegadeCommand#O | 43585-43969 | src/device/shared/mlp.rs:232:5  ctx.main .begin(x.view()) .fetch::<m![ |
| 97,426 | 97,913 | 487 | TuExec | 315 | Main RenegadeCommand#O | 43969-44310 | src/device/shared/mlp.rs:255:5  ctx.main .begin(x.view()) .fetch::<m![ |
| 120,790 | 144,048 | 23,258 | TuExec | 321 | Main RenegadeCommand#O | 59554-68817 | src/device/shared/mlp.rs:57:99  $ctx .main .begin($chunk.0.view()) .fe |
| 120,797 | 130,990 | 10,193 | DMA | 322 | DmaCommand | 59554-64434 | src/device/shared/mlp.rs:46:13  $weight_scale.to_dm(&mut $ctx.tdma) |
| 120,803 | 132,068 | 11,265 | DMA | 328 | DmaCommand | 64434-65094 | src/device/shared/mlp.rs:247:84  global_scale.to_dm(&mut ctx.tdma) |
| 131,292 | 132,251 | 959 | TuExec | 333 | Sub RenegadeCommand#O1 | 64513-64956 | src/device/shared/mlp.rs:91:107  $ctx.sub .begin(s_tile) .fetch::<m![L |
| 132,253 | 132,603 | 350 | StoVrf | 342 | Sub RenegadeCommand#O2 | 65094-65361 | src/device/shared/mlp.rs:248:89  ctx .sub .begin(global_scale.view())  |
| 132,284 | 133,870 | 1,586 | DMA | 343 | DmaCommand | 65094-65813 |  |
| 132,604 | 135,816 | 3,212 | DMA | 346 | DmaCommand | 65813-66651 |  |
| 135,820 | 144,046 | 8,226 | StoTab | 350 | Sub RenegadeCommand#O2 | 66654-67947 |  |
| 135,875 | 185,428 | 49,553 | DMA | 351 | DmaCommand | 66654-91507 | src/device/shared/mlp.rs:545:67  down_weight_packed.to_dm(&mut ctx.tdm |
| 144,444 | 146,609 | 2,165 | TuExec | 366 | Main RenegadeCommand#O | 69017-69657 | src/device/shared/mlp.rs:100:17  $ctx.main.begin(p_tile) .fetch::<m![L |
| 146,610 | 147,584 | 974 | TuExec | 374 | Sub RenegadeCommand#O1 | 69657-70100 | src/device/shared/mlp.rs:91:107  $ctx.sub .begin(s_tile) .fetch::<m![L |
| 147,585 | 149,752 | 2,167 | TuExec | 382 | Main RenegadeCommand#O | 70179-70819 | src/device/shared/mlp.rs:100:17  $ctx.main.begin(p_tile) .fetch::<m![L |
| 149,753 | 150,728 | 975 | TuExec | 386 | Sub RenegadeCommand#O1 | 70819-71262 | src/device/shared/mlp.rs:91:107  $ctx.sub .begin(s_tile) .fetch::<m![L |
| 150,729 | 152,896 | 2,167 | TuExec | 394 | Main RenegadeCommand#O | 71341-71981 | src/device/shared/mlp.rs:100:17  $ctx.main.begin(p_tile) .fetch::<m![L |
| 152,897 | 153,872 | 975 | TuExec | 398 | Sub RenegadeCommand#O1 | 71981-72424 | src/device/shared/mlp.rs:91:107  $ctx.sub .begin(s_tile) .fetch::<m![L |
| 153,873 | 156,040 | 2,167 | TuExec | 406 | Main RenegadeCommand#O | 72503-73143 | src/device/shared/mlp.rs:100:17  $ctx.main.begin(p_tile) .fetch::<m![L |
| 156,041 | 156,715 | 674 | TuExec | 410 | Sub RenegadeCommand#O2 | 73143-73586 | src/device/shared/mlp.rs:91:107  $ctx.sub .begin(s_tile) .fetch::<m![L |
| 156,716 | 158,881 | 2,165 | TuExec | 417 | Main RenegadeCommand#O | 73586-74226 | src/device/shared/mlp.rs:100:17  $ctx.main.begin(p_tile) .fetch::<m![L |
| 158,882 | 159,397 | 515 | TuExec | 423 | Main RenegadeCommand#O | 74226-74581 | src/device/shared/mlp.rs:119:9  $ctx.main.begin(tile_sums.view()) .fet |
| 159,398 | 159,997 | 599 | TuExec | 428 | Main RenegadeCommand#O | 74581-74965 | src/device/shared/mlp.rs:232:5  ctx.main .begin(x.view()) .fetch::<m![ |
| 159,998 | 160,711 | 713 | TuExec | 433 | Main RenegadeCommand#O | 74965-75306 | src/device/shared/mlp.rs:255:5  ctx.main .begin(x.view()) .fetch::<m![ |
| 160,712 | 161,349 | 637 | TuExec | 438 | Sub RenegadeCommand#O2 | 75306-75709 | src/device/shared/mlp.rs:282:78  ctx .sub .begin(gate.view()) .fetch:: |
| 161,351 | 161,927 | 576 | StoVrf | 442 | Sub RenegadeCommand#O2 | 75709-76092 | src/device/shared/mlp.rs:301:83  ctx .sub .begin(gelu.view()) .fetch:: |
| 161,928 | 162,589 | 661 | TuExec | 447 | Main RenegadeCommand#O | 76092-76433 | src/device/shared/mlp.rs:308:5  ctx.main .begin(up.view()) .fetch::<m! |
| 162,590 | 163,321 | 731 | TuExec | 453 | Main RenegadeCommand#O | 76433-76774 | src/device/shared/mlp.rs:339:81  ctx .main .begin(x.view()) .fetch::<m |
| 163,322 | 166,091 | 2,769 | TuExec | 459 | Main RenegadeCommand#O | 76774-77805 | src/device/shared/mlp.rs:355:88  ctx .main .begin(slice_max.view()) .f |
| 166,092 | 166,521 | 429 | TuExec | 464 | Main RenegadeCommand#O | 77805-78118 | src/device/shared/mlp.rs:363:75  ctx .main .begin(maxima.view()) .fetc |
| 166,522 | 166,907 | 385 | TuExec | 469 | Main RenegadeCommand#O | 78118-78400 | src/device/shared/mlp.rs:376:75  ctx .main .begin(unscale.view()) .fet |
| 166,909 | 167,345 | 436 | StoVrf | 475 | Sub RenegadeCommand#O2 | 78400-78667 | src/device/shared/mlp.rs:392:84  ctx .sub .begin(unscale_live) .fetch: |
| 167,346 | 167,840 | 494 | TuExec | 480 | Main RenegadeCommand#O | 78667-79008 | src/device/shared/mlp.rs:401:5  ctx.main .begin(x.view()) .fetch::<m![ |
| 167,352 | 167,699 | 347 | StoVrf | 482 | Sub RenegadeCommand#O2 | 78667-78934 | src/device/shared/mlp.rs:456:78  ctx .sub .begin(unscale_rows) .fetch: |
| 168,542 | 168,917 | 375 | TuExec | 493 | Sub RenegadeCommand#O2 | 79287-79580 | src/device/shared/mlp.rs:415:81  ctx .sub .begin(lanes.view().tile::<m |
| 168,918 | 169,589 | 671 | TuExec | 497 | Main RenegadeCommand#O | 79580-79921 | src/device/shared/mlp.rs:422:5  ctx.main .begin(x.view()) .fetch::<m![ |
| 169,590 | 178,647 | 9,057 | TuExec | 504 | Main RenegadeCommand#O | 79921-84536 | src/device/shared/mlp.rs:439:118  ctx .main .begin(lanes.view()) .fetc |
| 178,648 | 182,799 | 4,151 | TuExec | 510 | Sub RenegadeCommand#O2 | 84536-86719 | src/device/shared/mlp.rs:449:89  ctx .sub .begin(gathered) .fetch::<m! |
| 185,432 | 208,211 | 22,779 | TuExec | 517 | Main RenegadeCommand#O | 91510-100773 | src/device/shared/mlp.rs:467:98  ctx .main .begin(w.view()) .fetch::<m |
| 185,439 | 196,122 | 10,683 | DMA | 518 | DmaCommand | 91510-96471 | src/device/shared/mlp.rs:546:67  down_weight_scale.to_dm(&mut ctx.tdma |
| 196,124 | 203,116 | 6,992 | TuExec | 522 | Sub RenegadeCommand#O2 | 96471-98535 | src/device/shared/mlp.rs:484:68  ctx .sub .begin(s.view()) .fetch::<m! |
| 196,130 | 198,118 | 1,988 | DMA | 523 | DmaCommand | 96471-97213 | src/device/shared/hidden.rs:476:9  scalar.to_dm(&mut ctx.tdma) |
| 203,092 | 204,288 | 1,196 | DMA | 526 | DmaCommand | 98735-99351 | src/device/shared/hidden.rs:451:9  rms_weight.to_dm(&mut ctx.tdma) |
| 203,098 | 205,372 | 2,274 | DMA | 527 | DmaCommand | 99351-99967 | src/device/shared/hidden.rs:498:78  residual.to_dm(&mut ctx.tdma) |
| 203,104 | 206,978 | 3,874 | DMA | 528 | DmaCommand | 99967-100709 | src/device/shared/hidden.rs:542:74  scalar.to_dm(&mut ctx.tdma) |
| 203,117 | 208,177 | 5,060 | StoTrf | 532 | Sub RenegadeCommand#O2 | 98735-100798 | src/device/shared/mlp.rs:493:96  ctx .sub .begin(scale_f32.view()) .fe |
| 208,179 | 208,866 | 687 | StoVrf | 537 | Sub RenegadeCommand#O1 | 100798-101065 | src/device/shared/hidden.rs:477:5  ctx.sub .begin(all.view()) .fetch:: |
| 208,212 | 211,197 | 2,985 | TuExec | 542 | Main RenegadeCommand#O | 100798-101982 | src/device/shared/mlp.rs:502:74  ctx .main .begin(partials.view()) .fe |
| 208,868 | 209,622 | 754 | TuExec | 548 | Sub RenegadeCommand#O2 | 101144-101437 | src/device/shared/hidden.rs:377:5  ctx.sub .begin(x.view()) .fetch::<m |
| 209,624 | 210,378 | 754 | TuExec | 554 | Sub RenegadeCommand#O2 | 101516-101809 | src/device/shared/hidden.rs:377:5  ctx.sub .begin(x.view()) .fetch::<m |
| 210,380 | 210,699 | 319 | TuExec | 560 | Sub RenegadeCommand#O2 | 101888-102153 | src/device/shared/hidden.rs:543:5  ctx.sub .begin(scalar.view()) .fetc |
| 211,199 | 213,364 | 2,165 | DMA | 567 | DmaCommand | 102153-102779 | src/device/shared/mlp.rs:565:90  rows.to_dm(&mut ctx.tdma) |
| 213,366 | 217,141 | 3,775 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 2 |
| 217,446 | 218,001 | 555 | TuExec | 577 | Sub RenegadeCommand#O2 | 104464-104847 | src/device/shared/mlp.rs:568:87  ctx .sub .begin(theirs) .fetch::<m![H |
| 218,002 | 219,325 | 1,323 | TuExec | 583 | Main RenegadeCommand#O | 104847-105249 | src/device/shared/mlp.rs:575:5  ctx.main .begin(mine) .fetch::<m![H %  |
| 219,326 | 219,885 | 559 | TuExec | 588 | Main RenegadeCommand#O | 105249-105560 | src/device/shared/hidden.rs:389:68  ctx .main .begin(x.view()) .fetch: |
| 219,886 | 220,588 | 702 | TuExec | 593 | Main RenegadeCommand#O | 105560-105873 | src/device/shared/hidden.rs:411:69  ctx .main .begin(partial.view()) . |
| 220,589 | 221,029 | 440 | TuExec | 599 | Main RenegadeCommand#O | 105873-106154 | src/device/shared/hidden.rs:431:5  ctx.main .begin(ms.view()) .fetch:: |
| 221,030 | 221,547 | 517 | TuExec | 604 | Main RenegadeCommand#O | 106154-106465 | src/device/shared/hidden.rs:454:5  ctx.main .begin(x.view()) .fetch::< |
| 221,548 | 222,017 | 469 | TuExec | 609 | Main RenegadeCommand#O | 106465-106761 | src/device/shared/hidden.rs:501:5  ctx.main .begin(x.view()) .fetch::< |
| 222,018 | 222,445 | 427 | TuExec | 614 | Main RenegadeCommand#O | 106761-107072 | src/device/shared/hidden.rs:520:5  ctx.main .begin(x.view()) .fetch::< |
| 222,446 | 223,308 | 862 | DMA | 617 | DmaCommand | 107072-107854 | src/device/shared/hidden.rs:490:5  x.view().to_hbm_view(&mut ctx.tdma, |
| 223,310 | 224,271 | 961 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 2 |

### decoder_feedforward — cluster 1: Task 222,361 cycles
DMA busy union 193,638 (87.1%), TU busy union 159,614 (71.8%), spans 102

| begin | end | dur | kind | emit# | op | static lifetime | source |
|---:|---:|---:|---|---:|---|---|---|
| 97 | 1,267 | 1,170 | DMA | 8 | DmaCommand | 1003-1693 | src/device/shared/hidden.rs:95:65  x.to_dm(&mut ctx.tdma) |
| 2,407 | 212,385 | 209,978 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 7 |
| 5,340 | 5,894 | 554 | TuExec | 87 | Main RenegadeCommand#O | 1693-2004 | src/device/shared/hidden.rs:31:68  ctx .main .begin(x.view()) .fetch:: |
| 5,346 | 6,499 | 1,153 | DMA | 88 | DmaCommand | 1693-2383 | src/device/shared/hidden.rs:96:70  rms_weight.to_dm(&mut ctx.tdma) |
| 5,895 | 11,008 | 5,113 | TuExec | 93 | Main RenegadeCommand#O | 2004-4583 | src/device/shared/hidden.rs:49:78  ctx .main .begin(partial.view()) .f |
| 6,501 | 9,439 | 2,938 | TuExec | 96 | Sub RenegadeCommand#O6 | 2383-2676 | src/device/shared/hidden.rs:99:74  ctx .sub .begin(weight.view()) .fet |
| 8,737 | 10,205 | 1,468 | DMA | 97 | DmaCommand | 2383-3102 |  |
| 9,440 | 12,099 | 2,659 | DMA | 101 | DmaCommand | 3102-3940 |  |
| 11,001 | 58,997 | 47,996 | DMA | 103 | DmaCommand | 3940-28552 | src/device/shared/mlp.rs:44:13  $weight_packed.to_dm(&mut $ctx.tdma) |
| 11,009 | 11,416 | 407 | TuExec | 108 | Main RenegadeCommand#O | 4583-4864 | src/device/shared/hidden.rs:66:5  ctx .main .begin(mean_square.view()) |
| 11,417 | 18,250 | 6,833 | TuExec | 113 | Main RenegadeCommand#O | 4864-5175 | src/device/shared/hidden.rs:107:74  ctx .main .begin(x.view()) .fetch: |
| 12,101 | 18,758 | 6,657 | StoTab | 116 | Sub RenegadeCommand#O8 | 4864-6157 |  |
| 18,253 | 20,358 | 2,105 | TuExec | 124 | Main RenegadeCommand#O | 6239-9069 | src/device/shared/hidden.rs:350:75  ctx .main .begin(chunks.view()) .f |
| 20,359 | 21,632 | 1,273 | TuExec | 131 | Main RenegadeCommand#O | 9069-9813 | src/device/shared/hidden.rs:361:95  ctx .main .begin(gathered.view())  |
| 22,031 | 23,354 | 1,323 | TuExec | 139 | Main RenegadeCommand#O | 10013-10774 | src/device/shared/mlp.rs:152:9  ctx.main .begin(x_half) .fetch::<m![H  |
| 24,055 | 24,846 | 791 | TuExec | 154 | Sub RenegadeCommand#O3 | 11053-11556 | src/device/shared/mlp.rs:182:13  ctx .sub .begin(hi_half_raw) .fetch:: |
| 24,847 | 27,606 | 2,759 | TuExec | 158 | Main RenegadeCommand#O | 11556-12317 | src/device/shared/mlp.rs:192:9  ctx.main .begin(x_half) .fetch::<m![H  |
| 28,005 | 29,328 | 1,323 | TuExec | 167 | Main RenegadeCommand#O | 12517-13278 | src/device/shared/mlp.rs:152:9  ctx.main .begin(x_half) .fetch::<m![H  |
| 30,029 | 30,820 | 791 | TuExec | 179 | Sub RenegadeCommand#O5 | 13557-14060 | src/device/shared/mlp.rs:182:13  ctx .sub .begin(hi_half_raw) .fetch:: |
| 30,821 | 33,580 | 2,759 | TuExec | 183 | Main RenegadeCommand#O | 14060-14821 | src/device/shared/mlp.rs:192:9  ctx.main .begin(x_half) .fetch::<m![H  |
| 33,581 | 35,832 | 2,251 | StoTrf | 191 | Sub RenegadeCommand#O7 | 14821-16044 | src/device/shared/mlp.rs:220:5  ctx.sub .begin(lanes) .fetch::<m![Dumm |
| 59,003 | 82,069 | 23,066 | TuExec | 200 | Main RenegadeCommand#O | 28558-37821 | src/device/shared/mlp.rs:57:99  $ctx .main .begin($chunk.0.view()) .fe |
| 59,010 | 69,469 | 10,459 | DMA | 201 | DmaCommand | 28558-33438 | src/device/shared/mlp.rs:46:13  $weight_scale.to_dm(&mut $ctx.tdma) |
| 59,016 | 70,517 | 11,501 | DMA | 211 | DmaCommand | 33438-34098 | src/device/shared/mlp.rs:247:84  global_scale.to_dm(&mut ctx.tdma) |
| 69,771 | 70,766 | 995 | TuExec | 216 | Sub RenegadeCommand#O8 | 33517-33960 | src/device/shared/mlp.rs:91:107  $ctx.sub .begin(s_tile) .fetch::<m![L |
| 70,768 | 71,112 | 344 | StoVrf | 225 | Sub RenegadeCommand#O2 | 34098-34365 | src/device/shared/mlp.rs:248:89  ctx .sub .begin(global_scale.view())  |
| 70,799 | 73,615 | 2,816 | DMA | 226 | DmaCommand | 34098-34936 |  |
| 73,617 | 82,067 | 8,450 | StoTab | 231 | Sub RenegadeCommand#O1 | 34936-36229 |  |
| 73,675 | 121,235 | 47,560 | DMA | 233 | DmaCommand | 34942-59554 | src/device/shared/mlp.rs:44:13  $weight_packed.to_dm(&mut $ctx.tdma) |
| 82,467 | 84,628 | 2,161 | TuExec | 248 | Main RenegadeCommand#O | 38021-38661 | src/device/shared/mlp.rs:100:17  $ctx.main.begin(p_tile) .fetch::<m![L |
| 84,629 | 85,599 | 970 | TuExec | 256 | Sub RenegadeCommand#O1 | 38661-39104 | src/device/shared/mlp.rs:91:107  $ctx.sub .begin(s_tile) .fetch::<m![L |
| 85,600 | 87,763 | 2,163 | TuExec | 264 | Main RenegadeCommand#O | 39183-39823 | src/device/shared/mlp.rs:100:17  $ctx.main.begin(p_tile) .fetch::<m![L |
| 87,764 | 88,735 | 971 | TuExec | 268 | Sub RenegadeCommand#O1 | 39823-40266 | src/device/shared/mlp.rs:91:107  $ctx.sub .begin(s_tile) .fetch::<m![L |
| 88,736 | 90,899 | 2,163 | TuExec | 276 | Main RenegadeCommand#O | 40345-40985 | src/device/shared/mlp.rs:100:17  $ctx.main.begin(p_tile) .fetch::<m![L |
| 90,900 | 91,871 | 971 | TuExec | 280 | Sub RenegadeCommand#O1 | 40985-41428 | src/device/shared/mlp.rs:91:107  $ctx.sub .begin(s_tile) .fetch::<m![L |
| 91,872 | 94,035 | 2,163 | TuExec | 288 | Main RenegadeCommand#O | 41507-42147 | src/device/shared/mlp.rs:100:17  $ctx.main.begin(p_tile) .fetch::<m![L |
| 94,036 | 94,706 | 670 | TuExec | 292 | Sub RenegadeCommand#O1 | 42147-42590 | src/device/shared/mlp.rs:91:107  $ctx.sub .begin(s_tile) .fetch::<m![L |
| 94,707 | 96,868 | 2,161 | TuExec | 299 | Main RenegadeCommand#O | 42590-43230 | src/device/shared/mlp.rs:100:17  $ctx.main.begin(p_tile) .fetch::<m![L |
| 96,869 | 97,380 | 511 | TuExec | 305 | Main RenegadeCommand#O | 43230-43585 | src/device/shared/mlp.rs:119:9  $ctx.main.begin(tile_sums.view()) .fet |
| 97,381 | 97,976 | 595 | TuExec | 310 | Main RenegadeCommand#O | 43585-43969 | src/device/shared/mlp.rs:232:5  ctx.main .begin(x.view()) .fetch::<m![ |
| 97,977 | 98,460 | 483 | TuExec | 315 | Main RenegadeCommand#O | 43969-44310 | src/device/shared/mlp.rs:255:5  ctx.main .begin(x.view()) .fetch::<m![ |
| 121,237 | 144,969 | 23,732 | TuExec | 321 | Main RenegadeCommand#O | 59554-68817 | src/device/shared/mlp.rs:57:99  $ctx .main .begin($chunk.0.view()) .fe |
| 121,244 | 131,775 | 10,531 | DMA | 322 | DmaCommand | 59554-64434 | src/device/shared/mlp.rs:46:13  $weight_scale.to_dm(&mut $ctx.tdma) |
| 121,250 | 133,183 | 11,933 | DMA | 328 | DmaCommand | 64434-65094 | src/device/shared/mlp.rs:247:84  global_scale.to_dm(&mut ctx.tdma) |
| 132,079 | 133,038 | 959 | TuExec | 333 | Sub RenegadeCommand#O1 | 64513-64956 | src/device/shared/mlp.rs:91:107  $ctx.sub .begin(s_tile) .fetch::<m![L |
| 133,185 | 133,542 | 357 | StoVrf | 342 | Sub RenegadeCommand#O2 | 65094-65361 | src/device/shared/mlp.rs:248:89  ctx .sub .begin(global_scale.view())  |
| 133,217 | 135,173 | 1,956 | DMA | 343 | DmaCommand | 65094-65813 |  |
| 133,543 | 137,087 | 3,544 | DMA | 346 | DmaCommand | 65813-66651 |  |
| 137,091 | 144,967 | 7,876 | StoTab | 350 | Sub RenegadeCommand#O2 | 66654-67947 |  |
| 137,146 | 186,359 | 49,213 | DMA | 351 | DmaCommand | 66654-91507 | src/device/shared/mlp.rs:545:67  down_weight_packed.to_dm(&mut ctx.tdm |
| 145,365 | 147,526 | 2,161 | TuExec | 366 | Main RenegadeCommand#O | 69017-69657 | src/device/shared/mlp.rs:100:17  $ctx.main.begin(p_tile) .fetch::<m![L |
| 147,527 | 148,497 | 970 | TuExec | 374 | Sub RenegadeCommand#O1 | 69657-70100 | src/device/shared/mlp.rs:91:107  $ctx.sub .begin(s_tile) .fetch::<m![L |
| 148,498 | 150,661 | 2,163 | TuExec | 382 | Main RenegadeCommand#O | 70179-70819 | src/device/shared/mlp.rs:100:17  $ctx.main.begin(p_tile) .fetch::<m![L |
| 150,662 | 151,633 | 971 | TuExec | 386 | Sub RenegadeCommand#O1 | 70819-71262 | src/device/shared/mlp.rs:91:107  $ctx.sub .begin(s_tile) .fetch::<m![L |
| 151,634 | 153,797 | 2,163 | TuExec | 394 | Main RenegadeCommand#O | 71341-71981 | src/device/shared/mlp.rs:100:17  $ctx.main.begin(p_tile) .fetch::<m![L |
| 153,798 | 154,769 | 971 | TuExec | 398 | Sub RenegadeCommand#O1 | 71981-72424 | src/device/shared/mlp.rs:91:107  $ctx.sub .begin(s_tile) .fetch::<m![L |
| 154,770 | 156,933 | 2,163 | TuExec | 406 | Main RenegadeCommand#O | 72503-73143 | src/device/shared/mlp.rs:100:17  $ctx.main.begin(p_tile) .fetch::<m![L |
| 156,934 | 157,604 | 670 | TuExec | 410 | Sub RenegadeCommand#O2 | 73143-73586 | src/device/shared/mlp.rs:91:107  $ctx.sub .begin(s_tile) .fetch::<m![L |
| 157,605 | 159,766 | 2,161 | TuExec | 417 | Main RenegadeCommand#O | 73586-74226 | src/device/shared/mlp.rs:100:17  $ctx.main.begin(p_tile) .fetch::<m![L |
| 159,767 | 160,278 | 511 | TuExec | 423 | Main RenegadeCommand#O | 74226-74581 | src/device/shared/mlp.rs:119:9  $ctx.main.begin(tile_sums.view()) .fet |
| 160,279 | 160,874 | 595 | TuExec | 428 | Main RenegadeCommand#O | 74581-74965 | src/device/shared/mlp.rs:232:5  ctx.main .begin(x.view()) .fetch::<m![ |
| 160,875 | 161,592 | 717 | TuExec | 433 | Main RenegadeCommand#O | 74965-75306 | src/device/shared/mlp.rs:255:5  ctx.main .begin(x.view()) .fetch::<m![ |
| 161,593 | 162,226 | 633 | TuExec | 438 | Sub RenegadeCommand#O2 | 75306-75709 | src/device/shared/mlp.rs:282:78  ctx .sub .begin(gate.view()) .fetch:: |
| 162,228 | 162,800 | 572 | StoVrf | 442 | Sub RenegadeCommand#O2 | 75709-76092 | src/device/shared/mlp.rs:301:83  ctx .sub .begin(gelu.view()) .fetch:: |
| 162,801 | 163,458 | 657 | TuExec | 447 | Main RenegadeCommand#O | 76092-76433 | src/device/shared/mlp.rs:308:5  ctx.main .begin(up.view()) .fetch::<m! |
| 163,459 | 164,186 | 727 | TuExec | 453 | Main RenegadeCommand#O | 76433-76774 | src/device/shared/mlp.rs:339:81  ctx .main .begin(x.view()) .fetch::<m |
| 164,187 | 166,946 | 2,759 | TuExec | 459 | Main RenegadeCommand#O | 76774-77805 | src/device/shared/mlp.rs:355:88  ctx .main .begin(slice_max.view()) .f |
| 166,947 | 167,372 | 425 | TuExec | 464 | Main RenegadeCommand#O | 77805-78118 | src/device/shared/mlp.rs:363:75  ctx .main .begin(maxima.view()) .fetc |
| 167,373 | 167,754 | 381 | TuExec | 469 | Main RenegadeCommand#O | 78118-78400 | src/device/shared/mlp.rs:376:75  ctx .main .begin(unscale.view()) .fet |
| 167,756 | 168,192 | 436 | StoVrf | 475 | Sub RenegadeCommand#O2 | 78400-78667 | src/device/shared/mlp.rs:392:84  ctx .sub .begin(unscale_live) .fetch: |
| 168,193 | 168,683 | 490 | TuExec | 480 | Main RenegadeCommand#O | 78667-79008 | src/device/shared/mlp.rs:401:5  ctx.main .begin(x.view()) .fetch::<m![ |
| 168,199 | 168,542 | 343 | StoVrf | 482 | Sub RenegadeCommand#O2 | 78667-78934 | src/device/shared/mlp.rs:456:78  ctx .sub .begin(unscale_rows) .fetch: |
| 169,389 | 169,760 | 371 | TuExec | 493 | Sub RenegadeCommand#O2 | 79287-79580 | src/device/shared/mlp.rs:415:81  ctx .sub .begin(lanes.view().tile::<m |
| 169,761 | 170,458 | 697 | TuExec | 497 | Main RenegadeCommand#O | 79580-79921 | src/device/shared/mlp.rs:422:5  ctx.main .begin(x.view()) .fetch::<m![ |
| 170,459 | 179,508 | 9,049 | TuExec | 504 | Main RenegadeCommand#O | 79921-84536 | src/device/shared/mlp.rs:439:118  ctx .main .begin(lanes.view()) .fetc |
| 179,509 | 183,656 | 4,147 | TuExec | 510 | Sub RenegadeCommand#O2 | 84536-86719 | src/device/shared/mlp.rs:449:89  ctx .sub .begin(gathered) .fetch::<m! |
| 186,363 | 209,386 | 23,023 | TuExec | 517 | Main RenegadeCommand#O | 91510-100773 | src/device/shared/mlp.rs:467:98  ctx .main .begin(w.view()) .fetch::<m |
| 186,370 | 197,413 | 11,043 | DMA | 518 | DmaCommand | 91510-96471 | src/device/shared/mlp.rs:546:67  down_weight_scale.to_dm(&mut ctx.tdma |
| 197,415 | 204,453 | 7,038 | TuExec | 522 | Sub RenegadeCommand#O2 | 96471-98535 | src/device/shared/mlp.rs:484:68  ctx .sub .begin(s.view()) .fetch::<m! |
| 197,421 | 197,585 | 164 | DMA | 523 | DmaCommand | 96471-97213 | src/device/shared/hidden.rs:476:9  scalar.to_dm(&mut ctx.tdma) |
| 204,429 | 204,591 | 162 | DMA | 526 | DmaCommand | 98735-99351 | src/device/shared/hidden.rs:451:9  rms_weight.to_dm(&mut ctx.tdma) |
| 204,435 | 204,597 | 162 | DMA | 527 | DmaCommand | 99351-99967 | src/device/shared/hidden.rs:498:78  residual.to_dm(&mut ctx.tdma) |
| 204,441 | 204,609 | 168 | DMA | 528 | DmaCommand | 99967-100709 | src/device/shared/hidden.rs:542:74  scalar.to_dm(&mut ctx.tdma) |
| 204,454 | 209,352 | 4,898 | StoTrf | 532 | Sub RenegadeCommand#O2 | 98735-100798 | src/device/shared/mlp.rs:493:96  ctx .sub .begin(scale_f32.view()) .fe |
| 209,354 | 209,991 | 637 | StoVrf | 537 | Sub RenegadeCommand#O1 | 100798-101065 | src/device/shared/hidden.rs:477:5  ctx.sub .begin(all.view()) .fetch:: |
| 209,387 | 212,384 | 2,997 | TuExec | 542 | Main RenegadeCommand#O | 100798-101982 | src/device/shared/mlp.rs:502:74  ctx .main .begin(partials.view()) .fe |
| 209,993 | 210,751 | 758 | TuExec | 548 | Sub RenegadeCommand#O2 | 101144-101437 | src/device/shared/hidden.rs:377:5  ctx.sub .begin(x.view()) .fetch::<m |
| 210,753 | 211,505 | 752 | TuExec | 554 | Sub RenegadeCommand#O2 | 101516-101809 | src/device/shared/hidden.rs:377:5  ctx.sub .begin(x.view()) .fetch::<m |
| 211,507 | 211,866 | 359 | TuExec | 560 | Sub RenegadeCommand#O2 | 101888-102153 | src/device/shared/hidden.rs:543:5  ctx.sub .begin(scalar.view()) .fetc |
| 212,386 | 214,985 | 2,599 | DMA | 567 | DmaCommand | 102153-102779 | src/device/shared/mlp.rs:565:90  rows.to_dm(&mut ctx.tdma) |
| 214,987 | 214,989 | 2 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 2 |
| 215,291 | 215,842 | 551 | TuExec | 577 | Sub RenegadeCommand#O2 | 104464-104847 | src/device/shared/mlp.rs:568:87  ctx .sub .begin(theirs) .fetch::<m![H |
| 215,843 | 217,162 | 1,319 | TuExec | 583 | Main RenegadeCommand#O | 104847-105249 | src/device/shared/mlp.rs:575:5  ctx.main .begin(mine) .fetch::<m![H %  |
| 217,163 | 217,718 | 555 | TuExec | 588 | Main RenegadeCommand#O | 105249-105560 | src/device/shared/hidden.rs:389:68  ctx .main .begin(x.view()) .fetch: |
| 217,719 | 218,423 | 704 | TuExec | 593 | Main RenegadeCommand#O | 105560-105873 | src/device/shared/hidden.rs:411:69  ctx .main .begin(partial.view()) . |
| 218,424 | 218,862 | 438 | TuExec | 599 | Main RenegadeCommand#O | 105873-106154 | src/device/shared/hidden.rs:431:5  ctx.main .begin(ms.view()) .fetch:: |
| 218,863 | 219,376 | 513 | TuExec | 604 | Main RenegadeCommand#O | 106154-106465 | src/device/shared/hidden.rs:454:5  ctx.main .begin(x.view()) .fetch::< |
| 219,377 | 219,842 | 465 | TuExec | 609 | Main RenegadeCommand#O | 106465-106761 | src/device/shared/hidden.rs:501:5  ctx.main .begin(x.view()) .fetch::< |
| 219,843 | 220,266 | 423 | TuExec | 614 | Main RenegadeCommand#O | 106761-107072 | src/device/shared/hidden.rs:520:5  ctx.main .begin(x.view()) .fetch::< |
| 220,267 | 220,431 | 164 | DMA | 617 | DmaCommand | 107072-107854 | src/device/shared/hidden.rs:490:5  x.view().to_hbm_view(&mut ctx.tdma, |
| 220,433 | 222,358 | 1,925 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 2 |
