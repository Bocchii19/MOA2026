### sliding_project_qkv — cluster 0: Task 76,625 cycles
DMA busy union 65,847 (85.9%), TU busy union 57,287 (74.8%), spans 92

| begin | end | dur | kind | emit# | op | static lifetime | source |
|---:|---:|---:|---|---:|---|---|---|
| 97 | 1,469 | 1,372 | DMA | 18 | DmaCommand | 1003-1724 | src/device/sliding/qkv_s.rs:174:34  rms_weight.to_dm(&mut device.tdma) |
| 4,383 | 4,794 | 411 | TuExec | 102 | Sub RenegadeCommand#O4 | 1724-2025 | src/device/sliding/qkv_s.rs:178:107  device .sub .begin(weight_seed.vi |
| 4,389 | 5,757 | 1,368 | DMA | 103 | DmaCommand | 1724-2445 | src/device/sliding/qkv_s.rs:201:29  x.to_dm(&mut device.tdma) |
| 4,795 | 5,138 | 343 | StoVrf | 109 | Sub RenegadeCommand#O6 | 2025-2292 | src/device/sliding/qkv_s.rs:194:73  device .sub .begin(zero.view()) .f |
| 5,139 | 7,384 | 2,245 | TuExec | 113 | Main RenegadeCommand#O | 2292-3067 | src/device/sliding/qkv_s.rs:117:5  device .main .begin(q.view()) .fetc |
| 6,983 | 20,281 | 13,298 | DMA | 115 | DmaCommand | 2445-9411 | src/device/sliding/qkv_s.rs:396:23  k_weight.to_dm(&mut device.tdma) |
| 7,386 | 9,611 | 2,225 | TuExec | 121 | Main RenegadeCommand#O | 3067-3842 | src/device/sliding/qkv_s.rs:144:5  device .main .begin(q.view()) .fetc |
| 8,466 | 9,609 | 1,143 | TuExec | 123 | Sub RenegadeCommand#O1 | 3067-3394 | src/device/sliding/qkv_s.rs:237:77  device .sub .begin(weight_dm.view( |
| 9,613 | 11,303 | 1,690 | TuExec | 130 | Sub RenegadeCommand#O1 | 3842-4321 | src/device/sliding/qkv_s.rs:203:73  device .sub .begin(x.view()) .fetc |
| 11,304 | 11,697 | 393 | TuExec | 135 | Sub RenegadeCommand#O1 | 4521-4809 | src/device/sliding/qkv_s.rs:222:65  device .sub .begin(mean_square.vie |
| 11,698 | 12,136 | 438 | StoVrf | 140 | Sub RenegadeCommand#O1 | 4809-5076 | src/device/sliding/qkv_s.rs:244:72  device .sub .begin(rms.view()) .fe |
| 12,137 | 13,670 | 1,533 | TuExec | 145 | Main RenegadeCommand#O | 5076-5421 | src/device/sliding/qkv_s.rs:250:5  device .main .begin(x.view()) .fetc |
| 13,671 | 14,164 | 493 | TuExec | 151 | Main RenegadeCommand#O | 5421-5766 | src/device/sliding/qkv_s.rs:271:5  device .main .begin(x.view()) .fetc |
| 14,165 | 14,658 | 493 | TuExec | 156 | Main RenegadeCommand#O | 5766-6111 | src/device/sliding/qkv_s.rs:286:24  device .main .begin(x.view()) .fet |
| 14,659 | 15,070 | 411 | TuExec | 162 | Sub RenegadeCommand#O2 | 6111-6406 | src/device/sliding/qkv_s.rs:301:77  device .sub .begin(hi_neg.view())  |
| 15,071 | 17,209 | 2,138 | TuExec | 167 | Main RenegadeCommand#O | 6406-6751 | src/device/sliding/qkv_s.rs:308:5  device .main .begin(x.view()) .fetc |
| 17,210 | 17,670 | 460 | StoTrf | 173 | Sub RenegadeCommand#O2 | 6751-7078 | src/device/sliding/qkv_s.rs:43:5  device .sub .begin(x.view()) .fetch: |
| 17,241 | 21,103 | 3,862 | DMA | 175 | DmaCommand | 9411-9963 | src/device/sliding/rope_h4.rs:125:77  rope_offset.to_dm(&mut device.td |
| 21,105 | 21,523 | 418 | TuExec | 180 | Main RenegadeCommand#O | 9963-10245 | src/device/sliding/rope_h4.rs:126:35  device .main .begin(off.view())  |
| 21,111 | 22,314 | 1,203 | DMA | 181 | DmaCommand | 9963-10518 | src/device/sliding/qkv_s.rs:383:9  weight_scale.to_dm(&mut device.tdma |
| 21,524 | 24,084 | 2,560 | TuExec | 187 | Main RenegadeCommand#O | 10245-11174 | src/device/sliding/qkv_s.rs:341:26  device .main .begin(weight.view()) |
| 21,531 | 22,211 | 680 | StoVrf | 189 | Sub RenegadeCommand#O5 | 10245-10512 | src/device/sliding/rope_h4.rs:140:82  device .sub .begin(pos.view()) . |
| 21,715 | 23,430 | 1,715 | DMA | 192 | DmaCommand | 10518-11073 | src/device/sliding/qkv_h4_norm.rs:153:64  weight.to_dm(&mut device.tdm |
| 21,723 | 49,559 | 27,836 | DMA | 194 | DmaCommand | 11073-24456 | src/device/sliding/qkv_s.rs:39:5  weight.to_dm(&mut device.tdma) |
| 24,085 | 25,254 | 1,169 | TuExec | 202 | Main RenegadeCommand#O | 11174-11693 | src/device/sliding/qkv_s.rs:374:100  device .main .begin(r.view()) .fe |
| 25,255 | 26,974 | 1,719 | TuExec | 208 | Main RenegadeCommand#O | 11693-12038 | src/device/sliding/rope_h4.rs:102:37  device .main .begin(after.view() |
| 25,261 | 26,972 | 1,711 | TuExec | 210 | Sub RenegadeCommand#O1 | 11693-12020 | src/device/sliding/qkv_h4_norm.rs:101:63  device .sub .begin(x.view()) |
| 26,975 | 27,318 | 343 | StoVrf | 217 | Sub RenegadeCommand#O5 | 12038-12305 | src/device/sliding/rope_h4.rs:118:82  device .sub .begin(one_k.view()) |
| 27,319 | 27,692 | 373 | TuExec | 221 | Main RenegadeCommand#O | 12305-12587 | src/device/sliding/rope_h4.rs:147:35  device .main .begin(pos.view())  |
| 27,693 | 28,010 | 317 | TuExec | 226 | Main RenegadeCommand#O | 12587-12852 | src/device/sliding/rope_h4.rs:55:13  device.main .begin(v.view()) .fet |
| 28,885 | 29,250 | 365 | TuExec | 233 | Main RenegadeCommand#O | 13052-13334 | src/device/sliding/rope_h4.rs:61:13  device.main .begin(v.view()) .fet |
| 29,251 | 29,570 | 319 | TuExec | 240 | Main RenegadeCommand#O | 13334-13600 | src/device/sliding/rope_h4.rs:55:13  device.main .begin(v.view()) .fet |
| 30,073 | 30,440 | 367 | TuExec | 247 | Main RenegadeCommand#O | 13800-14083 | src/device/sliding/rope_h4.rs:61:13  device.main .begin(v.view()) .fet |
| 30,441 | 32,132 | 1,691 | TuExec | 254 | Main RenegadeCommand#O | 14083-14351 | src/device/sliding/rope_h4.rs:55:13  device.main .begin(v.view()) .fet |
| 32,134 | 32,806 | 672 | TuExec | 260 | Main RenegadeCommand#O | 14351-14695 | src/device/sliding/qkv_h4_norm.rs:140:60  device .main .begin(scale.vi |
| 32,807 | 33,578 | 771 | TuExec | 267 | Main RenegadeCommand#O | 14695-15040 | src/device/sliding/qkv_h4_norm.rs:108:71  device .main .begin(scale.vi |
| 33,579 | 34,278 | 699 | TuExec | 272 | Main RenegadeCommand#O | 15040-15325 | src/device/sliding/rope_h4.rs:61:13  device.main .begin(v.view()) .fet |
| 34,279 | 34,648 | 369 | TuExec | 277 | Main RenegadeCommand#O | 15325-15606 | src/device/sliding/qkv_h4_norm.rs:127:64  device .main .begin(mean_squ |
| 34,649 | 34,981 | 332 | TuExec | 284 | Main RenegadeCommand#O | 15606-15878 | src/device/sliding/rope_h4.rs:55:13  device.main .begin(v.view()) .fet |
| 35,377 | 35,756 | 379 | TuExec | 291 | Main RenegadeCommand#O | 16078-16367 | src/device/sliding/rope_h4.rs:61:13  device.main .begin(v.view()) .fet |
| 35,757 | 36,104 | 347 | TuExec | 298 | Main RenegadeCommand#O | 16367-16647 | src/device/sliding/rope_h4.rs:55:13  device.main .begin(v.view()) .fet |
| 36,503 | 36,898 | 395 | TuExec | 305 | Main RenegadeCommand#O | 16847-17144 | src/device/sliding/rope_h4.rs:61:13  device.main .begin(v.view()) .fet |
| 36,899 | 37,278 | 379 | TuExec | 312 | Main RenegadeCommand#O | 17144-17440 | src/device/sliding/rope_h4.rs:55:13  device.main .begin(v.view()) .fet |
| 37,673 | 38,100 | 427 | TuExec | 319 | Main RenegadeCommand#O | 17640-17953 | src/device/sliding/rope_h4.rs:61:13  device.main .begin(v.view()) .fet |
| 38,102 | 38,788 | 686 | TuExec | 325 | Main RenegadeCommand#O | 17953-18298 | src/device/sliding/qkv_h4_norm.rs:154:5  device.main .begin(weight.vie |
| 38,789 | 39,232 | 443 | TuExec | 332 | Main RenegadeCommand#O | 18298-18626 | src/device/sliding/rope_h4.rs:55:13  device.main .begin(v.view()) .fet |
| 39,631 | 40,122 | 491 | TuExec | 341 | Main RenegadeCommand#O | 18826-19171 | src/device/sliding/rope_h4.rs:61:13  device.main .begin(v.view()) .fet |
| 40,123 | 40,694 | 571 | TuExec | 348 | Main RenegadeCommand#O | 19171-19563 | src/device/sliding/rope_h4.rs:55:13  device.main .begin(v.view()) .fet |
| 41,127 | 41,746 | 619 | TuExec | 355 | Main RenegadeCommand#O | 19763-20172 | src/device/sliding/rope_h4.rs:61:13  device.main .begin(v.view()) .fet |
| 41,747 | 42,628 | 881 | TuExec | 362 | Main RenegadeCommand#O | 20172-20710 | src/device/sliding/rope_h4.rs:170:30  device .main .begin(v.view()) .f |
| 42,629 | 43,120 | 491 | TuExec | 367 | Main RenegadeCommand#O | 20710-21055 | src/device/sliding/rope_h4.rs:190:34  device .main .begin(f.view()) .f |
| 43,121 | 43,612 | 491 | TuExec | 372 | Main RenegadeCommand#O | 21055-21400 | src/device/sliding/rope_h4.rs:209:76  device .main .begin(angle.view() |
| 43,127 | 43,744 | 617 | StoVrf | 374 | Sub RenegadeCommand#O1 | 21055-21446 | src/device/sliding/rope_h4.rs:203:81  device .sub .begin(angle.view()) |
| 43,745 | 44,440 | 695 | TuExec | 385 | Main RenegadeCommand#O | 21446-21791 | src/device/sliding/rope_h4.rs:225:5  device.main .begin(turns.view())  |
| 43,751 | 44,226 | 475 | StoVrf | 387 | Sub RenegadeCommand#O1 | 21446-21773 | src/device/sliding/rope_h4.rs:247:96  device .sub .begin(angle.view(). |
| 44,441 | 44,984 | 543 | TuExec | 394 | Main RenegadeCommand#O | 21791-22104 | src/device/sliding/rope_h4.rs:253:13  device.main .begin(turns.view(). |
| 44,448 | 45,386 | 938 | StoVrf | 396 | Sub RenegadeCommand#O1 | 21791-22118 | src/device/sliding/rope_h4.rs:247:96  device .sub .begin(angle.view(). |
| 45,387 | 45,928 | 541 | TuExec | 407 | Main RenegadeCommand#O | 22304-22617 | src/device/sliding/rope_h4.rs:253:13  device.main .begin(turns.view(). |
| 45,929 | 46,340 | 411 | TuExec | 413 | Main RenegadeCommand#O | 22617-22897 | src/device/sliding/rope_h4.rs:274:5  device.main .begin(sin_full.view( |
| 46,647 | 47,329 | 682 | TuExec | 422 | Sub RenegadeCommand#O2 | 22976-23271 | src/device/sliding/rope_h4.rs:303:84  device .sub .begin(sin.tile::<m! |
| 47,330 | 48,029 | 699 | TuExec | 429 | Main RenegadeCommand#O | 23350-23663 | src/device/sliding/rope_h4.rs:371:5  device.main .begin(k.view().tile: |
| 47,337 | 48,027 | 690 | TuExec | 431 | Sub RenegadeCommand#O2 | 23350-23645 | src/device/sliding/rope_h4.rs:310:84  device .sub .begin(sin.tile::<m! |
| 48,427 | 48,938 | 511 | TuExec | 441 | Main RenegadeCommand#O | 23924-24237 | src/device/sliding/rope_h4.rs:384:5  device.main .begin(k.view().tile: |
| 49,561 | 55,824 | 6,263 | TuExec | 448 | Main RenegadeCommand#O | 24456-25897 | src/device/sliding/qkv_s.rs:59:5  device .main .begin(weight.view()) . |
| 53,110 | 54,311 | 1,201 | DMA | 449 | DmaCommand | 24456-25017 | src/device/sliding/qkv_s.rs:424:9  q_weight_scale.to_dm(&mut device.td |
| 54,313 | 55,235 | 922 | TuExec | 454 | Sub RenegadeCommand#O1 | 25017-25408 | src/device/sliding/qkv_h4_norm.rs:18:67  device .sub .begin(scale.view |
| 54,319 | 55,569 | 1,250 | DMA | 455 | DmaCommand | 25017-25578 | src/device/sliding/qkv_h4_norm.rs:70:68  weight.to_dm(&mut device.tdma |
| 54,325 | 68,051 | 13,726 | DMA | 456 | DmaCommand | 25578-32544 | src/device/sliding/qkv_s.rs:397:23  v_weight.to_dm(&mut device.tdma) |
| 55,825 | 57,507 | 1,682 | TuExec | 463 | Main RenegadeCommand#O | 25897-26672 | src/device/sliding/qkv_s.rs:89:5  device .main .begin(r.view()) .fetch |
| 57,508 | 58,666 | 1,158 | TuExec | 469 | Main RenegadeCommand#O | 26672-27081 | src/device/sliding/qkv_h4_norm.rs:25:75  device .main .begin(x.view()) |
| 57,514 | 58,092 | 578 | TuExec | 471 | Sub RenegadeCommand#O2 | 26672-27063 | src/device/sliding/qkv_h4_norm.rs:63:67  device .sub .begin(x.view())  |
| 58,667 | 59,039 | 372 | TuExec | 478 | Main RenegadeCommand#O | 27081-27364 | src/device/sliding/qkv_h4_norm.rs:44:67  device .main .begin(mean_squa |
| 59,040 | 59,476 | 436 | StoVrf | 482 | Sub RenegadeCommand#O1 | 27364-27635 | src/device/sliding/qkv_h4_norm.rs:57:68  device .sub .begin(rms.view() |
| 59,478 | 60,490 | 1,012 | TuExec | 488 | Main RenegadeCommand#O | 27635-28044 | src/device/sliding/qkv_h4_norm.rs:71:5  device.main .begin(weight.view |
| 59,484 | 60,399 | 915 | StoVrf | 490 | Sub RenegadeCommand#O2 | 27635-28026 | src/device/sliding/rope_h4.rs:397:81  device .sub .begin(k_sin.view()) |
| 60,491 | 60,930 | 439 | TuExec | 497 | Sub RenegadeCommand#O2 | 28105-28432 | src/device/sliding/rope_h4.rs:295:79  device .sub .begin(rows.view().t |
| 60,897 | 61,472 | 575 | TuExec | 501 | Main RenegadeCommand#O | 28305-28650 | src/device/sliding/rope_h4.rs:320:5  device.main .begin(q.view().tile: |
| 61,873 | 62,448 | 575 | TuExec | 511 | Main RenegadeCommand#O | 28850-29195 | src/device/sliding/rope_h4.rs:334:5  device.main .begin(q.view().tile: |
| 62,449 | 63,134 | 685 | TuExec | 516 | Main RenegadeCommand#O | 29195-29540 | src/device/sliding/rope_h4.rs:403:80  device .main .begin(k.view()) .f |
| 62,852 | 63,698 | 846 | StoVrf | 520 | Sub RenegadeCommand#O2 | 29395-29914 | src/device/sliding/rope_h4.rs:347:85  device .sub .begin(q_sin.view()) |
| 63,699 | 64,692 | 993 | TuExec | 527 | Main RenegadeCommand#O | 29914-30323 | src/device/sliding/rope_h4.rs:353:84  device .main .begin(q.view()) .f |
| 68,053 | 70,556 | 2,503 | TuExec | 535 | Main RenegadeCommand#O | 32544-33473 | src/device/sliding/qkv_s.rs:341:26  device .main .begin(weight.view()) |
| 68,060 | 69,286 | 1,226 | DMA | 536 | DmaCommand | 32544-33099 | src/device/sliding/qkv_s.rs:383:9  weight_scale.to_dm(&mut device.tdma |
| 69,288 | 69,924 | 636 | TuExec | 541 | Sub RenegadeCommand#O4 | 33099-33426 | src/device/sliding/qkv_h4_norm.rs:243:63  device .sub .begin(scale.vie |
| 69,294 | 70,385 | 1,091 | DMA | 542 | DmaCommand | 33099-33550 | src/ops.rs:78:5  q.view().to_hbm_view(&mut device.tdma, q_out.view_mut |
| 70,558 | 71,728 | 1,170 | TuExec | 549 | Main RenegadeCommand#O | 33473-33992 | src/device/sliding/qkv_s.rs:374:100  device .main .begin(r.view()) .fe |
| 70,564 | 72,209 | 1,645 | DMA | 550 | DmaCommandScatter | 33550-34479 | src/ops.rs:79:5  k.dma_scatter::<m![1], _, _>(kv_offset, k_cache) |
| 71,729 | 72,894 | 1,165 | TuExec | 558 | Main RenegadeCommand#O | 33992-34336 | src/device/sliding/qkv_h4_norm.rs:250:68  device .main .begin(x.view() |
| 72,895 | 73,650 | 755 | TuExec | 563 | Main RenegadeCommand#O | 34336-34681 | src/device/sliding/qkv_h4_norm.rs:268:5  device.main .begin(x.view())  |
| 73,651 | 75,295 | 1,644 | DMA | 567 | DmaCommandScatter | 34681-35610 | src/ops.rs:80:5  v.dma_scatter::<m![1], _, _>(kv_offset, v_cache) |
| 75,297 | 76,622 | 1,325 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 2 |

### sliding_project_qkv — cluster 1: Task 75,412 cycles
DMA busy union 64,634 (85.7%), TU busy union 57,267 (75.9%), spans 92

| begin | end | dur | kind | emit# | op | static lifetime | source |
|---:|---:|---:|---|---:|---|---|---|
| 100 | 1,464 | 1,364 | DMA | 18 | DmaCommand | 1003-1724 | src/device/sliding/qkv_s.rs:174:34  rms_weight.to_dm(&mut device.tdma) |
| 4,465 | 4,881 | 416 | TuExec | 102 | Sub RenegadeCommand#O4 | 1724-2025 | src/device/sliding/qkv_s.rs:178:107  device .sub .begin(weight_seed.vi |
| 4,471 | 5,834 | 1,363 | DMA | 103 | DmaCommand | 1724-2445 | src/device/sliding/qkv_s.rs:201:29  x.to_dm(&mut device.tdma) |
| 4,882 | 5,225 | 343 | StoVrf | 109 | Sub RenegadeCommand#O6 | 2025-2292 | src/device/sliding/qkv_s.rs:194:73  device .sub .begin(zero.view()) .f |
| 5,226 | 7,667 | 2,441 | TuExec | 113 | Main RenegadeCommand#O | 2292-3067 | src/device/sliding/qkv_s.rs:117:5  device .main .begin(q.view()) .fetc |
| 7,262 | 20,490 | 13,228 | DMA | 115 | DmaCommand | 2445-9411 | src/device/sliding/qkv_s.rs:396:23  k_weight.to_dm(&mut device.tdma) |
| 7,669 | 9,954 | 2,285 | TuExec | 121 | Main RenegadeCommand#O | 3067-3842 | src/device/sliding/qkv_s.rs:144:5  device .main .begin(q.view()) .fetc |
| 8,807 | 9,952 | 1,145 | TuExec | 123 | Sub RenegadeCommand#O1 | 3067-3394 | src/device/sliding/qkv_s.rs:237:77  device .sub .begin(weight_dm.view( |
| 9,956 | 11,642 | 1,686 | TuExec | 130 | Sub RenegadeCommand#O1 | 3842-4321 | src/device/sliding/qkv_s.rs:203:73  device .sub .begin(x.view()) .fetc |
| 11,643 | 12,034 | 391 | TuExec | 135 | Sub RenegadeCommand#O1 | 4521-4809 | src/device/sliding/qkv_s.rs:222:65  device .sub .begin(mean_square.vie |
| 12,035 | 12,473 | 438 | StoVrf | 140 | Sub RenegadeCommand#O1 | 4809-5076 | src/device/sliding/qkv_s.rs:244:72  device .sub .begin(rms.view()) .fe |
| 12,474 | 14,169 | 1,695 | TuExec | 145 | Main RenegadeCommand#O | 5076-5421 | src/device/sliding/qkv_s.rs:250:5  device .main .begin(x.view()) .fetc |
| 14,170 | 14,663 | 493 | TuExec | 151 | Main RenegadeCommand#O | 5421-5766 | src/device/sliding/qkv_s.rs:271:5  device .main .begin(x.view()) .fetc |
| 14,664 | 15,157 | 493 | TuExec | 156 | Main RenegadeCommand#O | 5766-6111 | src/device/sliding/qkv_s.rs:286:24  device .main .begin(x.view()) .fet |
| 15,158 | 15,571 | 413 | TuExec | 162 | Sub RenegadeCommand#O2 | 6111-6406 | src/device/sliding/qkv_s.rs:301:77  device .sub .begin(hi_neg.view())  |
| 15,572 | 17,800 | 2,228 | TuExec | 167 | Main RenegadeCommand#O | 6406-6751 | src/device/sliding/qkv_s.rs:308:5  device .main .begin(x.view()) .fetc |
| 17,801 | 18,261 | 460 | StoTrf | 173 | Sub RenegadeCommand#O2 | 6751-7078 | src/device/sliding/qkv_s.rs:43:5  device .sub .begin(x.view()) .fetch: |
| 17,832 | 21,312 | 3,480 | DMA | 175 | DmaCommand | 9411-9963 | src/device/sliding/rope_h4.rs:125:77  rope_offset.to_dm(&mut device.td |
| 21,314 | 21,728 | 414 | TuExec | 180 | Main RenegadeCommand#O | 9963-10245 | src/device/sliding/rope_h4.rs:126:35  device .main .begin(off.view())  |
| 21,320 | 22,993 | 1,673 | DMA | 181 | DmaCommand | 9963-10518 | src/device/sliding/qkv_s.rs:383:9  weight_scale.to_dm(&mut device.tdma |
| 21,729 | 24,516 | 2,787 | TuExec | 187 | Main RenegadeCommand#O | 10245-11174 | src/device/sliding/qkv_s.rs:341:26  device .main .begin(weight.view()) |
| 21,736 | 24,514 | 2,778 | StoVrf | 189 | Sub RenegadeCommand#O5 | 10245-10512 | src/device/sliding/rope_h4.rs:140:82  device .sub .begin(pos.view()) . |
| 21,924 | 24,146 | 2,222 | DMA | 192 | DmaCommand | 10518-11073 | src/device/sliding/qkv_h4_norm.rs:153:64  weight.to_dm(&mut device.tdm |
| 21,932 | 48,576 | 26,644 | DMA | 194 | DmaCommand | 11073-24456 | src/device/sliding/qkv_s.rs:39:5  weight.to_dm(&mut device.tdma) |
| 24,517 | 25,689 | 1,172 | TuExec | 202 | Main RenegadeCommand#O | 11174-11693 | src/device/sliding/qkv_s.rs:374:100  device .main .begin(r.view()) .fe |
| 25,690 | 26,195 | 505 | TuExec | 208 | Main RenegadeCommand#O | 11693-12038 | src/device/sliding/rope_h4.rs:102:37  device .main .begin(after.view() |
| 25,696 | 26,141 | 445 | TuExec | 210 | Sub RenegadeCommand#O1 | 11693-12020 | src/device/sliding/qkv_h4_norm.rs:101:63  device .sub .begin(x.view()) |
| 26,196 | 26,539 | 343 | StoVrf | 217 | Sub RenegadeCommand#O5 | 12038-12305 | src/device/sliding/rope_h4.rs:118:82  device .sub .begin(one_k.view()) |
| 26,540 | 26,913 | 373 | TuExec | 221 | Main RenegadeCommand#O | 12305-12587 | src/device/sliding/rope_h4.rs:147:35  device .main .begin(pos.view())  |
| 26,914 | 27,231 | 317 | TuExec | 226 | Main RenegadeCommand#O | 12587-12852 | src/device/sliding/rope_h4.rs:55:13  device.main .begin(v.view()) .fet |
| 28,244 | 28,609 | 365 | TuExec | 233 | Main RenegadeCommand#O | 13052-13334 | src/device/sliding/rope_h4.rs:61:13  device.main .begin(v.view()) .fet |
| 28,610 | 28,929 | 319 | TuExec | 240 | Main RenegadeCommand#O | 13334-13600 | src/device/sliding/rope_h4.rs:55:13  device.main .begin(v.view()) .fet |
| 29,434 | 29,801 | 367 | TuExec | 247 | Main RenegadeCommand#O | 13800-14083 | src/device/sliding/rope_h4.rs:61:13  device.main .begin(v.view()) .fet |
| 29,802 | 31,485 | 1,683 | TuExec | 254 | Main RenegadeCommand#O | 14083-14351 | src/device/sliding/rope_h4.rs:55:13  device.main .begin(v.view()) .fet |
| 31,487 | 32,159 | 672 | TuExec | 260 | Main RenegadeCommand#O | 14351-14695 | src/device/sliding/qkv_h4_norm.rs:140:60  device .main .begin(scale.vi |
| 32,160 | 32,931 | 771 | TuExec | 267 | Main RenegadeCommand#O | 14695-15040 | src/device/sliding/qkv_h4_norm.rs:108:71  device .main .begin(scale.vi |
| 32,932 | 33,627 | 695 | TuExec | 272 | Main RenegadeCommand#O | 15040-15325 | src/device/sliding/rope_h4.rs:61:13  device.main .begin(v.view()) .fet |
| 33,628 | 33,997 | 369 | TuExec | 277 | Main RenegadeCommand#O | 15325-15606 | src/device/sliding/qkv_h4_norm.rs:127:64  device .main .begin(mean_squ |
| 33,998 | 34,330 | 332 | TuExec | 284 | Main RenegadeCommand#O | 15606-15878 | src/device/sliding/rope_h4.rs:55:13  device.main .begin(v.view()) .fet |
| 34,728 | 35,107 | 379 | TuExec | 291 | Main RenegadeCommand#O | 16078-16367 | src/device/sliding/rope_h4.rs:61:13  device.main .begin(v.view()) .fet |
| 35,108 | 35,455 | 347 | TuExec | 298 | Main RenegadeCommand#O | 16367-16647 | src/device/sliding/rope_h4.rs:55:13  device.main .begin(v.view()) .fet |
| 35,854 | 36,249 | 395 | TuExec | 305 | Main RenegadeCommand#O | 16847-17144 | src/device/sliding/rope_h4.rs:61:13  device.main .begin(v.view()) .fet |
| 36,250 | 36,629 | 379 | TuExec | 312 | Main RenegadeCommand#O | 17144-17440 | src/device/sliding/rope_h4.rs:55:13  device.main .begin(v.view()) .fet |
| 37,028 | 37,455 | 427 | TuExec | 319 | Main RenegadeCommand#O | 17640-17953 | src/device/sliding/rope_h4.rs:61:13  device.main .begin(v.view()) .fet |
| 37,457 | 38,143 | 686 | TuExec | 325 | Main RenegadeCommand#O | 17953-18298 | src/device/sliding/qkv_h4_norm.rs:154:5  device.main .begin(weight.vie |
| 38,144 | 38,587 | 443 | TuExec | 332 | Main RenegadeCommand#O | 18298-18626 | src/device/sliding/rope_h4.rs:55:13  device.main .begin(v.view()) .fet |
| 38,984 | 39,475 | 491 | TuExec | 341 | Main RenegadeCommand#O | 18826-19171 | src/device/sliding/rope_h4.rs:61:13  device.main .begin(v.view()) .fet |
| 39,476 | 40,047 | 571 | TuExec | 348 | Main RenegadeCommand#O | 19171-19563 | src/device/sliding/rope_h4.rs:55:13  device.main .begin(v.view()) .fet |
| 40,480 | 41,099 | 619 | TuExec | 355 | Main RenegadeCommand#O | 19763-20172 | src/device/sliding/rope_h4.rs:61:13  device.main .begin(v.view()) .fet |
| 41,100 | 41,981 | 881 | TuExec | 362 | Main RenegadeCommand#O | 20172-20710 | src/device/sliding/rope_h4.rs:170:30  device .main .begin(v.view()) .f |
| 41,982 | 42,473 | 491 | TuExec | 367 | Main RenegadeCommand#O | 20710-21055 | src/device/sliding/rope_h4.rs:190:34  device .main .begin(f.view()) .f |
| 42,474 | 42,965 | 491 | TuExec | 372 | Main RenegadeCommand#O | 21055-21400 | src/device/sliding/rope_h4.rs:209:76  device .main .begin(angle.view() |
| 42,480 | 43,097 | 617 | StoVrf | 374 | Sub RenegadeCommand#O1 | 21055-21446 | src/device/sliding/rope_h4.rs:203:81  device .sub .begin(angle.view()) |
| 43,098 | 43,793 | 695 | TuExec | 385 | Main RenegadeCommand#O | 21446-21791 | src/device/sliding/rope_h4.rs:225:5  device.main .begin(turns.view())  |
| 43,104 | 43,579 | 475 | StoVrf | 387 | Sub RenegadeCommand#O1 | 21446-21773 | src/device/sliding/rope_h4.rs:247:96  device .sub .begin(angle.view(). |
| 43,794 | 44,337 | 543 | TuExec | 394 | Main RenegadeCommand#O | 21791-22104 | src/device/sliding/rope_h4.rs:253:13  device.main .begin(turns.view(). |
| 43,801 | 45,113 | 1,312 | StoVrf | 396 | Sub RenegadeCommand#O1 | 21791-22118 | src/device/sliding/rope_h4.rs:247:96  device .sub .begin(angle.view(). |
| 45,114 | 45,655 | 541 | TuExec | 407 | Main RenegadeCommand#O | 22304-22617 | src/device/sliding/rope_h4.rs:253:13  device.main .begin(turns.view(). |
| 45,656 | 46,069 | 413 | TuExec | 413 | Main RenegadeCommand#O | 22617-22897 | src/device/sliding/rope_h4.rs:274:5  device.main .begin(sin_full.view( |
| 46,370 | 47,048 | 678 | TuExec | 422 | Sub RenegadeCommand#O2 | 22976-23271 | src/device/sliding/rope_h4.rs:303:84  device .sub .begin(sin.tile::<m! |
| 47,049 | 47,744 | 695 | TuExec | 429 | Main RenegadeCommand#O | 23350-23663 | src/device/sliding/rope_h4.rs:371:5  device.main .begin(k.view().tile: |
| 47,056 | 47,742 | 686 | TuExec | 431 | Sub RenegadeCommand#O2 | 23350-23645 | src/device/sliding/rope_h4.rs:310:84  device .sub .begin(sin.tile::<m! |
| 48,174 | 48,685 | 511 | TuExec | 441 | Main RenegadeCommand#O | 23924-24237 | src/device/sliding/rope_h4.rs:384:5  device.main .begin(k.view().tile: |
| 48,687 | 55,567 | 6,880 | TuExec | 448 | Main RenegadeCommand#O | 24456-25897 | src/device/sliding/qkv_s.rs:59:5  device .main .begin(weight.view()) . |
| 52,854 | 54,072 | 1,218 | DMA | 449 | DmaCommand | 24456-25017 | src/device/sliding/qkv_s.rs:424:9  q_weight_scale.to_dm(&mut device.td |
| 54,074 | 55,002 | 928 | TuExec | 454 | Sub RenegadeCommand#O1 | 25017-25408 | src/device/sliding/qkv_h4_norm.rs:18:67  device .sub .begin(scale.view |
| 54,080 | 55,462 | 1,382 | DMA | 455 | DmaCommand | 25017-25578 | src/device/sliding/qkv_h4_norm.rs:70:68  weight.to_dm(&mut device.tdma |
| 54,086 | 68,282 | 14,196 | DMA | 456 | DmaCommand | 25578-32544 | src/device/sliding/qkv_s.rs:397:23  v_weight.to_dm(&mut device.tdma) |
| 55,568 | 57,250 | 1,682 | TuExec | 463 | Main RenegadeCommand#O | 25897-26672 | src/device/sliding/qkv_s.rs:89:5  device .main .begin(r.view()) .fetch |
| 57,251 | 58,409 | 1,158 | TuExec | 469 | Main RenegadeCommand#O | 26672-27081 | src/device/sliding/qkv_h4_norm.rs:25:75  device .main .begin(x.view()) |
| 57,257 | 57,835 | 578 | TuExec | 471 | Sub RenegadeCommand#O2 | 26672-27063 | src/device/sliding/qkv_h4_norm.rs:63:67  device .sub .begin(x.view())  |
| 58,410 | 58,782 | 372 | TuExec | 478 | Main RenegadeCommand#O | 27081-27364 | src/device/sliding/qkv_h4_norm.rs:44:67  device .main .begin(mean_squa |
| 58,783 | 59,221 | 438 | StoVrf | 482 | Sub RenegadeCommand#O1 | 27364-27635 | src/device/sliding/qkv_h4_norm.rs:57:68  device .sub .begin(rms.view() |
| 59,223 | 60,235 | 1,012 | TuExec | 488 | Main RenegadeCommand#O | 27635-28044 | src/device/sliding/qkv_h4_norm.rs:71:5  device.main .begin(weight.view |
| 59,229 | 60,140 | 911 | StoVrf | 490 | Sub RenegadeCommand#O2 | 27635-28026 | src/device/sliding/rope_h4.rs:397:81  device .sub .begin(k_sin.view()) |
| 60,236 | 60,675 | 439 | TuExec | 497 | Sub RenegadeCommand#O2 | 28105-28432 | src/device/sliding/rope_h4.rs:295:79  device .sub .begin(rows.view().t |
| 60,642 | 61,217 | 575 | TuExec | 501 | Main RenegadeCommand#O | 28305-28650 | src/device/sliding/rope_h4.rs:320:5  device.main .begin(q.view().tile: |
| 61,620 | 62,195 | 575 | TuExec | 511 | Main RenegadeCommand#O | 28850-29195 | src/device/sliding/rope_h4.rs:334:5  device.main .begin(q.view().tile: |
| 62,196 | 62,881 | 685 | TuExec | 516 | Main RenegadeCommand#O | 29195-29540 | src/device/sliding/rope_h4.rs:403:80  device .main .begin(k.view()) .f |
| 62,603 | 63,756 | 1,153 | StoVrf | 520 | Sub RenegadeCommand#O2 | 29395-29914 | src/device/sliding/rope_h4.rs:347:85  device .sub .begin(q_sin.view()) |
| 63,757 | 65,089 | 1,332 | TuExec | 527 | Main RenegadeCommand#O | 29914-30323 | src/device/sliding/rope_h4.rs:353:84  device .main .begin(q.view()) .f |
| 68,284 | 70,015 | 1,731 | TuExec | 535 | Main RenegadeCommand#O | 32544-33473 | src/device/sliding/qkv_s.rs:341:26  device .main .begin(weight.view()) |
| 68,291 | 69,572 | 1,281 | DMA | 536 | DmaCommand | 32544-33099 | src/device/sliding/qkv_s.rs:383:9  weight_scale.to_dm(&mut device.tdma |
| 69,574 | 70,013 | 439 | TuExec | 541 | Sub RenegadeCommand#O4 | 33099-33426 | src/device/sliding/qkv_h4_norm.rs:243:63  device .sub .begin(scale.vie |
| 69,580 | 70,550 | 970 | DMA | 542 | DmaCommand | 33099-33550 | src/ops.rs:78:5  q.view().to_hbm_view(&mut device.tdma, q_out.view_mut |
| 70,017 | 71,187 | 1,170 | TuExec | 549 | Main RenegadeCommand#O | 33473-33992 | src/device/sliding/qkv_s.rs:374:100  device .main .begin(r.view()) .fe |
| 70,023 | 71,820 | 1,797 | DMA | 550 | DmaCommandScatter | 33550-34479 | src/ops.rs:79:5  k.dma_scatter::<m![1], _, _>(kv_offset, k_cache) |
| 71,188 | 71,959 | 771 | TuExec | 558 | Main RenegadeCommand#O | 33992-34336 | src/device/sliding/qkv_h4_norm.rs:250:68  device .main .begin(x.view() |
| 71,960 | 72,715 | 755 | TuExec | 563 | Main RenegadeCommand#O | 34336-34681 | src/device/sliding/qkv_h4_norm.rs:268:5  device.main .begin(x.view())  |
| 72,716 | 74,376 | 1,660 | DMA | 567 | DmaCommandScatter | 34681-35610 | src/ops.rs:80:5  v.dma_scatter::<m![1], _, _>(kv_offset, v_cache) |
| 74,378 | 75,409 | 1,031 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 2 |
