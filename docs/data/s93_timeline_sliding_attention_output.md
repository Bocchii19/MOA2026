### sliding_attention_output — cluster 0: Task 46,535 cycles
DMA busy union 34,335 (73.8%), TU busy union 39,973 (85.9%), spans 62

| begin | end | dur | kind | emit# | op | static lifetime | source |
|---:|---:|---:|---|---:|---|---|---|
| 96 | 1,718 | 1,622 | DMA | 6 | DmaCommand | 1003-1611 | src/device/sliding/projection_g3.rs:42:24  x.to_dm(&mut device.tdma) |
| 2,033 | 5,315 | 3,282 | TuExec | 27 | Main RenegadeCommand#O | 1611-3410 | src/device/sliding/projection_g3.rs:43:94  device .main .begin(seed.vi |
| 2,039 | 21,950 | 19,911 | DMA | 28 | DmaCommand | 1611-11999 | src/device/sliding/projection_g3.rs:127:17  v.tile::<m![G3R], $rows, m |
| 5,316 | 5,663 | 347 | TuExec | 33 | Main RenegadeCommand#O | 3410-3690 | src/device/sliding/projection_g3.rs:51:89  device .main .begin(padded. |
| 5,664 | 6,147 | 483 | TuExec | 39 | Main RenegadeCommand#O | 3690-3970 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 6,138 | 6,876 | 738 | TuExec | 44 | Sub RenegadeCommand#O7 | 3848-4173 | src/device/sliding/projection_g3.rs:226:75  device .sub .begin(x.view( |
| 6,148 | 9,437 | 3,289 | TuExec | 48 | Main RenegadeCommand#O | 3970-4250 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 6,877 | 38,806 | 31,929 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 3 |
| 9,438 | 9,785 | 347 | TuExec | 60 | Main RenegadeCommand#O | 4331-4611 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 9,786 | 11,009 | 1,223 | TuExec | 65 | Main RenegadeCommand#O | 4611-4891 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 11,010 | 11,357 | 347 | TuExec | 70 | Main RenegadeCommand#O | 4891-5171 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 11,358 | 11,804 | 446 | TuExec | 75 | Main RenegadeCommand#O | 5171-5451 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 11,805 | 12,153 | 348 | TuExec | 80 | Main RenegadeCommand#O | 5451-5731 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 12,154 | 13,723 | 1,569 | TuExec | 85 | Main RenegadeCommand#O | 5731-6011 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 13,724 | 14,071 | 347 | TuExec | 90 | Main RenegadeCommand#O | 6011-6291 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 14,072 | 15,222 | 1,150 | TuExec | 95 | Main RenegadeCommand#O | 6291-6571 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 15,223 | 15,571 | 348 | TuExec | 100 | Main RenegadeCommand#O | 6571-6851 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 15,572 | 15,919 | 347 | TuExec | 105 | Main RenegadeCommand#O | 6851-7131 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 16,850 | 17,197 | 347 | TuExec | 110 | Main RenegadeCommand#O | 7131-7411 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 17,198 | 17,545 | 347 | TuExec | 115 | Main RenegadeCommand#O | 7411-7691 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 17,546 | 18,009 | 463 | TuExec | 120 | Main RenegadeCommand#O | 7691-7971 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 18,010 | 18,357 | 347 | TuExec | 125 | Main RenegadeCommand#O | 7971-8251 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 18,358 | 18,873 | 515 | TuExec | 130 | Main RenegadeCommand#O | 8251-8531 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 18,874 | 19,221 | 347 | TuExec | 135 | Main RenegadeCommand#O | 8531-8811 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 19,222 | 20,283 | 1,061 | TuExec | 140 | Main RenegadeCommand#O | 8811-9091 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 20,284 | 20,631 | 347 | TuExec | 145 | Main RenegadeCommand#O | 9091-9371 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 20,632 | 20,986 | 354 | TuExec | 150 | Main RenegadeCommand#O | 9371-9651 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 20,987 | 21,335 | 348 | TuExec | 155 | Main RenegadeCommand#O | 9651-9931 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 21,336 | 22,449 | 1,113 | TuExec | 160 | Main RenegadeCommand#O | 9931-10211 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 22,450 | 22,797 | 347 | TuExec | 165 | Main RenegadeCommand#O | 10211-10491 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 22,798 | 23,146 | 348 | TuExec | 170 | Main RenegadeCommand#O | 10491-10771 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 23,147 | 23,495 | 348 | TuExec | 175 | Main RenegadeCommand#O | 10771-11051 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 23,496 | 23,843 | 347 | TuExec | 180 | Main RenegadeCommand#O | 11051-11331 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 24,458 | 24,805 | 347 | TuExec | 185 | Main RenegadeCommand#O | 11331-11611 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 24,806 | 25,153 | 347 | TuExec | 190 | Main RenegadeCommand#O | 11611-11891 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 25,154 | 26,627 | 1,473 | TuExec | 195 | Main RenegadeCommand#O | 11891-12171 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 26,287 | 33,182 | 6,895 | DMA | 198 | DmaCommand | 11999-15543 | src/device/sliding/projection_g3.rs:127:17  v.tile::<m![G3R], $rows, m |
| 26,628 | 26,976 | 348 | TuExec | 203 | Main RenegadeCommand#O | 12171-12451 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 26,977 | 27,383 | 406 | TuExec | 208 | Main RenegadeCommand#O | 12451-12731 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 27,384 | 27,877 | 493 | TuExec | 213 | Main RenegadeCommand#O | 12731-13076 | src/device/sliding/projection_g3.rs:64:5  device.main .begin(x.view()) |
| 27,879 | 28,373 | 494 | TuExec | 218 | Main RenegadeCommand#O | 13076-13421 | src/device/sliding/projection_g3.rs:78:77  device .main .begin(x.view( |
| 28,374 | 28,783 | 409 | TuExec | 224 | Sub RenegadeCommand#O4 | 13421-13716 | src/device/sliding/projection_g3.rs:93:79  device .sub .begin(hi_neg.v |
| 28,784 | 29,466 | 682 | TuExec | 229 | Main RenegadeCommand#O | 13716-14061 | src/device/sliding/projection_g3.rs:100:5  device.main .begin(x.view() |
| 29,467 | 29,927 | 460 | StoTrf | 235 | Sub RenegadeCommand#O5 | 14061-14388 | src/device/sliding/projection_g3.rs:250:31  device .sub .begin(x2.view |
| 29,929 | 34,579 | 4,650 | TuExec | 240 | Main RenegadeCommand#O | 14388-15538 | src/device/sliding/projection_g3.rs:137:13  device.main .begin(w.view( |
| 34,979 | 37,140 | 2,161 | TuExec | 250 | Main RenegadeCommand#O | 15738-16376 | src/device/sliding/projection_g3.rs:137:13  device.main .begin(w.view( |
| 34,986 | 37,131 | 2,145 | DMA | 251 | DmaCommand | 15738-16354 | src/device/sliding/projection_g3.rs:256:35  weight_scale.to_dm(&mut de |
| 37,133 | 39,138 | 2,005 | TuExec | 255 | Sub RenegadeCommand#O8 | 16354-16647 | src/device/sliding/projection_g3.rs:214:5  device .sub .begin(t.view() |
| 38,807 | 39,962 | 1,155 | DMA | 261 | DmaCommand | 17582-18041 | src/device/sliding/projection_g3.rs:265:5  result.view().to_dm_view(&m |
| 39,139 | 40,546 | 1,407 | DMA | 265 | DmaCommand | 18041-18657 | src/device/sliding/projection_g3.rs:258:31  residual.to_dm(&mut device |
| 39,964 | 42,414 | 2,450 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 8 |
| 40,548 | 45,658 | 5,110 | Cluster | None | Cluster |  | {"Synchronization": {"DramReuse": {"path": [], "index": 80}} |
| 40,550 | 42,422 | 1,872 | TuExec | 274 | Sub RenegadeCommand#O8 | 18657-18950 | src/device/sliding/projection_g3.rs:214:5  device .sub .begin(t.view() |
| 40,556 | 41,710 | 1,154 | DMA | 275 | DmaCommand | 18657-19273 | src/device/sliding/projection_g3.rs:257:34  norm_weight.to_dm(&mut dev |
| 42,415 | 42,977 | 562 | TuExec | 279 | Main RenegadeCommand#O | 19641-19952 | src/device/sliding/projection_g3.rs:276:90  device .main .begin(y.view |
| 42,423 | 44,458 | 2,035 | StoVrf | 283 | Sub RenegadeCommand#O8 | 19641-19964 | src/device/sliding/projection_g3.rs:269:79  device .sub .begin(y.view( |
| 42,978 | 44,460 | 1,482 | TuExec | 288 | Main RenegadeCommand#O | 19952-20265 | src/device/sliding/projection_g3.rs:293:92  device .main .begin(partia |
| 44,461 | 44,870 | 409 | TuExec | 296 | Main RenegadeCommand#O | 20265-20546 | src/device/sliding/projection_g3.rs:308:79  device .main .begin(mean_s |
| 44,872 | 45,405 | 533 | TuExec | 302 | Main RenegadeCommand#O | 20546-20857 | src/device/sliding/projection_g3.rs:321:5  device.main .begin(norm_wei |
| 45,659 | 46,528 | 869 | DMA | 307 | DmaCommand | 20857-21639 | src/ops.rs:150:5  out.view().to_hbm_view(&mut device.tdma, residual_hb |
| 46,530 | 46,532 | 2 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 9 |

### sliding_attention_output — cluster 1: Task 46,824 cycles
DMA busy union 30,021 (64.1%), TU busy union 39,387 (84.1%), spans 62

| begin | end | dur | kind | emit# | op | static lifetime | source |
|---:|---:|---:|---|---:|---|---|---|
| 100 | 1,468 | 1,368 | DMA | 6 | DmaCommand | 1003-1611 | src/device/sliding/projection_g3.rs:42:24  x.to_dm(&mut device.tdma) |
| 2,093 | 5,375 | 3,282 | TuExec | 27 | Main RenegadeCommand#O | 1611-3410 | src/device/sliding/projection_g3.rs:43:94  device .main .begin(seed.vi |
| 2,099 | 22,182 | 20,083 | DMA | 28 | DmaCommand | 1611-11999 | src/device/sliding/projection_g3.rs:127:17  v.tile::<m![G3R], $rows, m |
| 5,376 | 5,723 | 347 | TuExec | 33 | Main RenegadeCommand#O | 3410-3690 | src/device/sliding/projection_g3.rs:51:89  device .main .begin(padded. |
| 5,724 | 6,209 | 485 | TuExec | 39 | Main RenegadeCommand#O | 3690-3970 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 6,200 | 6,942 | 742 | TuExec | 44 | Sub RenegadeCommand#O7 | 3848-4173 | src/device/sliding/projection_g3.rs:226:75  device .sub .begin(x.view( |
| 6,210 | 9,377 | 3,167 | TuExec | 48 | Main RenegadeCommand#O | 3970-4250 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 6,943 | 38,253 | 31,310 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 3 |
| 9,378 | 9,725 | 347 | TuExec | 60 | Main RenegadeCommand#O | 4331-4611 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 9,726 | 10,950 | 1,224 | TuExec | 65 | Main RenegadeCommand#O | 4611-4891 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 10,951 | 11,299 | 348 | TuExec | 70 | Main RenegadeCommand#O | 4891-5171 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 11,300 | 11,720 | 420 | TuExec | 75 | Main RenegadeCommand#O | 5171-5451 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 11,721 | 12,069 | 348 | TuExec | 80 | Main RenegadeCommand#O | 5451-5731 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 12,070 | 13,670 | 1,600 | TuExec | 85 | Main RenegadeCommand#O | 5731-6011 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 13,671 | 14,019 | 348 | TuExec | 90 | Main RenegadeCommand#O | 6011-6291 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 14,020 | 15,045 | 1,025 | TuExec | 95 | Main RenegadeCommand#O | 6291-6571 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 15,046 | 15,393 | 347 | TuExec | 100 | Main RenegadeCommand#O | 6571-6851 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 15,394 | 15,741 | 347 | TuExec | 105 | Main RenegadeCommand#O | 6851-7131 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 16,647 | 16,995 | 348 | TuExec | 110 | Main RenegadeCommand#O | 7131-7411 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 16,996 | 17,343 | 347 | TuExec | 115 | Main RenegadeCommand#O | 7411-7691 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 17,344 | 17,809 | 465 | TuExec | 120 | Main RenegadeCommand#O | 7691-7971 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 17,810 | 18,157 | 347 | TuExec | 125 | Main RenegadeCommand#O | 7971-8251 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 18,158 | 18,987 | 829 | TuExec | 130 | Main RenegadeCommand#O | 8251-8531 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 18,988 | 19,335 | 347 | TuExec | 135 | Main RenegadeCommand#O | 8531-8811 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 19,336 | 19,695 | 359 | TuExec | 140 | Main RenegadeCommand#O | 8811-9091 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 19,696 | 20,043 | 347 | TuExec | 145 | Main RenegadeCommand#O | 9091-9371 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 20,044 | 21,133 | 1,089 | TuExec | 150 | Main RenegadeCommand#O | 9371-9651 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 21,134 | 21,481 | 347 | TuExec | 155 | Main RenegadeCommand#O | 9651-9931 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 21,482 | 21,829 | 347 | TuExec | 160 | Main RenegadeCommand#O | 9931-10211 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 21,830 | 22,177 | 347 | TuExec | 165 | Main RenegadeCommand#O | 10211-10491 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 22,178 | 22,794 | 616 | TuExec | 170 | Main RenegadeCommand#O | 10491-10771 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 22,795 | 23,143 | 348 | TuExec | 175 | Main RenegadeCommand#O | 10771-11051 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 23,144 | 23,491 | 347 | TuExec | 180 | Main RenegadeCommand#O | 11051-11331 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 24,342 | 24,689 | 347 | TuExec | 185 | Main RenegadeCommand#O | 11331-11611 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 24,690 | 25,037 | 347 | TuExec | 190 | Main RenegadeCommand#O | 11611-11891 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 25,038 | 26,547 | 1,509 | TuExec | 195 | Main RenegadeCommand#O | 11891-12171 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 26,207 | 33,070 | 6,863 | DMA | 198 | DmaCommand | 11999-15543 | src/device/sliding/projection_g3.rs:127:17  v.tile::<m![G3R], $rows, m |
| 26,548 | 26,896 | 348 | TuExec | 203 | Main RenegadeCommand#O | 12171-12451 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 26,897 | 27,305 | 408 | TuExec | 208 | Main RenegadeCommand#O | 12451-12731 | src/device/sliding/projection_g3.rs:164:5  device.main .begin(x.view() |
| 27,306 | 27,799 | 493 | TuExec | 213 | Main RenegadeCommand#O | 12731-13076 | src/device/sliding/projection_g3.rs:64:5  device.main .begin(x.view()) |
| 27,801 | 28,295 | 494 | TuExec | 218 | Main RenegadeCommand#O | 13076-13421 | src/device/sliding/projection_g3.rs:78:77  device .main .begin(x.view( |
| 28,296 | 28,707 | 411 | TuExec | 224 | Sub RenegadeCommand#O4 | 13421-13716 | src/device/sliding/projection_g3.rs:93:79  device .sub .begin(hi_neg.v |
| 28,708 | 29,390 | 682 | TuExec | 229 | Main RenegadeCommand#O | 13716-14061 | src/device/sliding/projection_g3.rs:100:5  device.main .begin(x.view() |
| 29,391 | 29,851 | 460 | StoTrf | 235 | Sub RenegadeCommand#O5 | 14061-14388 | src/device/sliding/projection_g3.rs:250:31  device .sub .begin(x2.view |
| 29,853 | 34,363 | 4,510 | TuExec | 240 | Main RenegadeCommand#O | 14388-15538 | src/device/sliding/projection_g3.rs:137:13  device.main .begin(w.view( |
| 34,765 | 36,445 | 1,680 | TuExec | 250 | Main RenegadeCommand#O | 15738-16376 | src/device/sliding/projection_g3.rs:137:13  device.main .begin(w.view( |
| 34,772 | 34,934 | 162 | DMA | 251 | DmaCommand | 15738-16354 | src/device/sliding/projection_g3.rs:256:35  weight_scale.to_dm(&mut de |
| 35,341 | 38,590 | 3,249 | TuExec | 255 | Sub RenegadeCommand#O8 | 16354-16647 | src/device/sliding/projection_g3.rs:214:5  device .sub .begin(t.view() |
| 38,254 | 39,472 | 1,218 | DMA | 261 | DmaCommand | 17582-18041 | src/device/sliding/projection_g3.rs:265:5  result.view().to_dm_view(&m |
| 38,591 | 39,474 | 883 | DMA | 265 | DmaCommand | 18041-18657 | src/device/sliding/projection_g3.rs:258:31  residual.to_dm(&mut device |
| 39,475 | 41,659 | 2,184 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 8 |
| 39,479 | 44,589 | 5,110 | Cluster | None | Cluster |  | {"Synchronization": {"DramReuse": {"path": [], "index": 80}} |
| 39,481 | 41,667 | 2,186 | TuExec | 274 | Sub RenegadeCommand#O8 | 18657-18950 | src/device/sliding/projection_g3.rs:214:5  device .sub .begin(t.view() |
| 39,487 | 39,650 | 163 | DMA | 275 | DmaCommand | 18657-19273 | src/device/sliding/projection_g3.rs:257:34  norm_weight.to_dm(&mut dev |
| 41,660 | 42,221 | 561 | TuExec | 279 | Main RenegadeCommand#O | 19641-19952 | src/device/sliding/projection_g3.rs:276:90  device .main .begin(y.view |
| 41,668 | 43,620 | 1,952 | StoVrf | 283 | Sub RenegadeCommand#O8 | 19641-19964 | src/device/sliding/projection_g3.rs:269:79  device .sub .begin(y.view( |
| 42,222 | 43,622 | 1,400 | TuExec | 288 | Main RenegadeCommand#O | 19952-20265 | src/device/sliding/projection_g3.rs:293:92  device .main .begin(partia |
| 43,623 | 44,036 | 413 | TuExec | 296 | Main RenegadeCommand#O | 20265-20546 | src/device/sliding/projection_g3.rs:308:79  device .main .begin(mean_s |
| 44,038 | 44,571 | 533 | TuExec | 302 | Main RenegadeCommand#O | 20546-20857 | src/device/sliding/projection_g3.rs:321:5  device.main .begin(norm_wei |
| 44,590 | 44,752 | 162 | DMA | 307 | DmaCommand | 20857-21639 | src/ops.rs:150:5  out.view().to_hbm_view(&mut device.tdma, residual_hb |
| 44,754 | 46,821 | 2,067 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 9 |
