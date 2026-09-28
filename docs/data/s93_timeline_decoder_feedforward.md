### decoder_feedforward — cluster 0: Task 204,858 cycles
DMA busy union 191,590 (93.5%), TU busy union 156,981 (76.6%), spans 85

| begin | end | dur | kind | emit# | op | static lifetime | source |
|---:|---:|---:|---|---:|---|---|---|
| 102 | 1,218 | 1,116 | DMA | 16 | DmaCommand | 1003-1623 | src/device/shared/mlp.rs:1287:83  rms_weight.to_dm(&mut device.tdma) |
| 2,873 | 197,797 | 194,924 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 4 |
| 6,913 | 7,327 | 414 | TuExec | 60 | Main RenegadeCommand#O | 1623-1917 | src/device/shared/mlp.rs:1291:83  device .main .begin(weight_dm.view() |
| 6,919 | 14,040 | 7,121 | DMA | 61 | DmaCommand | 1623-5180 | src/device/shared/mlp.rs:384:5  vu.to_dm_view(&mut device.tdma, src.vi |
| 7,328 | 7,879 | 551 | TuExec | 66 | Sub RenegadeCommand#O6 | 1917-2300 | src/device/shared/mlp.rs:1298:84  device .sub .begin(weight_dm.view()) |
| 7,334 | 14,816 | 7,482 | DMA | 67 | DmaCommand | 5180-5800 | src/device/shared/mlp.rs:1250:75  x.to_dm(&mut device.tdma) |
| 14,818 | 20,281 | 5,463 | TuExec | 75 | Main RenegadeCommand#O | 5800-8494 | src/device/shared/mlp.rs:1252:94  device .main .begin(x.view()) .fetch |
| 14,824 | 22,184 | 7,360 | DMA | 76 | DmaCommand | 5800-9357 | src/device/shared/mlp.rs:386:5  vg.to_dm_view(&mut device.tdma, src.vi |
| 20,282 | 20,659 | 377 | TuExec | 81 | Main RenegadeCommand#O | 8494-8776 | src/device/shared/mlp.rs:1271:78  device .main .begin(reduced_mean_squ |
| 20,660 | 21,003 | 343 | StoVrf | 86 | Sub RenegadeCommand#O7 | 8776-9043 | src/device/shared/mlp.rs:1306:79  device .sub .begin(rms.view()) .fetc |
| 20,846 | 23,768 | 2,922 | DMA | 88 | DmaCommand | 9357-10195 |  |
| 22,186 | 37,905 | 15,719 | TuExec | 94 | Main RenegadeCommand#O | 9357-17300 | src/device/shared/mlp.rs:387:26  device.main .begin(src.view()) .fetch |
| 23,770 | 29,371 | 5,601 | StoTab | 97 | Sub RenegadeCommand#O6 | 10195-11488 |  |
| 23,829 | 71,294 | 47,465 | DMA | 99 | DmaCommand | 10201-34813 | src/device/shared/mlp.rs:365:5  weight.to_dm(&mut device.tdma) |
| 38,060 | 40,175 | 2,115 | TuExec | 110 | Main RenegadeCommand#O | 17379-18543 | src/device/shared/mlp.rs:475:5  device.main .begin(     scale          |
| 40,176 | 41,141 | 965 | TuExec | 116 | Main RenegadeCommand#O | 18543-18944 | src/device/shared/mlp.rs:1313:84  device .main .begin(x.view()) .fetch |
| 40,182 | 44,682 | 4,500 | StoTrf | 118 | Sub RenegadeCommand#O6 | 18543-20606 | src/device/shared/mlp.rs:491:5  device.sub .begin(s.view()) .fetch::<m |
| 41,143 | 41,749 | 606 | TuExec | 123 | Main RenegadeCommand#O | 18944-19345 | src/device/shared/mlp.rs:39:68  device .main .begin(x.view()) .fetch:: |
| 41,750 | 42,161 | 411 | TuExec | 128 | Main RenegadeCommand#O | 19345-19624 | src/device/shared/mlp.rs:95:5  device.main .begin(hi.view()) .fetch::< |
| 42,162 | 43,410 | 1,248 | TuExec | 133 | Main RenegadeCommand#O | 19624-20025 | src/device/shared/mlp.rs:56:72  device .main .begin(x.view()) .fetch:: |
| 44,683 | 45,119 | 436 | TuExec | 143 | Sub RenegadeCommand#O1 | 20685-21008 | src/device/shared/mlp.rs:71:74  device .sub .begin(hi_neg.view()) .fet |
| 45,120 | 46,081 | 961 | TuExec | 147 | Main RenegadeCommand#O | 21008-21409 | src/device/shared/mlp.rs:78:68  device .main .begin(x.view()) .fetch:: |
| 46,082 | 46,427 | 345 | TuExec | 152 | Main RenegadeCommand#O | 21409-21688 | src/device/shared/mlp.rs:101:5  device.main .begin(lo.view()) .fetch:: |
| 46,428 | 63,703 | 17,275 | TuExec | 158 | Main RenegadeCommand#O | 21688-30143 | src/device/shared/mlp.rs:569:116  device .main .begin(x_half.view()) . |
| 63,704 | 64,541 | 837 | TuExec | 163 | Main RenegadeCommand#O | 30143-30647 | src/device/shared/mlp.rs:577:90  device .main .begin(x_rep.view()) .fe |
| 64,542 | 66,657 | 2,115 | TuExec | 169 | Main RenegadeCommand#O | 30647-31811 | src/device/shared/mlp.rs:475:5  device.main .begin(     scale          |
| 64,548 | 66,929 | 2,381 | StoTrf | 171 | Sub RenegadeCommand#O2 | 30647-31870 | src/device/shared/mlp.rs:587:31  device .sub .begin(x.view()) .fetch:: |
| 71,296 | 92,253 | 20,957 | TuExec | 179 | Main RenegadeCommand#O | 34813-44076 | src/device/shared/mlp.rs:460:5  device.main .begin(packed.view()) .fet |
| 71,303 | 73,071 | 1,768 | DMA | 180 | DmaCommand | 34813-35473 | src/device/shared/mlp.rs:679:9  gate_global_scale.to_dm(&mut device.td |
| 71,309 | 74,572 | 3,263 | DMA | 181 | DmaCommand | 35473-36311 |  |
| 74,576 | 84,196 | 9,620 | StoTab | 185 | Sub RenegadeCommand#O5 | 36314-37607 |  |
| 74,634 | 122,646 | 48,012 | DMA | 187 | DmaCommand | 36320-60932 | src/device/shared/mlp.rs:365:5  weight.to_dm(&mut device.tdma) |
| 92,254 | 92,693 | 439 | TuExec | 195 | Main RenegadeCommand#O | 44076-44357 | src/device/shared/mlp.rs:680:84  device .main .begin(gate_scale.view() |
| 92,694 | 94,713 | 2,019 | TuExec | 200 | Main RenegadeCommand#O | 44357-45090 | src/device/shared/mlp.rs:499:5  device.main .begin(blocks.view()) .fet |
| 94,714 | 95,344 | 630 | TuExec | 205 | Main RenegadeCommand#O | 45090-45474 | src/device/shared/mlp.rs:652:5  device.main .begin(x.view()) .fetch::< |
| 94,720 | 99,439 | 4,719 | StoTrf | 207 | Sub RenegadeCommand#O4 | 45090-47153 | src/device/shared/mlp.rs:491:5  device.sub .begin(s.view()) .fetch::<m |
| 95,345 | 95,829 | 484 | TuExec | 212 | Main RenegadeCommand#O | 45474-45815 | src/device/shared/mlp.rs:693:33  device .main .begin(gate.view()) .fet |
| 99,440 | 100,131 | 691 | TuExec | 219 | Sub RenegadeCommand#O7 | 47353-47758 | src/device/shared/mlp.rs:707:26  device .sub .begin(gate_scaled.view() |
| 100,132 | 100,707 | 575 | StoVrf | 223 | Sub RenegadeCommand#O7 | 47758-48141 | src/device/shared/mlp.rs:724:80  device .sub .begin(gelu.view()) .fetc |
| 122,648 | 143,601 | 20,953 | TuExec | 229 | Main RenegadeCommand#O | 60932-70195 | src/device/shared/mlp.rs:460:5  device.main .begin(packed.view()) .fet |
| 122,655 | 124,627 | 1,972 | DMA | 230 | DmaCommand | 60932-61770 |  |
| 124,629 | 133,936 | 9,307 | StoTab | 234 | Sub RenegadeCommand#O1 | 61770-63063 |  |
| 124,729 | 164,966 | 40,237 | DMA | 236 | DmaCommand | 61773-81766 | src/device/shared/mlp.rs:834:17  v.tile::<m![KhR], $rows, m![KhG, KhR  |
| 143,602 | 145,633 | 2,031 | TuExec | 243 | Main RenegadeCommand#O | 70195-70928 | src/device/shared/mlp.rs:499:5  device.main .begin(blocks.view()) .fet |
| 145,634 | 146,263 | 629 | TuExec | 248 | Main RenegadeCommand#O | 70928-71312 | src/device/shared/mlp.rs:652:5  device.main .begin(x.view()) .fetch::< |
| 146,264 | 146,925 | 661 | TuExec | 253 | Main RenegadeCommand#O | 71312-71653 | src/device/shared/mlp.rs:730:75  device .main .begin(up.view()) .fetch |
| 146,926 | 147,411 | 485 | TuExec | 258 | Main RenegadeCommand#O | 71653-71994 | src/device/shared/mlp.rs:141:72  device .main .begin(x.view()) .fetch: |
| 147,412 | 147,897 | 485 | TuExec | 263 | Main RenegadeCommand#O | 71994-72335 | src/device/shared/mlp.rs:124:5  device .main .begin(x.view()) .fetch:: |
| 147,419 | 147,795 | 376 | TuExec | 265 | Sub RenegadeCommand#O8 | 71994-72287 | src/device/shared/mlp.rs:156:74  device .sub .begin(hi_neg.view()) .fe |
| 148,294 | 148,955 | 661 | TuExec | 274 | Main RenegadeCommand#O | 72535-72876 | src/device/shared/mlp.rs:163:5  device .main .begin(x.view()) .fetch:: |
| 148,956 | 149,637 | 681 | TuExec | 280 | Main RenegadeCommand#O | 72876-73302 | src/device/shared/mlp.rs:751:113  device.main .begin(out.view()) .fetc |
| 149,638 | 161,797 | 12,159 | TuExec | 285 | Main RenegadeCommand#O | 73302-79199 | src/device/shared/mlp.rs:757:94  device.main .begin(rep.view()) .fetch |
| 161,798 | 163,071 | 1,273 | TuExec | 291 | Main RenegadeCommand#O | 79199-79942 | src/device/shared/mlp.rs:769:5  device.main.begin(all.view()) .fetch:: |
| 164,968 | 181,961 | 16,993 | TuExec | 298 | Main RenegadeCommand#O | 81766-89229 | src/device/shared/mlp.rs:1019:5  device.main.begin(packed.view()) .fet |
| 165,272 | 174,020 | 8,748 | DMA | 300 | DmaCommand | 81845-85924 | src/device/shared/mlp.rs:860:17  v.tile::<m![KhR], $rows, m![KhG, KhR  |
| 165,513 | 176,620 | 11,107 | DMA | 303 | DmaCommand | 85924-87356 | src/device/shared/mlp.rs:860:17  v.tile::<m![KhR], $rows, m![KhG, KhR  |
| 176,622 | 177,617 | 995 | TuExec | 307 | Sub RenegadeCommand#O1 | 87356-87799 | src/device/shared/mlp.rs:959:100  device .sub .begin(scale.view()) .fe |
| 176,629 | 178,574 | 1,945 | DMA | 308 | DmaCommand | 87356-88194 |  |
| 178,578 | 185,848 | 7,270 | StoTab | 314 | Sub RenegadeCommand#O1 | 88197-89490 |  |
| 178,632 | 189,912 | 11,280 | DMA | 315 | DmaCommand | 88197-93606 | src/device/shared/mlp.rs:834:17  v.tile::<m![KhR], $rows, m![KhG, KhR  |
| 181,962 | 185,850 | 3,888 | TuExec | 320 | Main RenegadeCommand#O | 89229-90213 | src/device/shared/mlp.rs:1005:29  device.main.begin(s.view()) .fetch:: |
| 185,851 | 186,883 | 1,032 | TuExec | 327 | Main RenegadeCommand#O | 90213-90836 | src/device/shared/mlp.rs:1012:5  device.main.begin(s.view()) .fetch::< |
| 186,884 | 188,542 | 1,658 | TuExec | 332 | Main RenegadeCommand#O | 90836-91480 | src/device/shared/mlp.rs:1032:5  device.main.begin(b.view()) .fetch::< |
| 189,914 | 195,079 | 5,165 | TuExec | 341 | Main RenegadeCommand#O | 93606-95669 | src/device/shared/mlp.rs:966:5  device.main .begin(packed.view()) .fet |
| 189,921 | 191,483 | 1,562 | DMA | 342 | DmaCommand | 93606-94155 | src/device/shared/mlp.rs:804:81  down_global_scale.to_dm(&mut device.t |
| 191,485 | 191,901 | 416 | StoVrf | 346 | Sub RenegadeCommand#O1 | 94155-94422 | src/device/shared/mlp.rs:1112:86  device .sub .begin(global_scale.view |
| 191,537 | 192,926 | 1,389 | DMA | 347 | DmaCommand | 94155-94704 | src/device/shared/mlp.rs:805:77  up_global_scale.to_dm(&mut device.tdm |
| 192,928 | 193,720 | 792 | StoVrf | 352 | Sub RenegadeCommand#O1 | 94704-94971 | src/device/shared/mlp.rs:1121:82  device .sub .begin(up_scale.view())  |
| 193,118 | 194,452 | 1,334 | DMA | 353 | DmaCommand | 94704-95253 | src/device/shared/mlp.rs:806:82  layer_scalar.to_dm(&mut device.tdma) |
| 194,454 | 194,769 | 315 | TuExec | 359 | Sub RenegadeCommand#O1 | 95253-95518 | src/device/shared/mlp.rs:1127:86  device .sub .begin(layer_scalar.view |
| 194,460 | 195,606 | 1,146 | DMA | 360 | DmaCommand | 95253-95822 | src/device/shared/mlp.rs:802:80  residual.to_dm(&mut device.tdma) |
| 195,080 | 195,786 | 706 | TuExec | 368 | Main RenegadeCommand#O | 95669-96011 | src/device/shared/mlp.rs:1067:5  device.main .begin(result.view()) .fe |
| 195,788 | 198,504 | 2,716 | TuExec | 374 | Sub RenegadeCommand#O1 | 96011-96352 | src/device/shared/mlp.rs:1138:90  device .sub .begin(residual.view())  |
| 197,798 | 199,198 | 1,400 | DMA | 377 | DmaCommand | 96875-97357 | src/device/shared/mlp.rs:1101:83  down.to_dm(&mut device.tdma) |
| 198,505 | 199,630 | 1,125 | DMA | 382 | DmaCommand | 97357-97926 | src/device/shared/mlp.rs:803:83  post_ff_rms_weight.to_dm(&mut device. |
| 199,200 | 200,764 | 1,564 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 1 |
| 199,632 | 200,063 | 431 | TuExec | 389 | Sub RenegadeCommand#O1 | 97926-98249 | src/device/shared/mlp.rs:1153:82  device .sub .begin(norm_weight.view( |
| 200,765 | 201,369 | 604 | TuExec | 395 | Main RenegadeCommand#O | 98957-99357 | src/device/shared/mlp.rs:1102:72  device.main .begin(y2.view()) .fetch |
| 201,370 | 202,112 | 742 | TuExec | 400 | Main RenegadeCommand#O | 99357-99698 | src/device/shared/mlp.rs:1167:88  device .main .begin(y.view()) .fetch |
| 202,113 | 202,847 | 734 | TuExec | 405 | Main RenegadeCommand#O | 99698-99995 | src/device/shared/mlp.rs:1187:91  device .main .begin(partial_mean_squ |
| 202,848 | 203,323 | 475 | TuExec | 411 | Main RenegadeCommand#O | 99995-100276 | src/device/shared/mlp.rs:1204:77  device .main .begin(mean_square.view |
| 203,324 | 204,007 | 683 | TuExec | 416 | Main RenegadeCommand#O | 100276-100617 | src/device/shared/mlp.rs:1221:5  device.main .begin(y.view()) .fetch:: |
| 204,008 | 204,850 | 842 | DMA | 419 | DmaCommand | 100617-101308 | src/ops.rs:242:5  out.view().to_hbm_view(&mut device.tdma, residual_hb |
| 204,852 | 204,855 | 3 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 1 |

### decoder_feedforward — cluster 1: Task 204,704 cycles
DMA busy union 185,807 (90.8%), TU busy union 154,750 (75.6%), spans 85

| begin | end | dur | kind | emit# | op | static lifetime | source |
|---:|---:|---:|---|---:|---|---|---|
| 96 | 1,208 | 1,112 | DMA | 16 | DmaCommand | 1003-1623 | src/device/shared/mlp.rs:1287:83  rms_weight.to_dm(&mut device.tdma) |
| 2,753 | 195,530 | 192,777 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 4 |
| 6,681 | 7,099 | 418 | TuExec | 60 | Main RenegadeCommand#O | 1623-1917 | src/device/shared/mlp.rs:1291:83  device .main .begin(weight_dm.view() |
| 6,687 | 13,796 | 7,109 | DMA | 61 | DmaCommand | 1623-5180 | src/device/shared/mlp.rs:384:5  vu.to_dm_view(&mut device.tdma, src.vi |
| 7,100 | 7,651 | 551 | TuExec | 66 | Sub RenegadeCommand#O6 | 1917-2300 | src/device/shared/mlp.rs:1298:84  device .sub .begin(weight_dm.view()) |
| 7,106 | 14,548 | 7,442 | DMA | 67 | DmaCommand | 5180-5800 | src/device/shared/mlp.rs:1250:75  x.to_dm(&mut device.tdma) |
| 14,550 | 20,013 | 5,463 | TuExec | 75 | Main RenegadeCommand#O | 5800-8494 | src/device/shared/mlp.rs:1252:94  device .main .begin(x.view()) .fetch |
| 14,556 | 21,868 | 7,312 | DMA | 76 | DmaCommand | 5800-9357 | src/device/shared/mlp.rs:386:5  vg.to_dm_view(&mut device.tdma, src.vi |
| 20,014 | 20,391 | 377 | TuExec | 81 | Main RenegadeCommand#O | 8494-8776 | src/device/shared/mlp.rs:1271:78  device .main .begin(reduced_mean_squ |
| 20,392 | 20,735 | 343 | StoVrf | 86 | Sub RenegadeCommand#O7 | 8776-9043 | src/device/shared/mlp.rs:1306:79  device .sub .begin(rms.view()) .fetc |
| 20,578 | 23,446 | 2,868 | DMA | 88 | DmaCommand | 9357-10195 |  |
| 21,870 | 37,589 | 15,719 | TuExec | 94 | Main RenegadeCommand#O | 9357-17300 | src/device/shared/mlp.rs:387:26  device.main .begin(src.view()) .fetch |
| 23,448 | 29,039 | 5,591 | StoTab | 97 | Sub RenegadeCommand#O6 | 10195-11488 |  |
| 23,507 | 71,026 | 47,519 | DMA | 99 | DmaCommand | 10201-34813 | src/device/shared/mlp.rs:365:5  weight.to_dm(&mut device.tdma) |
| 37,744 | 39,859 | 2,115 | TuExec | 110 | Main RenegadeCommand#O | 17379-18543 | src/device/shared/mlp.rs:475:5  device.main .begin(     scale          |
| 39,860 | 40,825 | 965 | TuExec | 116 | Main RenegadeCommand#O | 18543-18944 | src/device/shared/mlp.rs:1313:84  device .main .begin(x.view()) .fetch |
| 39,866 | 44,364 | 4,498 | StoTrf | 118 | Sub RenegadeCommand#O6 | 18543-20606 | src/device/shared/mlp.rs:491:5  device.sub .begin(s.view()) .fetch::<m |
| 40,827 | 41,433 | 606 | TuExec | 123 | Main RenegadeCommand#O | 18944-19345 | src/device/shared/mlp.rs:39:68  device .main .begin(x.view()) .fetch:: |
| 41,434 | 41,845 | 411 | TuExec | 128 | Main RenegadeCommand#O | 19345-19624 | src/device/shared/mlp.rs:95:5  device.main .begin(hi.view()) .fetch::< |
| 41,846 | 42,932 | 1,086 | TuExec | 133 | Main RenegadeCommand#O | 19624-20025 | src/device/shared/mlp.rs:56:72  device .main .begin(x.view()) .fetch:: |
| 44,365 | 44,801 | 436 | TuExec | 143 | Sub RenegadeCommand#O1 | 20685-21008 | src/device/shared/mlp.rs:71:74  device .sub .begin(hi_neg.view()) .fet |
| 44,802 | 45,763 | 961 | TuExec | 147 | Main RenegadeCommand#O | 21008-21409 | src/device/shared/mlp.rs:78:68  device .main .begin(x.view()) .fetch:: |
| 45,764 | 46,109 | 345 | TuExec | 152 | Main RenegadeCommand#O | 21409-21688 | src/device/shared/mlp.rs:101:5  device.main .begin(lo.view()) .fetch:: |
| 46,110 | 63,385 | 17,275 | TuExec | 158 | Main RenegadeCommand#O | 21688-30143 | src/device/shared/mlp.rs:569:116  device .main .begin(x_half.view()) . |
| 63,386 | 64,223 | 837 | TuExec | 163 | Main RenegadeCommand#O | 30143-30647 | src/device/shared/mlp.rs:577:90  device .main .begin(x_rep.view()) .fe |
| 64,224 | 66,339 | 2,115 | TuExec | 169 | Main RenegadeCommand#O | 30647-31811 | src/device/shared/mlp.rs:475:5  device.main .begin(     scale          |
| 64,230 | 66,611 | 2,381 | StoTrf | 171 | Sub RenegadeCommand#O2 | 30647-31870 | src/device/shared/mlp.rs:587:31  device .sub .begin(x.view()) .fetch:: |
| 71,028 | 91,979 | 20,951 | TuExec | 179 | Main RenegadeCommand#O | 34813-44076 | src/device/shared/mlp.rs:460:5  device.main .begin(packed.view()) .fet |
| 71,035 | 73,061 | 2,026 | DMA | 180 | DmaCommand | 34813-35473 | src/device/shared/mlp.rs:679:9  gate_global_scale.to_dm(&mut device.td |
| 71,041 | 74,552 | 3,511 | DMA | 181 | DmaCommand | 35473-36311 |  |
| 74,556 | 84,154 | 9,598 | StoTab | 185 | Sub RenegadeCommand#O5 | 36314-37607 |  |
| 74,614 | 122,340 | 47,726 | DMA | 187 | DmaCommand | 36320-60932 | src/device/shared/mlp.rs:365:5  weight.to_dm(&mut device.tdma) |
| 91,980 | 92,419 | 439 | TuExec | 195 | Main RenegadeCommand#O | 44076-44357 | src/device/shared/mlp.rs:680:84  device .main .begin(gate_scale.view() |
| 92,420 | 94,363 | 1,943 | TuExec | 200 | Main RenegadeCommand#O | 44357-45090 | src/device/shared/mlp.rs:499:5  device.main .begin(blocks.view()) .fet |
| 94,364 | 94,994 | 630 | TuExec | 205 | Main RenegadeCommand#O | 45090-45474 | src/device/shared/mlp.rs:652:5  device.main .begin(x.view()) .fetch::< |
| 94,370 | 99,085 | 4,715 | StoTrf | 207 | Sub RenegadeCommand#O4 | 45090-47153 | src/device/shared/mlp.rs:491:5  device.sub .begin(s.view()) .fetch::<m |
| 94,995 | 95,479 | 484 | TuExec | 212 | Main RenegadeCommand#O | 45474-45815 | src/device/shared/mlp.rs:693:33  device .main .begin(gate.view()) .fet |
| 99,086 | 99,779 | 693 | TuExec | 219 | Sub RenegadeCommand#O7 | 47353-47758 | src/device/shared/mlp.rs:707:26  device .sub .begin(gate_scaled.view() |
| 99,780 | 100,355 | 575 | StoVrf | 223 | Sub RenegadeCommand#O7 | 47758-48141 | src/device/shared/mlp.rs:724:80  device .sub .begin(gelu.view()) .fetc |
| 122,342 | 143,297 | 20,955 | TuExec | 229 | Main RenegadeCommand#O | 60932-70195 | src/device/shared/mlp.rs:460:5  device.main .begin(packed.view()) .fet |
| 122,349 | 124,313 | 1,964 | DMA | 230 | DmaCommand | 60932-61770 |  |
| 124,315 | 133,584 | 9,269 | StoTab | 234 | Sub RenegadeCommand#O1 | 61770-63063 |  |
| 124,415 | 164,860 | 40,445 | DMA | 236 | DmaCommand | 61773-81766 | src/device/shared/mlp.rs:834:17  v.tile::<m![KhR], $rows, m![KhG, KhR  |
| 143,298 | 145,323 | 2,025 | TuExec | 243 | Main RenegadeCommand#O | 70195-70928 | src/device/shared/mlp.rs:499:5  device.main .begin(blocks.view()) .fet |
| 145,324 | 145,953 | 629 | TuExec | 248 | Main RenegadeCommand#O | 70928-71312 | src/device/shared/mlp.rs:652:5  device.main .begin(x.view()) .fetch::< |
| 145,954 | 146,615 | 661 | TuExec | 253 | Main RenegadeCommand#O | 71312-71653 | src/device/shared/mlp.rs:730:75  device .main .begin(up.view()) .fetch |
| 146,616 | 147,101 | 485 | TuExec | 258 | Main RenegadeCommand#O | 71653-71994 | src/device/shared/mlp.rs:141:72  device .main .begin(x.view()) .fetch: |
| 147,102 | 147,587 | 485 | TuExec | 263 | Main RenegadeCommand#O | 71994-72335 | src/device/shared/mlp.rs:124:5  device .main .begin(x.view()) .fetch:: |
| 147,109 | 147,485 | 376 | TuExec | 265 | Sub RenegadeCommand#O8 | 71994-72287 | src/device/shared/mlp.rs:156:74  device .sub .begin(hi_neg.view()) .fe |
| 147,988 | 148,649 | 661 | TuExec | 274 | Main RenegadeCommand#O | 72535-72876 | src/device/shared/mlp.rs:163:5  device .main .begin(x.view()) .fetch:: |
| 148,650 | 149,351 | 701 | TuExec | 280 | Main RenegadeCommand#O | 72876-73302 | src/device/shared/mlp.rs:751:113  device.main .begin(out.view()) .fetc |
| 149,352 | 161,855 | 12,503 | TuExec | 285 | Main RenegadeCommand#O | 73302-79199 | src/device/shared/mlp.rs:757:94  device.main .begin(rep.view()) .fetch |
| 161,856 | 163,129 | 1,273 | TuExec | 291 | Main RenegadeCommand#O | 79199-79942 | src/device/shared/mlp.rs:769:5  device.main.begin(all.view()) .fetch:: |
| 164,862 | 181,047 | 16,185 | TuExec | 298 | Main RenegadeCommand#O | 81766-89229 | src/device/shared/mlp.rs:1019:5  device.main.begin(packed.view()) .fet |
| 165,162 | 173,754 | 8,592 | DMA | 300 | DmaCommand | 81845-85924 | src/device/shared/mlp.rs:860:17  v.tile::<m![KhR], $rows, m![KhG, KhR  |
| 165,324 | 176,428 | 11,104 | DMA | 303 | DmaCommand | 85924-87356 | src/device/shared/mlp.rs:860:17  v.tile::<m![KhR], $rows, m![KhG, KhR  |
| 176,430 | 177,383 | 953 | TuExec | 307 | Sub RenegadeCommand#O1 | 87356-87799 | src/device/shared/mlp.rs:959:100  device .sub .begin(scale.view()) .fe |
| 176,437 | 178,386 | 1,949 | DMA | 308 | DmaCommand | 87356-88194 |  |
| 178,390 | 185,244 | 6,854 | StoTab | 314 | Sub RenegadeCommand#O1 | 88197-89490 |  |
| 178,444 | 189,584 | 11,140 | DMA | 315 | DmaCommand | 88197-93606 | src/device/shared/mlp.rs:834:17  v.tile::<m![KhR], $rows, m![KhG, KhR  |
| 181,048 | 185,246 | 4,198 | TuExec | 320 | Main RenegadeCommand#O | 89229-90213 | src/device/shared/mlp.rs:1005:29  device.main.begin(s.view()) .fetch:: |
| 185,247 | 186,279 | 1,032 | TuExec | 327 | Main RenegadeCommand#O | 90213-90836 | src/device/shared/mlp.rs:1012:5  device.main.begin(s.view()) .fetch::< |
| 186,280 | 187,936 | 1,656 | TuExec | 332 | Main RenegadeCommand#O | 90836-91480 | src/device/shared/mlp.rs:1032:5  device.main.begin(b.view()) .fetch::< |
| 189,586 | 194,753 | 5,167 | TuExec | 341 | Main RenegadeCommand#O | 93606-95669 | src/device/shared/mlp.rs:966:5  device.main .begin(packed.view()) .fet |
| 189,593 | 189,756 | 163 | DMA | 342 | DmaCommand | 93606-94155 | src/device/shared/mlp.rs:804:81  down_global_scale.to_dm(&mut device.t |
| 189,781 | 190,151 | 370 | StoVrf | 346 | Sub RenegadeCommand#O1 | 94155-94422 | src/device/shared/mlp.rs:1112:86  device .sub .begin(global_scale.view |
| 189,812 | 189,993 | 181 | DMA | 347 | DmaCommand | 94155-94704 | src/device/shared/mlp.rs:805:77  up_global_scale.to_dm(&mut device.tdm |
| 190,153 | 190,864 | 711 | StoVrf | 352 | Sub RenegadeCommand#O1 | 94704-94971 | src/device/shared/mlp.rs:1121:82  device .sub .begin(up_scale.view())  |
| 190,206 | 190,405 | 199 | DMA | 353 | DmaCommand | 94704-95253 | src/device/shared/mlp.rs:806:82  layer_scalar.to_dm(&mut device.tdma) |
| 190,866 | 192,641 | 1,775 | TuExec | 359 | Sub RenegadeCommand#O1 | 95253-95518 | src/device/shared/mlp.rs:1127:86  device .sub .begin(layer_scalar.view |
| 190,895 | 191,093 | 198 | DMA | 360 | DmaCommand | 95253-95822 | src/device/shared/mlp.rs:802:80  residual.to_dm(&mut device.tdma) |
| 194,754 | 195,460 | 706 | TuExec | 368 | Main RenegadeCommand#O | 95669-96011 | src/device/shared/mlp.rs:1067:5  device.main .begin(result.view()) .fe |
| 195,462 | 196,240 | 778 | TuExec | 374 | Sub RenegadeCommand#O1 | 96011-96352 | src/device/shared/mlp.rs:1138:90  device .sub .begin(residual.view())  |
| 195,531 | 197,040 | 1,509 | DMA | 377 | DmaCommand | 96875-97357 | src/device/shared/mlp.rs:1101:83  down.to_dm(&mut device.tdma) |
| 196,241 | 197,046 | 805 | DMA | 382 | DmaCommand | 97357-97926 | src/device/shared/mlp.rs:803:83  post_ff_rms_weight.to_dm(&mut device. |
| 197,042 | 199,163 | 2,121 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 1 |
| 197,810 | 198,241 | 431 | TuExec | 389 | Sub RenegadeCommand#O1 | 97926-98249 | src/device/shared/mlp.rs:1153:82  device .sub .begin(norm_weight.view( |
| 199,164 | 199,767 | 603 | TuExec | 395 | Main RenegadeCommand#O | 98957-99357 | src/device/shared/mlp.rs:1102:72  device.main .begin(y2.view()) .fetch |
| 199,768 | 200,510 | 742 | TuExec | 400 | Main RenegadeCommand#O | 99357-99698 | src/device/shared/mlp.rs:1167:88  device .main .begin(y.view()) .fetch |
| 200,511 | 201,235 | 724 | TuExec | 405 | Main RenegadeCommand#O | 99698-99995 | src/device/shared/mlp.rs:1187:91  device .main .begin(partial_mean_squ |
| 201,236 | 201,649 | 413 | TuExec | 411 | Main RenegadeCommand#O | 99995-100276 | src/device/shared/mlp.rs:1204:77  device .main .begin(mean_square.view |
| 201,650 | 202,333 | 683 | TuExec | 416 | Main RenegadeCommand#O | 100276-100617 | src/device/shared/mlp.rs:1221:5  device.main .begin(y.view()) .fetch:: |
| 202,334 | 202,496 | 162 | DMA | 419 | DmaCommand | 100617-101308 | src/ops.rs:242:5  out.view().to_hbm_view(&mut device.tdma, residual_hb |
| 202,498 | 204,701 | 2,203 | Cluster | None | Cluster |  | {"Synchronization": {"ExplicitSync": {"path": [], "index": 1 |
