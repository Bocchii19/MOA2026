# Audit descriptor DMA — bản best b162deec (compile --dump-summary)

Nguồn: `text_form/resourcelir.desc.json`, công cụ `campaign/dma_descriptors.py`. Lifetime là lịch tĩnh (1 GHz), không phải phần cứng.

## sliding_project_qkv

Tổng payload descriptor: 31,558,656 B; cân bằng 8 engine: 0.9968624660370035

| lệnh | loại | byte | số engine | packet (B) | căn 256 | lifetime tĩnh |
|---|---|---:|---:|---|---|---|
| DmaCommand#O3 | dma.dto_s | 15,360 | 2 | [240] | False | [0, 690] |
| DmaCommand#O4 | dma.dto_s | 15,360 | 2 | [240] | False | [690, 1380] |
| DmaCommand#O11 | dma.dto_s | 16,384 | 8 | [32] | False | [1380, 2099] |
| DmaCommand#O0 | dma.dto_s | 15,728,640 | 8 | [256] | True | [2099, 15482] |
| DmaCommand#O113 | dma.dto_s | 8,192 | 8 | [256] | True | [15482, 16043] |
| DmaCommand#O41 | dma.dto_s | 8,192 | 8 | [256] | True | [17911, 18472] |
| DmaCommand#O34 | dma.sto_d | 1,024 | 1 | [256] | True | [18472, 18815] |
| DmaCommand#O1 | dma.dto_s | 7,864,320 | 8 | [256] | True | [18815, 25806] |
| DmaCommand#O36 | dma.dto_s | 16,384 | 8 | [256] | True | [25806, 26378] |
| DmaCommand#O114 | dma.dto_s | 4,096 | 8 | [256] | True | [26378, 26933] |
| DmaCommand#O67 | dma.dto_s | 4,096 | 8 | [256] | True | [26933, 27488] |
| DmaCommand#O2 | dma.dto_s | 7,864,320 | 8 | [256] | True | [27488, 34479] |
| DmaCommand#O148 | dma.sto_d | 8,192 | 8 | [256] | True | [34479, 34930] |
| DmaCommand#O115 | dma.dto_s | 4,096 | 8 | [256] | True | [34930, 35485] |

## sliding_attention_output

Tổng payload descriptor: 15,890,432 B; cân bằng 8 engine: 0.986648016276704

| lệnh | loại | byte | số engine | packet (B) | căn 256 | lifetime tĩnh |
|---|---|---:|---:|---|---|---|
| DmaCommand#O2 | dma.dto_s | 131,072 | 8 | [256] | True | [0, 739] |
| DmaCommand#O0 | dma.dto_s | 15,728,640 | 8 | [256] | True | [739, 14122] |
| DmaCommand#O26 | dma.dto_s | 7,680 | 1 | [240] | False | [14122, 14738] |
| DmaCommand#O24 | dma.dto_s | 7,680 | 1 | [240] | False | [14738, 15354] |
| DmaCommand#O25 | dma.dto_s | 7,680 | 1 | [240] | False | [15942, 16558] |
| DmaCommand#O40 | dma.sto_d | 7,680 | 1 | [240] | False | [18658, 19440] |

## decoder_feedforward

Tổng payload descriptor: 99,637,248 B; cân bằng 8 engine: 0.9973554735547355

| lệnh | loại | byte | số engine | packet (B) | căn 256 | lifetime tĩnh |
|---|---|---:|---:|---|---|---|
| DmaCommand#O0 | dma.dto_s | 15,360 | 2 | [240] | False | [0, 690] |
| DmaCommand#O1 | dma.dto_s | 15,360 | 2 | [240] | False | [690, 1380] |
| DmaCommand#O8 | dma.dto_s | 16,384 | 8 | [32] | False | [1380, 2099] |
| DmaCommand#O80 | dma.dto_s | 4,096 | 8 | [8] | False | [2099, 2937] |
| DmaCommand#O74 | dma.dto_s | 29,491,200 | 8 | [256] | True | [2937, 27549] |
| DmaCommand#O75 | dma.dto_s | 3,686,400 | 8 | [800] | False | [27549, 32429] |
| DmaCommand#O213 | dma.dto_s | 2,048 | 8 | [32] | True | [32429, 33089] |
| DmaCommand#O146 | dma.dto_s | 4,096 | 8 | [8] | False | [33089, 33927] |
| DmaCommand#O76 | dma.dto_s | 29,491,200 | 8 | [256] | True | [33927, 58539] |
| DmaCommand#O77 | dma.dto_s | 3,686,400 | 8 | [800] | False | [58539, 63419] |
| DmaCommand#O217 | dma.dto_s | 2,048 | 8 | [32] | True | [63419, 64079] |
| DmaCommand#O224 | dma.dto_s | 16,384 | 8 | [32] | False | [64079, 64798] |
| DmaCommand#O249 | dma.dto_s | 4,096 | 8 | [8] | False | [64798, 65636] |
| DmaCommand#O78 | dma.dto_s | 29,491,200 | 8 | [256] | True | [65636, 90489] |
| DmaCommand#O79 | dma.dto_s | 3,686,400 | 8 | [480] | False | [90489, 95450] |
| DmaCommand#O12 | dma.dto_s | 1,024 | 1 | [32] | True | [95450, 96192] |
| DmaCommand#O270 | dma.dto_s | 7,680 | 1 | [240] | False | [96192, 96808] |
| DmaCommand#O273 | dma.dto_s | 7,680 | 1 | [240] | False | [96808, 97424] |
| DmaCommand#O276 | dma.dto_s | 512 | 1 | [16] | True | [97424, 98166] |
| DmaCommand#O279 | dma.sto_d | 7,680 | 1 | [240] | False | [105676, 106458] |
