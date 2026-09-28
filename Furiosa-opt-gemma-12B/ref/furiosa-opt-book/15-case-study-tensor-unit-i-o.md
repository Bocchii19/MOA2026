# Case Study: Tensor Unit I/O

This end-to-end Tensor Unit I/O pattern uses Fetch and Commit to stage a tensor through DM while preserving every logical index.
It is a composed movement pattern, distinct from the individual Fetch Engine and Commit Engine reference pages.

The canonical public case is `fetch_commit_simple`.
It accepts an `i8` HBM tensor with mapping `[A, B]`, widens values to `i32` through the Tensor Unit, and returns an HBM tensor with mapping `[B, A]`.
The final `to_hbm` DMA call selects the output HBM mapping, so the test proves the complete HBM-to-DM, Fetch, Collect, Commit, and DM-to-HBM pipeline.
It intentionally does not claim a streaming permutation: the public example’s `[A, B]` to `[B, A]` relayout occurs at the final DMA boundary.
The device source is included below.

```rust
#![expect(clippy::type_complexity)]

use furiosa_opt_std::prelude::*;

axes![A = 4096, B = 8];

type Chip = m![1];
type Cluster = m![1 # 2];

#[device(chip = 1)]
pub fn fetch_commit_simple(
    device: &mut Device,
    input: &HbmTensor<i8, m![1], m![A, B]>,
) -> HbmTensor<i32, m![1], m![B, A]> {
    // Element's innermost axis must match the source's innermost (B, stride 1) so that the
    // DMA tail contains the full B axis (8 × i8 = 8 bytes), satisfying min_align = 8.
    let input_dm = input.to_dm::<Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]>(&mut device.tdma);

    let fetch_and_commit_tensor: DmTensor<i32, Chip, Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![A / 8 % 2], m![A % 8, B]>()
        .fetch_cast::<i32>()
        .collect::<m![A / 8 % 2, A % 8], m![B]>()
        .commit_trim::<m![B]>()
        .commit();

    fetch_and_commit_tensor.to_hbm(&mut device.tdma)
}
```

The host oracle and test are included in the examples test target.

```rust
use furiosa_opt_examples::fetch_commit::fetch_commit_simple;
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn test_fetch_commit_simple_host() {
    use furiosa_opt_examples::fetch_commit::{A, B};

    let mut device = Device::new(fetch_commit_simple.topology()).unwrap();

    // Create input tensor with shape (A=4096)(B=8).
    let input = HostTensor::<i8, m![A, B]>::from_vec((0..32768).map(|x| x as i8).collect::<Vec<_>>())
        .to_hbm::<m![1], m![A, B]>(&mut device.pdma)
        .await
        .unwrap();

    // Call the device function.
    let output = launch(fetch_commit_simple, (&mut device, &input)).await.unwrap();

    let mut expected = vec![0i32; 4096 * 8];
    let mut idx = 0;
    for b in 0..8 {
        for a in 0..4096 {
            expected[idx] = ((a * 8 + b) as i8) as i32;
            idx += 1;
        }
    }

    assert_eq!(
        output.to_host::<m![B, A]>(&mut device.pdma).await.unwrap().into_inner(),
        Tensor::<_, m![B, A], CurrentBackend>::from_vec(expected)
    );
}
```

Run `cargo furiosa-opt test --test fetch_commit_tests` to execute the canonical case.
The host oracle checks every `output[b, a]` value against the widened `input[a, b]` value.
The streaming path is input DM, Fetch, Cast, Collect, Commit, and output DM with the original `[A, B]` DM layout.
The Fetch Engine, Collect Engine, and Commit Engine document the APIs used here.
