# Opt-in Drive model cache

This local adapter feature has offline tests. Live Drive persistence and mount
compatibility remain unverified, awaiting the user's Drive OAuth consent. It is
optional; the default batch path is unchanged. It supports only G4 with the
upstream-selected Ref2VA `fp8_scaled` branch. CUDA/GPU selection of another
variant stops the cache launch rather than downloading different weights.

Add the same option to both reviewed **plan** and authorized **run**:

```bash
--drive-cache /content/drive/MyDrive/ColabModelCache/MiniMax-H3/e5eb578a89295337b8ff433a035929ce0279e0b6
```

The plan records the remote root and SHA256 of `assets/cache-manifest.json`.
Run refuses an absent plan, a changed root/profile, enabling/disabling the cache
after planning, or an unsupported GPU before allocation. The manifest contains
five public LFS SHA256s and exact sizes, pinned to the public repository commit.

After allocating its single owned GPU, the existing watchdog covers the entire
child launch: official `colab drivemount`, extracted first setup cell (index 3)
from the external pinned notebook, helper/manifest upload, local staging,
confirmed `google.colab.drive.flush_and_unmount()`, then the normal upstream
runner in that same session. ComfyUI is installed **before** creating its model
directories. The revision cache root is created only after asserting the mount.
The parent watchdog releases its owned VM on failure, consent timeout or budget
expiry; no interactive browser or bespoke credentials are handled by this code.
The official mount can require the user to complete OAuth; blocked consent is a
failure, never permission to fall back to a cold download. No live OAuth is
performed while implementing/reviewing this feature.

The helper checks size and SHA256 before atomic publication and reuses valid
cache entries without HF calls. It passes `token=False` for public downloads.
Verified local weights make the pinned notebook's existing-file checks skip
all five HF weight downloads. It does not vendor or alter the upstream notebook.
The notebook still applies its own free-disk requirements when it starts; local
capacity must also accommodate those checks after weights have been staged.

Cold priming uses 44,414,604,463 bytes in Drive; local staging needs that amount
again locally. Temporary cache downloads are renamed, so no second whole cache
copy is required. Repairing an existing corrupt file retains it until its
replacement verifies; space for that replacement is needed. Warm staging still
reads/hashes/copies 44.4 GB and consumes GPU idle time. CPU priming can also be
billed. Drive quota is reported as unknown and remains a parent preflight concern.
Manual/private CPU priming is outside this public integration.

Read `cache-report.json` beside the run artifacts: separate cache-hit, download,
staging byte/counter/timing evidence is under `model_cache`. Actual transferred
HTTP bytes are unknown, and copy time overlaps verify-and-copy time. Persistence
is `flush_unmount_confirmed` only after the official flush call returns and the
mount is absent. Mount/stage/flush failure leaves persistence `unknown`; the
overall watchdog result still reports runtime cleanup independently. Offline
success does not establish live Drive durability or client acceptance.
