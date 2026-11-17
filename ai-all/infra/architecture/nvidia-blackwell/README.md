# NVIDIA Blackwell (B200): hardware hierarchy × CUDA / PTX

[`blackwell-topology.html`](blackwell-topology.html) is a self-contained, interactive diagram of the data-center Blackwell B200 (sm_100a). Five nested panels zoom from the package down to a single thread:

| Panel | Hardware level | Software level |
|---|---|---|
| A | Package: two dies, NV-HBI, 8 GPCs, L2, HBM3e, NVLink / PCIe / C2C | Grid |
| B | One GPC: 10 TPCs, 20 SMs, SM-to-SM network | Thread block cluster |
| C | One SM: 4 SMSPs, L1 / SMEM split by CTA, TMEM split by lane quarter, TMA | CTA |
| D | One SMSP: warp scheduler, register file, FP32 / FP64 / SFU / LD-ST, Tensor Core | Warp |
| E | One warp: 32 lanes with their registers, active mask, shuffle network and TMEM → register path | Thread |

Solid blocks are compute units, striped blocks are memory, and thinner links carry more bandwidth. Eighteen numbered cards around the diagram pair the most common CUDA C++ at each level with its PTX equivalent, from kernel launch and clusters to TMA, `tcgen05.alloc` / `.mma` / `.ld`, warp shuffles and registers.

## Viewing

GitHub shows HTML files as source, so download the file (or clone the repo) and open it in a browser. The page loads its fonts from Google Fonts and otherwise works offline.

- **Fit width** and **1:1** switch between the overview and full size; Ctrl / ⌘ + wheel zooms, and dragging pans.
- Hover a card or a numbered marker to highlight its hardware block; click either one to jump to the other.

## Scope and caveats

- Targets data-center Blackwell (B200, sm_100a). Consumer Blackwell (RTX 50 series, sm_120) has no TMEM and no `tcgen05` instructions.
- Block placement inside each die, which SMs are fused off, and the example kernel and cluster assignment are illustrative. The per-SMSP FP64 ×16 and SFU ×4 counts are inferred from published throughput.
- Sources are listed in the page footer.
