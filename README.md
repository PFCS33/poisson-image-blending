# Image Reconstruction & Poisson Blending

Sparse linear algebra meets image editing. This project implements two foundational computer vision algorithms from scratch — rebuilding images from their derivatives and blending patches seamlessly into new scenes.

Built for **CMPT 742: Practices in Visual Computing I** at Simon Fraser University.

---

## Part 1 — Image Reconstruction from Second-Order Derivatives

Given only the second-order derivatives of an image, can we recover the original? Yes — by formulating it as a sparse linear system `Av = b` and solving with a direct solver.

### How It Works

The discrete Laplacian at each pixel gives us one linear equation. Stack all pixels together, and we get a large sparse system. Four corner pixels are pinned to known values so the system has a unique solution.

Boundary pixels near edges use partial derivative stencils (horizontal-only or vertical-only) instead of the full 4-neighbor Laplacian. This keeps the matrix well-conditioned.

The solver uses `scipy.sparse.linalg.spsolve` on a CSR-format sparse matrix.

### Results

All reconstructions achieve near-zero residual error (order of 1e-12), confirming the solver works correctly.

<!--
  📸 IMAGE PLACEHOLDER: reconstruction_comparison.png
  Suggested image: a 2×2 grid showing original vs. reconstructed for two test images
  (e.g., the moon image and the pagoda image side by side with their reconstructions).
  You can screenshot or export from the report.pdf in Reconstruction/report.pdf
-->

| Test Image | Resolution | Residual ‖Av − b‖ |
|:----------:|:----------:|:------------------:|
| target.jpg | small | 4.09e-12 |
| target1.jpg | small | 3.64e-12 |
| large.jpg | large | 9.56e-12 |
| large1.jpg | large | 1.91e-11 |

### Usage

```bash
cd Reconstruction
python reconstruction.py
```

Configure via variables at the top of `reconstruction.py`:

- `SOURCE_PATH` — input image
- `AUTO_CORNER` — use original corner values (`True`) or set manually (`False`)
- `CORNER_VALUES` — manual corner intensities when `AUTO_CORNER=False`

---

## Part 2 — Poisson Image Blending (Seamless Cloning)

Poisson blending lets you paste a patch from one image into another while making the seam invisible. Instead of copying raw pixels, it preserves the gradient field of the source inside the mask region and forces boundary pixels to match the target. The result: the patch adopts the target's lighting and color naturally.

### How It Works

1. **Select a mask** — draw a polygon around the region of interest in the source image
2. **Align the patch** — position, rotate, and scale it over the target image interactively
3. **Solve the Poisson equation** — build the Laplacian system for masked pixels, set boundary conditions from the target, and solve with `spsolve`

The key insight: by matching gradients rather than raw pixel values, the blended region inherits the illumination of its surroundings.

### Results

Two experiments blend different source patches into a Shanghai night skyline photo. Both achieve near-zero solver residuals.

<!--
  📸 IMAGE PLACEHOLDER: poisson_blending_result.png
  Suggested image: a side-by-side comparison showing:
  Left — the source image with the selected patch highlighted
  Right — the blended result in the target scene
  You can export from Poisson Blending/report.pdf or use the images in results/
-->

| Source | Blended Into | Residual ‖Av − b‖ |
|:------:|:------------:|:------------------:|
| source1.jpg (boat) | Shanghai skyline | 7.75e-12 |
| source2.jpg (moon) | Shanghai skyline | 8.09e-12 |

### Usage

```bash
cd "Poisson Blending"
python main.py
```

Interactive controls during alignment:

| Key | Action |
|:---:|:------:|
| Click | Draw mask polygon vertices |
| Arrow keys | Move patch |
| `r` | Rotate +10° |
| `s` | Scale up |
| `a` | Scale down |
| `q` | Confirm and blend |

Configure `SOURCE_PATH` and `TARGET_PATH` at the top of `main.py`.

---

## Tech Stack

- **Python 3** — core language
- **NumPy** — array operations and linear algebra
- **SciPy** — sparse matrix construction (`lil_matrix`, `csr_matrix`) and direct solving (`spsolve`)
- **OpenCV** — image I/O, Laplacian computation, interactive GUI (mouse callbacks, keyboard events)
- **Matplotlib** — result visualization and comparison plots

## Project Structure

```
├── assignment_1.pdf                        # Assignment specification
├── Reconstruction/
│   ├── reconstruction.py                   # Laplacian system solver
│   ├── report.pdf                          # Results report
│   ├── *.jpg                               # Test images
│   └── results/                            # Reconstructed outputs
├── Poisson Blending/
│   ├── main.py                             # Blending pipeline entry point
│   ├── get_mask.py                         # Interactive mask selection
│   ├── align_target.py                     # Interactive patch alignment
│   ├── report.pdf                          # Results report
│   ├── *.jpg                               # Source and target images
│   └── results/                            # Blended outputs
└── README.md
```

## What I Learned

- Formulating image processing problems as sparse linear systems
- Building and solving large sparse matrices efficiently with SciPy
- The mathematics behind Poisson editing — why gradient-domain methods produce seamless blends
- Handling boundary conditions correctly in discrete PDE solvers
- Interactive OpenCV GUI for mask selection and geometric alignment

## License

This project was completed as coursework for CMPT 742 at SFU. Feel free to reference it for learning purposes.
