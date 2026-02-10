# Project Context

## Overview

CMPT 742 (Practices in Visual Computing I) — Assignment 1. This project implements two core computer vision algorithms from scratch using sparse linear algebra: image reconstruction from second-order derivatives and Poisson image blending (seamless cloning).

## Course

- **Course**: CMPT 742 — Practices in Visual Computing I
- **Institution**: Simon Fraser University
- **Assignment**: Assignment 1 — Image Reconstruction + Poisson Blending
- **Total Points**: 20

## Project Structure

```
.
├── assignment_1.pdf              # Assignment specification
├── Reconstruction/               # Part 1: Image reconstruction (10 pts)
│   ├── reconstruction.py         # Solver: builds Laplacian system Av = b
│   ├── target.jpg                # Test image (small)
│   ├── target1.jpg               # Test image (small)
│   ├── large.jpg                 # Test image (large)
│   ├── large1.jpg                # Test image (large)
│   ├── report.pdf                # Results report with comparison plots
│   └── results/                  # Output images and comparison plots
├── Poisson Blending/             # Part 2: Poisson blending (10 pts)
│   ├── main.py                   # Entry point: runs full blending pipeline
│   ├── get_mask.py               # Interactive polygon mask selection (OpenCV)
│   ├── align_target.py           # Interactive patch alignment (rotate/scale/move)
│   ├── source1.jpg               # Source patch 1 (boat)
│   ├── source2.jpg               # Source patch 2 (moon)
│   ├── target.jpg                # Target image (Shanghai skyline)
│   ├── report.pdf                # Results report with comparison plots
│   └── results/                  # Output blended images and comparison plots
```

## Technical Details

### Part 1 — Image Reconstruction

Reconstructs a grayscale image from its second-order derivatives by solving the sparse linear system `Av = b`.

**Key concepts**:
- Builds a Laplacian coefficient matrix `A` (sparse, K×K) encoding the discrete second-order derivative stencil
- Boundary handling: horizontal edges use only vertical derivatives, vertical edges use only horizontal derivatives, four corners are pinned to constant values for full rank
- Solves with `scipy.sparse.linalg.spsolve`
- Reports least-square residual `‖Av − b‖`

**Dependencies**: NumPy, OpenCV, Matplotlib, SciPy

### Part 2 — Poisson Blending

Implements seamless image cloning via the Poisson equation. The gradient field of the source patch is preserved while boundary pixels match the target image.

**Key concepts**:
- Interactive polygon mask selection (`get_mask.py`)
- Interactive patch alignment with rotation, scaling, and translation (`align_target.py`)
- Constructs Laplacian coefficient matrix for mask region
- Right-hand side vector `b` combines source Laplacian inside the mask with target pixel values at the boundary
- Solves with `scipy.sparse.linalg.spsolve`
- Reports least-square residual `‖Av − b‖`

**Dependencies**: NumPy, OpenCV, Matplotlib, SciPy

## Results Summary

### Reconstruction

| Image       | Residual Error          |
|-------------|-------------------------|
| target.jpg  | 4.092258539410570e-12   |
| target1.jpg | 3.639983595038776e-12   |
| large.jpg   | 9.558208171381165e-12   |
| large1.jpg  | 1.907675261508617e-11   |

### Poisson Blending

| Source       | Residual Error          |
|-------------|-------------------------|
| source1.jpg | 7.754151862786744e-12   |
| source2.jpg | 8.092129729494018e-12   |

All residual errors are on the order of 1e-12 to 1e-11, confirming numerical accuracy of the solver.
