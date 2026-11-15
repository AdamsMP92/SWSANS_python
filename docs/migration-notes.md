# MATLAB-to-Python migration notes

The Python implementation is based on the numerical model in the original
`SWSANS.mlapp`, while deliberately separating the scientific core from the
graphical interface.

## Preserved behavior

- The triangular `+Bmax -> -Bmax -> +Bmax` field path.
- The Sobol-to-Euler-angle transformation and legacy rotation coefficients.
- Uniaxial and cubic anisotropy-field formulas.
- Projected explicit Euler Landau–Lifshitz relaxation.
- First moments, all six independent second moments, and both SANS geometries.
- Analytic sphere expressions for `I(q)`, `p(r)`, and `c(r)`.

## Intentional changes

1. **Bounded convergence:** each field point has a maximum iteration count and
   retains diagnostics. The MATLAB `while` loop had no upper bound.
2. **Streaming storage:** individual `N x K` magnetization trajectories are no
   longer retained by default. This reduces browser memory dramatically.
3. **Determinism:** the initial azimuth has a configurable random seed.
4. **Explicit physical inputs:** `M_0`, particle radius, and detector range are
   no longer hidden plotting constants.
5. **Second-order option:** projected Heun is available alongside the legacy
   Euler path.
6. **Remanent cross-correlation correction:** the MATLAB assignment
   `Gxy_rem = Gyz_rem` is treated as a transcription error. Python retains and
   interpolates `Gamma_xy` independently.
7. **Mixed anisotropy extension:** Python can add uniaxial and cubic energy
   terms in a shared local particle frame, with the uniaxial axis aligned to
   cubic `[001]`. The original pure modes and their parameter meanings remain
   unchanged.

SciPy and MATLAB may generate different Sobol direction numbers. Stored
orientation arrays are therefore the preferred basis for exact cross-language
regression fixtures.

