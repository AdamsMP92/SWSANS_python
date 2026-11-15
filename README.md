# SWSANS

SWSANS is a tested Python implementation of randomly oriented
Stoner–Wohlfarth ensembles and their magnetic small-angle neutron scattering
(SANS) response. It connects a Landau–Lifshitz field sweep to hysteresis,
second-order magnetization correlations, two-dimensional detector maps,
azimuthally averaged intensities, pair-distance distributions, and correlation
functions.

[Open the browser app](https://adamsmp92.github.io/SWSANS_python/) ·
[Read the associated publication](https://doi.org/10.1107/S205225252300180X)

## Beta scope

The current beta includes:

- uniaxial and cubic magnetocrystalline anisotropy;
- deterministic low-discrepancy Sobol orientation sampling;
- projected Euler and projected Heun Landau–Lifshitz relaxation;
- complete triangular field cycles and convergence diagnostics;
- first moments and the full correlation tensor
  `Gamma[alpha, beta]`;
- saturation, descending remanence, and descending coercive states;
- perpendicular and parallel magnetic SANS geometries;
- `I(q)`, `p(r)`, `c(r)`, and CSV export;
- a Shiny/Shinylive interface that runs entirely in the browser.

The classical single-particle SW animation and the standalone symbolic
remanence calculator from the original MATLAB application are planned as a
second development stage. They are deliberately kept out of the first beta so
that the numerical core and browser workflow remain transparent and testable.

## Quick start

The reproducible development environment is managed by
[Pixi](https://pixi.sh/):

```bash
pixi run test
pixi run app
```

The second command serves the app at `http://127.0.0.1:8000`.

Other useful tasks are:

```bash
pixi run check
pixi run test-all
pixi run export-site
```

`test-all` includes the slower scientific reference check. The regular test
suite is intentionally lightweight for continuous integration.

## Python API

```python
from swsans import SimulationConfig, critical_states, simulate_hysteresis

result = simulate_hysteresis(
    SimulationConfig(
        k1_t=1.0,
        k2_t=0.0,
        field_max_t=2.0,
        n_orientations=256,
        n_field_points=101,
    )
)
states = critical_states(result)

print(states.remanence.mean_m[2])
print(abs(states.coercive.field_t))
```

`k1_t` and `k2_t` are the reduced anisotropy constants
`k_i = 2 K_i / M_0` in tesla. The field sweep starts at `+Bmax`, descends to
`-Bmax`, and returns to `+Bmax`.

## Numerical design

The original MATLAB app retained three dense orientation-by-field arrays.
At its defaults (`N = 10,000`, `K = 2,001`) those arrays alone occupy about
458 MiB. The Python core accumulates the first and second ensemble moments at
each field point and discards the individual orientation state after it has
been advanced. This keeps browser memory approximately proportional to `N + K`
rather than `N * K`.

Every field point has a finite iteration cap and reports its iteration count,
residual, and convergence flag. The legacy projected-Euler path remains
available for comparison with the MATLAB implementation; projected Heun gives
a second-order alternative.

The SciPy and MATLAB Sobol generators do not guarantee identical direction
numbers. Exact cross-language regression tests should therefore use stored
orientation fixtures rather than relying on equal seeds alone.

## Scientific model

The implementation follows:

> M. P. Adams, E. Pratami Sinaga and A. Michels, “On the angular anisotropy of
> the randomly averaged magnetic neutron scattering cross section of
> nanoparticles,” *IUCrJ* **10**, 261–269 (2023).
> [doi:10.1107/S205225252300180X](https://doi.org/10.1107/S205225252300180X)

The model assumes identical spherical single-domain particles, a dilute and
noninteracting ensemble, random crystal orientations, uniform magnetization
inside each particle, and no thermal fluctuations. The effective anisotropy
field is obtained from the anisotropy-energy gradient. No derivation assumes
the generally non-universal identity `B = mu_0 (H + M)`.

See [docs/model.md](docs/model.md) for the equations, coordinate conventions,
and units. See [docs/migration-notes.md](docs/migration-notes.md) for documented
differences from the MATLAB app.

## GitHub Pages

The deployment workflow exports `src/` with Shinylive and publishes the static
result to GitHub Pages. In the repository settings, configure **Pages → Source**
as **GitHub Actions**. Every push to `main` that changes the app or environment
then creates a new deployment.

## License

MIT. See [LICENSE](LICENSE).
