# Scientific model

## Scope

SWSANS represents a dilute ensemble of identical, spherical, uniformly
magnetized single-domain particles. The particles do not interact, their
crystal frames are randomly oriented, and thermal fluctuations are neglected.
The applied field points along the laboratory `z` axis.

These assumptions define a Stoner–Wohlfarth macrospin model. They do not
describe inhomogeneous internal spin textures, dense-particle structure
factors, interparticle dipolar coupling, or finite-temperature switching.

## Energy and effective field

For a particle with saturation magnetization `M_0`, reduced magnetization
`m`, and applied induction `B`, the energy density is

```math
\omega(\mathbf m)=-M_0\,\mathbf m\cdot\mathbf B+\omega_{\mathrm{ani}}(\mathbf m).
```

In the local particle frame, the uniaxial energy is

```math
\omega_{\mathrm{ani,u}}=K_{\mathrm{u}1}(1-m_z^2)
+K_{\mathrm{u}2}(1-m_z^2)^2,
```

and the cubic energy is

```math
\omega_{\mathrm{ani,c}}=K_{\mathrm{c}1}
(m_x^2m_y^2+m_x^2m_z^2+m_y^2m_z^2)
+K_{\mathrm{c}2}m_x^2m_y^2m_z^2.
```

The anisotropy field is obtained directly from the energy gradient,

```math
\mathbf B_{\mathrm{ani}}=-\frac{1}{M_0}
\frac{\partial\omega_{\mathrm{ani}}}{\partial\mathbf m}.
```

The user-facing constants are `k_i = 2 K_i / M_0` in tesla. Random particle
orientations are produced from a three-dimensional Sobol sequence and a
uniform `z-y-z` Euler-angle construction.

## Landau–Lifshitz relaxation

At every field value, the static state is approached through

```math
\frac{d\mathbf m}{dt}
=-\gamma_{\mathrm G}\,\mathbf m\times\mathbf B_{\mathrm{eff}}
-\eta\,\mathbf m\times
(\mathbf m\times\mathbf B_{\mathrm{eff}}).
```

The numerical state is projected onto `|m| = 1` after every Euler or Heun
step. Convergence is assessed from the maximum squared rate over the whole
orientation ensemble. A finite iteration cap prevents an unbounded loop and
is part of every result's diagnostics.

## Ensemble observables

At field index `k`, the first and second moments are

```math
\overline{\mathbf m}^{\,k}=\frac{1}{N}\sum_i\mathbf m^{i,k},
\qquad
\Gamma_{\alpha\beta}^{k}=\frac{1}{N}\sum_i
m_{\alpha}^{i,k}m_{\beta}^{i,k}.
```

Since every macrospin has unit length,

```math
\operatorname{tr}\boldsymbol\Gamma
=\Gamma_{xx}+\Gamma_{yy}+\Gamma_{zz}=1.
```

The hysteresis loop is described by the first moments. Magnetic SANS depends
on the second moments and therefore retains information that is absent from
the ensemble-averaged magnetization alone.

## Magnetic SANS

For a spherical particle of radius `R`,

```math
C(q)=\frac{16\pi^2R^6M_0^2b_H^2}{V}
\left[\frac{j_1(qR)}{qR}\right]^2,
\qquad b_H=2.91\times10^8\,\mathrm{A^{-1}m^{-1}}.
```

For `B` perpendicular to the incident neutron beam,

```math
\frac{d\Sigma_{M,\perp}}{d\Omega}=C(q)
\left(\Gamma_{xx}+\Gamma_{yy}\cos^2\theta
+\Gamma_{zz}\sin^2\theta
-2\Gamma_{yz}\sin\theta\cos\theta\right).
```

The detector-plane convention is
`q = (0, q_y, q_z) = q(0, sin(theta), cos(theta))`. The two-dimensional map
therefore displays `q_z` horizontally and `q_y` vertically.

For `B` parallel to the incident neutron beam,

```math
\frac{d\Sigma_{M,\parallel}}{d\Omega}=C(q)
\left(\Gamma_{xx}\sin^2\theta+\Gamma_{yy}\cos^2\theta
+\Gamma_{zz}-2\Gamma_{xy}\sin\theta\cos\theta\right).
```

Here the convention is
`q = (q_x, q_y, 0) = q(cos(theta), sin(theta), 0)`, with `q_x` horizontal and
`q_y` vertical on the detector map.

The code also evaluates the analytic azimuthal averages and the spherical
particle expressions for `p(r)` and `c(r) = p(r) / r^2` for the selected
cross section.

### Polarization-averaged spin-flip cross section

The app also provides the polarization-independent average of the two
spin-flip channels,

```math
\frac{d\Sigma_{\mathrm{sf}}}{d\Omega}
\equiv \frac{1}{2}\left(
\frac{d\Sigma^{+-}}{d\Omega}
+\frac{d\Sigma^{-+}}{d\Omega}\right).
```

The average removes the chiral term that changes sign between the two
channels. In the perpendicular geometry, with the field and polarization
axis along laboratory `z`, it is

```math
\frac{d\Sigma_{\mathrm{sf},\perp}}{d\Omega}=C(q)
\left(\Gamma_{xx}+\Gamma_{yy}\cos^4\theta
+\Gamma_{zz}\sin^2\theta\cos^2\theta
-2\Gamma_{yz}\sin\theta\cos^3\theta\right).
```

For the app's parallel convention `B || k_0 || z`, it is

```math
\frac{d\Sigma_{\mathrm{sf},\parallel}}{d\Omega}=C(q)
\left(\Gamma_{xx}\sin^2\theta+\Gamma_{yy}\cos^2\theta
-2\Gamma_{xy}\sin\theta\cos\theta\right).
```

The NuMagSANS reference writes the parallel expression for
`B || k_0 || x`. Exchanging its `x` and `z` axes gives the formula above and
keeps the SWSANS field direction fixed along `z`. Consequently,
`Gamma_zz` is longitudinal and does not contribute to the parallel
spin-flip average.

The corresponding analytic azimuthal moments are

```math
\left\langle A_{\mathrm{sf},\perp}\right\rangle_\theta
=\Gamma_{xx}+\frac{3}{8}\Gamma_{yy}+\frac{1}{8}\Gamma_{zz},
\qquad
\left\langle A_{\mathrm{sf},\parallel}\right\rangle_\theta
=\frac{1}{2}(\Gamma_{xx}+\Gamma_{yy}).
```

Only the magnetic, polarization-independent spin-flip contribution is
included. Nuclear spin-incoherent background, finite beam polarization,
polarization efficiency, and leakage corrections are outside this model.

## Magnetostatic convention

The model is formulated from the particle energy and its effective-field
gradient. It does not require the generally non-universal starting assumption
`B = mu_0 (H + M)`. It also makes no unique mathematical identification of
free versus bound currents; those concepts are outside the scope of this
macrospin/SANS calculation.

## Reference

M. P. Adams, E. Pratami Sinaga and A. Michels, *IUCrJ* **10**, 261–269
(2023), [doi:10.1107/S205225252300180X](https://doi.org/10.1107/S205225252300180X).

The polarized-SANS convention follows M. P. Adams and A. Michels,
*J. Appl. Cryst.* **59**, 943–959 (2026),
[doi:10.1107/S160057672600258X](https://doi.org/10.1107/S160057672600258X),
especially equations (20) and (22).

