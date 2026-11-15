"""Browser-native Shiny interface for the SWSANS scientific core."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui

from swsans import (
    Anisotropy,
    CriticalState,
    CriticalStates,
    HysteresisResult,
    InputValidationError,
    Integrator,
    SANSConfig,
    SimulationConfig,
    critical_states,
    field_point_state,
    pair_distance_curves,
    sans_intensity_curve,
    sans_map,
    simulate_hysteresis,
)

APP_DIR = Path(__file__).resolve().parent

PRESETS: dict[str, dict[str, str | float]] = {
    "uniaxial_axis": {
        "anisotropy": "uniaxial",
        "k1": 1.0,
        "k2": 0.0,
        "field_max": 2.0,
    },
    "uniaxial_plane": {
        "anisotropy": "uniaxial",
        "k1": -1.0,
        "k2": 0.0,
        "field_max": 2.0,
    },
    "uniaxial_cone": {
        "anisotropy": "uniaxial",
        "k1": -0.5,
        "k2": 0.5,
        "field_max": 2.0,
    },
    "cubic_100": {
        "anisotropy": "cubic",
        "k1": 1.0,
        "k2": 0.0,
        "field_max": 2.0,
    },
    "cubic_111": {
        "anisotropy": "cubic",
        "k1": -1.0,
        "k2": 0.0,
        "field_max": 2.0,
    },
    "cubic_110": {
        "anisotropy": "cubic",
        "k1": -1.0,
        "k2": 9.0,
        "field_max": 4.0,
    },
}


def _descending_zero_index(result: HysteresisResult) -> int:
    turning_index = int(np.argmin(result.field_t))
    return int(np.argmin(np.abs(result.field_t[: turning_index + 1])))


def _field_point_choices(result: HysteresisResult) -> dict[str, str]:
    total = result.field_t.size
    choices: dict[str, str] = {}
    for index, field in enumerate(result.field_t):
        branch = field_point_state(result, index).name
        choices[str(index)] = (
            f"{index + 1:03d}/{total:03d} · {branch} · B = {field:+.4f} T"
        )
    return choices


def _numeric_field(
    input_id: str,
    label: str,
    value: float,
    *,
    step: float,
    note: str,
    minimum: float | None = None,
) -> ui.Tag:
    return ui.div(
        ui.input_numeric(
            input_id,
            label,
            value,
            min=minimum,
            step=step,
            update_on="change",
        ),
        ui.tags.small(note, class_="field-note"),
        class_="field-block",
    )


def _metric_card(label: str, value: str, detail: str, accent: str) -> ui.Tag:
    return ui.div(
        ui.div(
            ui.tags.span(label, class_="metric-label"),
            ui.tags.span(class_=f"metric-dot metric-dot-{accent}"),
            class_="metric-heading",
        ),
        ui.tags.strong(value, class_="metric-value"),
        ui.tags.p(detail, class_="metric-detail"),
        class_=f"metric-card metric-card-{accent}",
    )


def _equation(latex: str, caption: str) -> ui.Tag:
    return ui.div(
        ui.HTML(rf"\[{latex}\]"),
        ui.tags.p(caption),
        class_="equation-block",
    )


controls = ui.div(
    ui.div(
        ui.tags.span("ENSEMBLE MODEL", class_="eyebrow"),
        ui.h2("Simulation inputs"),
        ui.tags.p(
            "Choose an anisotropy symmetry and relax a random ensemble along "
            "a complete field cycle.",
            class_="section-intro",
        ),
        class_="panel-heading",
    ),
    ui.div(
        ui.input_select(
            "preset",
            "Published model preset",
            {
                "uniaxial_axis": "Uniaxial · easy axis",
                "uniaxial_plane": "Uniaxial · easy plane",
                "uniaxial_cone": "Uniaxial · easy cone",
                "cubic_100": "Cubic · <100> easy axes",
                "cubic_111": "Cubic · <111> easy axes",
                "cubic_110": "Cubic · <110> easy axes",
            },
            selected="uniaxial_axis",
        ),
        ui.input_select(
            "anisotropy",
            "Anisotropy symmetry",
            {"uniaxial": "Uniaxial", "cubic": "Cubic"},
            selected="uniaxial",
        ),
        class_="field-grid field-grid-one",
    ),
    ui.div(
        _numeric_field(
            "k1",
            "First reduced constant k₁ [T]",
            1.0,
            step=0.1,
            note="k₁ = 2K₁/M₀",
        ),
        _numeric_field(
            "k2",
            "Second reduced constant k₂ [T]",
            0.0,
            step=0.1,
            note="k₂ = 2K₂/M₀",
        ),
        _numeric_field(
            "field_max",
            "Maximum applied field [T]",
            2.0,
            step=0.1,
            minimum=0.01,
            note="Sweep: +Bmax → −Bmax → +Bmax",
        ),
        class_="field-grid field-grid-one",
    ),
    ui.div(class_="panel-divider"),
    ui.div(
        ui.tags.span("BROWSER PROFILE", class_="eyebrow"),
        ui.div(
            ui.input_select(
                "orientation_count",
                "Particle orientations",
                {
                    "64": "64 · preview",
                    "128": "128 · fast",
                    "256": "256 · balanced",
                    "512": "512 · detailed",
                    "1024": "1024 · local recommended",
                },
                selected="128",
            ),
            _numeric_field(
                "field_points",
                "Field points",
                81,
                step=4,
                minimum=5,
                note="Odd values; 4m + 1 includes both zero fields",
            ),
            class_="field-grid field-grid-one",
        ),
    ),
    ui.accordion(
        ui.accordion_panel(
            "Advanced relaxation controls",
            ui.div(
                ui.input_select(
                    "integrator",
                    "Integrator",
                    {
                        "legacy_euler": "Projected Euler · MATLAB compatible",
                        "heun": "Projected Heun · second order",
                    },
                    selected="legacy_euler",
                ),
                _numeric_field(
                    "gamma",
                    "Reduced gyromagnetic factor",
                    1.0,
                    step=0.1,
                    minimum=0.0,
                    note="Precession weight",
                ),
                _numeric_field(
                    "damping",
                    "Reduced damping",
                    6.0,
                    step=0.5,
                    minimum=0.01,
                    note="Dissipative relaxation weight",
                ),
                _numeric_field(
                    "time_step",
                    "Reduced time step",
                    0.01,
                    step=0.001,
                    minimum=0.0001,
                    note="Original MATLAB value: 0.005",
                ),
                _numeric_field(
                    "tolerance",
                    "Squared-rate tolerance",
                    1.0e-8,
                    step=1.0e-8,
                    minimum=1.0e-14,
                    note="Maximum over all orientations",
                ),
                _numeric_field(
                    "max_iterations",
                    "Maximum iterations per field",
                    5000,
                    step=500,
                    minimum=100,
                    note="Prevents unbounded relaxation loops",
                ),
                class_="advanced-grid field-grid-one",
            ),
        ),
        open=False,
        class_="advanced-accordion",
    ),
    ui.input_action_button(
        "run",
        "Run SW–SANS simulation",
        class_="run-button",
    ),
    ui.output_ui("calculation_message"),
    class_="control-card",
)

results = ui.div(
    ui.output_ui("metric_cards"),
    ui.div(
        ui.div(
            ui.div(
                ui.tags.span("MAGNETIZATION", class_="eyebrow"),
                ui.h3("Hysteresis loop"),
                class_="plot-heading",
            ),
            ui.output_plot("hysteresis_plot", height="390px"),
            class_="plot-card",
        ),
        ui.div(
            ui.div(
                ui.tags.span("SECOND MOMENTS", class_="eyebrow"),
                ui.h3("Correlation tensor & anisotropy"),
                class_="plot-heading",
            ),
            ui.output_plot("moments_plot", height="390px"),
            class_="plot-card",
        ),
        class_="plot-grid",
    ),
    ui.div(
        ui.div(
            ui.tags.span("MAGNETIC SANS", class_="eyebrow"),
            ui.h2("Detector response"),
            ui.tags.p(
                "Evaluate any calculated field point for a dilute ensemble "
                "of uniformly magnetized spherical particles.",
                class_="section-intro",
            ),
            class_="panel-heading sans-heading",
        ),
        ui.div(
            ui.input_radio_buttons(
                "cross_section",
                "Magnetic cross section",
                {
                    "unpolarized": "Unpolarized",
                    "spin_flip": "Spin-flip average",
                },
                selected="unpolarized",
                inline=True,
            ),
            ui.tags.p(
                "Choose the unpolarized magnetic signal or "
                "½(Σ⁺⁻ + Σ⁻⁺), for which the chiral difference cancels.",
                class_="cross-section-note",
            ),
            class_="cross-section-switch",
        ),
        ui.div(
            ui.input_select(
                "sans_state",
                "Calculated field point",
                {"auto": "Descending zero-field point"},
                selected="auto",
            ),
            ui.input_select(
                "geometry",
                "Scattering geometry",
                {
                    "perpendicular": "B ⟂ k₀",
                    "parallel": "B ∥ k₀",
                },
                selected="perpendicular",
            ),
            ui.input_select(
                "reciprocal_scale",
                "Reciprocal-space scaling",
                {"log": "Logarithmic", "linear": "Linear"},
                selected="log",
            ),
            _numeric_field(
                "radius_nm",
                "Particle radius R [nm]",
                5.0,
                step=0.5,
                minimum=0.1,
                note="Uniform sphere form factor",
            ),
            _numeric_field(
                "magnetization",
                "Saturation magnetization M₀ [kA/m]",
                310.0,
                step=10.0,
                minimum=0.1,
                note="Converted internally to A/m",
            ),
            _numeric_field(
                "q_max",
                "Detector qmax [nm⁻¹]",
                1.0,
                step=0.1,
                minimum=0.01,
                note="Symmetric detector extent",
            ),
            class_="sans-controls",
        ),
        ui.div(
            ui.div(
                ui.output_plot("sans_map_plot", height="450px"),
                class_="plot-card plot-card-flat",
            ),
            ui.div(
                ui.output_plot("sans_curves_plot", height="450px"),
                class_="plot-card plot-card-flat",
            ),
            class_="plot-grid sans-plot-grid",
        ),
        class_="sans-card",
    ),
    ui.div(
        ui.div(
            ui.tags.span("NUMERICS", class_="eyebrow"),
            ui.h3("Convergence diagnostics"),
            ui.output_plot("convergence_plot", height="300px"),
            class_="diagnostic-card",
        ),
        ui.div(
            ui.tags.span("CRITICAL STATES", class_="eyebrow"),
            ui.h3("State summary"),
            ui.output_ui("critical_table"),
            ui.download_button(
                "download_csv",
                "Download complete field sweep",
                class_="download-button",
            ),
            class_="diagnostic-card",
        ),
        class_="diagnostic-grid",
    ),
    class_="results-column",
)

simulator_panel = ui.div(controls, results, class_="simulator-grid")

documentation_panel = ui.div(
    ui.div(
        ui.tags.span("MODEL & DOCUMENTATION", class_="eyebrow"),
        ui.h2("From a random macrospin ensemble to magnetic SANS"),
        ui.tags.p(
            "The implementation follows the Stoner–Wohlfarth and "
            "Landau–Lifshitz model used in the associated IUCrJ article. "
            "The polarized extension follows the NuMagSANS convention. "
            "The equations below define the conventions used by the code.",
            class_="docs-lead",
        ),
        class_="docs-hero",
    ),
    ui.div(
        ui.tags.section(
            ui.tags.span("01", class_="docs-index"),
            ui.h3("Scope and assumptions"),
            ui.tags.ul(
                ui.tags.li("Identical, spherical, uniformly magnetized particles."),
                ui.tags.li("Dilute, noninteracting ensemble with random crystal axes."),
                ui.tags.li("No thermal fluctuations or particle-size distribution."),
                ui.tags.li("Applied field along the laboratory z axis."),
            ),
            class_="docs-section",
        ),
        ui.tags.section(
            ui.tags.span("02", class_="docs-index"),
            ui.h3("Anisotropy energy"),
            _equation(
                r"\omega_{\mathrm{u}}=K_{\mathrm{u}1}(1-m_z^2)+"
                r"K_{\mathrm{u}2}(1-m_z^2)^2",
                "Uniaxial anisotropy in the particle frame.",
            ),
            _equation(
                r"\omega_{\mathrm{c}}=K_{\mathrm{c}1}(m_x^2m_y^2+"
                r"m_x^2m_z^2+m_y^2m_z^2)+K_{\mathrm{c}2}m_x^2m_y^2m_z^2",
                "Cubic anisotropy in the particle frame.",
            ),
            class_="docs-section",
        ),
        ui.tags.section(
            ui.tags.span("03", class_="docs-index"),
            ui.h3("Static states from Landau–Lifshitz relaxation"),
            _equation(
                r"\frac{d\mathbf m}{dt}=-\gamma_{\mathrm G}\,\mathbf m"
                r"\times\mathbf B_{\mathrm{eff}}-\eta\,\mathbf m\times"
                r"(\mathbf m\times\mathbf B_{\mathrm{eff}})",
                "Every time step is projected back onto |m| = 1.",
            ),
            ui.tags.p(
                "The effective anisotropy field is obtained from the energy "
                "gradient and rotated between the particle and laboratory frames."
            ),
            class_="docs-section",
        ),
        ui.tags.section(
            ui.tags.span("04", class_="docs-index"),
            ui.h3("Ensemble moments"),
            _equation(
                r"\overline{\mathbf m}^{\,k}=\frac{1}{N}\sum_i\mathbf m^{i,k},"
                r"\qquad \Gamma_{\alpha\beta}^{k}=\frac{1}{N}\sum_i"
                r"m_{\alpha}^{i,k}m_{\beta}^{i,k}",
                "The hysteresis loop uses first moments; SANS uses second moments.",
            ),
            class_="docs-section",
        ),
        ui.tags.section(
            ui.tags.span("05", class_="docs-index"),
            ui.h3("Perpendicular magnetic SANS"),
            _equation(
                r"\frac{d\Sigma_{M,\perp}}{d\Omega}=C(q)\left[\Gamma_{xx}+"
                r"\Gamma_{yy}\cos^2\theta+\Gamma_{zz}\sin^2\theta-"
                r"2\Gamma_{yz}\sin\theta\cos\theta\right]",
                "Here qᵧ = q sin θ and q_z = q cos θ; the detector map uses "
                "q_z horizontally and qᵧ vertically.",
            ),
            class_="docs-section",
        ),
        ui.tags.section(
            ui.tags.span("06", class_="docs-index"),
            ui.h3("Parallel magnetic SANS"),
            _equation(
                r"\frac{d\Sigma_{M,\parallel}}{d\Omega}=C(q)\left["
                r"\Gamma_{xx}\sin^2\theta+\Gamma_{yy}\cos^2\theta+"
                r"\Gamma_{zz}-2\Gamma_{xy}\sin\theta\cos\theta\right]",
                "Here qₓ = q cos θ and qᵧ = q sin θ; the detector map uses "
                "qₓ horizontally and qᵧ vertically.",
            ),
            class_="docs-section",
        ),
        ui.tags.section(
            ui.tags.span("07", class_="docs-index"),
            ui.h3("Polarization-averaged spin-flip SANS"),
            _equation(
                r"\frac{d\Sigma_{\mathrm{sf}}}{d\Omega}\equiv\frac{1}{2}"
                r"\left(\frac{d\Sigma^{+-}}{d\Omega}+"
                r"\frac{d\Sigma^{-+}}{d\Omega}\right)",
                "Averaging the two spin-flip channels removes the chiral term.",
            ),
            _equation(
                r"\frac{d\Sigma_{\mathrm{sf},\perp}}{d\Omega}=C(q)"
                r"\left[\Gamma_{xx}+\Gamma_{yy}\cos^4\theta+"
                r"\Gamma_{zz}\sin^2\theta\cos^2\theta-"
                r"2\Gamma_{yz}\sin\theta\cos^3\theta\right]",
                "Perpendicular geometry with B₀ and the polarization axis along z.",
            ),
            _equation(
                r"\frac{d\Sigma_{\mathrm{sf},\parallel}}{d\Omega}=C(q)"
                r"\left[\Gamma_{xx}\sin^2\theta+"
                r"\Gamma_{yy}\cos^2\theta-"
                r"2\Gamma_{xy}\sin\theta\cos\theta\right]",
                "NuMagSANS uses B₀ ∥ k₀ ∥ x for this geometry; the expression "
                "shown here is rotated into the app's fixed B₀ ∥ z convention.",
            ),
            ui.tags.p(
                "Only the magnetic polarization-independent spin-flip term is "
                "included. Nuclear spin-incoherent background, finite polarization, "
                "and spin leakage are outside this model."
            ),
            class_="docs-section docs-section-wide",
        ),
    ),
    ui.div(
        ui.tags.span("REFERENCE", class_="eyebrow"),
        ui.h3("Model references"),
        ui.tags.p(
            "M. P. Adams, E. Pratami Sinaga and A. Michels, IUCrJ 10, 261–269 (2023)."
        ),
        ui.tags.a(
            "Open DOI 10.1107/S205225252300180X",
            href="https://doi.org/10.1107/S205225252300180X",
            target="_blank",
            rel="noopener noreferrer",
            class_="source-link",
        ),
        ui.tags.p(
            "Spin-flip convention: M. P. Adams and A. Michels, "
            "J. Appl. Cryst. 59, 943–959 (2026).",
            class_="source-detail",
        ),
        ui.tags.a(
            "Open NuMagSANS reference",
            href="https://doi.org/10.1107/S160057672600258X",
            target="_blank",
            rel="noopener noreferrer",
            class_="source-link source-link-secondary",
        ),
        ui.tags.a(
            "View source code",
            href="https://github.com/AdamsMP92/SWSANS_python",
            target="_blank",
            rel="noopener noreferrer",
            class_="source-link source-link-secondary",
        ),
        class_="source-card",
    ),
    class_="documentation-panel",
)

app_ui = ui.page_bootstrap(
    ui.head_content(
        ui.tags.meta(
            name="description",
            content=(
                "Interactive Stoner-Wohlfarth ensemble and magnetic SANS simulator."
            ),
        ),
        ui.tags.link(rel="icon", type="image/svg+xml", href="magnet.svg"),
        ui.tags.link(rel="preconnect", href="https://fonts.googleapis.com"),
        ui.tags.link(
            rel="preconnect",
            href="https://fonts.gstatic.com",
            crossorigin="anonymous",
        ),
        ui.tags.link(
            rel="stylesheet",
            href=(
                "https://fonts.googleapis.com/css2?"
                "family=DM+Sans:wght@400;500;600;700&"
                "family=Newsreader:opsz,wght@6..72,600&display=swap"
            ),
        ),
        ui.tags.script(
            id="MathJax-script",
            src="https://cdn.jsdelivr.net/npm/mathjax@4.1.3/tex-chtml.js",
            defer=True,
        ),
    ),
    ui.include_css(APP_DIR / "www" / "styles.css"),
    ui.div(
        ui.div(
            ui.tags.header(
                ui.div(
                    ui.div(
                        ui.img(
                            src="magnet.svg",
                            alt="Red horseshoe magnet",
                            class_="hero-mark",
                        ),
                        ui.div(
                            ui.tags.span(
                                "MICHAEL P. ADAMS · SCIENTIFIC SOFTWARE",
                                class_="hero-kicker",
                            ),
                            ui.h1("SWSANS"),
                            ui.tags.p(
                                "Connect Stoner–Wohlfarth reversal, ensemble "
                                "correlations, and the magnetic SANS detector response."
                            ),
                        ),
                        class_="hero-copy",
                    ),
                    ui.div(
                        ui.tags.span(class_="runtime-dot"),
                        ui.div(
                            ui.tags.strong("Browser-native Python"),
                            ui.tags.small("Shinylive · no calculation server"),
                        ),
                        class_="runtime-badge",
                    ),
                    class_="hero-inner",
                ),
                class_="app-hero",
            ),
            ui.div(
                ui.navset_card_underline(
                    ui.nav_panel("Simulator", simulator_panel, value="simulator"),
                    ui.nav_panel(
                        "Model & documentation",
                        documentation_panel,
                        value="documentation",
                    ),
                    id="main_navigation",
                    selected="simulator",
                ),
                class_="main-nav-wrap",
            ),
            ui.tags.footer(
                ui.tags.span("SWSANS"),
                ui.tags.span("·"),
                ui.tags.span("Michael P. Adams"),
                ui.tags.span("·"),
                ui.tags.a(
                    "Source code",
                    href="https://github.com/AdamsMP92/SWSANS_python",
                    target="_blank",
                    rel="noopener noreferrer",
                ),
                class_="app-footer",
            ),
            class_="app-shell",
        ),
        class_="container-fluid app-page",
    ),
    title="SWSANS",
    lang="en",
)


def _style_axes(axis: plt.Axes) -> None:
    axis.set_facecolor("#ffffff")
    axis.grid(True, color="#e8eaf3", linewidth=0.8)
    axis.tick_params(colors="#667085", labelsize=9)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#cfd4e3")


def server(input: Inputs, output: Outputs, session: Session) -> None:
    @reactive.effect
    @reactive.event(input.preset)
    def _apply_preset() -> None:
        preset = PRESETS[str(input.preset())]
        ui.update_select("anisotropy", selected=str(preset["anisotropy"]))
        ui.update_numeric("k1", value=float(preset["k1"]))
        ui.update_numeric("k2", value=float(preset["k2"]))
        ui.update_numeric("field_max", value=float(preset["field_max"]))

    def _collect_config() -> SimulationConfig:
        return SimulationConfig(
            anisotropy=Anisotropy(str(input.anisotropy())),
            k1_t=float(input.k1()),
            k2_t=float(input.k2()),
            field_max_t=float(input.field_max()),
            n_orientations=int(input.orientation_count()),
            n_field_points=int(input.field_points()),
            gyromagnetic_factor=float(input.gamma()),
            damping=float(input.damping()),
            time_step=float(input.time_step()),
            tolerance=float(input.tolerance()),
            max_iterations=int(input.max_iterations()),
            integrator=Integrator(str(input.integrator())),
        )

    def _collect_sans_config() -> SANSConfig:
        return SANSConfig(
            radius_nm=float(input.radius_nm()),
            saturation_magnetization_a_per_m=float(input.magnetization()) * 1.0e3,
            q_max_per_nm=float(input.q_max()),
        )

    @reactive.calc
    @reactive.event(input.run, ignore_none=False)
    def calculation() -> tuple[HysteresisResult | None, str | None]:
        try:
            config = _collect_config()
            with ui.Progress(min=0, max=1) as progress:
                progress.set(
                    0.0,
                    message="Relaxing Stoner–Wohlfarth ensemble",
                    detail="Preparing Sobol orientations…",
                )

                def update(done: int, total: int) -> None:
                    stride = max(1, total // 20)
                    if done == total or done % stride == 0:
                        progress.set(
                            done / total,
                            detail=f"Field point {done} of {total}",
                        )

                result = simulate_hysteresis(config, progress=update)
                progress.set(1.0, detail="Simulation complete")
            return result, None
        except (InputValidationError, ArithmeticError, ValueError) as exc:
            return None, str(exc)
        except Exception as exc:
            return None, f"Numerical simulation failed: {exc}"

    def _result() -> HysteresisResult:
        result, _ = calculation()
        req(result is not None)
        assert result is not None
        return result

    @reactive.calc
    def states() -> CriticalStates:
        return critical_states(_result())

    @reactive.effect
    def _update_sans_field_choices() -> None:
        result = _result()
        ui.update_select(
            "sans_state",
            choices=_field_point_choices(result),
            selected=str(_descending_zero_index(result)),
        )

    @reactive.calc
    def selected_state() -> tuple[int, CriticalState]:
        result = _result()
        raw_index = str(input.sans_state())
        try:
            index = int(raw_index)
        except ValueError:
            index = _descending_zero_index(result)
        if index < 0 or index >= result.field_t.size:
            index = _descending_zero_index(result)
        return index, field_point_state(result, index)

    @output
    @render.ui
    def calculation_message() -> ui.Tag | None:
        result, error = calculation()
        if error is not None:
            return ui.div(error, class_="message message-error")
        if result is not None and not result.diagnostics.all_converged:
            percent = 100.0 * result.diagnostics.converged_fraction
            return ui.div(
                f"{percent:.1f}% of field points reached the requested tolerance. "
                "Inspect the diagnostics or increase the iteration limit.",
                class_="message message-warning",
            )
        return ui.div(
            "All field points reached the requested tolerance.",
            class_="message message-success",
        )

    @output
    @render.ui
    def metric_cards() -> ui.Tag:
        result = _result()
        state = states()
        remanence = float(state.remanence.mean_m[2])
        coercivity = abs(float(state.coercive.field_t))
        gamma = state.remanence.gamma
        denominator = gamma[0, 0] + gamma[2, 2]
        anisotropy = (
            (gamma[0, 0] + gamma[1, 1]) / denominator if denominator != 0.0 else np.nan
        )
        return ui.div(
            _metric_card(
                "REDUCED REMANENCE",
                f"{remanence:.4f}",
                "first descending B = 0 state",
                "violet",
            ),
            _metric_card(
                "COERCIVE FIELD",
                f"{coercivity:.4f} T",
                "m̄z = 0 on the descending branch",
                "cyan",
            ),
            _metric_card(
                "SANS ANISOTROPY",
                f"{anisotropy:.4f}",
                "A⊥ at remanence",
                "gold",
            ),
            _metric_card(
                "CONVERGENCE",
                f"{100.0 * result.diagnostics.converged_fraction:.1f}%",
                f"N = {result.config.n_orientations:,} orientations",
                "green",
            ),
            class_="metric-grid",
        )

    @output
    @render.plot
    def hysteresis_plot():
        result = _result()
        state = states()
        figure, axis = plt.subplots(figsize=(7.2, 4.0))
        axis.plot(
            result.field_t,
            result.mean_m[:, 2],
            color="#6656d9",
            linewidth=2.4,
            label=r"$\overline{m}_z$",
        )
        axis.plot(
            result.field_t,
            result.mean_m[:, 0],
            color="#17a7b5",
            linewidth=1.4,
            alpha=0.8,
            label=r"$\overline{m}_x$",
        )
        axis.scatter(
            [state.remanence.field_t],
            [state.remanence.mean_m[2]],
            color="#d99632",
            edgecolor="white",
            linewidth=1.0,
            s=70,
            zorder=4,
            label="remanence",
        )
        axis.scatter(
            [state.coercive.field_t],
            [0.0],
            color="#d95670",
            edgecolor="white",
            linewidth=1.0,
            s=60,
            marker="s",
            zorder=4,
            label="coercive state",
        )
        axis.axhline(0.0, color="#aeb5c4", linewidth=0.8)
        axis.set_xlabel("Applied field B [T]")
        axis.set_ylabel("Ensemble magnetization")
        axis.set_ylim(-1.05, 1.05)
        axis.legend(frameon=False, fontsize=9, ncol=2)
        _style_axes(axis)
        figure.tight_layout()
        return figure

    @output
    @render.plot
    def moments_plot():
        result = _result()
        figure, axes = plt.subplots(2, 1, figsize=(7.2, 4.2), sharex=True)
        colors = ("#6656d9", "#17a7b5", "#d99632")
        labels = (r"$\Gamma_{xx}$", r"$\Gamma_{yy}$", r"$\Gamma_{zz}$")
        for index, (color, label) in enumerate(zip(colors, labels, strict=True)):
            axes[0].plot(
                result.field_t,
                result.gamma[:, index, index],
                color=color,
                linewidth=2.0,
                label=label,
            )
        axes[0].set_ylabel("Second moment")
        axes[0].set_ylim(-0.03, 1.03)
        axes[0].legend(frameon=False, fontsize=9, ncol=3)
        axes[1].plot(
            result.field_t,
            result.anisotropy_parameter_perpendicular,
            color="#d95670",
            linewidth=2.0,
        )
        axes[1].axhline(1.0, color="#aeb5c4", linewidth=0.8, linestyle="--")
        axes[1].set_xlabel("Applied field B [T]")
        axes[1].set_ylabel(r"$A_\perp$")
        for axis in axes:
            _style_axes(axis)
        figure.tight_layout()
        return figure

    @output
    @render.plot
    def sans_map_plot():
        index, state = selected_state()
        result = _result()
        config = _collect_sans_config()
        geometry = str(input.geometry())
        scale = str(input.reciprocal_scale())
        cross_section = str(input.cross_section())
        x_axis, y_axis, intensity = sans_map(
            state.gamma,
            config,
            geometry,
            cross_section,
        )
        positive = intensity[intensity > 0.0]
        maximum = float(np.max(positive)) if positive.size else 1.0
        minimum = max(
            float(np.min(positive)) if positive.size else 1.0e-8, maximum * 1e-4
        )
        is_logarithmic = scale == "log"
        displayed_intensity = (
            np.maximum(intensity, minimum) if is_logarithmic else intensity
        )
        norm = (
            LogNorm(vmin=minimum, vmax=maximum)
            if is_logarithmic
            else Normalize(vmin=0.0, vmax=maximum)
        )
        figure, axis = plt.subplots(figsize=(5.5, 4.7))
        figure.subplots_adjust(left=0.17, right=0.84, bottom=0.15, top=0.88)
        image = axis.imshow(
            displayed_intensity,
            origin="lower",
            extent=[x_axis[0], x_axis[-1], y_axis[0], y_axis[-1]],
            cmap="magma",
            norm=norm,
            interpolation="bilinear",
        )
        if not positive.size:
            axis.text(
                0.5,
                0.5,
                "Zero magnetic signal",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="white",
                fontsize=10,
            )
        if geometry == "perpendicular":
            axis.set_xlabel(r"$q_z$ [nm$^{-1}$]")
            axis.set_ylabel(r"$q_y$ [nm$^{-1}$]", labelpad=8)
            title = r"$B\perp k_0$"
        else:
            axis.set_xlabel(r"$q_x$ [nm$^{-1}$]")
            axis.set_ylabel(r"$q_y$ [nm$^{-1}$]", labelpad=8)
            title = r"$B\parallel k_0$"
        channel_title = (
            "Spin-flip average" if cross_section == "spin_flip" else "Unpolarized"
        )
        axis.set_title(
            f"{channel_title} · point {index + 1}/{result.field_t.size} · "
            f"B = {state.field_t:+.4f} T · {title}",
            fontsize=10.5,
        )
        axis.set_aspect("equal")
        colorbar = figure.colorbar(image, ax=axis, fraction=0.047, pad=0.04)
        colorbar_label = (
            r"$d\Sigma_{\mathrm{sf}}/d\Omega$ [cm$^{-1}$]"
            if cross_section == "spin_flip"
            else r"$d\Sigma_M/d\Omega$ [cm$^{-1}$]"
        )
        colorbar.set_label(colorbar_label)
        return figure

    @output
    @render.plot
    def sans_curves_plot():
        _, state = selected_state()
        config = _collect_sans_config()
        geometry = str(input.geometry())
        scale = str(input.reciprocal_scale())
        cross_section = str(input.cross_section())
        spacing = "linear" if scale == "linear" else "log"
        q, intensity = sans_intensity_curve(
            state.gamma,
            config,
            geometry,
            spacing=spacing,
            cross_section=cross_section,
        )
        r, pair, correlation = pair_distance_curves(
            state.gamma,
            config,
            geometry,
            cross_section,
        )
        figure, axes = plt.subplots(3, 1, figsize=(6.2, 5.0))
        axes[0].plot(q, intensity, color="#6656d9", linewidth=2.0)
        intensity_label = (
            r"$I_{\mathrm{sf}}(q)$ [cm$^{-1}$]"
            if cross_section == "spin_flip"
            else r"$I_M(q)$ [cm$^{-1}$]"
        )
        axes[0].set_ylabel(intensity_label)
        axes[1].plot(r, pair, color="#17a7b5", linewidth=2.0)
        axes[1].set_ylabel(r"$p(r)$ [nm$^{-2}$]")
        axes[2].plot(r, correlation, color="#d99632", linewidth=2.0)
        axes[2].set_ylabel(r"$c(r)$ [nm$^{-4}$]")
        axes[2].set_xlabel("r [nm]")
        axes[0].set_xlabel(r"q [nm$^{-1}$]")
        for axis in axes:
            _style_axes(axis)
        for axis in axes[1:]:
            axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        if scale == "log":
            axes[0].set_xscale("log")
            if np.any(intensity > 0.0):
                axes[0].set_yscale("log")
            else:
                axes[0].set_ylim(-0.05, 1.0)
                axes[0].set_yticks([0.0])
                axes[0].text(
                    0.5,
                    0.65,
                    "Zero signal; logarithmic y-axis is undefined",
                    transform=axes[0].transAxes,
                    ha="center",
                    color="#7a8398",
                    fontsize=8,
                )
        else:
            axes[0].set_xlim(0.0, config.q_max_per_nm)
        figure.tight_layout()
        return figure

    @output
    @render.plot
    def convergence_plot():
        result = _result()
        figure, left = plt.subplots(figsize=(6.6, 3.0))
        right = left.twinx()
        left.plot(
            result.field_t,
            result.diagnostics.iterations,
            color="#6656d9",
            linewidth=1.8,
        )
        safe_residual = np.maximum(result.diagnostics.residual, 1.0e-30)
        right.semilogy(
            result.field_t,
            safe_residual,
            color="#d95670",
            linewidth=1.2,
            alpha=0.85,
        )
        right.axhline(
            result.config.tolerance,
            color="#d99632",
            linestyle="--",
            linewidth=1.0,
        )
        left.set_xlabel("Applied field B [T]")
        left.set_ylabel("Iterations", color="#6656d9")
        right.set_ylabel("Squared-rate residual", color="#d95670")
        _style_axes(left)
        right.spines["top"].set_visible(False)
        right.spines["right"].set_color("#cfd4e3")
        right.tick_params(colors="#667085", labelsize=8)
        figure.tight_layout()
        return figure

    @output
    @render.ui
    def critical_table() -> ui.Tag:
        state = states()
        rows = (
            state.saturation,
            state.remanence,
            state.coercive,
        )
        return ui.tags.table(
            ui.tags.thead(
                ui.tags.tr(
                    ui.tags.th("State"),
                    ui.tags.th("B [T]"),
                    ui.tags.th("m̄z"),
                    ui.tags.th("Γxx"),
                    ui.tags.th("Γyy"),
                    ui.tags.th("Γzz"),
                )
            ),
            ui.tags.tbody(
                *[
                    ui.tags.tr(
                        ui.tags.td(item.name.capitalize()),
                        ui.tags.td(f"{item.field_t:.4f}"),
                        ui.tags.td(f"{item.mean_m[2]:.4f}"),
                        ui.tags.td(f"{item.gamma[0, 0]:.4f}"),
                        ui.tags.td(f"{item.gamma[1, 1]:.4f}"),
                        ui.tags.td(f"{item.gamma[2, 2]:.4f}"),
                    )
                    for item in rows
                ]
            ),
            class_="state-table",
        )

    @render.download_button(
        filename="swsans_field_sweep.csv",
        media_type="text/csv",
    )
    def download_csv():
        result = _result()
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "field_T",
                "mean_mx",
                "mean_my",
                "mean_mz",
                "Gamma_xx",
                "Gamma_xy",
                "Gamma_xz",
                "Gamma_yy",
                "Gamma_yz",
                "Gamma_zz",
                "A_perpendicular",
                "iterations",
                "residual",
                "converged",
            ]
        )
        for index, field in enumerate(result.field_t):
            gamma = result.gamma[index]
            writer.writerow(
                [
                    field,
                    *result.mean_m[index],
                    gamma[0, 0],
                    gamma[0, 1],
                    gamma[0, 2],
                    gamma[1, 1],
                    gamma[1, 2],
                    gamma[2, 2],
                    result.anisotropy_parameter_perpendicular[index],
                    result.diagnostics.iterations[index],
                    result.diagnostics.residual[index],
                    result.diagnostics.converged[index],
                ]
            )
        yield buffer.getvalue()


app = App(app_ui, server, static_assets=APP_DIR / "www")
