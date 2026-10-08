import marimo

__generated_with = "0.25.1"

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import special, stats

    return mo, np, plt, special, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reliability analysis of a simply supported beam — FORM

    The beam has a span of 5 m, with a concentrated load applied 3 m from the left support.

    Two independent random variables are considered. The load $X_1$ follows a
    Gumbel maximum distribution with a mean of 50 N and a standard deviation of 5 N.
    The limiting stress $X_2$ follows a shifted lognormal distribution with a lower
    bound of 199,000 Pa, a mean of 288,000 Pa, and a standard deviation of 26,400 Pa.

    ## 1. Parameters and limit state

    $$g(X_1,X_2)=X_2-\frac{L_1L_2}{(L_1+L_2)Z_p}X_1.$$
    Failure is defined by $g\leq0$. Independence is assumed, consistent with the
    product of the marginal densities used in the original code.
    """)


@app.cell
def _(np, stats):
    L1, L2, Zp = 3.0, 2.0, 3.66e-4
    EX1, sX1 = 50.0, 5.0
    X2_min, EX2, sX2 = 19.9e4, 28.8e4, 2.64e4
    stress_factor = L1 * L2 / ((L1 + L2) * Zp)
    beta_X1 = sX1 * np.sqrt(6) / np.pi
    mu_X1 = EX1 - 0.5772156649 * beta_X1
    EX2s = EX2 - X2_min
    m_X2 = np.log(EX2s**2 / np.sqrt(sX2**2 + EX2s**2))
    sigma_X2 = np.sqrt(np.log1p(sX2**2 / EX2s**2))
    load_distribution = stats.gumbel_r(loc=mu_X1, scale=beta_X1)
    stress_distribution = stats.lognorm(s=sigma_X2, loc=X2_min, scale=np.exp(m_X2))

    def g(x):
        x = np.asarray(x, dtype=float)
        return -stress_factor * x[..., 0] + x[..., 1]

    return (
        EX1,
        EX2,
        X2_min,
        beta_X1,
        g,
        load_distribution,
        m_X2,
        mu_X1,
        sX1,
        sX2,
        sigma_X2,
        stress_distribution,
        stress_factor,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Transformation to standard normal space

    $$X_1(z_1)=\mu_1-b_1\log[-\log\Phi(z_1)],\qquad
    X_2(z_2)=X_{2,\min}+\exp(m_2+\sigma_2z_2).$$

    $$h(\mathbf z)=g(X_1(z_1),X_2(z_2)),\qquad
    \frac{\partial h}{\partial z_i}=\frac{\partial g}{\partial x_i}
    \frac{\partial x_i}{\partial z_i}.$$

    The Gumbel density is $f(x)=b^{-1}\exp[-t-\exp(-t)]$, where
    $t=(x-\mu)/b$. This corrects a sign error in the original explanatory text.
    The transformation uses `log_ndtr` to reduce rounding errors in the normal CDF.
    """)


@app.cell
def _(X2_min, beta_X1, g, m_X2, mu_X1, np, sigma_X2, special, stats, stress_factor):
    def x1(z1):
        return mu_X1 - beta_X1 * np.log(-special.log_ndtr(z1))

    def x2(z2):
        return X2_min + np.exp(sigma_X2 * np.asarray(z2) + m_X2)

    def h(z):
        z = np.asarray(z, dtype=float)
        return g(np.stack((x1(z[..., 0]), x2(z[..., 1])), axis=-1))

    def grad_h_exact(z):
        z = np.asarray(z, dtype=float)
        log_cdf = special.log_ndtr(z[0])
        dx1 = beta_X1 * np.exp(stats.norm.logpdf(z[0]) - log_cdf) / (-log_cdf)
        dx2 = sigma_X2 * np.exp(sigma_X2 * z[1] + m_X2)
        return np.array([-stress_factor * dx1, dx2])

    def grad_h_FD(z, delta_z=0.001):
        z = np.asarray(z, dtype=float)
        return (h(z + delta_z * np.eye(2)) - h(z)) / delta_z

    return grad_h_FD, grad_h_exact, h, x1, x2


@app.cell
def _(EX2, X2_min, load_distribution, np, plt, sX2, stress_distribution, stress_factor):
    _x = np.linspace(35, 100, 101)
    _y = np.linspace(X2_min, EX2 + 8 * sX2, 101)
    _xx, _yy = np.meshgrid(_x, _y)
    fig_physical, _ax = plt.subplots(figsize=(7, 5))
    _ax.contour(
        _xx,
        _yy,
        load_distribution.pdf(_xx) * stress_distribution.pdf(_yy),
        levels=np.logspace(-10, -5, 11),
        cmap="viridis",
    )
    _ax.plot(_x, stress_factor * _x, color="black", lw=2, label="g(X1, X2) = 0")
    _ax.set(
        xlabel="Load X1 [N]",
        ylabel="Limit stress X2 [Pa]",
        title="Physical space: limit state and joint PDF",
        ylim=(_y[0], _y[-1]),
    )
    _ax.legend()
    fig_physical.tight_layout()
    fig_physical


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. FORM iterations (HLRF)

    $$\boldsymbol\alpha_k=\frac{\nabla h(\mathbf z_k)}{\|\nabla h(\mathbf z_k)\|},\quad
    \beta_k=\frac{h(\mathbf z_k)-\nabla h(\mathbf z_k)^T\mathbf z_k}{\|\nabla h(\mathbf z_k)\|},\quad
    \mathbf z_{k+1}=-\beta_k\boldsymbol\alpha_k.$$

    The iterations start at the origin and stop when
    $|\beta_k-\beta_{k-1}|<10^{-3}$, as in the original code.
    A maximum iteration count and checks for invalid values have been added.
    $\boldsymbol\alpha$ is the normalized gradient pointing toward the safe domain.
    """)


@app.cell
def _(mo):
    gradient_choice = mo.ui.dropdown(
        options=["Finite difference", "Analytical"],
        value="Finite difference",
        label="Gradient",
    )
    gradient_choice
    return (gradient_choice,)


@app.cell
def _(grad_h_FD, grad_h_exact, gradient_choice, h, np, stats):
    def solve_form(gradient, tolerance=1e-3, max_iterations=100):
        z = np.zeros(2)
        path = [z.copy()]
        history = []
        previous_beta = np.inf
        for iteration in range(1, max_iterations + 1):
            gradient_value = gradient(z)
            norm_gradient = np.linalg.norm(gradient_value)
            if not np.isfinite(norm_gradient) or norm_gradient == 0:
                raise RuntimeError("Invalid or zero gradient in FORM iteration.")
            alpha = gradient_value / norm_gradient
            beta = float((h(z) - gradient_value @ z) / norm_gradient)
            next_z = -alpha * beta
            if not np.all(np.isfinite(next_z)):
                raise RuntimeError("Nonfinite FORM design point.")
            history.append(
                {
                    "iteration": iteration,
                    "beta": beta,
                    "alpha_1": float(alpha[0]),
                    "alpha_2": float(alpha[1]),
                    "z_1_next": float(next_z[0]),
                    "z_2_next": float(next_z[1]),
                }
            )
            path.append(next_z.copy())
            if abs(previous_beta - beta) < tolerance:
                return {
                    "beta": beta,
                    "pf": float(stats.norm.sf(beta)),
                    "z": next_z,
                    "path": np.array(path),
                    "history": history,
                    "residual": float(h(next_z)),
                }
            previous_beta, z = beta, next_z
        raise RuntimeError("FORM did not converge within the maximum iterations.")

    form_result = solve_form(
        grad_h_FD if gradient_choice.value == "Finite difference" else grad_h_exact
    )
    return form_result, solve_form


@app.cell(hide_code=True)
def _(form_result, mo, x1, x2):
    _r = form_result
    mo.md(f"""
    **Reliability index** β = {_r["beta"]:.8f}
    **FORM probability of failure** P_f = {_r["pf"]:.8e}
    **Design point (standard normal space)** z* = ({_r["z"][0]:.6f}, {_r["z"][1]:.6f})
    **Design point (physical space)** X* = ({x1(_r["z"][0]):.6f} N, {x2(_r["z"][1]):.6f} Pa)
    **Iterations** {len(_r["history"])} / **Limit-state residual** h(z*) = {_r["residual"]:.6g} Pa

    The original stopping criterion checks only the change in β, so the residual
    is also displayed for inspection. The probability of failure is evaluated
    using `norm.sf(beta)`.
    """)


@app.cell
def _(form_result, mo):
    mo.ui.table(form_result["history"], selection=None)


@app.cell
def _(form_result, h, np, plt):
    _xx, _yy = np.meshgrid(np.linspace(-5, 5, 1001), np.linspace(-5, 5, 1001))
    _zz = np.stack((_xx, _yy), axis=-1)
    fig_normal, _ax = plt.subplots(figsize=(7, 6))
    _ax.contour(
        _xx,
        _yy,
        np.exp(-0.5 * (_xx**2 + _yy**2)) / (2 * np.pi),
        levels=np.logspace(-5, -1, 6),
        cmap="viridis",
    )
    _ax.contour(_xx, _yy, h(_zz), levels=[0], colors="black", linewidths=2)
    _path = form_result["path"]
    _ax.plot(_path[:, 0], _path[:, 1], "x-", lw=2, label="FORM iterations")
    _ax.scatter(*form_result["z"], color="red", zorder=4, label="Design point")
    _ax.set(
        xlabel="Z1",
        ylabel="Z2",
        title="Standard normal space",
        xlim=(-5, 5),
        ylim=(-5, 5),
    )
    _ax.set_aspect("equal")
    _ax.legend()
    fig_normal.tight_layout()
    fig_normal


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Running the notebook

    From the repository root, run

    ```bash
    uv run marimo edit notebooks/01_form_beam.py
    ```

    The Python environment and dependencies are managed by uv using
    pyproject.toml and uv.lock. Input values can be changed in the
    parameters cell. FORM approximates the probability of failure by linearizing
    the limit state at the design point; it does not provide the exact probability.
    """)


if __name__ == "__main__":
    app.run()
