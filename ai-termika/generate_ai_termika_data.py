#!/usr/bin/env python3
"""
AI Termika — Generátor soutěžních dat
======================================
Česká AI Olympiáda 2026, linie AI Tech

Generuje kompletní datový balíček pro úlohu AI Termika:
  - train_measurements.csv   (zašuměná řídká měření)
  - boundary_partial.csv     (neúplné okrajové podmínky)
  - initial_sparse.csv       (řídká počáteční podmínka)
  - test_points.csv          (testovací body bez odpovědí)
  - test_ground_truth.csv    (skrytý ground truth pro hodnocení)
  - constants.json           (přibližné alpha)
  - metadata.json            (parametry generátoru pro reprodukovatelnost)

Ground truth: analytické řešení 1D rovnice vedení tepla
  ∂u/∂t = α · ∂²u/∂x²
na doméně [0,1] × [0,1] s Dirichletovými okrajovými podmínkami.

Řešení je Fourierova řada:
  u(x,t) = Σ_n  b_n · exp(-n²π²αt) · sin(nπx) + u_boundary(x,t)

kde u_boundary interpoluje okrajové podmínky a b_n jsou koeficienty
počáteční podmínky.

Autor: nvias, z.s. | www.aiolympiada.cz
"""

import numpy as np
import pandas as pd
import json
import os
import argparse
from pathlib import Path


# ============================================================================
# 1. ANALYTICKÉ ŘEŠENÍ
# ============================================================================

def initial_condition_func(x: np.ndarray) -> np.ndarray:
    """
    Počáteční podmínka u(x, 0).
    
    Kombinace sinusoid — dostatečně zajímavý tvar, aby nebyl triviální,
    ale stále hladký a fyzikálně rozumný.
    """
    return (
        np.sin(np.pi * x)
        + 0.5 * np.sin(3 * np.pi * x)
        + 0.3 * np.sin(2 * np.pi * x)
    )


def boundary_left(t: np.ndarray) -> np.ndarray:
    """Okrajová podmínka u(0, t) — pomalu klesající."""
    return 0.2 * np.exp(-2.0 * t)


def boundary_right(t: np.ndarray) -> np.ndarray:
    """Okrajová podmínka u(1, t) — oscilující, pak tlumená."""
    return 0.15 * np.sin(2 * np.pi * t) * np.exp(-3.0 * t)


def compute_fourier_coefficients(
    alpha_true: float,
    n_terms: int = 50,
    n_quad: int = 1000,
) -> np.ndarray:
    """
    Spočítá Fourierovy koeficienty b_n pro homogenní část řešení.
    
    Homogenní část = počáteční podmínka minus steady-state boundary interpolaci
    v čase t=0.
    """
    x_quad = np.linspace(0, 1, n_quad)
    dx = x_quad[1] - x_quad[0]

    # Boundary interpolace v t=0
    bl0 = boundary_left(np.array([0.0]))[0]
    br0 = boundary_right(np.array([0.0]))[0]
    u_boundary_t0 = bl0 * (1 - x_quad) + br0 * x_quad

    # Homogenní počáteční podmínka
    g = initial_condition_func(x_quad) - u_boundary_t0

    # Fourierovy koeficienty (sinová řada)
    coeffs = np.zeros(n_terms)
    for n in range(1, n_terms + 1):
        basis = np.sin(n * np.pi * x_quad)
        coeffs[n - 1] = 2.0 * np.sum(g * basis) * dx

    return coeffs


def exact_solution(
    x: np.ndarray,
    t: np.ndarray,
    alpha_true: float,
    coeffs: np.ndarray,
) -> np.ndarray:
    """
    Analytické řešení u(x, t) pro libovolné pole bodů.
    
    x, t: 1D pole stejné délky (ne meshgrid!)
    Vrací: 1D pole u(x_i, t_i)
    """
    n_terms = len(coeffs)

    # Boundary interpolace: lineární v x, závisí na t
    bl = boundary_left(t)
    br = boundary_right(t)
    u_boundary = bl * (1 - x) + br * x

    # Homogenní část: Fourierova řada
    u_homog = np.zeros_like(x, dtype=float)
    for n in range(1, n_terms + 1):
        decay = np.exp(-(n ** 2) * (np.pi ** 2) * alpha_true * t)
        u_homog += coeffs[n - 1] * decay * np.sin(n * np.pi * x)

    return u_homog + u_boundary


def exact_solution_grid(
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    alpha_true: float,
    coeffs: np.ndarray,
) -> np.ndarray:
    """
    Analytické řešení na 2D mřížce.
    
    Vrací matici [len(x_grid), len(t_grid)].
    """
    X, T = np.meshgrid(x_grid, t_grid, indexing="ij")
    return exact_solution(X.ravel(), T.ravel(), alpha_true, coeffs).reshape(X.shape)


# ============================================================================
# 2. GENEROVÁNÍ SOUTĚŽNÍCH SOUBORŮ
# ============================================================================

def generate_train_measurements(
    alpha_true: float,
    coeffs: np.ndarray,
    n_points: int = 80,
    noise_std: float = 0.05,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Zašuměná měření uvnitř domény.
    
    Body jsou vzorkované z Latin Hypercube-like schématu pro lepší pokrytí,
    s aditivním Gaussovským šumem.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Stratifikované vzorkování pro lepší pokrytí domény
    n_x_strata = int(np.sqrt(n_points * 2))
    n_t_strata = n_points // n_x_strata + 1

    x_edges = np.linspace(0.05, 0.95, n_x_strata + 1)
    t_edges = np.linspace(0.02, 0.98, n_t_strata + 1)

    points_x = []
    points_t = []

    for i in range(n_x_strata):
        for j in range(n_t_strata):
            if len(points_x) >= n_points:
                break
            px = rng.uniform(x_edges[i], x_edges[i + 1])
            pt = rng.uniform(t_edges[j], t_edges[j + 1])
            points_x.append(px)
            points_t.append(pt)
        if len(points_x) >= n_points:
            break

    # Doplnit zbývající body náhodně
    while len(points_x) < n_points:
        points_x.append(rng.uniform(0.05, 0.95))
        points_t.append(rng.uniform(0.02, 0.98))

    x = np.array(points_x[:n_points])
    t = np.array(points_t[:n_points])

    u_clean = exact_solution(x, t, alpha_true, coeffs)
    noise = rng.normal(0, noise_std, size=n_points)
    u_noisy = u_clean + noise

    return pd.DataFrame({"x": np.round(x, 6), "t": np.round(t, 6), "u": np.round(u_noisy, 6)})


def generate_boundary_partial(
    alpha_true: float,
    coeffs: np.ndarray,
    n_points_per_side: int = 12,
    coverage: float = 0.55,
    noise_std: float = 0.02,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Neúplné okrajové podmínky.
    
    Pokrývá jen ~55 % časového rozsahu na každé straně,
    s mírným šumem simulujícím měření na hranici.
    """
    if rng is None:
        rng = np.random.default_rng()

    rows = []

    for x_val in [0.0, 1.0]:
        # Vybrat náhodný podinterval časů, který pokrývá jen 'coverage' rozsahu
        gap_start = rng.uniform(0.1, 1.0 - coverage)
        gap_end = gap_start + (1.0 - coverage) * 0.6

        t_candidates = np.sort(rng.uniform(0.0, 1.0, size=n_points_per_side * 3))

        # Vyfiltrovat body mimo mezeru
        t_filtered = t_candidates[(t_candidates < gap_start) | (t_candidates > gap_end)]
        t_selected = t_filtered[:n_points_per_side]

        # Doplnit pokud málo bodů
        while len(t_selected) < n_points_per_side:
            t_extra = rng.uniform(0.0, gap_start)
            t_selected = np.append(t_selected, t_extra)

        x_arr = np.full_like(t_selected, x_val)
        u_clean = exact_solution(x_arr, t_selected, alpha_true, coeffs)
        noise = rng.normal(0, noise_std, size=len(t_selected))

        for i in range(len(t_selected)):
            rows.append({
                "x": round(float(x_arr[i]), 6),
                "t": round(float(t_selected[i]), 6),
                "u": round(float(u_clean[i] + noise[i]), 6),
            })

    df = pd.DataFrame(rows)
    return df.sort_values(["x", "t"]).reset_index(drop=True)


def generate_initial_sparse(
    alpha_true: float,
    coeffs: np.ndarray,
    n_points: int = 12,
    noise_std: float = 0.03,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    """
    Řídká počáteční podmínka.
    
    Jen několik bodů z t=0, ne rovnoměrně rozložených.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Náhodné body, ale zajistit alespoň krajní oblasti
    x_core = rng.uniform(0.05, 0.95, size=n_points - 2)
    x_endpoints = np.array([rng.uniform(0.0, 0.08), rng.uniform(0.92, 1.0)])
    x = np.concatenate([x_endpoints, x_core])
    x = np.sort(x)[:n_points]

    t = np.zeros(n_points)
    u_clean = exact_solution(x, t, alpha_true, coeffs)
    noise = rng.normal(0, noise_std, size=n_points)

    return pd.DataFrame({
        "x": np.round(x, 6),
        "t": np.round(t, 6),
        "u": np.round(u_clean + noise, 6),
    })


def generate_test_points(
    alpha_true: float,
    coeffs: np.ndarray,
    n_points: int = 500,
    rng: np.random.Generator = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Testovací body + skrytý ground truth.
    
    Body jsou náhodně rozložené v doméně, s důrazem na oblasti
    kde je řešení zajímavější (střed domény, raný čas).
    """
    if rng is None:
        rng = np.random.default_rng()

    # Směs dvou distribucí: uniformní + koncentrovaná na zajímavé oblasti
    n_uniform = n_points // 2
    n_focused = n_points - n_uniform

    # Uniformní body
    x_u = rng.uniform(0.0, 1.0, size=n_uniform)
    t_u = rng.uniform(0.0, 1.0, size=n_uniform)

    # Zaměřené body — více bodů v rané fázi a středu tyče
    x_f = rng.beta(2, 2, size=n_focused)  # Koncentrace kolem 0.5
    t_f = rng.beta(1.5, 3, size=n_focused)  # Koncentrace na malá t

    x = np.concatenate([x_u, x_f])
    t = np.concatenate([t_u, t_f])

    # Zamíchat
    perm = rng.permutation(n_points)
    x = x[perm]
    t = t[perm]

    u_true = exact_solution(x, t, alpha_true, coeffs)

    ids = np.arange(n_points)

    test_points = pd.DataFrame({
        "id": ids,
        "x": np.round(x, 6),
        "t": np.round(t, 6),
    })

    ground_truth = pd.DataFrame({
        "id": ids,
        "u": np.round(u_true, 6),
    })

    return test_points, ground_truth


# ============================================================================
# 3. VIZUALIZACE (pro kontrolu organizátorem)
# ============================================================================

def generate_visualizations(
    alpha_true: float,
    coeffs: np.ndarray,
    train_df: pd.DataFrame,
    boundary_df: pd.DataFrame,
    initial_df: pd.DataFrame,
    test_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    output_dir: Path,
):
    """Generuje kontrolní vizualizace pro organizátora."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("  [!] matplotlib není k dispozici, vizualizace přeskočeny.")
        return

    viz_dir = output_dir / "viz"
    viz_dir.mkdir(exist_ok=True)

    # --- 1. Celkové řešení jako heatmapa ---
    x_dense = np.linspace(0, 1, 200)
    t_dense = np.linspace(0, 1, 200)
    U = exact_solution_grid(x_dense, t_dense, alpha_true, coeffs)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    c = ax.pcolormesh(t_dense, x_dense, U, shading="auto", cmap="inferno")
    plt.colorbar(c, ax=ax, label="u(x,t)")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title("Ground truth: u(x,t)")
    fig.tight_layout()
    fig.savefig(viz_dir / "01_ground_truth_heatmap.png", dpi=150)
    plt.close(fig)

    # --- 2. Rozmístění trénovacích, boundary a initial bodů ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.scatter(train_df["t"], train_df["x"], c="blue", s=15, alpha=0.7, label="train_measurements")
    ax.scatter(boundary_df["t"], boundary_df["x"], c="red", s=25, marker="s", alpha=0.8, label="boundary_partial")
    ax.scatter(initial_df["t"], initial_df["x"], c="green", s=25, marker="^", alpha=0.8, label="initial_sparse")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title("Rozmístění datových bodů v doméně")
    ax.legend()
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(viz_dir / "02_data_points_distribution.png", dpi=150)
    plt.close(fig)

    # --- 3. Počáteční podmínka: úplná vs. řídká ---
    x_ic = np.linspace(0, 1, 500)
    u_ic = initial_condition_func(x_ic)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(x_ic, u_ic, "k-", linewidth=2, label="Skutečná počáteční podmínka")
    ax.scatter(initial_df["x"], initial_df["u"], c="green", s=60, marker="^",
               zorder=5, label="initial_sparse (co dostanou studenti)")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, 0)")
    ax.set_title("Počáteční podmínka — úplná vs. řídká")
    ax.legend()
    fig.tight_layout()
    fig.savefig(viz_dir / "03_initial_condition.png", dpi=150)
    plt.close(fig)

    # --- 4. Okrajové podmínky: úplné vs. neúplné ---
    t_bc = np.linspace(0, 1, 500)
    u_left = boundary_left(t_bc)
    u_right = boundary_right(t_bc)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bc_left = boundary_df[boundary_df["x"] == 0.0]
    bc_right = boundary_df[boundary_df["x"] == 1.0]

    axes[0].plot(t_bc, u_left, "k-", linewidth=2, label="Skutečné u(0,t)")
    axes[0].scatter(bc_left["t"], bc_left["u"], c="red", s=40, marker="s",
                    zorder=5, label="boundary_partial")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("u(0, t)")
    axes[0].set_title("Levý okraj (x=0)")
    axes[0].legend()

    axes[1].plot(t_bc, u_right, "k-", linewidth=2, label="Skutečné u(1,t)")
    axes[1].scatter(bc_right["t"], bc_right["u"], c="red", s=40, marker="s",
                    zorder=5, label="boundary_partial")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("u(1, t)")
    axes[1].set_title("Pravý okraj (x=1)")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(viz_dir / "04_boundary_conditions.png", dpi=150)
    plt.close(fig)

    # --- 5. Distribuce testovacích bodů ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    merged = test_df.merge(gt_df, on="id")
    sc = ax.scatter(merged["t"], merged["x"], c=merged["u"], s=10,
                    cmap="inferno", alpha=0.7)
    plt.colorbar(sc, ax=ax, label="u(x,t) — ground truth")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title("Testovací body (obarvené ground truth)")
    fig.tight_layout()
    fig.savefig(viz_dir / "05_test_points.png", dpi=150)
    plt.close(fig)

    # --- 6. Profily u(x) v několika časech ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    times_to_plot = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    colors = cm.viridis(np.linspace(0, 1, len(times_to_plot)))

    for t_val, color in zip(times_to_plot, colors):
        u_profile = exact_solution(x_dense, np.full_like(x_dense, t_val),
                                   alpha_true, coeffs)
        ax.plot(x_dense, u_profile, color=color, linewidth=2, label=f"t={t_val:.2f}")

    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title("Teplotní profily v různých časech")
    ax.legend()
    fig.tight_layout()
    fig.savefig(viz_dir / "06_temperature_profiles.png", dpi=150)
    plt.close(fig)

    print(f"  [✓] Vizualizace uloženy do {viz_dir}/")


# ============================================================================
# 4. HLAVNÍ GENERÁTOR
# ============================================================================

def generate_dataset(
    output_dir: str = "ai_termika_dataset",
    seed: int = 2026,
    alpha_true: float = 0.05,
    alpha_shift_pct: float = 0.08,
    n_train: int = 80,
    n_boundary_per_side: int = 12,
    n_initial: int = 12,
    n_test: int = 500,
    noise_train: float = 0.05,
    noise_boundary: float = 0.02,
    noise_initial: float = 0.03,
    n_fourier_terms: int = 50,
    visualize: bool = True,
):
    """
    Hlavní funkce — generuje kompletní datový balíček.
    
    Parametry
    ---------
    output_dir : str
        Cílová složka pro výstupní soubory.
    seed : int
        Seed pro reprodukovatelnost.
    alpha_true : float
        Skutečná hodnota difuzivity (skrytá).
    alpha_shift_pct : float
        Relativní odchylka alpha v constants.json (např. 0.08 = ±8 %).
    n_train : int
        Počet trénovacích měření.
    n_boundary_per_side : int
        Počet boundary bodů na každé straně tyče.
    n_initial : int
        Počet bodů počáteční podmínky.
    n_test : int
        Počet testovacích bodů.
    noise_train : float
        Směrodatná odchylka šumu v trénovacích datech.
    noise_boundary : float
        Směrodatná odchylka šumu v boundary datech.
    noise_initial : float
        Směrodatná odchylka šumu v initial datech.
    n_fourier_terms : int
        Počet členů Fourierovy řady.
    visualize : bool
        Zda generovat kontrolní vizualizace.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Složka pro studenty (co dostanou)
    student_dir = out / "student"
    student_dir.mkdir(exist_ok=True)

    # Složka pro organizátora (skrytá)
    organizer_dir = out / "organizer"
    organizer_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(seed)

    print("=" * 60)
    print("AI Termika — Generátor soutěžních dat")
    print("=" * 60)
    print(f"  Seed:           {seed}")
    print(f"  Alpha (true):   {alpha_true}")
    print(f"  Alpha shift:    ±{alpha_shift_pct*100:.0f} %")
    print(f"  Fourier terms:  {n_fourier_terms}")
    print()

    # --- Fourierovy koeficienty ---
    print("[1/7] Počítám Fourierovy koeficienty...")
    coeffs = compute_fourier_coefficients(alpha_true, n_fourier_terms)

    # --- Train measurements ---
    print(f"[2/7] Generuji train_measurements.csv ({n_train} bodů, σ={noise_train})...")
    train_df = generate_train_measurements(
        alpha_true, coeffs, n_train, noise_train, rng
    )
    train_df.to_csv(student_dir / "train_measurements.csv", index=False)

    # --- Boundary partial ---
    print(f"[3/7] Generuji boundary_partial.csv ({n_boundary_per_side}×2 bodů, σ={noise_boundary})...")
    boundary_df = generate_boundary_partial(
        alpha_true, coeffs, n_boundary_per_side, 0.55, noise_boundary, rng
    )
    boundary_df.to_csv(student_dir / "boundary_partial.csv", index=False)

    # --- Initial sparse ---
    print(f"[4/7] Generuji initial_sparse.csv ({n_initial} bodů, σ={noise_initial})...")
    initial_df = generate_initial_sparse(
        alpha_true, coeffs, n_initial, noise_initial, rng
    )
    initial_df.to_csv(student_dir / "initial_sparse.csv", index=False)

    # --- Test points ---
    print(f"[5/7] Generuji test_points.csv ({n_test} bodů)...")
    test_df, gt_df = generate_test_points(alpha_true, coeffs, n_test, rng)
    test_df.to_csv(student_dir / "test_points.csv", index=False)
    gt_df.to_csv(organizer_dir / "test_ground_truth.csv", index=False)

    # --- Constants (s posunutým alpha) ---
    print("[6/7] Generuji constants.json...")
    # Posunout alpha o náhodný offset v rozsahu ±shift
    alpha_direction = rng.choice([-1, 1])
    alpha_offset = alpha_direction * rng.uniform(
        alpha_shift_pct * 0.5, alpha_shift_pct
    )
    alpha_approx = round(alpha_true * (1 + alpha_offset), 6)

    constants = {
        "alpha": alpha_approx,
        "domain": {
            "x_min": 0.0,
            "x_max": 1.0,
            "t_min": 0.0,
            "t_max": 1.0,
        },
        "note": "Hodnota alpha je přibližná. Skutečná hodnota se může lišit o jednotky procent.",
    }
    with open(student_dir / "constants.json", "w", encoding="utf-8") as f:
        json.dump(constants, f, indent=2, ensure_ascii=False)

    # --- Metadata (pro organizátora) ---
    print("[7/7] Ukládám metadata...")
    metadata = {
        "generator": "AI Termika Dataset Generator v1.0",
        "competition": "Česká AI Olympiáda 2026",
        "seed": seed,
        "alpha_true": alpha_true,
        "alpha_approx": alpha_approx,
        "alpha_shift_actual": round(alpha_offset * 100, 2),
        "n_fourier_terms": n_fourier_terms,
        "n_train": n_train,
        "n_boundary_per_side": n_boundary_per_side,
        "n_initial": n_initial,
        "n_test": n_test,
        "noise_std": {
            "train": noise_train,
            "boundary": noise_boundary,
            "initial": noise_initial,
        },
        "initial_condition": "sin(πx) + 0.5·sin(3πx) + 0.3·sin(2πx)",
        "boundary_left": "0.2·exp(-2t)",
        "boundary_right": "0.15·sin(2πt)·exp(-3t)",
        "equation": "∂u/∂t = α · ∂²u/∂x²",
    }
    with open(organizer_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # --- Statistiky ---
    print()
    print("-" * 60)
    print("SOUHRN")
    print("-" * 60)
    print(f"  Alpha skutečné:    {alpha_true}")
    print(f"  Alpha přibližné:   {alpha_approx} ({alpha_offset*100:+.1f} %)")
    print(f"  Train body:        {len(train_df)}")
    print(f"  Boundary body:     {len(boundary_df)} (z toho x=0: {len(boundary_df[boundary_df['x']==0.0])}, x=1: {len(boundary_df[boundary_df['x']==1.0])})")
    print(f"  Initial body:      {len(initial_df)}")
    print(f"  Test body:         {len(test_df)}")
    print()
    print(f"  Rozsah u (train):  [{train_df['u'].min():.3f}, {train_df['u'].max():.3f}]")
    print(f"  Rozsah u (GT):     [{gt_df['u'].min():.3f}, {gt_df['u'].max():.3f}]")
    print()

    # --- Výstupní soubory ---
    print("VÝSTUPNÍ SOUBORY")
    print(f"  📁 {student_dir}/")
    for f in sorted(student_dir.iterdir()):
        size = f.stat().st_size
        print(f"     {f.name:30s} {size:>8,} B")
    print(f"  📁 {organizer_dir}/")
    for f in sorted(organizer_dir.iterdir()):
        size = f.stat().st_size
        print(f"     {f.name:30s} {size:>8,} B")

    # --- Vizualizace ---
    if visualize:
        print()
        print("VIZUALIZACE")
        generate_visualizations(
            alpha_true, coeffs,
            train_df, boundary_df, initial_df,
            test_df, gt_df,
            out,
        )

    # --- Validace ---
    print()
    print("VALIDACE")
    _validate_dataset(student_dir, organizer_dir)

    print()
    print("=" * 60)
    print("Hotovo! Dataset připraven.")
    print("=" * 60)


def _validate_dataset(student_dir: Path, organizer_dir: Path):
    """Základní kontroly konzistence dat."""
    checks_passed = 0
    checks_total = 0

    # 1. Všechny soubory existují
    expected_student = ["train_measurements.csv", "boundary_partial.csv",
                        "initial_sparse.csv", "test_points.csv", "constants.json"]
    expected_org = ["test_ground_truth.csv", "metadata.json"]

    for fname in expected_student:
        checks_total += 1
        if (student_dir / fname).exists():
            checks_passed += 1
        else:
            print(f"  [✗] Chybí {fname}")

    for fname in expected_org:
        checks_total += 1
        if (organizer_dir / fname).exists():
            checks_passed += 1
        else:
            print(f"  [✗] Chybí {fname}")

    # 2. CSV formát
    train = pd.read_csv(student_dir / "train_measurements.csv")
    checks_total += 1
    if list(train.columns) == ["x", "t", "u"]:
        checks_passed += 1
    else:
        print(f"  [✗] train_measurements.csv: špatné sloupce {list(train.columns)}")

    boundary = pd.read_csv(student_dir / "boundary_partial.csv")
    checks_total += 1
    if set(boundary["x"].unique()) <= {0.0, 1.0}:
        checks_passed += 1
    else:
        print(f"  [✗] boundary_partial.csv: x obsahuje jiné hodnoty než 0 a 1")

    test = pd.read_csv(student_dir / "test_points.csv")
    gt = pd.read_csv(organizer_dir / "test_ground_truth.csv")
    checks_total += 1
    if len(test) == len(gt) and list(test["id"]) == list(gt["id"]):
        checks_passed += 1
    else:
        print(f"  [✗] test_points a ground_truth nemají stejná ID")

    # 3. Žádné NaN
    for name, df in [("train", train), ("boundary", boundary),
                     ("test", test), ("gt", gt)]:
        checks_total += 1
        if not df.isnull().any().any():
            checks_passed += 1
        else:
            print(f"  [✗] {name} obsahuje NaN!")

    # 4. Rozsah domény
    checks_total += 1
    all_x = pd.concat([train["x"], boundary["x"],
                        pd.read_csv(student_dir / "initial_sparse.csv")["x"],
                        test["x"]])
    all_t = pd.concat([train["t"], boundary["t"],
                        pd.read_csv(student_dir / "initial_sparse.csv")["t"],
                        test["t"]])
    if all_x.min() >= 0.0 and all_x.max() <= 1.0 and all_t.min() >= 0.0 and all_t.max() <= 1.0:
        checks_passed += 1
    else:
        print(f"  [✗] Body mimo doménu [0,1]²!")

    print(f"  [{checks_passed}/{checks_total}] kontrol prošlo ✓")


# ============================================================================
# 5. CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AI Termika — Generátor soutěžních dat pro Českou AI Olympiádu 2026",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output", "-o", default="ai_termika_dataset",
                        help="Výstupní složka")
    parser.add_argument("--seed", "-s", type=int, default=2026,
                        help="Random seed")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Skutečná hodnota alpha")
    parser.add_argument("--alpha-shift", type=float, default=0.08,
                        help="Max relativní odchylka alpha v constants.json")
    parser.add_argument("--n-train", type=int, default=80,
                        help="Počet trénovacích bodů")
    parser.add_argument("--n-boundary", type=int, default=12,
                        help="Počet boundary bodů na každé straně")
    parser.add_argument("--n-initial", type=int, default=12,
                        help="Počet bodů počáteční podmínky")
    parser.add_argument("--n-test", type=int, default=500,
                        help="Počet testovacích bodů")
    parser.add_argument("--noise-train", type=float, default=0.05,
                        help="Šum v trénovacích datech (std)")
    parser.add_argument("--noise-boundary", type=float, default=0.02,
                        help="Šum v boundary datech (std)")
    parser.add_argument("--noise-initial", type=float, default=0.03,
                        help="Šum v initial datech (std)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Přeskočit generování vizualizací")

    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output,
        seed=args.seed,
        alpha_true=args.alpha,
        alpha_shift_pct=args.alpha_shift,
        n_train=args.n_train,
        n_boundary_per_side=args.n_boundary,
        n_initial=args.n_initial,
        n_test=args.n_test,
        noise_train=args.noise_train,
        noise_boundary=args.noise_boundary,
        noise_initial=args.noise_initial,
        visualize=not args.no_viz,
    )
