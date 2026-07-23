import numpy as np
import pandas as pd


def generate_field_data(n_samples=1000, random_state=42):
    rng = np.random.RandomState(random_state)

    well_count = rng.randint(5, 120, n_samples)
    water_injection_rate = rng.uniform(50, 5000, n_samples)

    lift_options = ["ESP", "Rod_Pump", "Gas_Lift", "PCP", "Jet_Pump"]
    lift_type = rng.choice(lift_options, n_samples)

    lift_power_map = {"ESP": (80, 250), "Rod_Pump": (30, 120), "Gas_Lift": (10, 60),
                      "PCP": (20, 90), "Jet_Pump": (15, 70)}
    artificial_lift_power_kw = np.zeros(n_samples)
    for i, lt in enumerate(lift_type):
        lo, hi = lift_power_map[lt]
        artificial_lift_power_kw[i] = rng.uniform(lo, hi)

    operating_cost_usd = (
        well_count * rng.uniform(800, 3000, n_samples)
        + artificial_lift_power_kw * rng.uniform(10, 50, n_samples)
        + water_injection_rate * rng.uniform(0.5, 3.0, n_samples)
    )

    total_oil_rate = (
        well_count * rng.uniform(20, 300, n_samples)
        + water_injection_rate * rng.uniform(0.01, 0.15, n_samples)
        - artificial_lift_power_kw * rng.uniform(0.1, 1.0, n_samples)
        + rng.normal(0, 200, n_samples)
    )
    total_oil_rate = np.clip(total_oil_rate, 10, None)

    water_cut = rng.uniform(0.2, 0.92, n_samples)
    total_water_rate = total_oil_rate * water_cut / (1 - water_cut)

    total_gas_rate = total_oil_rate * rng.uniform(0.5, 5.0, n_samples)

    revenue_per_bbl = rng.uniform(50, 120, n_samples)
    revenue = total_oil_rate * revenue_per_bbl
    net_profit = revenue - operating_cost_usd

    production_efficiency = np.clip(
        (total_oil_rate / (well_count * 300 + 1)) * rng.uniform(0.7, 1.3, n_samples),
        0.05, 0.99
    )

    df = pd.DataFrame({
        "well_count": well_count,
        "total_oil_rate": np.round(total_oil_rate, 2),
        "total_water_rate": np.round(total_water_rate, 2),
        "total_gas_rate": np.round(total_gas_rate, 2),
        "water_injection_rate": np.round(water_injection_rate, 2),
        "lift_type": lift_type,
        "operating_cost_usd": np.round(operating_cost_usd, 2),
        "revenue_per_bbl": np.round(revenue_per_bbl, 2),
        "net_profit": np.round(net_profit, 2),
        "production_efficiency": np.round(production_efficiency, 4),
        "artificial_lift_power_kw": np.round(artificial_lift_power_kw, 2),
    })

    return df
