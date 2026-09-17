<div align="center">

# Dynamic Pricing & Profit Optimization Dashboard

**Interactive Streamlit app for exploring the price–profit tradeoff in real time — powered by the demand elasticity models from the [main ML repo](https://github.com/rishav-singh15/demand-price-elasticity-ML).**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly&logoColor=white)

**🔗 [Main ML Repo & Full Analysis](https://github.com/rishav-singh15/demand-price-elasticity-ML) · [Live dashboard](https://price-optimisation2.streamlit.app)**

</div>

---

##  What This Is

A live "what-if" pricing simulator built on top of the demand elasticity models estimated in the main analysis. Instead of reading static plots, a recruiter or stakeholder can move a price slider and instantly see how quantity, revenue, profit, and margin respond — for each of the three products (A, B, C) — with 95% confidence bands on every profit curve.

This turns the econometric output of the main project into something non-technical stakeholders can actually interact with and trust.

##  Features

| Control | What it does |
|---|---|
|  **Product selector** | Switch between Product A / B / C, each with its own fitted elasticity |
|  **Price slider** | Set your price and instantly see predicted quantity, revenue, and profit |
|  **Competitor price slider** | Model competitive reactions using the estimated cross-price elasticity |
|  **Promotion toggle** | Simulate the promotional demand lift |
|  **Weekend toggle** | Apply the estimated weekend demand boost |
|  **Seasonality slider** (day of year) | See how demand shifts across the calendar |
|  **Confidence interval toggle** | Show/hide the 95% CI band around the profit curve |

**Live outputs:**
- Four synced charts — Profit, Revenue, Quantity, and Margin vs. Price — with your current price and the profit-maximizing price both marked
- Plain-language recommendation ("pricing below optimal, raise to $X for +Y% profit," etc.)
- Elasticity interpretation and competitive-position summary
- Scenario comparison table (your price vs. optimal vs. historical)
- One-click CSV export of the full price-profit curve, and a downloadable text summary report

##  Try It

**[👉 Open the live dashboard](#)** *(https://price-optimisation2.streamlit.app)*

## ▶️ Running Locally

```bash
git clone https://github.com/rishav-singh15/price2.git
cd price2
pip install -r requirements.txt
streamlit run dashboard.py
```

Requires `dashboard_parameters.csv` and `dashboard_main_grid.csv` (included in this repo) — these are pre-computed outputs from the regression models in the [main ML repo](https://github.com/rishav-singh15/demand-price-elasticity-ML), not re-fitted live.

##  Repository Structure

```
price2/
├── dashboard.py                  # Streamlit app
├── dashboard_parameters.csv      # fitted model coefficients per product
├── dashboard_main_grid.csv       # pre-computed price/quantity/profit grid
└── requirements.txt
```

##  Tech Stack

`Python` · `Streamlit` · `Plotly` (interactive charts) · `pandas` / `numpy`

## 🔗 Related

This dashboard is one part of a three-piece project:
1. **[Main ML repo](https://github.com/rishav-singh15/demand-price-elasticity-ML)** — data generation, demand elasticity modeling, bootstrap CIs, and profit optimization (the notebook and full write-up live here)
2. **This dashboard** — interactive scenario exploration for the fitted models above
