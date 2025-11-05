import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Dynamic Pricing Dashboard",
    page_icon="bar_chart",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Removed .main-header and .sub-header, now using st.title/header */
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        color: #212529; /* Dark text for light background */
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        color: #212529; /* Dark text for light background */
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# CORE FUNCTIONS
# ============================================================================
def predict_full_demand(price, competitor_price, promotion_flag, is_weekend, day_of_year, params):
    """
    Predicts quantity using the FULL regression model.
    """
    beta = params.get('beta_override', params['log_price'])
    
    sin_doy = np.sin(2 * np.pi * day_of_year / 365)
    cos_doy = np.cos(2 * np.pi * day_of_year / 365)
    
    log_quantity = (
        params['const'] +
        beta * np.log(price) +
        params['log_competitor_price'] * np.log(competitor_price) +
        params['promotion_flag'] * promotion_flag +
        params['is_weekend'] * is_weekend +
        params['sin_doy'] * sin_doy +
        params['cos_doy'] * cos_doy
    )
    
    return np.exp(log_quantity) - 0.1

def calculate_dashboard_metrics(price_grid, cost, competitor_price, promotion_flag, 
                                is_weekend, day_of_year, params):
    """
    Calculates Qty, Revenue, and Profit for a range of prices with confidence intervals.
    """
    results = []
    
    for p in price_grid:
        # Point estimate
        params_point = params.copy()
        params_point['beta_override'] = params['log_price']
        qty = predict_full_demand(p, competitor_price, promotion_flag, is_weekend, day_of_year, params_point)
        profit = (p - cost) * qty
        
        # Lower CI
        params_lower = params.copy()
        params_lower['beta_override'] = params['beta_ci_lower']
        qty_lower = predict_full_demand(p, competitor_price, promotion_flag, is_weekend, day_of_year, params_lower)
        profit_lower = (p - cost) * qty_lower
        
        # Upper CI
        params_upper = params.copy()
        params_upper['beta_override'] = params['beta_ci_upper']
        qty_upper = predict_full_demand(p, competitor_price, promotion_flag, is_weekend, day_of_year, params_upper)
        profit_upper = (p - cost) * qty_upper
        
        results.append({
            'price': p,
            'predicted_qty': max(0, qty),
            'revenue': p * max(0, qty),
            'profit': profit,
            'profit_lower_ci': profit_lower,
            'profit_upper_ci': profit_upper
        })
        
    return pd.DataFrame(results)

# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data
def load_data():
    """Load the pre-computed dashboard data."""
    try:
        params_df = pd.read_csv('dashboard_parameters.csv')
        grid_df = pd.read_csv('dashboard_main_grid.csv')
        return params_df, grid_df, None
    except Exception as e:
        return None, None, str(e)

# ============================================================================
# MAIN DASHBOARD
# ============================================================================
def main():
    # Header
    st.title("Dynamic Pricing & Profit Optimization Dashboard")
    
    # Load data
    params_df, grid_df, error = load_data()
    
    if error:
        st.error(f"Error loading data: {error}")
        st.info("Please ensure 'dashboard_parameters.csv' and 'dashboard_main_grid.csv' are in the same directory as this script.")
        return
    
    # ========================================================================
    # SIDEBAR: CONTROLS
    # ========================================================================
    st.sidebar.title("Dashboard Controls")
    
    # Product Selection
    st.sidebar.subheader("Product Selection")
    product_id = st.sidebar.selectbox(
        "Select Product",
        options=['A', 'B', 'C'],
        index=0
    )
    
    # Get parameters for selected product
    product_params = params_df[params_df['product_id'] == product_id].iloc[0].to_dict()
    
    st.sidebar.markdown("---")
    
    # Scenario Settings
    st.sidebar.subheader("Scenario Settings")
    
    # Price slider
    current_price = product_params['current_price']
    optimal_price = product_params['optimal_price']
    price_min = current_price * 0.5
    price_max = optimal_price * 1.5
    
    selected_price = st.sidebar.slider(
        "Selected Price (₹)",
        min_value=float(price_min),
        max_value=float(price_max),
        value=float(current_price),
        step=0.01,
        help="Adjust your product price to see the impact on demand and profit"
    )
    
    # Competitor price
    competitor_price = st.sidebar.slider(
        "Competitor Price (₹)",
        min_value=float(price_min * 0.8),
        max_value=float(price_max * 1.2),
        value=float(current_price * 1.05),
        step=0.01,
        help="Set competitor pricing to analyze competitive dynamics"
    )
    
    # Promotion flag
    promotion_flag = st.sidebar.checkbox(
        "Run Promotion",
        value=False,
        help="Enable to simulate promotional pricing impact"
    )
    
    # Weekend flag
    is_weekend = st.sidebar.checkbox(
        "Weekend",
        value=False,
        help="Toggle weekend vs weekday demand patterns"
    )
    
    # Day of year (seasonality)
    day_of_year = st.sidebar.slider(
        "Day of Year (Seasonality)",
        min_value=1,
        max_value=365,
        value=182,
        help="1=Jan 1, 182=Mid-year, 365=Dec 31"
    )
    
    # Month display for better UX
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    approx_month = min(11, day_of_year // 30)
    st.sidebar.caption(f"Approx: {month_names[approx_month]}")
    
    st.sidebar.markdown("---")
    
    # Confidence Interval Toggle
    show_ci = st.sidebar.checkbox(
        "Show Confidence Intervals",
        value=True,
        help="Display 95% confidence bands on profit curve"
    )
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    # Product Info Header
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Product",
            value=f"Product {product_id}",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Current Price",
            value=f"₹{current_price:.2f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Optimal Price",
            value=f"₹{optimal_price:.2f}",
            delta=f"{product_params['profit_uplift_pct']:.1f}% profit gain"
        )
    
    with col4:
        st.metric(
            label="Unit Cost",
            value=f"₹{product_params['cost']:.2f}",
            delta=None
        )
    
    st.markdown("---")
    
    # Calculate metrics for selected scenario
    params_scenario = product_params.copy()
    params_scenario['beta_override'] = params_scenario['log_price']
    
    qty = predict_full_demand(
        selected_price, 
        competitor_price, 
        1 if promotion_flag else 0,
        1 if is_weekend else 0,
        day_of_year,
        params_scenario
    )
    qty = max(0, qty)
    
    revenue = selected_price * qty
    profit = (selected_price - product_params['cost']) * qty
    margin = ((selected_price - product_params['cost']) / selected_price * 100) if selected_price > 0 else 0
    
    # Display current scenario metrics
    st.header("Current Scenario Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Predicted Quantity",
            value=f"{qty:.0f} units",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Expected Revenue",
            value=f"₹{revenue:,.2f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Expected Profit",
            value=f"₹{profit:,.2f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Profit Margin",
            value=f"{margin:.1f}%",
            delta=None
        )
    
    # ========================================================================
    # VISUALIZATION SECTION
    # ========================================================================
    
    st.header("Price-Profit Optimization Curve")
    
    # Generate price grid for current scenario
    price_grid = np.linspace(price_min, price_max, 300)
    
    scenario_df = calculate_dashboard_metrics(
        price_grid,
        product_params['cost'],
        competitor_price,
        1 if promotion_flag else 0,
        1 if is_weekend else 0,
        day_of_year,
        product_params
    )
    
    # Create interactive plot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Profit vs Price', 'Revenue vs Price', 
                       'Quantity vs Price', 'Profit Margin vs Price'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Plot 1: Profit vs Price (with CI)
    if show_ci:
        fig.add_trace(
            go.Scatter(
                x=scenario_df['price'],
                y=scenario_df['profit_upper_ci'],
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=scenario_df['price'],
                y=scenario_df['profit_lower_ci'],
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                name='95% CI',
                fillcolor='rgba(31, 119, 180, 0.2)',
                hoverinfo='skip'
            ),
            row=1, col=1
        )
    
    fig.add_trace(
        go.Scatter(
            x=scenario_df['price'],
            y=scenario_df['profit'],
            mode='lines',
            name='Profit',
            line=dict(color='#1f77b4', width=3),
            hovertemplate='<b>Price:</b> ₹%{x:.2f}<br><b>Profit:</b> ₹%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Mark current and optimal prices on profit curve
    fig.add_trace(
        go.Scatter(
            x=[selected_price],
            y=[profit],
            mode='markers',
            name='Selected Price',
            marker=dict(size=12, color='red', symbol='diamond'),
            hovertemplate='<b>Selected Price:</b> ₹%{x:.2f}<br><b>Profit:</b> ₹%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    optimal_idx = scenario_df['profit'].idxmax()
    optimal_profit = scenario_df.loc[optimal_idx, 'profit']
    optimal_price_actual = scenario_df.loc[optimal_idx, 'price']
    
    fig.add_trace(
        go.Scatter(
            x=[optimal_price_actual],
            y=[optimal_profit],
            mode='markers',
            name='Optimal Price',
            marker=dict(size=12, color='green', symbol='star'),
            hovertemplate='<b>Optimal Price:</b> ₹%{x:.2f}<br><b>Max Profit:</b> ₹%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Plot 2: Revenue vs Price
    fig.add_trace(
        go.Scatter(
            x=scenario_df['price'],
            y=scenario_df['revenue'],
            mode='lines',
            name='Revenue',
            line=dict(color='#2ca02c', width=3),
            showlegend=False,
            hovertemplate='<b>Price:</b> ₹%{x:.2f}<br><b>Revenue:</b> ₹%{y:,.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=[selected_price],
            y=[revenue],
            mode='markers',
            marker=dict(size=12, color='red', symbol='diamond'),
            showlegend=False,
            hovertemplate='<b>Selected Price:</b> ₹%{x:.2f}<br><b>Revenue:</b> ₹%{y:,.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Plot 3: Quantity vs Price
    fig.add_trace(
        go.Scatter(
            x=scenario_df['price'],
            y=scenario_df['predicted_qty'],
            mode='lines',
            name='Quantity',
            line=dict(color='#ff7f0e', width=3),
            showlegend=False,
            hovertemplate='<b>Price:</b> ₹%{x:.2f}<br><b>Quantity:</b> %{y:.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[selected_price],
            y=[qty],
            mode='markers',
            marker=dict(size=12, color='red', symbol='diamond'),
            showlegend=False,
            hovertemplate='<b>Selected Price:</b> ₹%{x:.2f}<br><b>Quantity:</b> %{y:.0f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Plot 4: Profit Margin vs Price
    scenario_df['margin'] = ((scenario_df['price'] - product_params['cost']) / scenario_df['price'] * 100)
    
    fig.add_trace(
        go.Scatter(
            x=scenario_df['price'],
            y=scenario_df['margin'],
            mode='lines',
            name='Margin %',
            line=dict(color='#9467bd', width=3),
            showlegend=False,
            hovertemplate='<b>Price:</b> ₹%{x:.2f}<br><b>Margin:</b> %{y:.1f}%<extra></extra>'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=[selected_price],
            y=[margin],
            mode='markers',
            marker=dict(size=12, color='red', symbol='diamond'),
            showlegend=False,
            hovertemplate='<b>Selected Price:</b> ₹%{x:.2f}<br><b>Margin:</b> %{y:.1f}%<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_xaxes(title_text="Price (₹)", row=1, col=2)
    fig.update_xaxes(title_text="Price (₹)", row=2, col=1)
    fig.update_xaxes(title_text="Price (₹)", row=2, col=2)
    
    fig.update_yaxes(title_text="Profit (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Revenue (₹)", row=1, col=2)
    fig.update_yaxes(title_text="Quantity (units)", row=2, col=1)
    fig.update_yaxes(title_text="Margin (%)", row=2, col=2)
    
    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # INSIGHTS & RECOMMENDATIONS
    # ========================================================================
    
    st.header("Insights & Recommendations")
    
    profit_gap = optimal_profit - profit
    profit_gap_pct = (profit_gap / profit * 100) if profit > 0 else 0
    
    if abs(selected_price - optimal_price_actual) < 0.35:
        st.markdown(f"""
        <div class="success-box">
        <strong>Optimal Pricing</strong><br>
        Your current price (₹{selected_price:.2f}) is very close to the optimal price (₹{optimal_price_actual:.2f}).
        You are maximizing profit under the current market conditions.
        </div>
        """, unsafe_allow_html=True)
    elif selected_price < optimal_price_actual:
        st.markdown(f"""
        <div class="warning-box">
        <strong>Pricing Below Optimal</strong><br>
        Your price is ₹{optimal_price_actual - selected_price:.2f} below the optimal price.<br>
        • <strong>Potential profit gain:</strong> ₹{profit_gap:,.2f} ({profit_gap_pct:.1f}% increase)<br>
        • <strong>Recommendation:</strong> Consider gradually raising prices to ₹{optimal_price_actual:.2f}<br>
        • <strong>Risk:</strong> Low - demand is relatively inelastic at this price point
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="warning-box">
        <strong>Pricing Above Optimal</strong><br>
        Your price is ₹{selected_price - optimal_price_actual:.2f} above the optimal price.<br>
        • <strong>Lost profit:</strong> ₹{abs(profit_gap):,.2f} ({abs(profit_gap_pct):.1f}% decrease)<br>
        • <strong>Recommendation:</strong> Consider lowering prices to ₹{optimal_price_actual:.2f}<br>
        • <strong>Benefit:</strong> Higher volume (estimated {scenario_df.loc[optimal_idx, 'predicted_qty']:.0f} units vs {qty:.0f} units)
        </div>
        """, unsafe_allow_html=True)
    
    # Additional insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Elasticity Insights")
        elasticity = product_params['log_price']
        if elasticity > -1:
            elasticity_msg = "Inelastic demand (|ε| < 1) - customers less sensitive to price changes"
        else:
            elasticity_msg = "Elastic demand (|ε| > 1) - customers highly sensitive to price changes"
        
        st.info(f"""
        **Price Elasticity:** {elasticity:.3f}  
        {elasticity_msg}
        
        **Interpretation:** A 1% price increase leads to a {abs(elasticity):.2f}% quantity decrease.
        """)
    
    with col2:
        st.subheader("Competitive Position")
        comp_diff = selected_price - competitor_price
        comp_pct = (comp_diff / competitor_price * 100) if competitor_price > 0 else 0
        
        if comp_diff > 0:
            comp_status = f"Your price is {comp_pct:.1f}% **higher** than competitors"
        elif comp_diff < 0:
            comp_status = f"Your price is {abs(comp_pct):.1f}% **lower** than competitors"
        else:
            comp_status = "Your price **matches** competitor pricing"
        
        st.info(f"""
        **Selected Price:** ₹{selected_price:.2f}  
        **Competitor Price:** ₹{competitor_price:.2f}  
        
        {comp_status}
        """)
    
    # ========================================================================
    # SCENARIO COMPARISON TABLE
    # ========================================================================
    
    st.header("Scenario Comparison")
    
    # Calculate scenarios
    scenarios = []
    
    # Current scenario
    scenarios.append({
        'Scenario': 'Selected Price',
        'Price': f'₹{selected_price:.2f}',
        'Quantity': f'{qty:.0f}',
        'Revenue': f'₹{revenue:,.2f}',
        'Profit': f'₹{profit:,.2f}',
        'Margin': f'{margin:.1f}%'
    })
    
    # Optimal scenario
    scenarios.append({
        'Scenario': 'Optimal Price',
        'Price': f'₹{optimal_price_actual:.2f}',
        'Quantity': f'{scenario_df.loc[optimal_idx, "predicted_qty"]:.0f}',
        'Revenue': f'₹{scenario_df.loc[optimal_idx, "revenue"]:,.2f}',
        'Profit': f'₹{optimal_profit:,.2f}',
        'Margin': f'{scenario_df.loc[optimal_idx, "margin"]:.1f}%'
    })
    
    # Current price from historical data
    params_current = product_params.copy()
    params_current['beta_override'] = params_current['log_price']
    qty_current = predict_full_demand(
        current_price, competitor_price, 
        1 if promotion_flag else 0, 1 if is_weekend else 0,
        day_of_year, params_current
    )
    qty_current = max(0, qty_current)
    rev_current = current_price * qty_current
    profit_current = (current_price - product_params['cost']) * qty_current
    margin_current = ((current_price - product_params['cost']) / current_price * 100)
    
    scenarios.append({
        'Scenario': 'Historical Price',
        'Price': f'₹{current_price:.2f}',
        'Quantity': f'{qty_current:.0f}',
        'Revenue': f'₹{rev_current:,.2f}',
        'Profit': f'₹{profit_current:,.2f}',
        'Margin': f'{margin_current:.1f}%'
    })
    
    comparison_df = pd.DataFrame(scenarios)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # ========================================================================
    # MODEL DETAILS (EXPANDABLE)
    # ========================================================================
    
    with st.expander("View Model Details & Coefficients"):
        st.subheader("Regression Model Coefficients")
        
        coef_data = {
            'Variable': [
                'Intercept',
                'log(Price)',
                'log(Competitor Price)',
                'Promotion Flag',
                'Weekend Flag',
                'sin(Day of Year)',
                'cos(Day of Year)'
            ],
            'Coefficient': [
                f"{product_params['const']:.4f}",
                f"{product_params['log_price']:.4f}",
                f"{product_params['log_competitor_price']:.4f}",
                f"{product_params['promotion_flag']:.4f}",
                f"{product_params['is_weekend']:.4f}",
                f"{product_params['sin_doy']:.4f}",
                f"{product_params['cos_doy']:.4f}"
            ],
            'Interpretation': [
                'Baseline demand level',
                'Price elasticity of demand',
                'Cross-price elasticity',
                'Promotion lift effect',
                'Weekend demand boost',
                'Seasonal variation (sine)',
                'Seasonal variation (cosine)'
            ]
        }
        
        coef_df = pd.DataFrame(coef_data)
        st.dataframe(coef_df, use_container_width=True, hide_index=True)
        
        st.markdown(f"""
        **Model Confidence:**
        - 95% CI for Price Elasticity: [{product_params['beta_ci_lower']:.4f}, {product_params['beta_ci_upper']:.4f}]
        - Product Cost: ₹{product_params['cost']:.2f}
        """)
    
    # ========================================================================
    # DOWNLOAD SECTION
    # ========================================================================
    
    st.markdown("---")
    st.header("Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prepare download data
        download_df = scenario_df[['price', 'predicted_qty', 'revenue', 'profit']].copy()
        download_df.columns = ['Price', 'Predicted Quantity', 'Revenue', 'Profit']
        
        csv_data = download_df.to_csv(index=False)
        st.download_button(
            label="Download Price-Profit Curve Data",
            data=csv_data,
            file_name=f"product_{product_id}_pricing_analysis.csv",
            mime="text/csv"
        )
    
    with col2:
        # Summary report
        summary_text = f"""
PRICING ANALYSIS REPORT
Product: {product_id}
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

CURRENT SCENARIO:
- Price: ₹{selected_price:.2f}
- Competitor Price: ₹{competitor_price:.2f}
- Promotion: {'Yes' if promotion_flag else 'No'}
- Weekend: {'Yes' if is_weekend else 'No'}
- Day of Year: {day_of_year}

RESULTS:
- Predicted Quantity: {qty:.0f} units
- Expected Revenue: ₹{revenue:,.2f}
- Expected Profit: ₹{profit:,.2f}
- Profit Margin: {margin:.1f}%

OPTIMIZATION:
- Optimal Price: ₹{optimal_price_actual:.2f}
- Maximum Profit: ₹{optimal_profit:,.2f}
- Potential Gain: ₹{profit_gap:,.2f} ({profit_gap_pct:.1f}%)

MODEL PARAMETERS:
- Price Elasticity: {product_params['log_price']:.4f}
- Cross-Price Elasticity: {product_params['log_competitor_price']:.4f}
- Promotion Effect: {product_params['promotion_flag']:.4f}
- Weekend Effect: {product_params['is_weekend']:.4f}
        """
        
        st.download_button(
            label="Download Summary Report",
            data=summary_text,
            file_name=f"product_{product_id}_summary_report.txt",
            mime="text/plain"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-allign: center; color: #7f8c8d; padding: 2rem 0;'>
    <p><strong>Dynamic Pricing Dashboard v1.0</strong></p>
    <p>Built with Streamlit • Powered by Advanced Analytics</p>
    <p style='font-size: 0.9rem;'>Adjust the controls in the sidebar to explore different pricing scenarios</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN THE APP
# ============================================================================
if __name__ == "__main__":
    main()