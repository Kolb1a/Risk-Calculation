import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import openpyxl
from scipy.stats import norm
from datetime import datetime, timedelta
import io

# Page configuration
st.set_page_config(
    page_title="Credit Portfolio Risk Calculator",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high { border-left-color: #ff4444 !important; }
    .risk-medium { border-left-color: #ffaa00 !important; }
    .risk-low { border-left-color: #00aa00 !important; }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏦 Credit Portfolio Risk Calculator</h1>', unsafe_allow_html=True)

# Sidebar for file upload and controls
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Upload your Excel model",
        type=['xlsx', 'xls'],
        help="Upload your credit portfolio Excel file"
    )
    
    st.header("⚙️ Simulation Settings")
    num_simulations = st.slider("Monte Carlo Simulations", 100, 10000, 1000, 100)
    confidence_level = st.selectbox("Confidence Level", [95, 99, 99.9], index=0)

# Helper Functions
@st.cache_data
def load_excel_data(uploaded_file):
    """Load and process Excel file data"""
    if uploaded_file is None:
        # Create sample data based on the Excel structure
        return create_sample_data()
    
    try:
        # Load all sheets
        loan_book = pd.read_excel(uploaded_file, sheet_name='LoanBook')
        inputs = pd.read_excel(uploaded_file, sheet_name='Inputs')
        
        # Try to load other sheets if they exist
        try:
            analytics = pd.read_excel(uploaded_file, sheet_name='Analytics')
            metrics = pd.read_excel(uploaded_file, sheet_name='Metrics')
        except:
            analytics = None
            metrics = None
            
        return loan_book, inputs, analytics, metrics
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        return create_sample_data()

def create_sample_data():
    """Create sample data matching the Excel structure"""
    # Sample loan book data
    loan_ids = [f"Loan_{i}" for i in range(1, 21)]
    borrowers = [f"Borrower_{i}" for i in range(1, 21)]
    
    np.random.seed(42)  # For reproducible results
    
    loan_book = pd.DataFrame({
        'LoanID': loan_ids,
        'Borrower Name': borrowers,
        'Customer Segment': np.random.choice(['SME', 'Corporate', 'Retail'], 20),
        'Industry Code': np.random.choice(['C10', 'C20', 'C40'], 20),
        'Facility Type': np.random.choice(['Term', 'Revolver', 'Bridge'], 20),
        'Collateral Type': np.random.choice(['Receivables', 'CRE', 'Securities'], 20),
        'Outstanding Balance': np.random.uniform(100000, 2000000, 20),
        'Undrawn Commitments': np.random.uniform(0, 500000, 20),
        'Credit Conv. Factor': np.random.uniform(0.3, 0.7, 20),
        'Net Operating Income': np.random.uniform(50000, 400000, 20),
        'Annual Interest Pmt': np.random.uniform(5000, 100000, 20),
        'Principal Payment': np.random.uniform(20000, 400000, 20),
        'Collateral Value': np.random.uniform(200000, 3000000, 20),
        'Haircut': np.random.uniform(0.2, 0.5, 20),
        'Debt': np.random.uniform(100000, 2000000, 20),
        'EBITDA': np.random.uniform(50000, 500000, 20),
        'Interest Expense': np.random.uniform(5000, 100000, 20)
    })
    
    # Calculate derived metrics
    loan_book['EAD'] = loan_book['Outstanding Balance'] + loan_book['Credit Conv. Factor'] * loan_book['Undrawn Commitments']
    loan_book['DSCR'] = loan_book['Net Operating Income'] / (loan_book['Annual Interest Pmt'] + loan_book['Principal Payment'])
    loan_book['LTV'] = loan_book['Outstanding Balance'] / loan_book['Collateral Value']
    loan_book['Debt/EBITDA'] = loan_book['Debt'] / loan_book['EBITDA']
    loan_book['Interest Coverage'] = loan_book['EBITDA'] / loan_book['Interest Expense']
    loan_book['LGD'] = loan_book['Haircut']
    
    # Risk inputs
    inputs = pd.DataFrame({
        'LoanID': loan_ids,
        'PD': np.random.uniform(0.01, 0.1, 20),
        'LGD': loan_book['LGD'].values,
        'EAD': loan_book['EAD'].values,
        'rho': np.random.uniform(0.15, 0.3, 20)
    })
    
    return loan_book, inputs, None, None

def calculate_portfolio_metrics(loan_book, inputs):
    """Calculate portfolio-level risk metrics"""
    total_exposure = inputs['EAD'].sum()
    weighted_pd = (inputs['PD'] * inputs['EAD']).sum() / total_exposure
    weighted_lgd = (inputs['LGD'] * inputs['EAD']).sum() / total_exposure
    expected_loss = (inputs['PD'] * inputs['LGD'] * inputs['EAD']).sum()
    
    return {
        'Total Exposure': total_exposure,
        'Weighted Average PD': weighted_pd,
        'Weighted Average LGD': weighted_lgd,
        'Expected Loss': expected_loss,
        'Number of Loans': len(inputs)
    }

def run_monte_carlo_simulation(inputs, num_simulations=1000):
    """Run Monte Carlo simulation for portfolio loss"""
    np.random.seed(42)
    
    losses = []
    
    for _ in range(num_simulations):
        # Generate systematic factor
        Z = np.random.normal(0, 1)
        
        # Generate idiosyncratic factors for each loan
        epsilons = np.random.normal(0, 1, len(inputs))
        
        scenario_losses = 0
        
        for i, row in inputs.iterrows():
            # Calculate asset value using single-factor model
            sqrt_rho = np.sqrt(row['rho'])
            sqrt_1_rho = np.sqrt(1 - row['rho'])
            
            asset_value = sqrt_rho * Z + sqrt_1_rho * epsilons[i]
            
            # Calculate default threshold
            default_threshold = norm.ppf(row['PD'])
            
            # Check if loan defaults
            if asset_value < default_threshold:
                loss = row['LGD'] * row['EAD']
                scenario_losses += loss
        
        losses.append(scenario_losses)
    
    return np.array(losses)

def create_risk_dashboard(loan_book, inputs, portfolio_metrics):
    """Create main risk dashboard"""
    
    # Portfolio Overview
    st.header("📊 Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Exposure</h3>
            <h2>${portfolio_metrics['Total Exposure']:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Expected Loss</h3>
            <h2>${portfolio_metrics['Expected Loss']:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg PD</h3>
            <h2>{portfolio_metrics['Weighted Average PD']:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg LGD</h3>
            <h2>{portfolio_metrics['Weighted Average LGD']:.2%}</h2>
        </div>
        """, unsafe_allow_html=True)

def create_visualizations(loan_book, inputs, mc_losses):
    """Create risk visualizations"""
    
    st.header("📈 Risk Analytics")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Composition", "Risk Metrics", "Monte Carlo Results", "Loan Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Exposure by segment
            segment_exposure = loan_book.groupby('Customer Segment')['Outstanding Balance'].sum()
            fig_segment = px.pie(
                values=segment_exposure.values,
                names=segment_exposure.index,
                title="Exposure by Customer Segment"
            )
            st.plotly_chart(fig_segment, use_container_width=True)
        
        with col2:
            # Exposure by industry
            industry_exposure = loan_book.groupby('Industry Code')['Outstanding Balance'].sum()
            fig_industry = px.bar(
                x=industry_exposure.index,
                y=industry_exposure.values,
                title="Exposure by Industry"
            )
            fig_industry.update_layout(xaxis_title="Industry Code", yaxis_title="Exposure")
            st.plotly_chart(fig_industry, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # PD vs EAD scatter
            fig_pd_ead = px.scatter(
                inputs, x='PD', y='EAD', size='LGD',
                title="PD vs EAD (bubble size = LGD)",
                hover_data=['LoanID']
            )
            fig_pd_ead.update_layout(xaxis_title="Probability of Default", yaxis_title="Exposure at Default")
            st.plotly_chart(fig_pd_ead, use_container_width=True)
        
        with col2:
            # DSCR distribution
            fig_dscr = px.histogram(
                loan_book, x='DSCR', nbins=20,
                title="Debt Service Coverage Ratio Distribution"
            )
            fig_dscr.update_layout(xaxis_title="DSCR", yaxis_title="Count")
            st.plotly_chart(fig_dscr, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Loss distribution
            fig_losses = px.histogram(
                x=mc_losses, nbins=50,
                title=f"Portfolio Loss Distribution ({len(mc_losses)} simulations)"
            )
            fig_losses.update_layout(xaxis_title="Portfolio Loss", yaxis_title="Frequency")
            st.plotly_chart(fig_losses, use_container_width=True)
        
        with col2:
            # VaR and CVaR
            var_95 = np.percentile(mc_losses, 95)
            var_99 = np.percentile(mc_losses, 99)
            cvar_95 = mc_losses[mc_losses >= var_95].mean()
            
            st.subheader("Risk Measures")
            st.metric("VaR 95%", f"${var_95:,.0f}")
            st.metric("VaR 99%", f"${var_99:,.0f}")
            st.metric("CVaR 95%", f"${cvar_95:,.0f}")
            st.metric("Expected Loss", f"${mc_losses.mean():,.0f}")
            st.metric("Max Loss", f"${mc_losses.max():,.0f}")
    
    with tab4:
        # Individual loan analysis
        selected_loan = st.selectbox("Select Loan for Analysis", inputs['LoanID'].tolist())
        
        if selected_loan:
            loan_data = inputs[inputs['LoanID'] == selected_loan].iloc[0]
            loan_details = loan_book[loan_book['LoanID'] == selected_loan].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Risk Parameters")
                st.write(f"**PD:** {loan_data['PD']:.3%}")
                st.write(f"**LGD:** {loan_data['LGD']:.1%}")
                st.write(f"**EAD:** ${loan_data['EAD']:,.0f}")
                st.write(f"**Correlation:** {loan_data['rho']:.3f}")
            
            with col2:
                st.subheader("Financial Metrics")
                st.write(f"**DSCR:** {loan_details['DSCR']:.2f}")
                st.write(f"**LTV:** {loan_details['LTV']:.1%}")
                st.write(f"**Debt/EBITDA:** {loan_details['Debt/EBITDA']:.2f}")
                st.write(f"**Interest Coverage:** {loan_details['Interest Coverage']:.2f}")
            
            with col3:
                st.subheader("Expected Loss")
                el = loan_data['PD'] * loan_data['LGD'] * loan_data['EAD']
                st.metric("Expected Loss", f"${el:,.0f}")
                
                risk_level = "High" if loan_data['PD'] > 0.05 else "Medium" if loan_data['PD'] > 0.02 else "Low"
                st.write(f"**Risk Level:** {risk_level}")

def create_reports_section(loan_book, inputs, portfolio_metrics, mc_losses):
    """Create downloadable reports"""
    
    st.header("📄 Reports")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Generate Executive Summary", use_container_width=True):
            # Create executive summary
            summary = f"""
# Credit Portfolio Risk Report
## Executive Summary

**Portfolio Overview:**
- Total Portfolio Exposure: ${portfolio_metrics['Total Exposure']:,.0f}
- Number of Loans: {portfolio_metrics['Number of Loans']}
- Expected Loss: ${portfolio_metrics['Expected Loss']:,.0f}

**Risk Metrics:**
- Weighted Average PD: {portfolio_metrics['Weighted Average PD']:.2%}
- Weighted Average LGD: {portfolio_metrics['Weighted Average LGD']:.1%}
- VaR 95%: ${np.percentile(mc_losses, 95):,.0f}
- VaR 99%: ${np.percentile(mc_losses, 99):,.0f}

**Key Findings:**
- {len(inputs[inputs['PD'] > 0.05])} loans have PD > 5% (high risk)
- {len(inputs[inputs['PD'] < 0.02])} loans have PD < 2% (low risk)
- Largest exposure: {inputs.loc[inputs['EAD'].idxmax(), 'LoanID']} (${inputs['EAD'].max():,.0f})

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            """
            
            st.download_button(
                label="📥 Download Summary",
                data=summary,
                file_name=f"portfolio_summary_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
    
    with col2:
        if st.button("📈 Generate Detailed Report", use_container_width=True):
            # Create detailed Excel report
            output = io.BytesIO()
            
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Portfolio summary
                summary_df = pd.DataFrame([portfolio_metrics]).T
                summary_df.columns = ['Value']
                summary_df.to_excel(writer, sheet_name='Portfolio Summary')
                
                # Loan details
                loan_book.to_excel(writer, sheet_name='Loan Book', index=False)
                inputs.to_excel(writer, sheet_name='Risk Inputs', index=False)
                
                # MC results
                mc_results = pd.DataFrame({
                    'Scenario': range(1, len(mc_losses) + 1),
                    'Portfolio Loss': mc_losses
                })
                mc_results.to_excel(writer, sheet_name='MC Simulation', index=False)
            
            st.download_button(
                label="📥 Download Excel Report",
                data=output.getvalue(),
                file_name=f"portfolio_risk_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# Input Parameter Adjustment Section
def create_scenario_analysis():
    """Create scenario analysis section"""
    
    st.header("🎯 Scenario Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Economic Scenarios")
        
        scenario = st.selectbox(
            "Select Economic Scenario",
            ["Base Case", "Stress Scenario", "Severe Stress", "Custom"]
        )
        
        if scenario == "Base Case":
            pd_multiplier = 1.0
            lgd_multiplier = 1.0
            correlation_adj = 0.0
        elif scenario == "Stress Scenario":
            pd_multiplier = 1.5
            lgd_multiplier = 1.2
            correlation_adj = 0.05
        elif scenario == "Severe Stress":
            pd_multiplier = 2.0
            lgd_multiplier = 1.5
            correlation_adj = 0.1
        else:  # Custom
            pd_multiplier = st.slider("PD Multiplier", 0.5, 3.0, 1.0, 0.1)
            lgd_multiplier = st.slider("LGD Multiplier", 0.8, 2.0, 1.0, 0.1)
            correlation_adj = st.slider("Correlation Adjustment", -0.1, 0.2, 0.0, 0.01)
        
        st.write(f"**PD Multiplier:** {pd_multiplier}x")
        st.write(f"**LGD Multiplier:** {lgd_multiplier}x")
        st.write(f"**Correlation Adjustment:** +{correlation_adj:.3f}")
    
    with col2:
        st.subheader("Industry Stress")
        
        industry_stress = {}
        industries = ['C10', 'C20', 'C40']
        
        for industry in industries:
            industry_stress[industry] = st.slider(
                f"{industry} Additional PD Stress",
                0.0, 0.05, 0.0, 0.001,
                format="%.3f"
            )
    
    return scenario, pd_multiplier, lgd_multiplier, correlation_adj, industry_stress

def apply_scenario_adjustments(inputs, loan_book, pd_mult, lgd_mult, corr_adj, industry_stress):
    """Apply scenario adjustments to inputs"""
    
    adjusted_inputs = inputs.copy()
    
    # Apply base scenario adjustments
    adjusted_inputs['PD'] *= pd_mult
    adjusted_inputs['LGD'] *= lgd_mult
    adjusted_inputs['rho'] = np.clip(adjusted_inputs['rho'] + corr_adj, 0.01, 0.99)
    
    # Apply industry-specific stress
    for industry, stress in industry_stress.items():
        industry_mask = loan_book['Industry Code'] == industry
        if industry_mask.any():
            adjusted_inputs.loc[industry_mask, 'PD'] += stress
    
    # Ensure PD doesn't exceed 100%
    adjusted_inputs['PD'] = np.clip(adjusted_inputs['PD'], 0.001, 0.99)
    
    return adjusted_inputs

# Risk Concentration Analysis
def create_concentration_analysis(loan_book, inputs):
    """Create concentration risk analysis"""
    
    st.header("🎯 Concentration Analysis")
    
    # Calculate concentration metrics
    total_exposure = inputs['EAD'].sum()
    
    # Top 10 exposures
    top_exposures = inputs.nlargest(10, 'EAD')
    top_10_concentration = top_exposures['EAD'].sum() / total_exposure
    
    # Industry concentration
    industry_exposure = loan_book.groupby('Industry Code')['Outstanding Balance'].sum()
    max_industry_conc = industry_exposure.max() / industry_exposure.sum()
    
    # Single name concentration
    max_single_exposure = inputs['EAD'].max()
    single_name_conc = max_single_exposure / total_exposure
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Top 10 Concentration", 
            f"{top_10_concentration:.1%}",
            help="Percentage of portfolio in top 10 exposures"
        )
    
    with col2:
        st.metric(
            "Max Industry Concentration", 
            f"{max_industry_conc:.1%}",
            help="Largest industry exposure as % of portfolio"
        )
    
    with col3:
        st.metric(
            "Largest Single Exposure", 
            f"{single_name_conc:.1%}",
            help="Largest single loan as % of portfolio"
        )
    
    # Concentration charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 exposures chart
        fig_top10 = px.bar(
            x=top_exposures['LoanID'],
            y=top_exposures['EAD'],
            title="Top 10 Exposures"
        )
        fig_top10.update_layout(xaxis_title="Loan ID", yaxis_title="Exposure at Default")
        st.plotly_chart(fig_top10, use_container_width=True)
    
    with col2:
        # Concentration curve
        sorted_exposures = inputs['EAD'].sort_values(ascending=False)
        cumulative_exposure = sorted_exposures.cumsum() / total_exposure
        
        fig_conc = go.Figure()
        fig_conc.add_trace(go.Scatter(
            x=list(range(1, len(cumulative_exposure) + 1)),
            y=cumulative_exposure.values,
            mode='lines+markers',
            name='Concentration Curve'
        ))
        
        # Add diagonal line for perfect diversification
        fig_conc.add_trace(go.Scatter(
            x=[1, len(inputs)],
            y=[0, 1],
            mode='lines',
            name='Perfect Diversification',
            line=dict(dash='dash')
        ))
        
        fig_conc.update_layout(
            title="Exposure Concentration Curve",
            xaxis_title="Number of Loans",
            yaxis_title="Cumulative Exposure %"
        )
        st.plotly_chart(fig_conc, use_container_width=True)

# Enhanced main function with all features
def enhanced_main():
    """Enhanced main function with all features"""
    
    # Load data
    loan_book, inputs, analytics, metrics = load_excel_data(uploaded_file)
    
    if loan_book is not None and inputs is not None:
        
        # Scenario Analysis Section
        st.markdown("---")
        scenario, pd_mult, lgd_mult, corr_adj, industry_stress = create_scenario_analysis()
        
        # Apply scenario adjustments
        adjusted_inputs = apply_scenario_adjustments(
            inputs, loan_book, pd_mult, lgd_mult, corr_adj, industry_stress
        )
        
        # Calculate metrics for both base and adjusted scenarios
        base_metrics = calculate_portfolio_metrics(loan_book, inputs)
        adjusted_metrics = calculate_portfolio_metrics(loan_book, adjusted_inputs)
        
        # Run simulations for both scenarios
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Base Case Results")
            with st.spinner('Running base case simulation...'):
                base_mc_losses = run_monte_carlo_simulation(inputs, num_simulations)
            
            # Base case dashboard
            create_risk_dashboard(loan_book, inputs, base_metrics)
        
        with col2:
            st.subheader("📊 Scenario Results")
            with st.spinner('Running scenario simulation...'):
                scenario_mc_losses = run_monte_carlo_simulation(adjusted_inputs, num_simulations)
            
            # Scenario dashboard
            create_risk_dashboard(loan_book, adjusted_inputs, adjusted_metrics)
        
        # Comparison section
        st.markdown("---")
        st.header("🔄 Scenario Comparison")
        
        comparison_df = pd.DataFrame({
            'Metric': ['Expected Loss', 'VaR 95%', 'VaR 99%', 'CVaR 95%'],
            'Base Case': [
                base_mc_losses.mean(),
                np.percentile(base_mc_losses, 95),
                np.percentile(base_mc_losses, 99),
                base_mc_losses[base_mc_losses >= np.percentile(base_mc_losses, 95)].mean()
            ],
            'Scenario': [
                scenario_mc_losses.mean(),
                np.percentile(scenario_mc_losses, 95),
                np.percentile(scenario_mc_losses, 99),
                scenario_mc_losses[scenario_mc_losses >= np.percentile(scenario_mc_losses, 95)].mean()
            ]
        })
        
        comparison_df['Change'] = comparison_df['Scenario'] - comparison_df['Base Case']
        comparison_df['Change %'] = (comparison_df['Change'] / comparison_df['Base Case']) * 100
        
        # Format currency columns
        for col in ['Base Case', 'Scenario', 'Change']:
            comparison_df[col] = comparison_df[col].apply(lambda x: f"${x:,.0f}")
        comparison_df['Change %'] = comparison_df['Change %'].apply(lambda x: f"{x:+.1f}%")
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualization comparison
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Histogram(
            x=base_mc_losses,
            name='Base Case',
            opacity=0.7,
            nbinsx=50
        ))
        
        fig_comparison.add_trace(go.Histogram(
            x=scenario_mc_losses,
            name=f'{scenario}',
            opacity=0.7,
            nbinsx=50
        ))
        
        fig_comparison.update_layout(
            title="Loss Distribution Comparison",
            xaxis_title="Portfolio Loss",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Concentration Analysis
        st.markdown("---")
        create_concentration_analysis(loan_book, inputs)
        
        # Original visualizations and reports
        st.markdown("---")
        create_visualizations(loan_book, inputs, base_mc_losses)
        
        st.markdown("---")
        create_reports_section(loan_book, inputs, base_metrics, base_mc_losses)
        
        # Data tables
        with st.expander("📋 View Raw Data"):
            tab1, tab2, tab3 = st.tabs(["Loan Book", "Base Risk Inputs", "Scenario Risk Inputs"])
            
            with tab1:
                st.dataframe(loan_book, use_container_width=True)
            
            with tab2:
                st.dataframe(inputs, use_container_width=True)
            
            with tab3:
                st.dataframe(adjusted_inputs, use_container_width=True)
    
    else:
        st.warning("Please upload your Excel file to begin analysis, or the app will use sample data.")
        
        # Show sample data info
        st.info("""
        **Sample Data Structure:**
        - 20 loans with realistic financial parameters
        - Industry codes: C10, C20, C40
        - Customer segments: SME, Corporate, Retail
        - Facility types: Term, Revolver, Bridge
        - Collateral types: Receivables, CRE, Securities
        
        Upload your Excel file to analyze your actual portfolio!
        """)

# Footer
def add_footer():
    """Add footer with instructions"""
    st.markdown("---")
    st.markdown("""
    ### 📋 Instructions
    
    1. **Upload Data**: Use the sidebar to upload your Excel file with loan portfolio data
    2. **Adjust Settings**: Configure Monte Carlo simulation parameters
    3. **Scenario Analysis**: Test different economic scenarios and stress conditions
    4. **Review Results**: Analyze portfolio metrics, visualizations, and concentration risks
    5. **Generate Reports**: Download executive summaries and detailed Excel reports
    
    **Excel File Format:**
    - Sheet 'LoanBook': Loan details and financial metrics
    - Sheet 'Inputs': Risk parameters (PD, LGD, EAD, correlation)
    - Optional sheets: 'Analytics', 'Metrics', 'MC Sim'
    
    **Created with Streamlit** 🚀 | **Powered by Monte Carlo Simulation** 📊
    """)

if __name__ == "__main__":
    enhanced_main()
    add_footer()