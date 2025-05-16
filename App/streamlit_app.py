import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from typing import Optional, Tuple

# Constants
MODEL_PATH = "C:/Users/Nour/OneDrive/Desktop/Streamlit/Models/best_demand_forecast_model.keras"

class DemandForecaster:
    """Handles model loading and prediction operations."""
    
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
    
    @st.cache_resource
    def _load_model(_self, path: str):
        """Load and cache the demand forecasting model."""
        return load_model(path)
    
    def predict(_self, input_data: pd.DataFrame) -> Tuple[int, int]:
        """Make predictions and return rounded integer results."""
        prediction = _self.model.predict(input_data)
        return (int(round(prediction[0][0])), int(round(prediction[0][1])))

def setup_page() -> None:
    """Configure page settings and apply custom styles."""
    st.set_page_config(
        page_title="Sales & Demand Forecasting",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
            /* Main layout adjustments */
            .block-container {padding-top: 2rem; padding-bottom: 2rem;}
            .stButton>button {border-radius: 4px; font-weight: 500;}
            
            /* Card-like containers */
            .metric-card {
                border-radius: 8px;
                padding: 1.5rem;
                background: rgba(240, 242, 246, 0.6);
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                margin-bottom: 1rem;
            }
            
            /* Better tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
            }
            .stTabs [data-baseweb="tab"] {
                padding: 8px 16px;
                border-radius: 4px 4px 0 0;
            }
            
            /* Responsive adjustments */
            @media (max-width: 768px) {
                .metric-card {padding: 1rem;}
            }
        </style>
    """, unsafe_allow_html=True)

def create_sidebar_inputs() -> pd.DataFrame:
    """Create and collect sidebar inputs, return as DataFrame."""
    with st.sidebar:
        st.header("🔧 Model Parameters")
        
        # Using columns for better organization
        col1, col2 = st.columns(2)
        
        with col1:
            inventory_level = st.number_input(
                "Inventory Level", 
                min_value=0, 
                value=0, 
                help="Current on-hand inventory units",
                key="inventory"
            )
            price = st.number_input(
                "Price (USD)", 
                min_value=0.0, 
                value=0.0, 
                format="%.2f",
                help="Selling price per unit",
                key="price"
            )
            
        with col2:
            seasonality = st.selectbox(
                "Summer Season", 
                options=[0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                help="Is it currently summer season?",
                key="season"
            )
            inventory_demand = st.number_input(
                "Demand", 
                min_value=0, 
                value=0,
                help="Current customer demand for inventory",
                key="demand"
            )
        
        units_sold_price = st.slider(
            "Price Sensitivity", 
            min_value=0.0, 
            max_value=10.0, 
            value=0.0, 
            step=0.5,
            help="Units sold influenced by price sensitivity",
            key="sensitivity"
        )
        
    return pd.DataFrame({
        'Inventory Level': [inventory_level],
        'Price': [price],
        'Seasonality_Summer': [seasonality],
        'Inventory_Demand': [inventory_demand],
        'UnitsSold_Price': [units_sold_price]
    })

def display_forecast_results(forecaster: DemandForecaster, data: pd.DataFrame) -> None:
    """Display forecast results in a visually appealing way."""
    with st.container():
        st.subheader("📊 Forecast Dashboard")
        
        if st.button("Run Forecast", type="primary", use_container_width=True):
            with st.spinner("Generating predictions..."):
                units_sold, demand_forecast = forecaster.predict(data)
                
            st.success("Forecast completed successfully!")
            
            # Display metrics in cards
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="font-size: 14px; color: #666; margin-bottom: 8px;">
                            <i class="fas fa-box-open"></i> Units Sold
                        </div>
                        <div style="font-size: 32px; font-weight: 600;">
                            {units_sold:,}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="font-size: 14px; color: #666; margin-bottom: 8px;">
                            <i class="fas fa-chart-line"></i> Demand Forecast
                        </div>
                        <div style="font-size: 32px; font-weight: 600;">
                            {demand_forecast:,}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Add some visual separation
            st.markdown("---")
            
            # Placeholder for future visualization
            st.write("📈 *Forecast visualization coming soon...*")

def display_input_data(data: pd.DataFrame) -> None:
    """Display input data in a clean format."""
    with st.container():
        st.subheader("🔍 Input Parameters")
        
        # Show data in a styled table
        st.dataframe(
            data.style.format({
                'Price': "${:,.2f}",
                'UnitsSold_Price': "{:,.1f}"
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Add some metrics about the inputs
        st.caption(f"Total potential inventory value: ${data['Inventory Level'][0] * data['Price'][0]:,.2f}")

def main():
    """Main application flow."""
    setup_page()
    
    # Load model and create forecaster instance
    forecaster = DemandForecaster(MODEL_PATH)
    
    # Get user inputs from sidebar
    input_data = create_sidebar_inputs()
    
    # Create tabbed interface
    tab1, tab2 = st.tabs(["Forecast Dashboard", "Input Analysis"])
    
    with tab1:
        display_forecast_results(forecaster, input_data)
    
    with tab2:
        display_input_data(input_data)
    
    # Minimal footer
    st.markdown("---")
    st.caption("© 2023 Sales Forecasting Suite | v2.0")

if __name__ == "__main__":
    main()