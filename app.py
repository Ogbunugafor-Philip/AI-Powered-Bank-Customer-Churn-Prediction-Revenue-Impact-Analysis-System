"""
AI-Powered Bank Customer Churn & Revenue Risk Dashboard
========================================================
A production-grade analytics dashboard for proactive customer retention
and revenue protection in the banking industry.

Author: Philip Ogbunugafor
License: MIT
Version: 2.0.0
"""

import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime
from typing import Optional, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

class DashboardConfig:
    """Centralized configuration for dashboard settings"""
    
    # Page Configuration
    PAGE_TITLE = "AI Churn & Revenue Risk Dashboard"
    PAGE_ICON = "📊"
    LAYOUT = "wide"
    
    # Database
    DB_PATH = "../../database/customer_analytics.db"
    
    # Color Schemes
    COLORS = {
        'primary': '#1f77b4',
        'success': '#2ecc71',
        'warning': '#f39c12',
        'danger': '#e74c3c',
        'info': '#3498db',
        'vip_critical': '#e74c3c',
        'vip_concern': '#f39c12',
        'general_risk': '#95a5a6'
    }
    
    # Priority Colors
    PRIORITY_COLORS = {
        'VIP Critical': '#e74c3c',
        'VIP Concern': '#f39c12',
        'General Risk': '#95a5a6'
    }
    
    # Chart Template
    CHART_TEMPLATE = "plotly_white"
    
    # Currency
    CURRENCY_SYMBOL = "₦"
    CURRENCY_FORMAT = "{:,.2f}"


# =============================================================================
# DATA ACCESS LAYER
# =============================================================================

class DataManager:
    """Handles all database operations and data loading"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def _get_connection(self) -> sqlite3.Connection:
        """Create database connection with error handling"""
        try:
            return sqlite3.connect(self.db_path)
        except sqlite3.Error as e:
            st.error(f"Database connection error: {e}")
            st.stop()
    
    @st.cache_data(ttl=300)
    def load_revenue_insights(_self) -> pd.DataFrame:
        """Load revenue insights data with caching"""
        try:
            conn = _self._get_connection()
            query = """
                SELECT 
                    customer_id,
                    churn_probability,
                    estimated_clv,
                    retention_priority,
                    revenue_risk
                FROM revenue_insights
                WHERE churn_probability IS NOT NULL
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                st.warning("No data found in revenue_insights table.")
                return pd.DataFrame()
            
            return df
        except Exception as e:
            st.error(f"Error loading revenue insights: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=300)
    def load_vip_recommendations(_self) -> pd.DataFrame:
        """Load VIP recommendations with caching"""
        try:
            conn = _self._get_connection()
            df = pd.read_sql_query("SELECT * FROM vip_recommendations", conn)
            conn.close()
            return df
        except Exception:
            return pd.DataFrame()


# =============================================================================
# ANALYTICS ENGINE
# =============================================================================

class ChurnAnalytics:
    """Business logic and analytics calculations"""
    
    @staticmethod
    def calculate_portfolio_metrics(df: pd.DataFrame) -> Dict:
        """Calculate key portfolio-level metrics"""
        if df.empty:
            return {
                'total_customers': 0,
                'total_revenue': 0.0,
                'avg_churn_prob': 0.0,
                'total_revenue_risk': 0.0,
                'high_risk_count': 0,
                'medium_risk_count': 0,
                'low_risk_count': 0,
                'vip_critical_count': 0,
                'avg_clv': 0.0,
                'median_churn_prob': 0.0,
                'total_clv_at_risk': 0.0
            }
        
        metrics = {
            'total_customers': int(len(df)),
            'total_revenue': float(df['estimated_clv'].sum()),
            'avg_churn_prob': float(df['churn_probability'].mean()),
            'total_revenue_risk': float(df['revenue_risk'].sum()),
            'high_risk_count': int((df['churn_probability'] >= 0.7).sum()),
            'medium_risk_count': int(((df['churn_probability'] >= 0.4) & 
                                  (df['churn_probability'] < 0.7)).sum()),
            'low_risk_count': int((df['churn_probability'] < 0.4).sum()),
            'vip_critical_count': int((df['retention_priority'] == 'VIP Critical').sum()),
            'avg_clv': float(df['estimated_clv'].mean()),
            'median_churn_prob': float(df['churn_probability'].median()),
            'total_clv_at_risk': float(df[df['churn_probability'] >= 0.5]['estimated_clv'].sum())
        }
        
        return metrics
    
    @staticmethod
    def calculate_risk_distribution(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk distribution by priority"""
        risk_dist = df.groupby('retention_priority').agg({
            'revenue_risk': 'sum',
            'customer_id': 'count',
            'estimated_clv': 'mean',
            'churn_probability': 'mean'
        }).reset_index()
        
        risk_dist.columns = ['retention_priority', 'total_risk', 'customer_count', 
                             'avg_clv', 'avg_churn_prob']
        risk_dist['risk_percentage'] = (risk_dist['total_risk'] / 
                                        risk_dist['total_risk'].sum() * 100)
        
        return risk_dist.sort_values('total_risk', ascending=False)


# =============================================================================
# VISUALIZATION COMPONENTS
# =============================================================================

class ChartBuilder:
    """Creates professional visualizations"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
    
    def create_risk_distribution_chart(self, risk_data: pd.DataFrame) -> go.Figure:
        """Create horizontal bar chart for risk distribution"""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=risk_data['retention_priority'],
            x=risk_data['total_risk'],
            orientation='h',
            text=[f"{val:,.0f}<br>({pct:.1f}%)" 
                  for val, pct in zip(risk_data['total_risk'], 
                                      risk_data['risk_percentage'])],
            textposition='outside',
            marker=dict(
                color=[self.config.PRIORITY_COLORS.get(p, '#95a5a6') 
                       for p in risk_data['retention_priority']],
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>%{y}</b><br>' +
                          'Revenue at Risk: ₦%{x:,.2f}<br>' +
                          '<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': 'Revenue Risk Distribution by Priority Tier',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            xaxis_title='Revenue at Risk (₦)',
            yaxis_title='Priority Tier',
            template=self.config.CHART_TEMPLATE,
            height=400,
            showlegend=False,
            margin=dict(l=20, r=150, t=60, b=20)
        )
        
        return fig
    
    def create_churn_vs_clv_scatter(self, df: pd.DataFrame) -> go.Figure:
        """Create scatter plot of churn probability vs CLV"""
        fig = px.scatter(
            df,
            x='churn_probability',
            y='estimated_clv',
            color='retention_priority',
            size='revenue_risk',
            hover_data={
                'customer_id': True,
                'churn_probability': ':.2%',
                'estimated_clv': ':,.2f',
                'revenue_risk': ':,.2f',
                'retention_priority': True
            },
            color_discrete_map=self.config.PRIORITY_COLORS,
            labels={
                'churn_probability': 'Churn Probability',
                'estimated_clv': f'Customer Lifetime Value ({self.config.CURRENCY_SYMBOL})',
                'retention_priority': 'Priority Tier',
                'revenue_risk': 'Revenue Risk'
            },
            template=self.config.CHART_TEMPLATE
        )
        
        fig.update_layout(
            title={
                'text': 'Customer Risk Matrix: Churn Probability vs Lifetime Value',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': '#2c3e50'}
            },
            height=500,
            xaxis=dict(tickformat='.0%'),
            yaxis=dict(tickformat=',.0f')
        )
        
        # Add quadrant lines
        fig.add_hline(y=df['estimated_clv'].median(), line_dash="dash", 
                     line_color="gray", opacity=0.5)
        fig.add_vline(x=df['churn_probability'].median(), line_dash="dash", 
                     line_color="gray", opacity=0.5)
        
        return fig
    
    def create_priority_pie_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create pie chart showing customer distribution by priority"""
        priority_counts = df['retention_priority'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=priority_counts.index,
            values=priority_counts.values,
            hole=0.4,
            marker=dict(
                colors=[self.config.PRIORITY_COLORS.get(p, '#95a5a6') 
                        for p in priority_counts.index],
                line=dict(color='white', width=2)
            ),
            textposition='auto',
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>' +
                          'Customers: %{value}<br>' +
                          'Percentage: %{percent}<br>' +
                          '<extra></extra>'
        )])
        
        fig.update_layout(
            title={
                'text': 'Customer Distribution by Priority',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            template=self.config.CHART_TEMPLATE,
            height=400,
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
        )
        
        return fig
    
    def create_churn_distribution_histogram(self, df: pd.DataFrame) -> go.Figure:
        """Create histogram of churn probability distribution"""
        fig = px.histogram(
            df,
            x='churn_probability',
            nbins=50,
            color='retention_priority',
            color_discrete_map=self.config.PRIORITY_COLORS,
            labels={
                'churn_probability': 'Churn Probability',
                'count': 'Number of Customers',
                'retention_priority': 'Priority Tier'
            },
            template=self.config.CHART_TEMPLATE
        )
        
        fig.update_layout(
            title={
                'text': 'Distribution of Churn Probability Across Portfolio',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            height=400,
            xaxis=dict(tickformat='.0%'),
            bargap=0.1
        )
        
        return fig


# =============================================================================
# UI COMPONENTS
# =============================================================================

class DashboardUI:
    """Manages dashboard UI components"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown("""
            <style>
            .main-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 10px;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .main-header h1 {
                color: white;
                margin: 0;
                font-size: 2.5rem;
                font-weight: 700;
            }
            .main-header p {
                color: #f0f0f0;
                margin: 0.5rem 0 0 0;
                font-size: 1.1rem;
            }
            </style>
            <div class="main-header">
                <h1>📊 AI-Powered Churn & Revenue Risk Dashboard</h1>
                <p>Proactive customer retention and revenue protection through advanced analytics</p>
            </div>
        """, unsafe_allow_html=True)
    
    def format_number(self, num: float) -> str:
        """Safely format number with commas"""
        try:
            return f"{num:,.0f}"
        except:
            return "0"
    
    def format_percentage(self, num: float) -> str:
        """Safely format percentage"""
        try:
            return f"{num*100:.2f}%"
        except:
            return "0.00%"
    
    def format_currency(self, num: float) -> str:
        """Safely format currency"""
        try:
            return f"₦{num:,.0f}"
        except:
            return "₦0"
    
    def render_kpi_cards(self, metrics: Dict):
        """Render executive KPI cards - 3 rows with theme-adaptive colors"""
        st.markdown("### 📈 Executive Summary")
        
        # Row 1: Total Customers, Total Revenue, Total Revenue at Risk
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div style='background-color: #ffffff; padding: 1.5rem; border-radius: 10px; 
                     border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;'>📊 Total Customers</div>
                    <div style='color: #1a1a1a; font-size: 2rem; font-weight: bold;'>{self.format_number(metrics['total_customers'])}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div style='background-color: #ffffff; padding: 1.5rem; border-radius: 10px; 
                     border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;'>💵 Total Revenue (CLV)</div>
                    <div style='color: #2ecc71; font-size: 2rem; font-weight: bold;'>{self.format_currency(metrics['total_revenue'])}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div style='background-color: #ffffff; padding: 1.5rem; border-radius: 10px; 
                     border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;'>💰 Total Revenue at Risk</div>
                    <div style='color: #e74c3c; font-size: 2rem; font-weight: bold;'>{self.format_currency(metrics['total_revenue_risk'])}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.write("")
        
        # Row 2: VIP Critical, High Risk, Average Customer Value
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div style='background-color: #ffffff; padding: 1.5rem; border-radius: 10px; 
                     border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;'>🚨 VIP Critical Customers</div>
                    <div style='color: #e74c3c; font-size: 2rem; font-weight: bold;'>{self.format_number(metrics['vip_critical_count'])}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div style='background-color: #ffffff; padding: 1.5rem; border-radius: 10px; 
                     border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;'>⚠️ High Risk Customers</div>
                    <div style='color: #f39c12; font-size: 2rem; font-weight: bold;'>{self.format_number(metrics['high_risk_count'])}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div style='background-color: #ffffff; padding: 1.5rem; border-radius: 10px; 
                     border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;'>💎 Average Customer Value</div>
                    <div style='color: #2ecc71; font-size: 2rem; font-weight: bold;'>{self.format_currency(metrics['avg_clv'])}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.write("")
        
        # Row 3: Average Churn Risk, Median Churn Probability, CLV at High Risk
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div style='background-color: #ffffff; padding: 1.5rem; border-radius: 10px; 
                     border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;'>🎯 Average Churn Risk</div>
                    <div style='color: #1a1a1a; font-size: 2rem; font-weight: bold;'>{self.format_percentage(metrics['avg_churn_prob'])}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div style='background-color: #ffffff; padding: 1.5rem; border-radius: 10px; 
                     border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;'>📉 Median Churn Probability</div>
                    <div style='color: #1a1a1a; font-size: 2rem; font-weight: bold;'>{self.format_percentage(metrics['median_churn_prob'])}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div style='background-color: #ffffff; padding: 1.5rem; border-radius: 10px; 
                     border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <div style='color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;'>🔥 CLV at High Risk</div>
                    <div style='color: #e74c3c; font-size: 2rem; font-weight: bold;'>{self.format_currency(metrics['total_clv_at_risk'])}</div>
                </div>
            """, unsafe_allow_html=True)
    
    def render_sidebar_filters(self, df: pd.DataFrame) -> Tuple:
        """Render sidebar filters and return filter values"""
        st.sidebar.markdown("## 🔎 Dashboard Filters")
        st.sidebar.markdown("---")
        
        # Priority Filter
        st.sidebar.markdown("### Retention Priority")
        priority_options = ["All"] + sorted(df['retention_priority'].unique().tolist())
        selected_priority = st.sidebar.selectbox(
            "Filter by Priority Tier",
            priority_options,
            key="priority_filter"
        )
        
        # Churn Probability Range
        st.sidebar.markdown("### Churn Probability Range")
        churn_range = st.sidebar.slider(
            "Select Range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.01,
            key="churn_slider"
        )
        
        # CLV Range
        st.sidebar.markdown("### Customer Lifetime Value")
        min_clv, max_clv = float(df['estimated_clv'].min()), float(df['estimated_clv'].max())
        clv_range = st.sidebar.slider(
            "CLV Range",
            min_value=min_clv,
            max_value=max_clv,
            value=(min_clv, max_clv),
            step=10000.0,
            key="clv_slider"
        )
        
        st.sidebar.markdown("---")
        
        # Data Refresh
        if st.sidebar.button("🔄 Refresh Data", key="refresh_btn"):
            st.cache_data.clear()
            st.rerun()
        
        return selected_priority, churn_range, clv_range
    
    def render_complete_portfolio(self, df: pd.DataFrame):
        """Render complete customer portfolio with all data"""
        st.markdown("### 📋 Complete Customer Portfolio Analysis")
        
        st.markdown("""
            <div style='background-color: #e8f4f8; padding: 1rem; border-radius: 5px; 
                 border-left: 5px solid #17a2b8; margin-bottom: 1rem;'>
                <strong style='color: #0c5460;'>📊 Full Dataset:</strong> 
                <span style='color: #0c5460;'>View and download complete churn analysis for all customers. 
                Use this data for deeper analysis, custom reporting, or integration with other systems.</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Summary stats for filtered data
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        with col2:
            st.metric("Avg Churn Risk", f"{df['churn_probability'].mean()*100:.2f}%")
        with col3:
            st.metric("Total CLV", f"₦{df['estimated_clv'].sum():,.0f}")
        with col4:
            st.metric("Total Risk", f"₦{df['revenue_risk'].sum():,.0f}")
        
        st.write("")
        
        # Prepare display dataframe
        display_df = df.copy()
        display_df = display_df.sort_values('revenue_risk', ascending=False)
        
        # Format for display
        display_df_formatted = display_df.copy()
        display_df_formatted['churn_probability'] = display_df_formatted['churn_probability'].apply(
            lambda x: self.format_percentage(x)
        )
        display_df_formatted['estimated_clv'] = display_df_formatted['estimated_clv'].apply(
            lambda x: self.format_currency(x)
        )
        display_df_formatted['revenue_risk'] = display_df_formatted['revenue_risk'].apply(
            lambda x: self.format_currency(x)
        )
        
        display_df_formatted.columns = [
            'Customer ID', 'Churn Probability', 'Customer Lifetime Value', 
            'Retention Priority', 'Revenue at Risk'
        ]
        
        # Show data with search
        st.markdown("#### Customer Data Table")
        st.caption(f"Showing all {len(display_df_formatted):,} customers (filtered by your selections)")
        
        st.dataframe(
            display_df_formatted,
            use_container_width=True,
            hide_index=True,
            height=500
        )
        
        # Download buttons
        st.markdown("#### Download Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download complete data as CSV
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Complete Dataset (CSV)",
                data=csv,
                file_name=f"complete_churn_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download all customer data with churn predictions for analysis in Excel, Python, etc."
            )
        
        with col2:
            # Download high risk only
            high_risk_df = display_df[display_df['churn_probability'] >= 0.7]
            if not high_risk_df.empty:
                csv_high = high_risk_df.to_csv(index=False)
                st.download_button(
                    label="⚠️ Download High Risk Only (CSV)",
                    data=csv_high,
                    file_name=f"high_risk_customers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help=f"Download {len(high_risk_df):,} customers with 70%+ churn probability"
                )
            else:
                st.info("No high-risk customers in current filter")
        
        with col3:
            # Download VIP only
            vip_df = display_df[display_df['retention_priority'] == 'VIP Critical']
            if not vip_df.empty:
                csv_vip = vip_df.to_csv(index=False)
                st.download_button(
                    label="🚨 Download VIP Critical (CSV)",
                    data=csv_vip,
                    file_name=f"vip_critical_customers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help=f"Download {len(vip_df):,} VIP Critical customers"
                )
            else:
                st.info("No VIP Critical customers in current filter")
    
    def render_vip_watchlist(self, df: pd.DataFrame):
        """Render VIP customer watchlist"""
        st.markdown("### 🧲 VIP Churn Watchlist (Quick View)")
        st.markdown("""
            <div style='background-color: #fff3cd; padding: 1rem; border-radius: 5px; 
                 border-left: 5px solid #ffc107; margin-bottom: 1rem;'>
                <strong style='color: #856404;'>⚠️ Priority Action Required:</strong> 
                <span style='color: #856404;'>Top 20 high-value customers with elevated churn risk. 
                These represent the highest individual revenue exposure and require immediate retention interventions.</span>
            </div>
        """, unsafe_allow_html=True)
        
        vip_critical = df[df['retention_priority'] == 'VIP Critical'].copy()
        vip_critical = vip_critical.sort_values('revenue_risk', ascending=False)
        
        if not vip_critical.empty:
            # Format the dataframe for display
            display_df = vip_critical[[
                'customer_id', 'churn_probability', 'estimated_clv', 'revenue_risk'
            ]].head(20).copy()
            
            display_df['churn_probability'] = display_df['churn_probability'].apply(
                lambda x: self.format_percentage(x)
            )
            display_df['estimated_clv'] = display_df['estimated_clv'].apply(
                lambda x: self.format_currency(x)
            )
            display_df['revenue_risk'] = display_df['revenue_risk'].apply(
                lambda x: self.format_currency(x)
            )
            
            display_df.columns = ['Customer ID', 'Churn Risk', 'Lifetime Value', 'Revenue at Risk']
            
            # Style the dataframe
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            st.caption(f"Showing top 20 of {len(vip_critical):,} VIP Critical customers. See 'Complete Customer Portfolio' section above for full data.")
        else:
            st.info("✅ No VIP Critical customers detected in current filters.")
    
    def render_recommendations_table(self, vip_df: pd.DataFrame):
        """Render AI recommendations table"""
        st.markdown("### 🧠 AI-Driven Retention Strategies")
        
        if not vip_df.empty and 'recommended_strategy' in vip_df.columns:
            st.markdown("""
                <div style='background-color: #d1ecf1; padding: 1rem; border-radius: 5px; 
                     border-left: 5px solid #17a2b8; margin-bottom: 1rem;'>
                    <strong style='color: #0c5460;'>💡 Intelligent Recommendations:</strong> 
                    <span style='color: #0c5460;'>AI-generated retention strategies based on customer behavior patterns, risk factors, and lifetime value.</span>
                </div>
            """, unsafe_allow_html=True)
            
            display_df = vip_df[[
                'customer_id', 'churn_probability', 'estimated_clv', 
                'revenue_risk', 'recommended_strategy'
            ]].head(15).copy()
            
            display_df['churn_probability'] = display_df['churn_probability'].apply(
                lambda x: self.format_percentage(x)
            )
            display_df['estimated_clv'] = display_df['estimated_clv'].apply(
                lambda x: self.format_currency(x)
            )
            display_df['revenue_risk'] = display_df['revenue_risk'].apply(
                lambda x: self.format_currency(x)
            )
            
            display_df.columns = [
                'Customer ID', 'Churn Risk', 'Lifetime Value', 
                'Revenue at Risk', 'Recommended Action'
            ]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=400
            )
        else:
            st.info("💡 Run Phase 4 Step 4.4 to generate AI recommendations.")


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point"""
    
    # Initialize configuration
    config = DashboardConfig()
    
    # Set page configuration
    st.set_page_config(
        page_title=config.PAGE_TITLE,
        page_icon=config.PAGE_ICON,
        layout=config.LAYOUT,
        initial_sidebar_state="expanded"
    )
    
    # Initialize components
    data_manager = DataManager(config.DB_PATH)
    analytics = ChurnAnalytics()
    chart_builder = ChartBuilder(config)
    ui = DashboardUI(config)
    
    # Render header
    ui.render_header()
    
    # Load data
    with st.spinner('🔄 Loading customer analytics data...'):
        revenue_df = data_manager.load_revenue_insights()
        vip_df = data_manager.load_vip_recommendations()
    
    # Check if data exists
    if revenue_df.empty:
        st.error("""
            ❌ **No Data Available**
            
            The revenue_insights table is empty or missing. Please ensure:
            1. Phase 2 (Model Development) is completed
            2. Phase 3 (Model Evaluation) is completed
            3. Phase 4 (Revenue Analysis) is completed
            
            Run all prerequisite scripts before launching the dashboard.
        """)
        st.stop()
    
    # Sidebar filters
    selected_priority, churn_range, clv_range = ui.render_sidebar_filters(revenue_df)
    
    # Apply filters
    filtered_df = revenue_df.copy()
    
    if selected_priority != "All":
        filtered_df = filtered_df[filtered_df['retention_priority'] == selected_priority]
    
    filtered_df = filtered_df[
        (filtered_df['churn_probability'] >= churn_range[0]) &
        (filtered_df['churn_probability'] <= churn_range[1]) &
        (filtered_df['estimated_clv'] >= clv_range[0]) &
        (filtered_df['estimated_clv'] <= clv_range[1])
    ]
    
    # Calculate metrics
    metrics = analytics.calculate_portfolio_metrics(filtered_df)
    
    # Render KPI cards
    ui.render_kpi_cards(metrics)
    
    st.markdown("---")
    
    # Analytics Section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Risk distribution chart
        risk_dist = analytics.calculate_risk_distribution(filtered_df)
        fig_risk = chart_builder.create_risk_distribution_chart(risk_dist)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Priority pie chart
        fig_pie = chart_builder.create_priority_pie_chart(filtered_df)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Churn vs CLV scatter plot
    fig_scatter = chart_builder.create_churn_vs_clv_scatter(filtered_df)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")
    
    # Churn distribution histogram
    fig_hist = chart_builder.create_churn_distribution_histogram(filtered_df)
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    
    # COMPLETE PORTFOLIO SECTION (NEW!)
    ui.render_complete_portfolio(filtered_df)
    
    st.markdown("---")
    
    # VIP Watchlist (Top 20 Quick View)
    ui.render_vip_watchlist(filtered_df)
    
    st.markdown("---")
    
    # AI Recommendations
    ui.render_recommendations_table(vip_df)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
        <div style='text-align: center; padding: 2rem; background-color: #f8f9fa; 
             border-radius: 10px; margin-top: 2rem;'>
            <p style='color: #1a1a1a; margin: 0;'>
                <strong>AI-Powered Bank Customer Churn Prevention System</strong><br>
                <span style='color: #666;'>Built with 🧠 Machine Learning • 📊 Advanced Analytics • 💡 Business Intelligence</span><br>
                <small style='color: #999;'>Transforming churn predictions into strategic revenue protection</small>
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
