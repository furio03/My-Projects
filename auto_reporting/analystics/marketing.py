import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from data_manipulation.read_data import identify_variables, correct_dtypes
from statistics_functions.clustering import auto_kmeans, auto_kmedoids
from LLM.llm_output import needed_variables


def analytics_marketing(df: pd.DataFrame, mapping: dict = None, user_objective: str = None,
                        save_plots: bool = False, output_dir: str = "domain_plots"):
    """
    Comprehensive marketing analytics to provide actionable insights.
    
    Analyses performed:
    - A: Channel Performance Analysis - Revenue, conversions, ROI by channel
    - B: Campaign Effectiveness - ROI, CAC, conversion rates by campaign
    - C: Customer Acquisition Cost (CAC) - Overall and by channel/campaign
    - D: Customer Lifetime Value (CLTV) - Revenue patterns and segmentation
    - E: Conversion Funnel Analysis - Drop-off rates at each stage
    - F: RFM Segmentation - Customer value segmentation with clustering
    - G: Customer Retention & Churn - Monthly trends and patterns
    - H: Budget Allocation Optimization - Recommendations based on ROI
    - I: Marketing Performance Trends - Time-based performance analysis
    - J: Marketing Forecasting - Predict future performance metrics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input marketing dataset
    mapping : dict, optional
        Column mapping for marketing roles
    user_objective : str, optional
        User's analysis objective
    save_plots : bool, default=False
        Whether to save plots to disk
    output_dir : str, default="marketing_analysis_plots"
        Directory to save plots
    """
    print("Running comprehensive marketing analytics")
    
    # Create output directory for plots if needed
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Plots will be saved to: {output_dir}/")
    
    # Helper function to save plots
    def save_plot(filename):
        """Save plot to output directory if save_plots is enabled"""
        if save_plots:
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"   ✓ Plot saved: {filename}")

    # Define marketing-specific roles for variable mapping
    roles = [
        'monetary_value',    # Revenue, Sales
        'volume_metrics',    # Conversions, Orders, Clicks
        'business_dimension', # Product, Category
        'channel',           # Marketing Channel
        'campaign',          # Campaign Name
        'customer_id',       # Customer Identifier
        'date',             # Transaction/Event Date
        'marketing_cost'     # Advertising Spend
    ]

    if mapping is None:
        mapping = needed_variables(df, roles)

    def get_single_col(key):
        """Extract single column name from mapping"""
        val = mapping.get(key)
        if not val:
            return None
        if isinstance(val, list):
            val = val[0] if len(val) > 0 else None
        if isinstance(val, str) and ',' in val:
            val = val.split(',')[0].strip()
        return val if val in df.columns else None

    # Extract variable names
    revenue_col = get_single_col('monetary_value')
    volume_col = get_single_col('volume_metrics')
    dim_col = get_single_col('business_dimension')
    channel_col = get_single_col('channel')
    campaign_col = get_single_col('campaign')
    customer_col = get_single_col('customer_id')
    date_col = get_single_col('date')
    cost_col = get_single_col('marketing_cost')

    var_types = identify_variables(df)
    numeric_cols = var_types.get('quantitative', [])

    # Fallback warnings
    missing_cols = []
    if not channel_col:
        missing_cols.append('channel (Marketing Channel)')
    if not revenue_col and not volume_col:
        missing_cols.append('monetary_value or volume_metrics (Revenue/Conversions)')
    
    if missing_cols:
        print(f"\n⚠️  WARNING: Missing columns detected:")
        for col in missing_cols:
            print(f"   - {col}")
        print("   Some analyses will be skipped.\n")

    results = {}

    # --- ANALYSIS A: Channel Performance ---
    print("\n📊 A. Analyzing Channel Performance...")
    
    if channel_col and (revenue_col or volume_col):
        try:
            temp_ch = df[[channel_col]].copy()
            agg_map = {}
            
            if revenue_col and revenue_col in numeric_cols:
                temp_ch[revenue_col] = pd.to_numeric(df[revenue_col], errors='coerce')
                agg_map[revenue_col] = 'sum'
            
            if volume_col and volume_col in numeric_cols:
                temp_ch[volume_col] = pd.to_numeric(df[volume_col], errors='coerce')
                agg_map[volume_col] = 'sum'
            
            if cost_col and cost_col in numeric_cols:
                temp_ch[cost_col] = pd.to_numeric(df[cost_col], errors='coerce')
                agg_map[cost_col] = 'sum'

            if agg_map:
                channel_perf = temp_ch.groupby(channel_col).agg(agg_map)
                
                if len(channel_perf) > 0:
                    # Rename columns
                    rename_dict = {}
                    if volume_col and volume_col in channel_perf.columns:
                        rename_dict[volume_col] = 'conversions'
                    if revenue_col and revenue_col in channel_perf.columns:
                        rename_dict[revenue_col] = 'revenue'
                    if cost_col and cost_col in channel_perf.columns:
                        rename_dict[cost_col] = 'cost'
                    
                    if rename_dict:
                        channel_perf = channel_perf.rename(columns=rename_dict)
                    
                    # Calculate metrics
                    if 'cost' in channel_perf.columns and 'conversions' in channel_perf.columns:
                        channel_perf['cost_per_conversion'] = channel_perf['cost'] / channel_perf['conversions'].replace(0, 1)
                    
                    if 'cost' in channel_perf.columns and 'revenue' in channel_perf.columns:
                        channel_perf['roi_pct'] = ((channel_perf['revenue'] - channel_perf['cost']) / channel_perf['cost'].replace(0, 1)) * 100
                    
                    channel_perf = channel_perf.sort_values(
                        by='roi_pct' if 'roi_pct' in channel_perf.columns else channel_perf.columns[0], 
                        ascending=False
                    )
                    
                    results['channel_performance'] = channel_perf
                    
                    # Visualization: Channel Performance
                    try:
                        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
                        
                        # Left: Revenue by channel
                        if 'revenue' in channel_perf.columns:
                            colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(channel_perf)))
                            bars1 = axes[0].barh(range(len(channel_perf)), channel_perf['revenue'], 
                                                color=colors, edgecolor='black', linewidth=1.5)
                            axes[0].set_yticks(range(len(channel_perf)))
                            axes[0].set_yticklabels(channel_perf.index, fontsize=11)
                            axes[0].set_xlabel('Total Revenue (€)', fontsize=12, fontweight='bold')
                            axes[0].set_title('Revenue by Marketing Channel', fontsize=14, fontweight='bold')
                            axes[0].grid(axis='x', alpha=0.3)
                            
                            for i, bar in enumerate(bars1):
                                width = bar.get_width()
                                axes[0].text(width + width*0.02, bar.get_y() + bar.get_height()/2.,
                                           f'€{int(width):,}', ha='left', va='center', fontweight='bold')
                        
                        # Right: ROI by channel
                        if 'roi_pct' in channel_perf.columns:
                            roi_sorted = channel_perf.sort_values('roi_pct', ascending=False)
                            colors2 = ['#2ecc71' if x > 0 else '#e74c3c' for x in roi_sorted['roi_pct']]
                            bars2 = axes[1].barh(range(len(roi_sorted)), roi_sorted['roi_pct'], 
                                                color=colors2, edgecolor='black', linewidth=1.5)
                            axes[1].set_yticks(range(len(roi_sorted)))
                            axes[1].set_yticklabels(roi_sorted.index, fontsize=11)
                            axes[1].set_xlabel('ROI (%)', fontsize=12, fontweight='bold')
                            axes[1].set_title('Return on Investment by Channel', fontsize=14, fontweight='bold')
                            axes[1].axvline(0, color='black', linestyle='-', linewidth=1)
                            axes[1].grid(axis='x', alpha=0.3)
                            
                            for i, bar in enumerate(bars2):
                                width = bar.get_width()
                                axes[1].text(width + (5 if width > 0 else -5), bar.get_y() + bar.get_height()/2.,
                                           f'{width:.1f}%', ha='left' if width > 0 else 'right', 
                                           va='center', fontweight='bold')
                        
                        plt.tight_layout()
                        save_plot('01_channel_performance.png')
                        plt.show()
                    except Exception as e:
                        print(f"   ⚠️ Could not generate channel visualization: {e}")
                    
                    print(f"   ✓ Analyzed {len(channel_perf)} marketing channels")
        except Exception as e:
            print(f"   ⚠️ Channel performance analysis failed: {e}")
    else:
        print("   ⚠️ Skipping channel analysis (channel or revenue/volume column not available)")

    # --- ANALYSIS B: Campaign Performance ---
    print("\n🎯 B. Analyzing Campaign Effectiveness...")
    
    if campaign_col and (revenue_col or volume_col or cost_col):
        try:
            temp_camp = df[[campaign_col]].copy()
            agg_map = {}
            
            if revenue_col and revenue_col in numeric_cols:
                temp_camp[revenue_col] = pd.to_numeric(df[revenue_col], errors='coerce')
                agg_map[revenue_col] = 'sum'
            
            if volume_col and volume_col in numeric_cols:
                temp_camp[volume_col] = pd.to_numeric(df[volume_col], errors='coerce')
                agg_map[volume_col] = 'sum'
            
            if cost_col and cost_col in numeric_cols:
                temp_camp[cost_col] = pd.to_numeric(df[cost_col], errors='coerce')
                agg_map[cost_col] = 'sum'

            if agg_map:
                camp_perf = temp_camp.groupby(campaign_col).agg(agg_map)
                
                if len(camp_perf) > 0:
                    rename_dict = {}
                    if revenue_col and revenue_col in camp_perf.columns:
                        rename_dict[revenue_col] = 'revenue'
                    if volume_col and volume_col in camp_perf.columns:
                        rename_dict[volume_col] = 'conversions'
                    if cost_col and cost_col in camp_perf.columns:
                        rename_dict[cost_col] = 'cost'
                    
                    if rename_dict:
                        camp_perf = camp_perf.rename(columns=rename_dict)
                    
                    # Calculate metrics
                    if 'cost' in camp_perf.columns and 'revenue' in camp_perf.columns:
                        camp_perf['roi_pct'] = ((camp_perf['revenue'] - camp_perf['cost']) / camp_perf['cost'].replace(0, 1)) * 100
                    
                    if 'cost' in camp_perf.columns and 'conversions' in camp_perf.columns:
                        camp_perf['cac'] = camp_perf['cost'] / camp_perf['conversions'].replace(0, 1)
                    
                    camp_perf = camp_perf.sort_values(
                        by='roi_pct' if 'roi_pct' in camp_perf.columns else 'revenue' if 'revenue' in camp_perf.columns else camp_perf.columns[0],
                        ascending=False
                    )
                    
                    results['campaign_performance'] = camp_perf
                    
                    # Visualization: Top 10 Campaigns
                    try:
                        top_campaigns = camp_perf.head(10)
                        
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
                        
                        # Left: Campaign ROI
                        if 'roi_pct' in top_campaigns.columns:
                            colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_campaigns['roi_pct']]
                            bars1 = ax1.bar(range(len(top_campaigns)), top_campaigns['roi_pct'], 
                                           color=colors, edgecolor='black', linewidth=1.5)
                            ax1.set_xticks(range(len(top_campaigns)))
                            ax1.set_xticklabels(top_campaigns.index, rotation=45, ha='right', fontsize=10)
                            ax1.set_ylabel('ROI (%)', fontsize=12, fontweight='bold')
                            ax1.set_title('Top 10 Campaigns by ROI', fontsize=14, fontweight='bold')
                            ax1.axhline(0, color='black', linestyle='-', linewidth=1)
                            ax1.grid(axis='y', alpha=0.3)
                            
                            for i, bar in enumerate(bars1):
                                height = bar.get_height()
                                ax1.text(bar.get_x() + bar.get_width()/2., height,
                                        f'{height:.1f}%', ha='center', 
                                        va='bottom' if height > 0 else 'top', 
                                        fontweight='bold', fontsize=9)
                        
                        # Right: CAC by campaign
                        if 'cac' in top_campaigns.columns:
                            cac_sorted = top_campaigns.sort_values('cac')
                            colors2 = plt.cm.Reds(np.linspace(0.9, 0.4, len(cac_sorted)))
                            bars2 = ax2.barh(range(len(cac_sorted)), cac_sorted['cac'], 
                                            color=colors2, edgecolor='black', linewidth=1.5)
                            ax2.set_yticks(range(len(cac_sorted)))
                            ax2.set_yticklabels(cac_sorted.index, fontsize=10)
                            ax2.set_xlabel('Customer Acquisition Cost (€)', fontsize=12, fontweight='bold')
                            ax2.set_title('CAC by Campaign (Lower is Better)', fontsize=14, fontweight='bold')
                            ax2.grid(axis='x', alpha=0.3)
                            
                            for i, bar in enumerate(bars2):
                                width = bar.get_width()
                                ax2.text(width + width*0.02, bar.get_y() + bar.get_height()/2.,
                                        f'€{width:.2f}', ha='left', va='center', fontweight='bold', fontsize=9)
                        
                        plt.tight_layout()
                        save_plot('02_campaign_effectiveness.png')
                        plt.show()
                    except Exception as e:
                        print(f"   ⚠️ Could not generate campaign visualization: {e}")
                    
                    print(f"   ✓ Analyzed {len(camp_perf)} campaigns")
        except Exception as e:
            print(f"   ⚠️ Campaign performance analysis failed: {e}")
    else:
        print("   ⚠️ Skipping campaign analysis (campaign column not available)")

    # --- ANALYSIS C: Customer Acquisition Cost (CAC) ---
    print("\n💰 C. Calculating Customer Acquisition Cost...")
    
    if cost_col and cost_col in numeric_cols and customer_col:
        try:
            total_cost = pd.to_numeric(df[cost_col], errors='coerce').sum(skipna=True)
            unique_customers = df[customer_col].nunique()
            if unique_customers > 0 and not pd.isna(total_cost) and total_cost > 0:
                cac_overall = float(round(total_cost / unique_customers, 2))
                results['cac_overall'] = cac_overall
                print(f"   ✓ Overall CAC: €{cac_overall:.2f} per customer")
        except Exception as e:
            print(f"   ⚠️ CAC calculation failed: {e}")
    else:
        print("   ⚠️ Skipping CAC calculation (cost or customer column not available)")

    # --- ANALYSIS D: Customer Lifetime Value (CLTV) ---
    print("\n💎 D. Analyzing Customer Lifetime Value...")
    
    if revenue_col and revenue_col in numeric_cols and customer_col and channel_col:
        try:
            temp_cltv = df[[channel_col, customer_col, revenue_col]].copy()
            temp_cltv[revenue_col] = pd.to_numeric(temp_cltv[revenue_col], errors='coerce')
            temp_cltv = temp_cltv.dropna(subset=[revenue_col])
            
            if len(temp_cltv) > 0:
                cust_metrics = temp_cltv.groupby([channel_col, customer_col]).agg({revenue_col: 'sum'})
                cust_metrics = cust_metrics.groupby(level=0).agg({revenue_col: ['mean', 'count']})
                cust_metrics.columns = ['avg_revenue_per_customer', 'customer_count']
                cust_metrics['cltv_proxy'] = cust_metrics['avg_revenue_per_customer'] * cust_metrics['customer_count']
                cust_metrics = cust_metrics.sort_values(by='cltv_proxy', ascending=False)
                results['cltv_by_channel'] = cust_metrics
                
                # Visualization: CLTV by Channel
                try:
                    plt.figure(figsize=(12, 7))
                    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(cust_metrics)))
                    bars = plt.barh(range(len(cust_metrics)), cust_metrics['avg_revenue_per_customer'],
                                   color=colors, edgecolor='black', linewidth=1.5)
                    plt.yticks(range(len(cust_metrics)), cust_metrics.index, fontsize=11)
                    plt.xlabel('Average Revenue per Customer (€)', fontsize=12, fontweight='bold')
                    plt.title('Customer Lifetime Value by Channel', fontsize=14, fontweight='bold')
                    plt.grid(axis='x', alpha=0.3)
                    
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        plt.text(width + width*0.02, bar.get_y() + bar.get_height()/2.,
                                f'€{width:.0f}', ha='left', va='center', fontweight='bold')
                    
                    plt.tight_layout()
                    save_plot('03_cltv_by_channel.png')
                    plt.show()
                except Exception as e:
                    print(f"   ⚠️ Could not generate CLTV visualization: {e}")
                
                print(f"   ✓ CLTV analysis completed across {len(cust_metrics)} channels")
        except Exception as e:
            print(f"   ⚠️ CLTV analysis failed: {e}")
    else:
        print("   ⚠️ Skipping CLTV analysis (revenue, customer, or channel column not available)")

    # --- ANALYSIS E: Conversion Funnel ---
    print("\n🔽 E. Analyzing Conversion Funnel...")
    
    try:
        funnel_cols = [c for c in df.columns if any(k in c.lower() for k in 
                      ['visit', 'impression', 'click', 'add_to_cart', 'checkout', 'purchase']) and c in numeric_cols]
        
        if funnel_cols:
            canonical = ['visit', 'impression', 'click', 'add_to_cart', 'checkout', 'purchase']
            ordered = [c for step in canonical for c in funnel_cols if step in c.lower()]
            ordered = list(dict.fromkeys(ordered))
            
            funnel_summary = {}
            funnel_values = []
            funnel_labels = []
            prev_total = None
            
            for c in ordered:
                numeric_sum = pd.to_numeric(df[c], errors='coerce').sum(skipna=True)
                total = int(numeric_sum) if not pd.isna(numeric_sum) else 0
                
                funnel_summary[c] = total
                funnel_values.append(total)
                funnel_labels.append(c.replace('_', ' ').title())
                
                if prev_total is not None and prev_total > 0:
                    dropoff_pct = round(((prev_total - total) / prev_total) * 100, 2)
                    funnel_summary[f'{c}_dropoff_pct'] = dropoff_pct
                
                prev_total = total
            
            if funnel_summary:
                results['funnel_summary'] = funnel_summary
                
                # Visualization: Funnel
                try:
                    plt.figure(figsize=(12, 8))
                    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(funnel_values)))
                    
                    # Create funnel chart
                    y_pos = np.arange(len(funnel_labels))
                    bars = plt.barh(y_pos, funnel_values, color=colors, edgecolor='black', linewidth=2)
                    
                    plt.yticks(y_pos, funnel_labels, fontsize=12, fontweight='bold')
                    plt.xlabel('Number of Users', fontsize=13, fontweight='bold')
                    plt.title('Marketing Conversion Funnel', fontsize=15, fontweight='bold', pad=20)
                    plt.grid(axis='x', alpha=0.3, linestyle='--')
                    
                    # Add values and conversion rates
                    for i, (bar, val) in enumerate(zip(bars, funnel_values)):
                        width = bar.get_width()
                        plt.text(width + width*0.02, bar.get_y() + bar.get_height()/2.,
                                f'{int(val):,}', ha='left', va='center', fontweight='bold', fontsize=11)
                        
                        # Add conversion rate
                        if i > 0 and funnel_values[i-1] > 0:
                            conv_rate = (val / funnel_values[i-1]) * 100
                            plt.text(width/2, bar.get_y() + bar.get_height()/2.,
                                    f'{conv_rate:.1f}%', ha='center', va='center', 
                                    color='white', fontweight='bold', fontsize=10,
                                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                    
                    plt.tight_layout()
                    save_plot('04_conversion_funnel.png')
                    plt.show()
                except Exception as e:
                    print(f"   ⚠️ Could not generate funnel visualization: {e}")
                
                print(f"   ✓ Funnel analysis completed with {len(ordered)} stages")
    except Exception as e:
        print(f"   ⚠️ Funnel analysis failed: {e}")

    # --- ANALYSIS F: RFM Segmentation ---
    print("\n👥 F. Performing RFM Segmentation...")
    
    if customer_col and revenue_col and revenue_col in numeric_cols and date_col:
        try:
            temp = df[[customer_col, revenue_col, date_col]].copy()
            temp[date_col] = pd.to_datetime(temp[date_col], errors='coerce')
            temp[revenue_col] = pd.to_numeric(temp[revenue_col], errors='coerce')
            temp = temp.dropna(subset=[date_col, revenue_col])
            
            if len(temp) > 0:
                snapshot = temp[date_col].max() + pd.Timedelta(days=1)
                rfm = temp.groupby(customer_col).agg({
                    date_col: lambda x: (snapshot - x.max()).days,
                    revenue_col: ['sum', 'count']
                })
                rfm.columns = ['recency', 'monetary', 'frequency']
                
                # Clustering if enough customers
                if len(rfm) >= 10:
                    clustering_data = (rfm - rfm.mean()) / rfm.std().replace(0, 1)
                    clustering_data = clustering_data.dropna()
                    
                    if len(clustering_data) > 0:
                        try:
                            cluster_labels = auto_kmeans(clustering_data)
                            rfm['segment'] = cluster_labels
                            
                            # Visualization: RFM Segments
                            try:
                                fig = plt.figure(figsize=(14, 6))
                                ax = fig.add_subplot(111, projection='3d')
                                
                                scatter = ax.scatter(rfm['recency'], rfm['frequency'], rfm['monetary'],
                                                    c=rfm['segment'], cmap='viridis', s=100, alpha=0.6,
                                                    edgecolors='black', linewidth=0.5)
                                
                                ax.set_xlabel('Recency (days)', fontsize=11, fontweight='bold')
                                ax.set_ylabel('Frequency (purchases)', fontsize=11, fontweight='bold')
                                ax.set_zlabel('Monetary (€)', fontsize=11, fontweight='bold')
                                ax.set_title('RFM Customer Segmentation', fontsize=14, fontweight='bold', pad=20)
                                
                                cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
                                cbar.set_label('Segment', fontsize=11, fontweight='bold')
                                
                                plt.tight_layout()
                                save_plot('05_rfm_segmentation.png')
                                plt.show()
                            except Exception as e:
                                print(f"   ⚠️ Could not generate RFM visualization: {e}")
                            
                            results['rfm_segments'] = rfm.sort_values(by='monetary', ascending=False)
                            print(f"   ✓ Segmented {len(rfm)} customers into {rfm['segment'].nunique()} clusters")
                        except Exception as cluster_err:
                            results['rfm_segments'] = rfm
                            print(f"   ✓ RFM calculated for {len(rfm)} customers (clustering unavailable)")
                elif len(rfm) > 0:
                    results['rfm_segments'] = rfm
                    print(f"   ✓ RFM calculated for {len(rfm)} customers (too few for clustering)")
        except Exception as e:
            print(f"   ⚠️ RFM segmentation failed: {e}")
    else:
        print("   ⚠️ Skipping RFM analysis (customer, revenue, or date column not available)")

    # --- ANALYSIS G: Customer Retention & Churn ---
    print("\n📈 G. Analyzing Customer Retention Trends...")
    
    if customer_col and date_col:
        try:
            temp = df[[customer_col, date_col]].copy()
            temp[date_col] = pd.to_datetime(temp[date_col], errors='coerce')
            temp = temp.dropna(subset=[date_col])
            
            if len(temp) > 0:
                temp['period'] = temp[date_col].dt.to_period('M')
                monthly_active = temp.groupby('period')[customer_col].nunique()
                monthly_active = monthly_active.to_frame('active_customers')
                monthly_active['pct_change'] = monthly_active['active_customers'].pct_change() * 100
                
                if len(monthly_active) > 0:
                    results['monthly_active_customers'] = monthly_active
                    
                    # Visualization: Retention Trends
                    try:
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
                        
                        # Top: Active customers over time
                        x = range(len(monthly_active))
                        ax1.plot(x, monthly_active['active_customers'], marker='o', 
                                linewidth=3, markersize=8, color='#3498db', label='Active Customers')
                        ax1.fill_between(x, monthly_active['active_customers'], alpha=0.3, color='#3498db')
                        ax1.set_xlabel('Period', fontsize=12, fontweight='bold')
                        ax1.set_ylabel('Active Customers', fontsize=12, fontweight='bold')
                        ax1.set_title('Monthly Active Customers Trend', fontsize=14, fontweight='bold')
                        ax1.set_xticks(x)
                        ax1.set_xticklabels([str(p) for p in monthly_active.index], rotation=45, ha='right')
                        ax1.grid(True, alpha=0.3, linestyle='--')
                        ax1.legend(fontsize=11)
                        
                        # Bottom: Growth rate
                        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in monthly_active['pct_change'].fillna(0)]
                        ax2.bar(x, monthly_active['pct_change'].fillna(0), color=colors, edgecolor='black', linewidth=1.5)
                        ax2.axhline(0, color='black', linestyle='-', linewidth=1)
                        ax2.set_xlabel('Period', fontsize=12, fontweight='bold')
                        ax2.set_ylabel('Growth Rate (%)', fontsize=12, fontweight='bold')
                        ax2.set_title('Customer Base Growth Rate', fontsize=14, fontweight='bold')
                        ax2.set_xticks(x)
                        ax2.set_xticklabels([str(p) for p in monthly_active.index], rotation=45, ha='right')
                        ax2.grid(True, alpha=0.3, linestyle='--')
                        
                        plt.tight_layout()
                        save_plot('06_retention_trends.png')
                        plt.show()
                    except Exception as e:
                        print(f"   ⚠️ Could not generate retention visualization: {e}")
                    
                    print(f"   ✓ Retention analysis completed for {len(monthly_active)} periods")
        except Exception as e:
            print(f"   ⚠️ Retention analysis failed: {e}")
    else:
        print("   ⚠️ Skipping retention analysis (customer or date column not available)")

    # --- ANALYSIS H: Budget Allocation Optimization ---
    print("\n💡 H. Generating Budget Allocation Recommendations...")
    
    if 'channel_performance' in results:
        try:
            ch = results['channel_performance'].copy()
            if len(ch) > 0:
                if 'roi_pct' in ch.columns:
                    recommended = ch.sort_values(by='roi_pct', ascending=False).head(3).index.tolist()
                    results['recommended_channels'] = recommended
                    
                    # Visualization: Budget Recommendations
                    try:
                        plt.figure(figsize=(12, 7))
                        
                        # Create comparison: current vs recommended
                        if 'cost' in ch.columns:
                            current_spend = ch['cost']
                            total_budget = current_spend.sum()
                            
                            # Recommended: redistribute based on ROI
                            if 'roi_pct' in ch.columns and ch['roi_pct'].sum() > 0:
                                roi_weights = ch['roi_pct'].clip(lower=0)
                                roi_weights = roi_weights / roi_weights.sum()
                                recommended_spend = roi_weights * total_budget
                            else:
                                recommended_spend = current_spend
                            
                            x = np.arange(len(ch))
                            width = 0.35
                            
                            bars1 = plt.bar(x - width/2, current_spend, width, label='Current Spend',
                                          color='#95a5a6', edgecolor='black', linewidth=1.5)
                            bars2 = plt.bar(x + width/2, recommended_spend, width, label='Recommended Spend',
                                          color='#3498db', edgecolor='black', linewidth=1.5)
                            
                            plt.xlabel('Channel', fontsize=12, fontweight='bold')
                            plt.ylabel('Marketing Spend (€)', fontsize=12, fontweight='bold')
                            plt.title('Budget Allocation: Current vs Recommended', fontsize=14, fontweight='bold')
                            plt.xticks(x, ch.index, rotation=45, ha='right')
                            plt.legend(fontsize=11)
                            plt.grid(axis='y', alpha=0.3)
                            
                            plt.tight_layout()
                            save_plot('07_budget_recommendations.png')
                            plt.show()
                    except Exception as e:
                        print(f"   ⚠️ Could not generate budget visualization: {e}")
                    
                    print(f"   ✓ Top 3 recommended channels: {', '.join(recommended)}")
        except Exception as e:
            print(f"   ⚠️ Budget optimization failed: {e}")
    else:
        print("   ⚠️ Skipping budget optimization (channel performance not available)")

    # --- ANALYSIS I: Marketing Forecasting ---
    print("\n🔮 I. Generating Marketing Performance Forecast...")
    
    if date_col and (revenue_col or volume_col):
        try:
            forecast_col = revenue_col if revenue_col else volume_col
            temp_forecast = df[[date_col, forecast_col]].copy()
            temp_forecast[date_col] = pd.to_datetime(temp_forecast[date_col], errors='coerce')
            temp_forecast[forecast_col] = pd.to_numeric(temp_forecast[forecast_col], errors='coerce')
            temp_forecast = temp_forecast.dropna()
            
            if len(temp_forecast) > 5:
                temp_forecast = temp_forecast.sort_values(date_col)
                temp_forecast['period'] = temp_forecast[date_col].dt.to_period('M')
                monthly_data = temp_forecast.groupby('period')[forecast_col].sum().reset_index()
                monthly_data['time_index'] = range(len(monthly_data))
                
                X = monthly_data[['time_index']].values
                y = monthly_data[forecast_col].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Forecast next 6 periods
                future_steps = np.array([[len(X) + i] for i in range(1, 7)])
                predictions = model.predict(future_steps)
                r2_score = model.score(X, y)
                
                results['marketing_forecast'] = {
                    'target_metric': forecast_col,
                    'forecast_next_6_periods': [float(round(p, 2)) for p in predictions],
                    'trend': 'Increasing' if predictions[-1] > predictions[0] else 'Decreasing',
                    'r2_score': float(round(r2_score, 3)),
                    'trend_coefficient': float(round(model.coef_[0], 4))
                }
                
                # Visualization: Forecast
                try:
                    plt.figure(figsize=(14, 7))
                    
                    # Actual data
                    plt.plot(range(len(y)), y, 'o-', label='Actual Performance',
                            linewidth=3, markersize=8, color='#3498db', alpha=0.8)
                    
                    # Forecast
                    forecast_x = range(len(y), len(y) + 6)
                    plt.plot(forecast_x, predictions, 's--', label='6-Period Forecast',
                            color='#e74c3c', linewidth=3, markersize=10, alpha=0.8)
                    
                    # Forecast area
                    plt.axvspan(len(y) - 0.5, len(y) + 5.5, alpha=0.1, color='orange')
                    
                    # Value labels on forecast
                    for i, (x, pred) in enumerate(zip(forecast_x, predictions)):
                        if i % 2 == 0:
                            plt.text(x, pred + (y.max() - y.min()) * 0.03, f'{pred:.0f}',
                                    ha='center', va='bottom', fontweight='bold', fontsize=10,
                                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                    
                    plt.xlabel('Time Period', fontsize=13, fontweight='bold')
                    plt.ylabel(forecast_col.replace('_', ' ').title(), fontsize=13, fontweight='bold')
                    plt.title(f'Marketing Forecast: {forecast_col.replace("_", " ").title()}\n(R² = {r2_score:.3f})',
                             fontsize=15, fontweight='bold', pad=20)
                    plt.legend(loc='best', fontsize=12)
                    plt.grid(True, alpha=0.3, linestyle='--')
                    
                    plt.tight_layout()
                    save_plot('08_marketing_forecast.png')
                    plt.show()
                except Exception as e:
                    print(f"   ⚠️ Could not generate forecast visualization: {e}")
                
                print(f"   ✓ Forecast generated (R² = {r2_score:.3f})")
        except Exception as e:
            print(f"   ⚠️ Forecasting failed: {e}")
    else:
        print("   ⚠️ Skipping forecast (date or revenue/volume column not available)")

    print("\n" + "="*60)
    print("✅ Marketing Analytics completed successfully!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    """Test the marketing analytics function with sample data"""
    print("\n" + "="*60)
    print("MARKETING ANALYTICS TEST MODE")
    print("="*60 + "\n")
    
    # Create sample data
    np.random.seed(42)
    n_records = 500
    
    channels = ['Email', 'Social Media', 'Google Ads', 'Display', 'Organic']
    campaigns = ['Spring_Sale', 'Summer_Campaign', 'Black_Friday', 'Holiday_Special']
    
    data = {
        'CustomerID': np.random.randint(1, 200, n_records),
        'Channel': np.random.choice(channels, n_records),
        'Campaign': np.random.choice(campaigns, n_records),
        'Revenue': np.random.gamma(5, 20, n_records).round(2),
        'Conversions': np.random.poisson(2, n_records),
        'Marketing_Spend': np.random.gamma(3, 15, n_records).round(2),
        'OrderDate': pd.date_range(end='2024-12-31', periods=n_records, freq='D')
    }
    
    df_test = pd.DataFrame(data)
    print(f"✅ Sample dataset created: {len(df_test)} rows\n")
    
    test_mapping = {
        'monetary_value': 'Revenue',
        'volume_metrics': 'Conversions',
        'channel': 'Channel',
        'campaign': 'Campaign',
        'customer_id': 'CustomerID',
        'date': 'OrderDate',
        'marketing_cost': 'Marketing_Spend'
    }
    
    try:
        results = analytics_marketing(df_test, mapping=test_mapping, 
                                     save_plots=True, output_dir="test_marketing_plots")
        
        print("\n" + "="*60)
        print("📊 TEST RESULTS SUMMARY")
        print("="*60 + "\n")
        
        for key, value in results.items():
            print(f"\n{'─'*60}")
            print(f"📈 {key.upper().replace('_', ' ')}")
            print(f"{'─'*60}")
            if isinstance(value, (pd.DataFrame, pd.Series)):
                print(value.head(10))
            else:
                print(value)
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()