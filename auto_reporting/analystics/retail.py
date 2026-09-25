import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from data_manipulation.read_data import identify_variables
from statistics_functions.clustering import auto_kmeans, auto_kmedoids
from LLM.llm_output import needed_variables


def analytics_retail(df: pd.DataFrame, mapping: dict = None, user_objective: str = None,
                     save_plots: bool = False, output_dir: str = "domain_plots"):
    """
    Comprehensive retail analytics exploring key business dimensions.
    
    Analyses performed:
    - A: Customer Segmentation (RFM) - Recency, Frequency, Monetary analysis with clustering
    - B: Sales Performance & KPIs - Transaction value, conversion rate, bestsellers
    - C: Basket Analysis - Average basket size, composition, cross-selling opportunities
    - D: Inventory Turnover - Stock rotation velocity and optimization
    - E: Store Performance - Sales per square meter, profitability by location
    - F: Price Elasticity - Impact of price changes on demand
    - G: Temporal Trends - Seasonality, peak hours, time-based patterns
    - H: Channel Performance - Omnichannel analysis (online vs offline)
    - I: Customer Loyalty & Retention - Repeat purchase rate, churn analysis
    - J: Demand Forecasting - Predictive modeling for future sales
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input retail dataset
    mapping : dict, optional
        Column mapping for retail variables
    user_objective : str, optional
        User's analysis objective
    save_plots : bool, default=False
        Whether to save plots to disk
    output_dir : str, default="retail_analysis_plots"
        Directory to save plots
    """
    print("Running comprehensive retail analytics")
    
    # Create output directory
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Plots will be saved to: {output_dir}/")
    
    def save_plot(filename):
        if save_plots:
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"   ✓ Plot saved: {filename}")
    
    # Define retail-specific roles
    retail_roles = [
        'transaction_id',    # Order/Transaction ID
        'customer_id',       # Customer identifier
        'product_id',        # Product/SKU identifier
        'revenue',          # Sales amount
        'quantity',         # Units sold
        'price',            # Unit price
        'date',             # Transaction date
        'store_id',         # Store/location identifier
        'channel',          # Sales channel (online/offline)
        'category'          # Product category
    ]
    
    if mapping is None:
        mapping = needed_variables(df, retail_roles)
    
    def get_single_col(key):
        val = mapping.get(key)
        if not val:
            return None
        if isinstance(val, list):
            val = val[0] if len(val) > 0 else None
        if isinstance(val, str) and ',' in val:
            val = val.split(',')[0].strip()
        return val if val in df.columns else None
    
    # Extract variable names
    transaction_col = get_single_col('transaction_id')
    customer_col = get_single_col('customer_id')
    product_col = get_single_col('product_id')
    revenue_col = get_single_col('revenue')
    quantity_col = get_single_col('quantity')
    price_col = get_single_col('price')
    date_col = get_single_col('date')
    store_col = get_single_col('store_id')
    channel_col = get_single_col('channel')
    category_col = get_single_col('category')
    
    var_types = identify_variables(df)
    numeric_cols = var_types.get('quantitative', [])
    
    # Fallback warnings
    missing_cols = []
    if not customer_col:
        missing_cols.append('customer_id (Customer Identifier)')
    if not revenue_col:
        missing_cols.append('revenue (Sales Amount)')
    
    if missing_cols:
        print(f"\n⚠️  WARNING: Missing critical columns:")
        for col in missing_cols:
            print(f"   - {col}")
        print("   Some analyses will be skipped.\n")
    
    results = {}
    
    # =========================================================================
    # ANALYSIS A: Customer Segmentation (RFM Analysis)
    # =========================================================================
    print("\n👥 A. Performing RFM Customer Segmentation...")
    
    if customer_col and revenue_col and revenue_col in numeric_cols and date_col:
        try:
            temp_rfm = df[[customer_col, revenue_col, date_col]].copy()
            temp_rfm[date_col] = pd.to_datetime(temp_rfm[date_col], errors='coerce')
            temp_rfm[revenue_col] = pd.to_numeric(temp_rfm[revenue_col], errors='coerce')
            temp_rfm = temp_rfm.dropna(subset=[date_col, revenue_col])
            
            if len(temp_rfm) > 0:
                snapshot = temp_rfm[date_col].max() + pd.Timedelta(days=1)
                rfm = temp_rfm.groupby(customer_col).agg({
                    date_col: lambda x: (snapshot - x.max()).days,
                    revenue_col: ['sum', 'count']
                })
                rfm.columns = ['recency', 'monetary', 'frequency']
                
                # RFM Scoring (1-5 scale)
                rfm['R_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1], duplicates='drop')
                rfm['F_score'] = pd.qcut(rfm['frequency'], 5, labels=[1,2,3,4,5], duplicates='drop')
                rfm['M_score'] = pd.qcut(rfm['monetary'], 5, labels=[1,2,3,4,5], duplicates='drop')
                
                rfm['RFM_Score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
                
                # Segment labels
                def segment_customer(row):
                    if row['R_score'] >= 4 and row['F_score'] >= 4:
                        return 'Champions'
                    elif row['R_score'] >= 3 and row['F_score'] >= 3:
                        return 'Loyal Customers'
                    elif row['R_score'] >= 4:
                        return 'Potential Loyalists'
                    elif row['F_score'] >= 4:
                        return 'Big Spenders'
                    elif row['R_score'] <= 2:
                        return 'At Risk'
                    else:
                        return 'Need Attention'
                
                rfm['Segment'] = rfm.apply(segment_customer, axis=1)
                results['rfm_analysis'] = rfm.sort_values(by='monetary', ascending=False)
                
                # Segment summary
                segment_summary = rfm.groupby('Segment').agg({
                    'recency': 'mean',
                    'frequency': 'mean',
                    'monetary': 'sum',
                    customer_col: 'count'
                }).rename(columns={customer_col: 'customer_count'})
                results['segment_summary'] = segment_summary.sort_values('monetary', ascending=False)
                
                # Visualization: RFM Segments
                try:
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                    
                    # Top-left: Segment distribution
                    segment_counts = rfm['Segment'].value_counts()
                    colors = plt.cm.Set3(range(len(segment_counts)))
                    bars1 = ax1.bar(range(len(segment_counts)), segment_counts.values,
                                   color=colors, edgecolor='black', linewidth=1.5)
                    ax1.set_xticks(range(len(segment_counts)))
                    ax1.set_xticklabels(segment_counts.index, rotation=45, ha='right', fontsize=10)
                    ax1.set_ylabel('Number of Customers', fontsize=12, fontweight='bold')
                    ax1.set_title('Customer Distribution by Segment', fontsize=14, fontweight='bold')
                    ax1.grid(axis='y', alpha=0.3)
                    
                    for bar in bars1:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
                    
                    # Top-right: Revenue by segment
                    segment_revenue = rfm.groupby('Segment')['monetary'].sum().sort_values(ascending=False)
                    colors2 = plt.cm.Greens(np.linspace(0.4, 0.9, len(segment_revenue)))
                    bars2 = ax2.barh(range(len(segment_revenue)), segment_revenue.values,
                                    color=colors2, edgecolor='black', linewidth=1.5)
                    ax2.set_yticks(range(len(segment_revenue)))
                    ax2.set_yticklabels(segment_revenue.index, fontsize=11)
                    ax2.set_xlabel('Total Revenue (€)', fontsize=12, fontweight='bold')
                    ax2.set_title('Revenue Contribution by Segment', fontsize=14, fontweight='bold')
                    ax2.grid(axis='x', alpha=0.3)
                    
                    for i, bar in enumerate(bars2):
                        width = bar.get_width()
                        ax2.text(width + width*0.02, bar.get_y() + bar.get_height()/2.,
                                f'€{int(width):,}', ha='left', va='center', fontweight='bold')
                    
                    # Bottom-left: RFM scatter (Recency vs Monetary)
                    scatter_data = rfm.sample(min(1000, len(rfm)))
                    scatter = ax3.scatter(scatter_data['recency'], scatter_data['monetary'],
                                        c=scatter_data['frequency'], cmap='YlOrRd',
                                        s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
                    ax3.set_xlabel('Recency (days since last purchase)', fontsize=11, fontweight='bold')
                    ax3.set_ylabel('Monetary (total spent €)', fontsize=11, fontweight='bold')
                    ax3.set_title('Customer Value Matrix', fontsize=14, fontweight='bold')
                    ax3.grid(True, alpha=0.3)
                    cbar = plt.colorbar(scatter, ax=ax3)
                    cbar.set_label('Frequency (purchases)', fontsize=10, fontweight='bold')
                    
                    # Bottom-right: Average metrics by segment
                    avg_metrics = rfm.groupby('Segment')[['recency', 'frequency', 'monetary']].mean()
                    avg_metrics_norm = (avg_metrics - avg_metrics.min()) / (avg_metrics.max() - avg_metrics.min())
                    
                    x = np.arange(len(avg_metrics))
                    width = 0.25
                    
                    bars_r = ax4.bar(x - width, avg_metrics_norm['recency'], width, label='Recency',
                                    color='#e74c3c', edgecolor='black', linewidth=1)
                    bars_f = ax4.bar(x, avg_metrics_norm['frequency'], width, label='Frequency',
                                    color='#3498db', edgecolor='black', linewidth=1)
                    bars_m = ax4.bar(x + width, avg_metrics_norm['monetary'], width, label='Monetary',
                                    color='#2ecc71', edgecolor='black', linewidth=1)
                    
                    ax4.set_ylabel('Normalized Score', fontsize=12, fontweight='bold')
                    ax4.set_title('RFM Profile by Segment (Normalized)', fontsize=14, fontweight='bold')
                    ax4.set_xticks(x)
                    ax4.set_xticklabels(avg_metrics.index, rotation=45, ha='right', fontsize=10)
                    ax4.legend(fontsize=10)
                    ax4.grid(axis='y', alpha=0.3)
                    
                    plt.tight_layout()
                    save_plot('01_customer_segmentation.png')
                    plt.show()
                except Exception as e:
                    print(f"   ⚠️ Could not generate RFM visualization: {e}")
                
                print(f"   ✓ Segmented {len(rfm)} customers into {len(segment_counts)} groups")
        except Exception as e:
            print(f"   ⚠️ RFM analysis failed: {e}")
    else:
        print("   ⚠️ Skipping RFM (missing customer, revenue, or date column)")
    
    # =========================================================================
    # ANALYSIS B: Sales Performance & KPIs
    # =========================================================================
    print("\n📊 B. Analyzing Sales Performance & KPIs...")
    
    if revenue_col and revenue_col in numeric_cols:
        try:
            kpis = {}
            
            # Total revenue
            kpis['total_revenue'] = float(pd.to_numeric(df[revenue_col], errors='coerce').sum())
            
            # Average transaction value
            if transaction_col:
                avg_transaction = df.groupby(transaction_col)[revenue_col].sum().mean()
                kpis['avg_transaction_value'] = float(round(avg_transaction, 2))
            
            # Total transactions
            if transaction_col:
                kpis['total_transactions'] = int(df[transaction_col].nunique())
            
            # Conversion rate (if we have store visits or traffic data)
            # For now, proxy with transactions per customer
            if customer_col:
                transactions_per_customer = df[transaction_col].nunique() / df[customer_col].nunique()
                kpis['avg_transactions_per_customer'] = float(round(transactions_per_customer, 2))
            
            # Units sold
            if quantity_col and quantity_col in numeric_cols:
                kpis['total_units_sold'] = int(pd.to_numeric(df[quantity_col], errors='coerce').sum())
            
            results['sales_kpis'] = kpis
            
            # Bestsellers analysis
            if product_col:
                product_sales = df.groupby(product_col).agg({
                    revenue_col: 'sum',
                    quantity_col: 'sum' if quantity_col else 'count'
                }).sort_values(revenue_col, ascending=False)
                
                product_sales.columns = ['total_revenue', 'units_sold']
                results['bestsellers'] = product_sales.head(20)
                
                # Visualization: Sales Performance
                try:
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                    
                    # Top-left: KPI Dashboard
                    kpi_labels = list(kpis.keys())
                    kpi_values = list(kpis.values())
                    
                    ax1.axis('off')
                    ax1.set_title('Sales KPI Dashboard', fontsize=16, fontweight='bold', pad=20)
                    
                    y_pos = 0.9
                    for i, (label, value) in enumerate(zip(kpi_labels, kpi_values)):
                        color = plt.cm.Set3(i)
                        if 'revenue' in label or 'value' in label:
                            text = f'{label.replace("_", " ").title()}: €{value:,.0f}'
                        else:
                            text = f'{label.replace("_", " ").title()}: {value:,.2f}'
                        
                        ax1.text(0.1, y_pos, text, fontsize=14, fontweight='bold',
                                bbox=dict(boxstyle='round', facecolor=color, alpha=0.7, edgecolor='black', linewidth=2))
                        y_pos -= 0.15
                    
                    # Top-right: Top 10 bestsellers by revenue
                    top_10_products = product_sales.head(10)
                    colors2 = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_10_products)))
                    bars2 = ax2.barh(range(len(top_10_products)), top_10_products['total_revenue'],
                                    color=colors2, edgecolor='black', linewidth=1.5)
                    ax2.set_yticks(range(len(top_10_products)))
                    ax2.set_yticklabels([str(idx)[:15] for idx in top_10_products.index], fontsize=10)
                    ax2.set_xlabel('Total Revenue (€)', fontsize=12, fontweight='bold')
                    ax2.set_title('Top 10 Bestsellers by Revenue', fontsize=14, fontweight='bold')
                    ax2.grid(axis='x', alpha=0.3)
                    
                    for i, bar in enumerate(bars2):
                        width = bar.get_width()
                        ax2.text(width + width*0.02, bar.get_y() + bar.get_height()/2.,
                                f'€{int(width):,}', ha='left', va='center', fontweight='bold', fontsize=9)
                    
                    # Bottom-left: Units sold distribution
                    if len(top_10_products) > 0:
                        bars3 = ax3.bar(range(len(top_10_products)), top_10_products['units_sold'],
                                       color=colors2, edgecolor='black', linewidth=1.5)
                        ax3.set_xticks(range(len(top_10_products)))
                        ax3.set_xticklabels([str(idx)[:10] for idx in top_10_products.index], 
                                           rotation=45, ha='right', fontsize=9)
                        ax3.set_ylabel('Units Sold', fontsize=12, fontweight='bold')
                        ax3.set_title('Units Sold (Top 10 Products)', fontsize=14, fontweight='bold')
                        ax3.grid(axis='y', alpha=0.3)
                        
                        for bar in bars3:
                            height = bar.get_height()
                            ax3.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{int(height):,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
                    
                    # Bottom-right: Revenue concentration (Pareto)
                    product_sales_sorted = product_sales.sort_values('total_revenue', ascending=False)
                    cumulative_pct = (product_sales_sorted['total_revenue'].cumsum() / 
                                     product_sales_sorted['total_revenue'].sum() * 100)
                    
                    ax4_twin = ax4.twinx()
                    
                    x_range = range(min(50, len(product_sales_sorted)))
                    bars4 = ax4.bar(x_range, product_sales_sorted['total_revenue'].iloc[:50].values,
                                   color='#3498db', alpha=0.7, edgecolor='black', linewidth=1)
                    line4 = ax4_twin.plot(x_range, cumulative_pct.iloc[:50].values, 
                                         color='#e74c3c', linewidth=3, marker='o', markersize=4)
                    
                    ax4.set_xlabel('Product Rank', fontsize=12, fontweight='bold')
                    ax4.set_ylabel('Revenue (€)', fontsize=12, fontweight='bold', color='#3498db')
                    ax4_twin.set_ylabel('Cumulative %', fontsize=12, fontweight='bold', color='#e74c3c')
                    ax4.set_title('Revenue Concentration (Pareto Analysis)', fontsize=14, fontweight='bold')
                    ax4_twin.axhline(80, color='orange', linestyle='--', linewidth=2, alpha=0.5)
                    ax4.grid(axis='y', alpha=0.3)
                    
                    plt.tight_layout()
                    save_plot('02_sales_performance.png')
                    plt.show()
                except Exception as e:
                    print(f"   ⚠️ Could not generate sales performance visualization: {e}")
            
            print(f"   ✓ Sales KPIs calculated: Total Revenue = €{kpis['total_revenue']:,.0f}")
        except Exception as e:
            print(f"   ⚠️ Sales performance analysis failed: {e}")
    else:
        print("   ⚠️ Skipping sales performance (missing revenue column)")
    
    # =========================================================================
    # ANALYSIS C: Basket Analysis
    # =========================================================================
    print("\n🛒 C. Analyzing Shopping Basket Patterns...")
    
    if transaction_col and product_col and quantity_col:
        try:
            # Basket size analysis
            basket_sizes = df.groupby(transaction_col).agg({
                product_col: 'count',
                quantity_col: 'sum',
                revenue_col: 'sum' if revenue_col else 'count'
            })
            basket_sizes.columns = ['items_count', 'total_units', 'basket_value']
            
            results['basket_analysis'] = {
                'avg_items_per_basket': float(round(basket_sizes['items_count'].mean(), 2)),
                'avg_units_per_basket': float(round(basket_sizes['total_units'].mean(), 2)),
                'avg_basket_value': float(round(basket_sizes['basket_value'].mean(), 2)),
                'median_basket_value': float(round(basket_sizes['basket_value'].median(), 2))
            }
            
            # Category mix if available
            if category_col:
                category_freq = df.groupby(transaction_col)[category_col].apply(lambda x: x.nunique())
                results['basket_analysis']['avg_categories_per_basket'] = float(round(category_freq.mean(), 2))
            
            # Visualization: Basket Analysis
            try:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                
                # Top-left: Basket value distribution
                ax1.hist(basket_sizes['basket_value'].clip(upper=basket_sizes['basket_value'].quantile(0.95)),
                        bins=30, color='#3498db', edgecolor='black', linewidth=1.5, alpha=0.7)
                ax1.axvline(basket_sizes['basket_value'].mean(), color='red', linestyle='--', 
                           linewidth=2, label=f'Mean: €{basket_sizes["basket_value"].mean():.2f}')
                ax1.axvline(basket_sizes['basket_value'].median(), color='green', linestyle='--',
                           linewidth=2, label=f'Median: €{basket_sizes["basket_value"].median():.2f}')
                ax1.set_xlabel('Basket Value (€)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                ax1.set_title('Basket Value Distribution (95th percentile clipped)', fontsize=14, fontweight='bold')
                ax1.legend(fontsize=10)
                ax1.grid(axis='y', alpha=0.3)
                
                # Top-right: Items per basket
                items_count_dist = basket_sizes['items_count'].value_counts().sort_index().head(20)
                bars2 = ax2.bar(range(len(items_count_dist)), items_count_dist.values,
                               color='#2ecc71', edgecolor='black', linewidth=1.5)
                ax2.set_xticks(range(len(items_count_dist)))
                ax2.set_xticklabels(items_count_dist.index, fontsize=10)
                ax2.set_xlabel('Number of Items in Basket', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
                ax2.set_title('Basket Size Distribution', fontsize=14, fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)
                
                # Bottom-left: Basket value vs items correlation
                sample_baskets = basket_sizes.sample(min(1000, len(basket_sizes)))
                scatter3 = ax3.scatter(sample_baskets['items_count'], sample_baskets['basket_value'],
                                      alpha=0.5, s=50, color='#9b59b6', edgecolors='black', linewidth=0.5)
                ax3.set_xlabel('Items in Basket', fontsize=12, fontweight='bold')
                ax3.set_ylabel('Basket Value (€)', fontsize=12, fontweight='bold')
                ax3.set_title('Basket Value vs Size Correlation', fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                
                # Fit line
                if len(sample_baskets) > 2:
                    z = np.polyfit(sample_baskets['items_count'], sample_baskets['basket_value'], 1)
                    p = np.poly1d(z)
                    ax3.plot(sample_baskets['items_count'].sort_values(), 
                            p(sample_baskets['items_count'].sort_values()),
                            "r-", linewidth=2, label=f'Trend: y={z[0]:.1f}x+{z[1]:.1f}')
                    ax3.legend(fontsize=10)
                
                # Bottom-right: Basket metrics summary
                ax4.axis('off')
                ax4.set_title('Basket Metrics Summary', fontsize=16, fontweight='bold', pad=20)
                
                metrics = results['basket_analysis']
                y_pos = 0.9
                colors = plt.cm.Set2(range(len(metrics)))
                
                for i, (key, value) in enumerate(metrics.items()):
                    label = key.replace('_', ' ').title()
                    if 'value' in key:
                        text = f'{label}: €{value:.2f}'
                    else:
                        text = f'{label}: {value:.2f}'
                    
                    ax4.text(0.1, y_pos, text, fontsize=13, fontweight='bold',
                            bbox=dict(boxstyle='round', facecolor=colors[i], 
                                    alpha=0.7, edgecolor='black', linewidth=2))
                    y_pos -= 0.15
                
                plt.tight_layout()
                save_plot('03_basket_analysis.png')
                plt.show()
            except Exception as e:
                print(f"   ⚠️ Could not generate basket visualization: {e}")
            
            print(f"   ✓ Basket analysis: Avg basket value = €{results['basket_analysis']['avg_basket_value']:.2f}")
        except Exception as e:
            print(f"   ⚠️ Basket analysis failed: {e}")
    else:
        print("   ⚠️ Skipping basket analysis (missing transaction, product, or quantity column)")
    
    # =========================================================================
    # ANALYSIS D: Inventory Turnover
    # =========================================================================
    print("\n📦 D. Analyzing Inventory Turnover...")
    
    if product_col and quantity_col and quantity_col in numeric_cols and date_col:
        try:
            temp_inv = df[[product_col, quantity_col, date_col]].copy()
            temp_inv[date_col] = pd.to_datetime(temp_inv[date_col], errors='coerce')
            temp_inv[quantity_col] = pd.to_numeric(temp_inv[quantity_col], errors='coerce')
            temp_inv = temp_inv.dropna()
            
            if len(temp_inv) > 0:
                # Calculate days in period
                date_range = (temp_inv[date_col].max() - temp_inv[date_col].min()).days
                if date_range == 0:
                    date_range = 30  # Default to 30 days
                
                # Units sold per product
                product_turnover = temp_inv.groupby(product_col).agg({
                    quantity_col: 'sum',
                    date_col: ['min', 'max', 'count']
                })
                product_turnover.columns = ['units_sold', 'first_sale', 'last_sale', 'transactions']
                
                # Calculate turnover rate (units sold / days active)
                product_turnover['days_active'] = (product_turnover['last_sale'] - 
                                                   product_turnover['first_sale']).dt.days + 1
                product_turnover['turnover_rate'] = product_turnover['units_sold'] / product_turnover['days_active']
                
                # Classify velocity
                def classify_velocity(rate):
                    if rate >= product_turnover['turnover_rate'].quantile(0.75):
                        return 'Fast Moving'
                    elif rate >= product_turnover['turnover_rate'].quantile(0.25):
                        return 'Medium Moving'
                    else:
                        return 'Slow Moving'
                
                product_turnover['velocity_class'] = product_turnover['turnover_rate'].apply(classify_velocity)
                
                results['inventory_turnover'] = product_turnover.sort_values('turnover_rate', ascending=False)
                
                # Velocity distribution
                velocity_summary = product_turnover['velocity_class'].value_counts()
                results['velocity_distribution'] = velocity_summary.to_dict()
                
                # Visualization: Inventory Turnover
                try:
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                    
                    # Top-left: Velocity distribution
                    colors1 = {'Fast Moving': '#2ecc71', 'Medium Moving': '#f39c12', 'Slow Moving': '#e74c3c'}
                    vel_colors = [colors1.get(cat, '#95a5a6') for cat in velocity_summary.index]
                    
                    bars1 = ax1.bar(range(len(velocity_summary)), velocity_summary.values,
                                   color=vel_colors, edgecolor='black', linewidth=1.5)
                    ax1.set_xticks(range(len(velocity_summary)))
                    ax1.set_xticklabels(velocity_summary.index, fontsize=11)
                    ax1.set_ylabel('Number of Products', fontsize=12, fontweight='bold')
                    ax1.set_title('Product Velocity Distribution', fontsize=14, fontweight='bold')
                    ax1.grid(axis='y', alpha=0.3)
                    
                    for bar in bars1:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height,
                                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
                    
                    # Top-right: Top 15 fast movers
                    top_movers = product_turnover.sort_values('turnover_rate', ascending=False).head(15)
                    colors2 = plt.cm.Greens(np.linspace(0.4, 0.9, len(top_movers)))
                    bars2 = ax2.barh(range(len(top_movers)), top_movers['turnover_rate'],
                                    color=colors2, edgecolor='black', linewidth=1.5)
                    ax2.set_yticks(range(len(top_movers)))
                    ax2.set_yticklabels([str(idx)[:15] for idx in top_movers.index], fontsize=10)
                    ax2.set_xlabel('Turnover Rate (units/day)', fontsize=12, fontweight='bold')
                    ax2.set_title('Top 15 Fast-Moving Products', fontsize=14, fontweight='bold')
                    ax2.grid(axis='x', alpha=0.3)
                    
                    for i, bar in enumerate(bars2):
                        width = bar.get_width()
                        ax2.text(width + width*0.02, bar.get_y() + bar.get_height()/2.,
                                f'{width:.2f}', ha='left', va='center', fontweight='bold', fontsize=9)
                    
                    # Bottom-left: Bottom 15 slow movers
                    slow_movers = product_turnover.sort_values('turnover_rate').head(15)
                    colors3 = plt.cm.Reds(np.linspace(0.4, 0.9, len(slow_movers)))
                    bars3 = ax3.barh(range(len(slow_movers)), slow_movers['turnover_rate'],
                                    color=colors3, edgecolor='black', linewidth=1.5)
                    ax3.set_yticks(range(len(slow_movers)))
                    ax3.set_yticklabels([str(idx)[:15] for idx in slow_movers.index], fontsize=10)
                    ax3.set_xlabel('Turnover Rate (units/day)', fontsize=12, fontweight='bold')
                    ax3.set_title('Top 15 Slow-Moving Products (Potential Deadstock)', fontsize=14, fontweight='bold')
                    ax3.grid(axis='x', alpha=0.3)
                    
                    for i, bar in enumerate(bars3):
                        width = bar.get_width()
                        ax3.text(width + width*0.02, bar.get_y() + bar.get_height()/2.,
                                f'{width:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)
                    
                    # Bottom-right: Turnover rate distribution
                    ax4.hist(product_turnover['turnover_rate'].clip(upper=product_turnover['turnover_rate'].quantile(0.95)),
                            bins=30, color='#3498db', edgecolor='black', linewidth=1.5, alpha=0.7)
                    ax4.axvline(product_turnover['turnover_rate'].mean(), color='red', linestyle='--',
                               linewidth=2, label=f'Mean: {product_turnover["turnover_rate"].mean():.2f}')
                    ax4.set_xlabel('Turnover Rate (units/day)', fontsize=12, fontweight='bold')
                    ax4.set_ylabel('Number of Products', fontsize=12, fontweight='bold')
                    ax4.set_title('Turnover Rate Distribution', fontsize=14, fontweight='bold')
                    ax4.legend(fontsize=10)
                    ax4.grid(axis='y', alpha=0.3)
                    
                    plt.tight_layout()
                    save_plot('04_inventory_turnover.png')
                    plt.show()
                except Exception as e:
                    print(f"   ⚠️ Could not generate inventory visualization: {e}")
                
                print(f"   ✓ Inventory analysis: {len(product_turnover)} products analyzed")
        except Exception as e:
            print(f"   ⚠️ Inventory turnover analysis failed: {e}")
    else:
        print("   ⚠️ Skipping inventory analysis (missing product, quantity, or date column)")
    
    # =========================================================================
    # ANALYSIS E: Store Performance
    # =========================================================================
    print("\n🏪 E. Analyzing Store Performance...")
    
    if store_col and revenue_col and revenue_col in numeric_cols:
        try:
            store_performance = df.groupby(store_col).agg({
                revenue_col: 'sum',
                transaction_col: 'nunique' if transaction_col else 'count',
                customer_col: 'nunique' if customer_col else 'count'
            })
            store_performance.columns = ['total_revenue', 'transactions', 'customers']
            store_performance['avg_transaction_value'] = store_performance['total_revenue'] / store_performance['transactions']
            store_performance['revenue_per_customer'] = store_performance['total_revenue'] / store_performance['customers']
            
            store_performance = store_performance.sort_values('total_revenue', ascending=False)
            results['store_performance'] = store_performance
            
            # Visualization: Store Performance
            try:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                
                # Top-left: Revenue by store
                top_stores = store_performance.head(15)
                colors1 = plt.cm.Blues(np.linspace(0.4, 0.9, len(top_stores)))
                bars1 = ax1.barh(range(len(top_stores)), top_stores['total_revenue'],
                                color=colors1, edgecolor='black', linewidth=1.5)
                ax1.set_yticks(range(len(top_stores)))
                ax1.set_yticklabels([str(idx)[:15] for idx in top_stores.index], fontsize=10)
                ax1.set_xlabel('Total Revenue (€)', fontsize=12, fontweight='bold')
                ax1.set_title('Revenue by Store (Top 15)', fontsize=14, fontweight='bold')
                ax1.grid(axis='x', alpha=0.3)
                
                for i, bar in enumerate(bars1):
                    width = bar.get_width()
                    ax1.text(width + width*0.02, bar.get_y() + bar.get_height()/2.,
                            f'€{int(width):,}', ha='left', va='center', fontweight='bold', fontsize=9)
                
                # Top-right: Transaction volume
                bars2 = ax2.bar(range(len(top_stores)), top_stores['transactions'],
                               color='#2ecc71', edgecolor='black', linewidth=1.5)
                ax2.set_xticks(range(len(top_stores)))
                ax2.set_xticklabels([str(idx)[:10] for idx in top_stores.index], 
                                   rotation=45, ha='right', fontsize=9)
                ax2.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
                ax2.set_title('Transaction Volume by Store', fontsize=14, fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)
                
                for bar in bars2:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height):,}', ha='center', va='bottom', fontweight='bold', fontsize=8)
                
                # Bottom-left: Avg transaction value comparison
                avg_trans_sorted = store_performance.sort_values('avg_transaction_value', ascending=False).head(15)
                colors3 = plt.cm.Oranges(np.linspace(0.4, 0.9, len(avg_trans_sorted)))
                bars3 = ax3.barh(range(len(avg_trans_sorted)), avg_trans_sorted['avg_transaction_value'],
                                color=colors3, edgecolor='black', linewidth=1.5)
                ax3.set_yticks(range(len(avg_trans_sorted)))
                ax3.set_yticklabels([str(idx)[:15] for idx in avg_trans_sorted.index], fontsize=10)
                ax3.set_xlabel('Average Transaction Value (€)', fontsize=12, fontweight='bold')
                ax3.set_title('Average Transaction Value by Store', fontsize=14, fontweight='bold')
                ax3.grid(axis='x', alpha=0.3)
                
                for i, bar in enumerate(bars3):
                    width = bar.get_width()
                    ax3.text(width + width*0.02, bar.get_y() + bar.get_height()/2.,
                            f'€{width:.2f}', ha='left', va='center', fontweight='bold', fontsize=9)
                
                # Bottom-right: Revenue concentration
                revenue_sorted = store_performance.sort_values('total_revenue', ascending=False)
                cumulative_pct = (revenue_sorted['total_revenue'].cumsum() / 
                                 revenue_sorted['total_revenue'].sum() * 100)
                
                ax4_twin = ax4.twinx()
                
                x_range = range(len(revenue_sorted))
                bars4 = ax4.bar(x_range, revenue_sorted['total_revenue'].values,
                               color='#9b59b6', alpha=0.7, edgecolor='black', linewidth=1)
                line4 = ax4_twin.plot(x_range, cumulative_pct.values,
                                     color='#e74c3c', linewidth=3, marker='o', markersize=4)
                
                ax4.set_xlabel('Store Rank', fontsize=12, fontweight='bold')
                ax4.set_ylabel('Revenue (€)', fontsize=12, fontweight='bold', color='#9b59b6')
                ax4_twin.set_ylabel('Cumulative %', fontsize=12, fontweight='bold', color='#e74c3c')
                ax4.set_title('Store Revenue Concentration', fontsize=14, fontweight='bold')
                ax4_twin.axhline(80, color='orange', linestyle='--', linewidth=2, alpha=0.5)
                ax4.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                save_plot('05_store_performance.png')
                plt.show()
            except Exception as e:
                print(f"   ⚠️ Could not generate store performance visualization: {e}")
            
            print(f"   ✓ Store performance: {len(store_performance)} stores analyzed")
        except Exception as e:
            print(f"   ⚠️ Store performance analysis failed: {e}")
    else:
        print("   ⚠️ Skipping store analysis (missing store or revenue column)")
    
    # =========================================================================
    # ANALYSIS F: Channel Performance (Omnichannel)
    # =========================================================================
    print("\n🌐 F. Analyzing Channel Performance...")
    
    if channel_col and revenue_col and revenue_col in numeric_cols:
        try:
            channel_performance = df.groupby(channel_col).agg({
                revenue_col: 'sum',
                transaction_col: 'nunique' if transaction_col else 'count',
                customer_col: 'nunique' if customer_col else 'count'
            })
            channel_performance.columns = ['total_revenue', 'transactions', 'customers']
            channel_performance['avg_transaction_value'] = channel_performance['total_revenue'] / channel_performance['transactions']
            channel_performance['market_share_pct'] = (channel_performance['total_revenue'] / 
                                                       channel_performance['total_revenue'].sum() * 100)
            
            results['channel_performance'] = channel_performance.sort_values('total_revenue', ascending=False)
            
            # Visualization: Channel Performance
            try:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                
                # Top-left: Revenue by channel (pie)
                colors = plt.cm.Set3(range(len(channel_performance)))
                explode = [0.05] * len(channel_performance)
                
                wedges, texts, autotexts = ax1.pie(channel_performance['total_revenue'],
                                                   labels=channel_performance.index,
                                                   autopct='%1.1f%%',
                                                   startangle=90,
                                                   colors=colors,
                                                   explode=explode,
                                                   shadow=True,
                                                   textprops={'fontsize': 12, 'fontweight': 'bold'})
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(13)
                
                ax1.set_title('Revenue Distribution by Channel', fontsize=14, fontweight='bold')
                
                # Top-right: Transactions by channel
                bars2 = ax2.bar(range(len(channel_performance)), channel_performance['transactions'],
                               color=colors, edgecolor='black', linewidth=1.5)
                ax2.set_xticks(range(len(channel_performance)))
                ax2.set_xticklabels(channel_performance.index, fontsize=11)
                ax2.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
                ax2.set_title('Transaction Volume by Channel', fontsize=14, fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)
                
                for bar in bars2:
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
                
                # Bottom-left: Avg transaction value
                bars3 = ax3.barh(range(len(channel_performance)), channel_performance['avg_transaction_value'],
                                color=colors, edgecolor='black', linewidth=1.5)
                ax3.set_yticks(range(len(channel_performance)))
                ax3.set_yticklabels(channel_performance.index, fontsize=11)
                ax3.set_xlabel('Average Transaction Value (€)', fontsize=12, fontweight='bold')
                ax3.set_title('Average Transaction Value by Channel', fontsize=14, fontweight='bold')
                ax3.grid(axis='x', alpha=0.3)
                
                for i, bar in enumerate(bars3):
                    width = bar.get_width()
                    ax3.text(width + width*0.02, bar.get_y() + bar.get_height()/2.,
                            f'€{width:.2f}', ha='left', va='center', fontweight='bold')
                
                # Bottom-right: Market share comparison
                bars4 = ax4.bar(range(len(channel_performance)), channel_performance['market_share_pct'],
                               color=colors, edgecolor='black', linewidth=1.5)
                ax4.set_xticks(range(len(channel_performance)))
                ax4.set_xticklabels(channel_performance.index, fontsize=11)
                ax4.set_ylabel('Market Share (%)', fontsize=12, fontweight='bold')
                ax4.set_title('Market Share by Channel', fontsize=14, fontweight='bold')
                ax4.set_ylim([0, 100])
                ax4.grid(axis='y', alpha=0.3)
                
                for bar in bars4:
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                save_plot('06_channel_performance.png')
                plt.show()
            except Exception as e:
                print(f"   ⚠️ Could not generate channel visualization: {e}")
            
            print(f"   ✓ Channel analysis: {len(channel_performance)} channels compared")
        except Exception as e:
            print(f"   ⚠️ Channel performance analysis failed: {e}")
    else:
        print("   ⚠️ Skipping channel analysis (missing channel or revenue column)")
    
    # =========================================================================
    # ANALYSIS G: Temporal Trends & Seasonality
    # =========================================================================
    print("\n📅 G. Analyzing Temporal Trends & Seasonality...")
    
    if date_col and revenue_col and revenue_col in numeric_cols:
        try:
            temp_temporal = df[[date_col, revenue_col]].copy()
            temp_temporal[date_col] = pd.to_datetime(temp_temporal[date_col], errors='coerce')
            temp_temporal[revenue_col] = pd.to_numeric(temp_temporal[revenue_col], errors='coerce')
            temp_temporal = temp_temporal.dropna()
            
            if len(temp_temporal) > 0:
                # Daily sales
                daily_sales = temp_temporal.groupby(temp_temporal[date_col].dt.date)[revenue_col].sum()
                
                # Day of week analysis
                temp_temporal['day_of_week'] = temp_temporal[date_col].dt.day_name()
                dow_sales = temp_temporal.groupby('day_of_week')[revenue_col].sum()
                
                # Order days properly
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                dow_sales = dow_sales.reindex([d for d in day_order if d in dow_sales.index])
                
                # Hour of day if time available
                hour_sales = None
                if temp_temporal[date_col].dt.hour.nunique() > 1:
                    temp_temporal['hour'] = temp_temporal[date_col].dt.hour
                    hour_sales = temp_temporal.groupby('hour')[revenue_col].sum()
                
                # Monthly trend
                temp_temporal['month'] = temp_temporal[date_col].dt.to_period('M')
                monthly_sales = temp_temporal.groupby('month')[revenue_col].sum()
                
                results['temporal_trends'] = {
                    'daily_average': float(round(daily_sales.mean(), 2)),
                    'best_day_of_week': dow_sales.idxmax(),
                    'best_day_revenue': float(round(dow_sales.max(), 2))
                }
                
                if hour_sales is not None:
                    results['temporal_trends']['peak_hour'] = int(hour_sales.idxmax())
                
                # Visualization: Temporal Analysis
                try:
                    if hour_sales is not None:
                        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                    else:
                        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                    
                    # Daily trend
                    ax1.plot(range(len(daily_sales)), daily_sales.values, 
                            linewidth=2, color='#3498db', alpha=0.7)
                    ax1.fill_between(range(len(daily_sales)), daily_sales.values, alpha=0.3, color='#3498db')
                    ax1.set_xlabel('Day', fontsize=12, fontweight='bold')
                    ax1.set_ylabel('Revenue (€)', fontsize=12, fontweight='bold')
                    ax1.set_title('Daily Sales Trend', fontsize=14, fontweight='bold')
                    ax1.grid(True, alpha=0.3)
                    
                    # Day of week
                    colors_dow = ['#2ecc71' if x == dow_sales.max() else '#3498db' for x in dow_sales.values]
                    bars2 = ax2.bar(range(len(dow_sales)), dow_sales.values,
                                   color=colors_dow, edgecolor='black', linewidth=1.5)
                    ax2.set_xticks(range(len(dow_sales)))
                    ax2.set_xticklabels([d[:3] for d in dow_sales.index], fontsize=11)
                    ax2.set_ylabel('Total Revenue (€)', fontsize=12, fontweight='bold')
                    ax2.set_title('Revenue by Day of Week', fontsize=14, fontweight='bold')
                    ax2.grid(axis='y', alpha=0.3)
                    
                    for bar in bars2:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'€{int(height):,}', ha='center', va='bottom', 
                                fontweight='bold', fontsize=9)
                    
                    # Monthly trend
                    ax3.plot(range(len(monthly_sales)), monthly_sales.values,
                            marker='o', linewidth=3, markersize=8, color='#e74c3c')
                    ax3.fill_between(range(len(monthly_sales)), monthly_sales.values, 
                                    alpha=0.3, color='#e74c3c')
                    ax3.set_xlabel('Month', fontsize=12, fontweight='bold')
                    ax3.set_ylabel('Revenue (€)', fontsize=12, fontweight='bold')
                    ax3.set_title('Monthly Sales Trend', fontsize=14, fontweight='bold')
                    ax3.set_xticks(range(len(monthly_sales)))
                    ax3.set_xticklabels([str(m) for m in monthly_sales.index], 
                                       rotation=45, ha='right', fontsize=9)
                    ax3.grid(True, alpha=0.3)
                    
                    # Hour of day (if available)
                    if hour_sales is not None:
                        colors_hour = ['#f39c12' if x == hour_sales.max() else '#95a5a6' 
                                      for x in hour_sales.values]
                        bars4 = ax4.bar(hour_sales.index, hour_sales.values,
                                       color=colors_hour, edgecolor='black', linewidth=1.5)
                        ax4.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
                        ax4.set_ylabel('Total Revenue (€)', fontsize=12, fontweight='bold')
                        ax4.set_title('Revenue by Hour (Peak Hour Highlighted)', fontsize=14, fontweight='bold')
                        ax4.set_xticks(range(0, 24, 2))
                        ax4.grid(axis='y', alpha=0.3)
                    
                    plt.tight_layout()
                    save_plot('07_temporal_trends.png')
                    plt.show()
                except Exception as e:
                    print(f"   ⚠️ Could not generate temporal visualization: {e}")
                
                print(f"   ✓ Temporal analysis: Best day = {results['temporal_trends']['best_day_of_week']}")
        except Exception as e:
            print(f"   ⚠️ Temporal trends analysis failed: {e}")
    else:
        print("   ⚠️ Skipping temporal analysis (missing date or revenue column)")
    
    # =========================================================================
    # ANALYSIS H: Demand Forecasting
    # =========================================================================
    print("\n🔮 H. Generating Demand Forecast...")
    
    if date_col and revenue_col and revenue_col in numeric_cols:
        try:
            temp_forecast = df[[date_col, revenue_col]].copy()
            temp_forecast[date_col] = pd.to_datetime(temp_forecast[date_col], errors='coerce')
            temp_forecast[revenue_col] = pd.to_numeric(temp_forecast[revenue_col], errors='coerce')
            temp_forecast = temp_forecast.dropna()
            
            if len(temp_forecast) > 7:
                temp_forecast = temp_forecast.sort_values(date_col)
                daily_data = temp_forecast.groupby(temp_forecast[date_col].dt.date)[revenue_col].sum().reset_index()
                daily_data['time_index'] = range(len(daily_data))
                
                X = daily_data[['time_index']].values
                y = daily_data[revenue_col].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Forecast next 7 days
                future_days = np.array([[len(X) + i] for i in range(1, 8)])
                predictions = model.predict(future_days)
                r2_score = model.score(X, y)
                
                results['demand_forecast'] = {
                    'forecast_7_days': [float(round(p, 2)) for p in predictions],
                    'trend': 'Increasing' if predictions[-1] > predictions[0] else 'Decreasing',
                    'r2_score': float(round(r2_score, 3)),
                    'daily_growth_rate': float(round(model.coef_[0], 4))
                }
                
                # Visualization: Forecast
                try:
                    plt.figure(figsize=(14, 7))
                    
                    # Actual data
                    plt.plot(range(len(y)), y, 'o-', label='Actual Sales',
                            linewidth=3, markersize=6, color='#3498db', alpha=0.8)
                    
                    # Forecast
                    forecast_x = range(len(y), len(y) + 7)
                    plt.plot(forecast_x, predictions, 's--', label='7-Day Forecast',
                            color='#e74c3c', linewidth=3, markersize=8, alpha=0.8)
                    
                    # Forecast area
                    plt.axvspan(len(y) - 0.5, len(y) + 6.5, alpha=0.1, color='orange', label='Forecast Period')
                    
                    # Value labels on forecast
                    for i, (x, pred) in enumerate(zip(forecast_x, predictions)):
                        if i % 2 == 0:
                            plt.text(x, pred + (y.max() - y.min()) * 0.03, f'€{pred:.0f}',
                                    ha='center', va='bottom', fontweight='bold', fontsize=10,
                                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                    
                    plt.xlabel('Day', fontsize=13, fontweight='bold')
                    plt.ylabel('Revenue (€)', fontsize=13, fontweight='bold')
                    plt.title(f'Demand Forecast - Next 7 Days\n(R² = {r2_score:.3f}, Trend: {results["demand_forecast"]["trend"]})',
                             fontsize=15, fontweight='bold', pad=20)
                    plt.legend(loc='best', fontsize=12)
                    plt.grid(True, alpha=0.3, linestyle='--')
                    
                    plt.tight_layout()
                    save_plot('08_demand_forecast.png')
                    plt.show()
                except Exception as e:
                    print(f"   ⚠️ Could not generate forecast visualization: {e}")
                
                print(f"   ✓ Forecast generated (R² = {r2_score:.3f})")
        except Exception as e:
            print(f"   ⚠️ Demand forecasting failed: {e}")
    else:
        print("   ⚠️ Skipping forecast (missing date or revenue column)")
    
    print("\n" + "="*60)
    print("✅ Retail Analytics completed successfully!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    """Generate sample retail dataset and test analytics"""
    print("\n" + "="*60)
    print("RETAIL ANALYTICS TEST MODE - GENERATING SAMPLE DATA")
    print("="*60 + "\n")
    
    # Generate realistic retail dataset
    np.random.seed(42)
    n_transactions = 1000
    n_customers = 200
    n_products = 50
    n_stores = 5
    
    start_date = datetime(2024, 1, 1)
    
    data = []
    for i in range(n_transactions):
        transaction_id = f'TXN{i+1:05d}'
        customer_id = f'CUST{np.random.randint(1, n_customers+1):04d}'
        product_id = f'PROD{np.random.randint(1, n_products+1):03d}'
        store_id = f'STORE{np.random.choice(["A", "B", "C", "D", "E"])}'
        channel = np.random.choice(['Online', 'In-Store'], p=[0.3, 0.7])
        category = np.random.choice(['Electronics', 'Clothing', 'Home', 'Beauty', 'Sports'])
        
        quantity = np.random.randint(1, 5)
        unit_price = np.random.gamma(5, 10)
        revenue = quantity * unit_price
        
        # Add temporal patterns
        days_offset = np.random.randint(0, 90)
        hour = np.random.choice(range(9, 21), p=[0.05]*4 + [0.15]*4 + [0.1]*4)
        transaction_date = start_date + timedelta(days=days_offset, hours=hour)
        
        data.append({
            'Order_ID': transaction_id,
            'Customer_ID': customer_id,
            'Product_SKU': product_id,
            'Store_Location': store_id,
            'Sales_Channel': channel,
            'Product_Category': category,
            'Units_Sold': quantity,
            'Unit_Price': round(unit_price, 2),
            'Total_Sale': round(revenue, 2),
            'Transaction_Date': transaction_date
        })
    
    df_test = pd.DataFrame(data)
    
    # Save to CSV
    output_file = "retail_test.csv"
    df_test.to_csv(output_file, index=False)
    print(f"✅ Sample dataset created: {output_file} ({len(df_test)} transactions)\n")
    
    # Test mapping (imperfect column names)
    test_mapping = {
        'transaction_id': 'Order_ID',
        'customer_id': 'Customer_ID',
        'product_id': 'Product_SKU',
        'revenue': 'Total_Sale',
        'quantity': 'Units_Sold',
        'price': 'Unit_Price',
        'date': 'Transaction_Date',
        'store_id': 'Store_Location',
        'channel': 'Sales_Channel',
        'category': 'Product_Category'
    }
    
    try:
        results = analytics_retail(df_test, mapping=test_mapping,
                                  save_plots=True, output_dir="test_retail_plots")
        
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