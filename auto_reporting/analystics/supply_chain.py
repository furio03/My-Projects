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
from statistics_functions.clustering import auto_kmeans
from LLM.llm_output import needed_variables


def analytics_supply_chain(df: pd.DataFrame, mapping: dict = None, user_objective: str = None,
                           save_plots: bool = False, output_dir: str = "domain_plots"):
    """
    Comprehensive supply chain analytics based on SCOR model.
    
    Analyses (9 dimensions):
    - A: Supplier Performance    - On-time delivery, quality, lead time (SOURCE)
    - B: Inventory Management    - Stock levels, turnover, safety stock (PLAN)
    - C: Demand Forecasting      - Predictive analytics for future demand (PLAN)
    - D: Logistics Performance   - Delivery times, transportation costs (DELIVER)
    - E: Production Efficiency   - Cycle time, capacity utilization (MAKE)
    - F: Perfect Order Rate      - Quality KPIs and customer satisfaction
    - G: Cost Analysis           - Total supply chain costs breakdown
    - H: Cash-to-Cash Cycle      - Working capital and financial efficiency
    - I: Risk Assessment         - Supplier risk, disruption probability (RESILIENCE)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Supply chain dataset
    mapping : dict, optional
        Column mapping
    user_objective : str, optional
        Analysis objective
    save_plots : bool, default=False
        Save visualizations
    output_dir : str, default="domain_plots"
        Output directory
    """
    print("Running comprehensive supply chain analytics (SCOR model)")
    
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Plots will be saved to: {output_dir}/")
    
    def save_plot(filename):
        if save_plots:
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"   ✓ Plot saved: {filename}")
        plt.close()
    
    # Supply chain variable roles
    sc_roles = [
        'order_id', 'supplier_id', 'product_id', 'quantity',
        'order_date', 'delivery_date', 'lead_time',
        'cost', 'inventory_level', 'demand',
        'delivery_status', 'quality_score', 'region'
    ]
    
    if mapping is None:
        mapping = needed_variables(df, sc_roles)
    
    def get_single_col(key):
        val = mapping.get(key)
        if not val:
            return None
        if isinstance(val, list):
            val = val[0] if len(val) > 0 else None
        if isinstance(val, str) and ',' in val:
            val = val.split(',')[0].strip()
        return val if isinstance(val, str) and val in df.columns else None
    
    order_id     = get_single_col('order_id')
    supplier_id  = get_single_col('supplier_id')
    product_id   = get_single_col('product_id')
    quantity_col = get_single_col('quantity')
    order_date   = get_single_col('order_date')
    delivery_date = get_single_col('delivery_date')
    lead_time    = get_single_col('lead_time')
    cost_col     = get_single_col('cost')
    inventory_col = get_single_col('inventory_level')
    demand_col   = get_single_col('demand')
    status_col   = get_single_col('delivery_status')
    quality_col  = get_single_col('quality_score')
    region_col   = get_single_col('region')
    
    var_types = identify_variables(df)
    numeric_cols = var_types.get('quantitative', [])
    results = {}
    
    # =========================================================================
    # A: SUPPLIER PERFORMANCE (SOURCE)
    # =========================================================================
    print("\n🏭 A. Analyzing Supplier Performance (SOURCE)...")
    
    if supplier_id and lead_time and lead_time in numeric_cols:
        try:
            supplier_perf = df.groupby(supplier_id).agg({
                lead_time: ['mean', 'std', 'count'],
                quality_col: 'mean' if quality_col and quality_col in numeric_cols else 'count'
            })
            supplier_perf.columns = ['avg_lead_time', 'std_lead_time', 'order_count', 'quality_metric']
            supplier_perf = supplier_perf.sort_values('avg_lead_time')
            
            # On-time delivery rate
            if status_col:
                otd = df.groupby(supplier_id)[status_col].apply(
                    lambda x: (x.str.lower().str.contains('on time|delivered', na=False).sum() / len(x) * 100)
                )
                supplier_perf['on_time_delivery_pct'] = otd
            
            results['supplier_performance'] = supplier_perf.head(20)
            
            # Visualization
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                
                top_10 = supplier_perf.head(10)
                colors = ['#2ecc71' if x < supplier_perf['avg_lead_time'].median() 
                         else '#e74c3c' for x in top_10['avg_lead_time']]
                
                bars1 = ax1.barh(range(len(top_10)), top_10['avg_lead_time'],
                                color=colors, edgecolor='black', linewidth=1.5)
                ax1.set_yticks(range(len(top_10)))
                ax1.set_yticklabels(top_10.index, fontsize=10)
                ax1.set_xlabel('Avg Lead Time (days)', fontsize=12, fontweight='bold')
                ax1.set_title('Supplier Lead Time Performance', fontsize=14, fontweight='bold')
                ax1.axvline(supplier_perf['avg_lead_time'].median(), 
                           color='orange', linestyle='--', linewidth=2, label='Median')
                ax1.legend()
                ax1.grid(axis='x', alpha=0.3)
                
                for i, bar in enumerate(bars1):
                    w = bar.get_width()
                    ax1.text(w + 0.2, bar.get_y() + bar.get_height()/2.,
                            f'{w:.1f}d', ha='left', va='center', fontweight='bold')
                
                # On-time delivery if available
                if 'on_time_delivery_pct' in supplier_perf.columns:
                    otd_top = supplier_perf.nlargest(10, 'on_time_delivery_pct')
                    colors2 = ['#2ecc71' if x >= 90 else '#f39c12' if x >= 75 else '#e74c3c'
                              for x in otd_top['on_time_delivery_pct']]
                    bars2 = ax2.bar(range(len(otd_top)), otd_top['on_time_delivery_pct'],
                                   color=colors2, edgecolor='black', linewidth=1.5)
                    ax2.set_xticks(range(len(otd_top)))
                    ax2.set_xticklabels(otd_top.index, rotation=45, ha='right', fontsize=9)
                    ax2.set_ylabel('On-Time Delivery (%)', fontsize=12, fontweight='bold')
                    ax2.set_title('Supplier Reliability (Top 10)', fontsize=14, fontweight='bold')
                    ax2.axhline(90, color='green', linestyle='--', linewidth=2, label='Target 90%')
                    ax2.legend()
                    ax2.grid(axis='y', alpha=0.3)
                    
                    for bar in bars2:
                        h = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., h,
                                f'{h:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                save_plot('01_supplier_performance.png')
            except Exception as e:
                print(f"   ⚠️ Supplier visualization failed: {e}")
            
            print(f"   ✓ {len(supplier_perf)} suppliers analyzed")
        except Exception as e:
            print(f"   ⚠️ Supplier analysis failed: {e}")
    else:
        print("   ⚠️ Skipping (supplier_id or lead_time not found)")
    
    # =========================================================================
    # B: INVENTORY MANAGEMENT (PLAN)
    # =========================================================================
    print("\n📦 B. Analyzing Inventory Management (PLAN)...")
    
    if inventory_col and inventory_col in numeric_cols:
        try:
            inv_stats = {
                'avg_inventory_level': float(df[inventory_col].mean()),
                'safety_stock_estimate': float(df[inventory_col].quantile(0.25)),
                'max_inventory': float(df[inventory_col].max()),
                'stockout_risk_pct': float((df[inventory_col] == 0).sum() / len(df) * 100)
            }
            
            if quantity_col and quantity_col in numeric_cols:
                inv_stats['turnover_ratio'] = float(
                    df[quantity_col].sum() / df[inventory_col].mean()
                )
            
            results['inventory_kpis'] = inv_stats
            
            # ABC Analysis if product available
            if product_id and quantity_col and cost_col:
                product_value = df.groupby(product_id).agg({
                    quantity_col: 'sum',
                    cost_col: 'mean'
                })
                product_value['total_value'] = (
                    product_value[quantity_col] * product_value[cost_col]
                )
                product_value = product_value.sort_values('total_value', ascending=False)
                
                cumsum = product_value['total_value'].cumsum()
                total = product_value['total_value'].sum()
                cumsum_pct = cumsum / total * 100
                
                product_value['ABC_class'] = 'C'
                product_value.loc[cumsum_pct <= 80, 'ABC_class'] = 'A'
                product_value.loc[(cumsum_pct > 80) & (cumsum_pct <= 95), 'ABC_class'] = 'B'
                
                results['abc_analysis'] = product_value
            
            # Visualization
            try:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                
                # Inventory distribution
                inv_data = df[inventory_col].clip(upper=df[inventory_col].quantile(0.95))
                ax1.hist(inv_data, bins=30, color='#3498db', edgecolor='black', 
                        linewidth=1.5, alpha=0.7)
                ax1.axvline(inv_stats['avg_inventory_level'], color='red', 
                           linestyle='--', linewidth=2, label=f"Mean: {inv_stats['avg_inventory_level']:.0f}")
                ax1.set_xlabel('Inventory Level', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                ax1.set_title('Inventory Level Distribution', fontsize=14, fontweight='bold')
                ax1.legend()
                ax1.grid(axis='y', alpha=0.3)
                
                # KPIs dashboard
                ax2.axis('off')
                ax2.set_title('Inventory KPIs', fontsize=14, fontweight='bold')
                y_pos = 0.85
                colors_kpi = ['#3498db', '#f39c12', '#2ecc71', '#e74c3c', '#9b59b6']
                for i, (k, v) in enumerate(inv_stats.items()):
                    label = k.replace('_', ' ').title()
                    text = f'{label}: {v:.2f}'
                    ax2.text(0.05, y_pos, text, fontsize=12, fontweight='bold',
                            transform=ax2.transAxes,
                            bbox=dict(boxstyle='round', facecolor=colors_kpi[i % len(colors_kpi)],
                                     alpha=0.7, edgecolor='black'))
                    y_pos -= 0.15
                
                # ABC Analysis
                if 'abc_analysis' in results:
                    abc = results['abc_analysis']
                    abc_dist = abc['ABC_class'].value_counts()
                    colors_abc = {'A': '#e74c3c', 'B': '#f39c12', 'C': '#2ecc71'}
                    pie_colors = [colors_abc[c] for c in abc_dist.index]
                    
                    wedges, texts, autotexts = ax3.pie(abc_dist.values, labels=abc_dist.index,
                                                       autopct='%1.1f%%', colors=pie_colors,
                                                       startangle=90, shadow=True,
                                                       textprops={'fontsize': 12, 'fontweight': 'bold'})
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    ax3.set_title('ABC Inventory Classification', fontsize=14, fontweight='bold')
                    
                    # Top products by value
                    top_prod = abc.head(10)
                    bars4 = ax4.barh(range(len(top_prod)), top_prod['total_value'],
                                    color='#3498db', edgecolor='black', linewidth=1.5)
                    ax4.set_yticks(range(len(top_prod)))
                    ax4.set_yticklabels([str(x)[:15] for x in top_prod.index], fontsize=10)
                    ax4.set_xlabel('Total Value', fontsize=12, fontweight='bold')
                    ax4.set_title('Top 10 Products by Value', fontsize=14, fontweight='bold')
                    ax4.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                save_plot('02_inventory_management.png')
            except Exception as e:
                print(f"   ⚠️ Inventory visualization failed: {e}")
            
            print(f"   ✓ Turnover ratio: {inv_stats.get('turnover_ratio', 0):.2f}x")
        except Exception as e:
            print(f"   ⚠️ Inventory analysis failed: {e}")
    else:
        print("   ⚠️ Skipping (inventory_level not found)")
    
    # =========================================================================
    # C: DEMAND FORECASTING (PLAN - PREDICTIVE)
    # =========================================================================
    print("\n🔮 C. Generating Demand Forecast (PREDICTIVE)...")
    
    if order_date and quantity_col and quantity_col in numeric_cols:
        try:
            temp_fc = df[[order_date, quantity_col]].copy()
            temp_fc[order_date] = pd.to_datetime(temp_fc[order_date], errors='coerce')
            temp_fc[quantity_col] = pd.to_numeric(temp_fc[quantity_col], errors='coerce')
            temp_fc = temp_fc.dropna()
            
            if len(temp_fc) > 10:
                temp_fc = temp_fc.sort_values(order_date)
                daily_demand = temp_fc.groupby(temp_fc[order_date].dt.date)[quantity_col].sum()
                daily_demand = daily_demand.reset_index()
                daily_demand['time_idx'] = range(len(daily_demand))
                
                X = daily_demand[['time_idx']].values
                y = daily_demand[quantity_col].values
                
                model = LinearRegression()
                model.fit(X, y)
                
                future = np.array([[len(X) + i] for i in range(1, 8)])
                preds = model.predict(future)
                r2 = model.score(X, y)
                
                results['demand_forecast'] = {
                    'forecast_7_days': [float(p) for p in preds],
                    'trend': 'Increasing' if preds[-1] > preds[0] else 'Decreasing',
                    'r2_score': float(r2),
                    'daily_growth': float(model.coef_[0])
                }
                
                # Visualization
                try:
                    plt.figure(figsize=(14, 7))
                    plt.plot(range(len(y)), y, 'o-', linewidth=3, markersize=6,
                            color='#3498db', alpha=0.8, label='Actual Demand')
                    fx = range(len(y), len(y) + 7)
                    plt.plot(fx, preds, 's--', linewidth=3, markersize=8,
                            color='#e74c3c', alpha=0.8, label='7-Day Forecast')
                    plt.axvspan(len(y) - 0.5, len(y) + 6.5, alpha=0.1, 
                               color='orange', label='Forecast Period')
                    
                    for i, (xi, p) in enumerate(zip(fx, preds)):
                        if i % 2 == 0:
                            plt.text(xi, p, f'{p:.0f}', ha='center', va='bottom',
                                    fontweight='bold', fontsize=10,
                                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                    
                    plt.xlabel('Time Period (days)', fontsize=13, fontweight='bold')
                    plt.ylabel('Demand (units)', fontsize=13, fontweight='bold')
                    plt.title(f'Demand Forecast - Next 7 Days (R²={r2:.3f})',
                             fontsize=15, fontweight='bold')
                    plt.legend(loc='best', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    save_plot('03_demand_forecast.png')
                except Exception as e:
                    print(f"   ⚠️ Forecast visualization failed: {e}")
                
                print(f"   ✓ Forecast: {results['demand_forecast']['trend']} (R²={r2:.3f})")
        except Exception as e:
            print(f"   ⚠️ Demand forecast failed: {e}")
    else:
        print("   ⚠️ Skipping (order_date or quantity not found)")
    
    # =========================================================================
    # D: LOGISTICS PERFORMANCE (DELIVER)
    # =========================================================================
    print("\n🚚 D. Analyzing Logistics Performance (DELIVER)...")
    
    if delivery_date and order_date:
        try:
            temp_log = df[[order_date, delivery_date]].copy()
            temp_log[order_date] = pd.to_datetime(temp_log[order_date], errors='coerce')
            temp_log[delivery_date] = pd.to_datetime(temp_log[delivery_date], errors='coerce')
            temp_log = temp_log.dropna()
            
            temp_log['actual_lead_time'] = (
                temp_log[delivery_date] - temp_log[order_date]
            ).dt.days
            
            logistics_kpis = {
                'avg_delivery_time_days': float(temp_log['actual_lead_time'].mean()),
                'median_delivery_time': float(temp_log['actual_lead_time'].median()),
                'on_time_rate_pct': float((temp_log['actual_lead_time'] <= 7).sum() / len(temp_log) * 100)
            }
            
            results['logistics_performance'] = logistics_kpis
            
            # Visualization
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                
                # Lead time distribution
                lead_data = temp_log['actual_lead_time'].clip(0, temp_log['actual_lead_time'].quantile(0.95))
                ax1.hist(lead_data, bins=30, color='#3498db', edgecolor='black',
                        linewidth=1.5, alpha=0.7)
                ax1.axvline(logistics_kpis['avg_delivery_time_days'], color='red',
                           linestyle='--', linewidth=2, 
                           label=f"Avg: {logistics_kpis['avg_delivery_time_days']:.1f}d")
                ax1.axvline(7, color='green', linestyle='--', linewidth=2, label='Target: 7d')
                ax1.set_xlabel('Delivery Time (days)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                ax1.set_title('Delivery Time Distribution', fontsize=14, fontweight='bold')
                ax1.legend()
                ax1.grid(axis='y', alpha=0.3)
                
                # KPI gauge
                ax2.axis('off')
                ax2.set_title('Logistics KPIs', fontsize=14, fontweight='bold')
                y_pos = 0.85
                colors_log = ['#3498db', '#2ecc71', '#f39c12']
                for i, (k, v) in enumerate(logistics_kpis.items()):
                    label = k.replace('_', ' ').title()
                    text = f'{label}: {v:.1f}{"%" if "pct" in k else "d"}'
                    ax2.text(0.05, y_pos, text, fontsize=13, fontweight='bold',
                            transform=ax2.transAxes,
                            bbox=dict(boxstyle='round', facecolor=colors_log[i],
                                     alpha=0.7, edgecolor='black', linewidth=2))
                    y_pos -= 0.2
                
                plt.tight_layout()
                save_plot('04_logistics_performance.png')
            except Exception as e:
                print(f"   ⚠️ Logistics visualization failed: {e}")
            
            print(f"   ✓ On-time rate: {logistics_kpis['on_time_rate_pct']:.1f}%")
        except Exception as e:
            print(f"   ⚠️ Logistics analysis failed: {e}")
    else:
        print("   ⚠️ Skipping (delivery_date or order_date not found)")
    
    # =========================================================================
    # E: PRODUCTION EFFICIENCY (MAKE)
    # =========================================================================
    print("\n⚙️  E. Analyzing Production Efficiency (MAKE)...")
    
    if quantity_col and quantity_col in numeric_cols and order_date:
        try:
            temp_prod = df[[order_date, quantity_col]].copy()
            temp_prod[order_date] = pd.to_datetime(temp_prod[order_date], errors='coerce')
            temp_prod[quantity_col] = pd.to_numeric(temp_prod[quantity_col], errors='coerce')
            temp_prod = temp_prod.dropna()
            
            temp_prod['month'] = temp_prod[order_date].dt.to_period('M')
            monthly_prod = temp_prod.groupby('month')[quantity_col].agg(['sum', 'mean', 'count'])
            
            production_kpis = {
                'total_units_produced': int(temp_prod[quantity_col].sum()),
                'avg_daily_output': float(temp_prod.groupby(temp_prod[order_date].dt.date)[quantity_col].sum().mean()),
                'production_variability_cv': float(monthly_prod['sum'].std() / monthly_prod['sum'].mean())
            }
            
            results['production_efficiency'] = production_kpis
            
            # Visualization
            try:
                plt.figure(figsize=(14, 7))
                x = range(len(monthly_prod))
                plt.plot(x, monthly_prod['sum'], marker='o', linewidth=3,
                        markersize=8, color='#2ecc71', label='Monthly Production')
                plt.fill_between(x, monthly_prod['sum'], alpha=0.3, color='#2ecc71')
                plt.axhline(monthly_prod['sum'].mean(), color='red', linestyle='--',
                           linewidth=2, label=f"Avg: {monthly_prod['sum'].mean():.0f}")
                plt.xticks(x, [str(m) for m in monthly_prod.index], rotation=45, ha='right')
                plt.xlabel('Month', fontsize=12, fontweight='bold')
                plt.ylabel('Units Produced', fontsize=12, fontweight='bold')
                plt.title('Production Output Trend', fontsize=14, fontweight='bold')
                plt.legend(fontsize=11)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                save_plot('05_production_efficiency.png')
            except Exception as e:
                print(f"   ⚠️ Production visualization failed: {e}")
            
            print(f"   ✓ Total units: {production_kpis['total_units_produced']:,}")
        except Exception as e:
            print(f"   ⚠️ Production analysis failed: {e}")
    else:
        print("   ⚠️ Skipping (quantity or order_date not found)")
    
    # =========================================================================
    # F: PERFECT ORDER RATE (QUALITY KPI)
    # =========================================================================
    print("\n✅ F. Calculating Perfect Order Rate...")
    
    if status_col:
        try:
            perfect_mask = df[status_col].str.lower().str.contains(
                'perfect|complete|on time|delivered', na=False)
            perfect_rate = perfect_mask.sum() / len(df) * 100
            
            results['perfect_order_rate'] = {
                'perfect_order_pct': float(perfect_rate),
                'total_orders': len(df),
                'perfect_orders': int(perfect_mask.sum())
            }
            
            print(f"   ✓ Perfect order rate: {perfect_rate:.1f}%")
        except Exception as e:
            print(f"   ⚠️ Perfect order calculation failed: {e}")
    else:
        print("   ⚠️ Skipping (delivery_status not found)")
    
    # =========================================================================
    # G: COST ANALYSIS (FINANCIAL)
    # =========================================================================
    print("\n💰 G. Analyzing Supply Chain Costs...")
    
    if cost_col and cost_col in numeric_cols:
        try:
            total_cost = df[cost_col].sum()
            
            cost_breakdown = {
                'total_supply_chain_cost': float(total_cost),
                'avg_cost_per_order': float(df[cost_col].mean()),
                'cost_per_unit': float(total_cost / df[quantity_col].sum()) if quantity_col else 0
            }
            
            # Cost by category if available
            if product_id:
                cost_by_prod = df.groupby(product_id)[cost_col].sum().sort_values(ascending=False)
                cost_breakdown['top_cost_driver'] = str(cost_by_prod.index[0])
                results['cost_by_product'] = cost_by_prod.head(10)
            
            results['cost_analysis'] = cost_breakdown
            
            # Visualization
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                
                # Cost distribution
                cost_data = df[cost_col].clip(upper=df[cost_col].quantile(0.95))
                ax1.hist(cost_data, bins=30, color='#e74c3c', edgecolor='black',
                        linewidth=1.5, alpha=0.7)
                ax1.axvline(cost_breakdown['avg_cost_per_order'], color='navy',
                           linestyle='--', linewidth=2,
                           label=f"Avg: €{cost_breakdown['avg_cost_per_order']:.2f}")
                ax1.set_xlabel('Cost per Order (€)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                ax1.set_title('Cost Distribution', fontsize=14, fontweight='bold')
                ax1.legend()
                ax1.grid(axis='y', alpha=0.3)
                
                # Cost by product
                if 'cost_by_product' in results:
                    cbp = results['cost_by_product']
                    colors2 = plt.cm.Reds(np.linspace(0.4, 0.9, len(cbp)))
                    bars2 = ax2.barh(range(len(cbp)), cbp.values,
                                    color=colors2, edgecolor='black', linewidth=1.5)
                    ax2.set_yticks(range(len(cbp)))
                    ax2.set_yticklabels([str(x)[:15] for x in cbp.index], fontsize=10)
                    ax2.set_xlabel('Total Cost (€)', fontsize=12, fontweight='bold')
                    ax2.set_title('Cost by Product (Top 10)', fontsize=14, fontweight='bold')
                    ax2.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                save_plot('06_cost_analysis.png')
            except Exception as e:
                print(f"   ⚠️ Cost visualization failed: {e}")
            
            print(f"   ✓ Total cost: €{total_cost:,.2f}")
        except Exception as e:
            print(f"   ⚠️ Cost analysis failed: {e}")
    else:
        print("   ⚠️ Skipping (cost column not found)")
    
    # =========================================================================
    # H: CASH-TO-CASH CYCLE (WORKING CAPITAL)
    # =========================================================================
    print("\n💵 H. Calculating Cash-to-Cash Cycle...")
    
    if order_date and delivery_date and inventory_col:
        try:
            # Simplified C2C = DIO + DSO - DPO
            # DIO = Days Inventory Outstanding
            if inventory_col in numeric_cols and quantity_col in numeric_cols:
                avg_inv = df[inventory_col].mean()
                daily_usage = df[quantity_col].sum() / 365
                dio = avg_inv / daily_usage if daily_usage > 0 else 0
                
                # DSO = Days Sales Outstanding (approximated by lead time)
                if lead_time and lead_time in numeric_cols:
                    dso = df[lead_time].mean()
                else:
                    dso = 30  # Default estimate
                
                dpo = 30  # Default payment terms estimate
                
                c2c = dio + dso - dpo
                
                results['cash_to_cash_cycle'] = {
                    'days_inventory_outstanding': float(dio),
                    'days_sales_outstanding': float(dso),
                    'days_payable_outstanding': float(dpo),
                    'cash_to_cash_days': float(c2c)
                }
                
                print(f"   ✓ Cash-to-Cash cycle: {c2c:.1f} days")
        except Exception as e:
            print(f"   ⚠️ C2C calculation failed: {e}")
    else:
        print("   ⚠️ Skipping (insufficient data for C2C)")
    
    # =========================================================================
    # I: RISK ASSESSMENT (RESILIENCE)
    # =========================================================================
    print("\n⚠️  I. Assessing Supply Chain Risk...")
    
    if supplier_id:
        try:
            supplier_counts = df.groupby(supplier_id).size()
            
            # Concentration risk (Herfindahl index)
            market_shares = supplier_counts / supplier_counts.sum()
            hhi = (market_shares ** 2).sum()
            
            # Lead time variability as risk proxy
            if lead_time and lead_time in numeric_cols:
                risk_scores = df.groupby(supplier_id)[lead_time].agg(['mean', 'std'])
                risk_scores['risk_score'] = risk_scores['std'] / risk_scores['mean']
                risk_scores = risk_scores.sort_values('risk_score', ascending=False)
                results['supplier_risk'] = risk_scores.head(10)
            
            results['concentration_risk'] = {
                'herfindahl_index': float(hhi),
                'risk_level': 'High' if hhi > 0.25 else ('Medium' if hhi > 0.15 else 'Low'),
                'n_suppliers': int(len(supplier_counts))
            }
            
            # Visualization
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                
                # Supplier concentration
                top_supp = supplier_counts.nlargest(10)
                colors1 = plt.cm.Oranges(np.linspace(0.4, 0.9, len(top_supp)))
                bars1 = ax1.bar(range(len(top_supp)), top_supp.values,
                               color=colors1, edgecolor='black', linewidth=1.5)
                ax1.set_xticks(range(len(top_supp)))
                ax1.set_xticklabels(top_supp.index, rotation=45, ha='right', fontsize=9)
                ax1.set_ylabel('Order Count', fontsize=12, fontweight='bold')
                ax1.set_title('Supplier Concentration (Top 10)', fontsize=14, fontweight='bold')
                ax1.grid(axis='y', alpha=0.3)
                
                # Risk scores
                if 'supplier_risk' in results:
                    risk = results['supplier_risk']
                    colors2 = ['#e74c3c' if x > risk['risk_score'].median() 
                              else '#2ecc71' for x in risk['risk_score']]
                    bars2 = ax2.barh(range(len(risk)), risk['risk_score'],
                                    color=colors2, edgecolor='black', linewidth=1.5)
                    ax2.set_yticks(range(len(risk)))
                    ax2.set_yticklabels(risk.index, fontsize=10)
                    ax2.set_xlabel('Risk Score (CV)', fontsize=12, fontweight='bold')
                    ax2.set_title('Supplier Risk Assessment', fontsize=14, fontweight='bold')
                    ax2.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                save_plot('07_risk_assessment.png')
            except Exception as e:
                print(f"   ⚠️ Risk visualization failed: {e}")
            
            print(f"   ✓ Risk: {results['concentration_risk']['risk_level']} (HHI={hhi:.3f})")
        except Exception as e:
            print(f"   ⚠️ Risk assessment failed: {e}")
    else:
        print("   ⚠️ Skipping (supplier_id not found)")
    
    # =========================================================================
    # SCOR DASHBOARD (SUMMARY)
    # =========================================================================
    print("\n📊 Generating SCOR Dashboard...")
    
    try:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # Color scheme for SCOR sections
        scor_colors = {
            'plan': '#E3F2FD',      # Light Blue
            'source': '#F3E5F5',    # Light Purple
            'make': '#E8F5E9',      # Light Green
            'deliver': '#FFF3E0',   # Light Orange
            'return': '#FCE4EC',    # Light Pink
            'enable': '#F1F8E9'     # Light Lime
        }
        
        # Plan
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=scor_colors['plan'], 
                                    edgecolor='#1976D2', linewidth=3))
        ax1.text(0.5, 0.92, 'PLAN', ha='center', fontsize=16, fontweight='bold',
                color='#1976D2')
        if 'inventory_kpis' in results:
            y = 0.75
            for k, v in results['inventory_kpis'].items():
                ax1.text(0.05, y, f"{k.replace('_',' ').title()}: {v:.1f}",
                        fontsize=9, wrap=True)
                y -= 0.12
        ax1.axis('off')
        
        # Source
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=scor_colors['source'],
                                    edgecolor='#7B1FA2', linewidth=3))
        ax2.text(0.5, 0.92, 'SOURCE', ha='center', fontsize=16, fontweight='bold',
                color='#7B1FA2')
        if 'supplier_performance' in results and len(results['supplier_performance']) > 0:
            ax2.text(0.05, 0.75, f"Suppliers: {len(results['supplier_performance'])}",
                    fontsize=10)
            ax2.text(0.05, 0.60, f"Avg Lead Time: {results['supplier_performance']['avg_lead_time'].mean():.1f}d",
                    fontsize=10)
        ax2.axis('off')
        
        # Make
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=scor_colors['make'],
                                    edgecolor='#388E3C', linewidth=3))
        ax3.text(0.5, 0.92, 'MAKE', ha='center', fontsize=16, fontweight='bold',
                color='#388E3C')
        if 'production_efficiency' in results:
            y = 0.75
            for k, v in results['production_efficiency'].items():
                ax3.text(0.05, y, f"{k.replace('_',' ').title()}: {v:,.0f}",
                        fontsize=9, wrap=True)
                y -= 0.12
        ax3.axis('off')
        
        # Deliver
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=scor_colors['deliver'],
                                    edgecolor='#F57C00', linewidth=3))
        ax4.text(0.5, 0.92, 'DELIVER', ha='center', fontsize=16, fontweight='bold',
                color='#F57C00')
        if 'logistics_performance' in results:
            y = 0.75
            for k, v in results['logistics_performance'].items():
                ax4.text(0.05, y, f"{k.replace('_',' ').title()}: {v:.1f}",
                        fontsize=9, wrap=True)
                y -= 0.12
        ax4.axis('off')
        
        # Return (placeholder)
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=scor_colors['return'],
                                    edgecolor='#C2185B', linewidth=3))
        ax5.text(0.5, 0.5, 'RETURN\n(Reverse Logistics)', ha='center', va='center',
                fontsize=14, fontweight='bold', color='#C2185B')
        ax5.axis('off')
        
        # Enable (Financial)
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=scor_colors['enable'],
                                    edgecolor='#689F38', linewidth=3))
        ax6.text(0.5, 0.92, 'ENABLE (Finance)', ha='center', fontsize=16, fontweight='bold',
                color='#689F38')
        if 'cost_analysis' in results:
            ax6.text(0.05, 0.75, f"Total Cost: €{results['cost_analysis']['total_supply_chain_cost']:,.0f}",
                    fontsize=10)
        if 'cash_to_cash_cycle' in results:
            ax6.text(0.05, 0.60, f"C2C: {results['cash_to_cash_cycle']['cash_to_cash_days']:.0f} days",
                    fontsize=10)
        ax6.axis('off')
        
        # Risk & Resilience
        ax7 = fig.add_subplot(gs[2, :])
        ax7.set_xlim(0, 1)
        ax7.set_ylim(0, 1)
        ax7.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor='#FFEBEE',
                                    edgecolor='#D32F2F', linewidth=3))
        ax7.text(0.5, 0.85, 'RESILIENCE & RISK', ha='center', fontsize=16, fontweight='bold',
                color='#D32F2F')
        if 'concentration_risk' in results:
            cr = results['concentration_risk']
            risk_color = '#e74c3c' if cr['risk_level'] == 'High' else ('#f39c12' if cr['risk_level'] == 'Medium' else '#2ecc71')
            ax7.text(0.2, 0.55, f"Risk Level: {cr['risk_level']}", fontsize=12,
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=risk_color, 
                             edgecolor='black', linewidth=2))
            ax7.text(0.5, 0.55, f"Suppliers: {cr['n_suppliers']}", fontsize=12,
                    fontweight='bold')
            ax7.text(0.7, 0.55, f"HHI: {cr['herfindahl_index']:.3f}", fontsize=12,
                    fontweight='bold')
        ax7.axis('off')
        
        plt.suptitle('SCOR Model Dashboard - Supply Chain Overview',
                    fontsize=18, fontweight='bold', y=0.98)
        save_plot('08_scor_dashboard.png')
    except Exception as e:
        print(f"   ⚠️ Dashboard generation failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Supply Chain Analytics completed!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Test data generation
    np.random.seed(42)
    n = 500
    
    start_date = datetime(2023, 1, 1)
    suppliers = [f'SUP{i:03d}' for i in range(1, 11)]
    products = [f'PROD{i:03d}' for i in range(1, 31)]
    
    data = []
    for i in range(n):
        order_dt = start_date + timedelta(days=np.random.randint(0, 365))
        lead = np.random.randint(3, 15)
        delivery_dt = order_dt + timedelta(days=lead)
        
        data.append({
            'OrderID': f'ORD{i+1:05d}',
            'SupplierID': np.random.choice(suppliers),
            'ProductID': np.random.choice(products),
            'Quantity': np.random.randint(10, 200),
            'OrderDate': order_dt.strftime('%Y-%m-%d'),
            'DeliveryDate': delivery_dt.strftime('%Y-%m-%d'),
            'LeadTime': lead,
            'Cost': round(np.random.gamma(3, 50), 2),
            'InventoryLevel': np.random.randint(50, 500),
            'DeliveryStatus': np.random.choice(['On Time', 'Delayed', 'On Time', 'On Time']),
            'QualityScore': round(np.random.uniform(85, 100), 1)
        })
    
    df = pd.DataFrame(data)
    df.to_csv("supply_chain_test.csv", index=False)
    print(f"✅ Test dataset: {len(df)} orders")
    
    mapping = {
        'order_id': 'OrderID', 'supplier_id': 'SupplierID',
        'product_id': 'ProductID', 'quantity': 'Quantity',
        'order_date': 'OrderDate', 'delivery_date': 'DeliveryDate',
        'lead_time': 'LeadTime', 'cost': 'Cost',
        'inventory_level': 'InventoryLevel',
        'delivery_status': 'DeliveryStatus',
        'quality_score': 'QualityScore'
    }
    
    results = analytics_supply_chain(df, mapping=mapping,
                                     save_plots=True, output_dir="domain_plots")
    
    print("\n" + "=" * 50)
    print("RESULTS:")
    for k, v in results.items():
        if isinstance(v, pd.DataFrame):
            print(f"\n{k}: {v.shape}")
            print(v.head(3))
        else:
            print(f"\n{k}: {v}")