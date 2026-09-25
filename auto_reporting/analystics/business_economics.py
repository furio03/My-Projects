import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from LLM.llm_output import identify_target_variable, needed_variables
from data_manipulation.read_data import identify_variables, correct_dtypes
from statistics_functions.clustering import auto_kmeans

def analytics_business_economics(df: pd.DataFrame, mapping: dict = None, user_objective: str = None,
                                 save_plots: bool = False, output_dir: str = "domain_plots"):
    """
    Perform analytics for the business_economics domain.

    Analyses performed:
    - A: Revenue Distribution by Dimension - Groups revenue metrics by business dimensions
    - B: Efficiency Analysis (Value per Volume) - Calculates average efficiency ratio
    - C: Pareto Analysis (80/20 Rule) - Identifies core business drivers
    - D: Customer Lifetime Value (CLV) - Calculates CLV score
    - E: Price Elasticity and Sensitivity - Determines price elasticity coefficient
    - F: Contribution Margin Analysis - Evaluates profitability
    - G: Customer K-Means Clustering - Segments customers/products
    - H: Break-Even Analysis - Calculates break-even units
    - I: Trend & Momentum Analysis - Analyzes time-series data
    - J: Price Optimization Strategy - Recommends pricing strategy
    - K: Forecasting (Linear Trend) - Generates forecasts
    """
    print("Running analysis for business_economics")
    
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
    
    # Firstly we divide variables based on their business_economics role
    economic_roles = ["monetary_value", "unit_cost", "volume_metrics", "business_dimension"]
    if mapping is None:
        mapping = needed_variables(df, economic_roles)

    # Function to clean mapping
    def get_single_col(key):
        val = mapping.get(key)
        if not val: return None
        if isinstance(val, list):
            val = val[0] if len(val) > 0 else None
        if isinstance(val, str) and "," in val:
            val = val.split(",")[0].strip()
        return val if val in df.columns else None

    # Clean Variables
    revenue_col = get_single_col('monetary_value')
    volume_col  = get_single_col('volume_metrics')
    dim_col     = get_single_col('business_dimension')
    cost_col    = get_single_col('unit_cost') 

    results = {}

    # ANALYSIS A: Revenue Distribution by Dimension
    if revenue_col and dim_col:
        print(f"Analyzing {revenue_col} by {dim_col}")
        results['performance_by_segment'] = df.groupby(dim_col)[revenue_col].agg(['mean', 'sum', 'count'])
        
        # Visualization: Revenue by Segment
        try:
            segment_data = df.groupby(dim_col)[revenue_col].sum().sort_values(ascending=False).head(10)
            
            plt.figure(figsize=(12, 7))
            colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(segment_data)))
            bars = plt.barh(range(len(segment_data)), segment_data.values, 
                           color=colors, edgecolor='black', linewidth=1.5)
            plt.yticks(range(len(segment_data)), segment_data.index, fontsize=11)
            plt.xlabel('Total Revenue (€)', fontsize=12, fontweight='bold')
            plt.title('Revenue Distribution by Business Segment (Top 10)', fontsize=14, fontweight='bold')
            plt.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + width*0.02, bar.get_y() + bar.get_height()/2.,
                        f'€{int(width):,}', ha='left', va='center', fontweight='bold')
            
            plt.tight_layout()
            save_plot('01_revenue_by_segment.png')
            plt.close()
        except Exception as e:
            print(f"   ⚠️ Could not generate revenue distribution visualization: {e}")
    
    # ANALYSIS B: Efficiency (Value per Volume)
    if revenue_col and volume_col:
        rev_single = revenue_col[0] if isinstance(revenue_col, list) else revenue_col
        vol_single = volume_col[0] if isinstance(volume_col, list) else volume_col
        
        print(f"Calculating efficiency: {rev_single} / {vol_single}")
        
        df['value_per_unit'] = df[rev_single] / df[vol_single].replace(0, 1)
        results['avg_efficiency'] = df['value_per_unit'].mean()

    # ANALYSIS C: Pareto (80/20 Rule)
    if revenue_col and dim_col:
        top_segments = df.groupby(dim_col)[revenue_col].sum().sort_values(ascending=False)
        cumulative = top_segments.cumsum() / top_segments.sum()
        results['core_business_drivers'] = cumulative[cumulative <= 0.8].index.tolist()
        
        # Visualization: Pareto Chart
        try:
            plt.figure(figsize=(14, 7))
            
            # Create dual axis
            fig, ax1 = plt.subplots(figsize=(14, 7))
            ax2 = ax1.twinx()
            
            # Take top 15 for visualization
            top_15 = top_segments.head(15)
            cumulative_15 = (top_15.cumsum() / top_segments.sum() * 100)
            
            x = range(len(top_15))
            
            # Bar chart
            colors = ['#2ecc71' if cumulative_15.iloc[i] <= 80 else '#95a5a6' for i in range(len(top_15))]
            bars = ax1.bar(x, top_15.values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.7)
            ax1.set_xlabel('Business Segment', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Revenue (€)', fontsize=12, fontweight='bold', color='black')
            ax1.set_xticks(x)
            ax1.set_xticklabels(top_15.index, rotation=45, ha='right', fontsize=10)
            ax1.tick_params(axis='y', labelcolor='black')
            
            # Cumulative line
            line = ax2.plot(x, cumulative_15.values, color='#e74c3c', marker='o', 
                           linewidth=3, markersize=8, label='Cumulative %')
            ax2.axhline(80, color='orange', linestyle='--', linewidth=2, label='80% Threshold')
            ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold', color='#e74c3c')
            ax2.tick_params(axis='y', labelcolor='#e74c3c')
            ax2.set_ylim([0, 105])
            ax2.legend(loc='lower right', fontsize=11)
            
            plt.title('Pareto Analysis: 80/20 Rule (Top 15 Segments)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            save_plot('02_pareto_analysis.png')
            plt.close()
        except Exception as e:
            print(f"   ⚠️ Could not generate Pareto visualization: {e}")

    # ANALYSIS D: Customer Lifetime Value (CLV)
    if revenue_col and volume_col and dim_col:
        clv_metrics = df.groupby(dim_col).agg({
            revenue_col: ['sum', 'count'],
        })
        clv_metrics.columns = ['total_revenue', 'order_count']
        clv_metrics['average_order_value'] = clv_metrics['total_revenue'] / clv_metrics['order_count']
        clv_metrics['clv_score'] = clv_metrics['average_order_value'] * clv_metrics['order_count']
        results['clv_analysis'] = clv_metrics.sort_values(by='clv_score', ascending=False)
        
        # Visualization: CLV Scatter
        try:
            top_clv = clv_metrics.sort_values(by='clv_score', ascending=False).head(20)
            
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(top_clv['average_order_value'], top_clv['order_count'],
                                 s=top_clv['clv_score']/top_clv['clv_score'].max()*1000,
                                 c=top_clv['clv_score'], cmap='YlOrRd', 
                                 alpha=0.6, edgecolors='black', linewidth=1.5)
            
            plt.xlabel('Average Order Value (€)', fontsize=12, fontweight='bold')
            plt.ylabel('Purchase Frequency (Orders)', fontsize=12, fontweight='bold')
            plt.title('Customer Lifetime Value Analysis (Top 20 Segments)', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, linestyle='--')
            
            cbar = plt.colorbar(scatter)
            cbar.set_label('CLV Score', fontsize=11, fontweight='bold')
            
            # Annotate top 5
            for idx, row in top_clv.head(5).iterrows():
                plt.annotate(str(idx)[:15], 
                           (row['average_order_value'], row['order_count']),
                           fontsize=9, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            plt.tight_layout()
            save_plot('03_clv_analysis.png')
            plt.close()
        except Exception as e:
            print(f"   ⚠️ Could not generate CLV visualization: {e}")

    # ANALYSIS E: Price Elasticity and Sensitivity
    price_col = mapping.get('price_metrics')
    volume_col_temp = mapping.get('volume_metrics')

    if isinstance(price_col, list): price_col = price_col[0]
    if isinstance(volume_col_temp, list): volume_col_temp = volume_col_temp[0]

    if price_col and volume_col_temp and price_col in df.columns and volume_col_temp in df.columns:
        print(f"Calculating Price Elasticity using {price_col} and {volume_col_temp}...")
        
        mask = (df[price_col] > 0) & (df[volume_col_temp] > 0)
        valid_data = df[mask].copy()
        
        if len(valid_data) > 5:
            log_p = np.log(valid_data[[price_col]])
            log_q = np.log(valid_data[volume_col_temp])
            
            model = LinearRegression()
            model.fit(log_p, log_q)
            
            elasticity_coef = model.coef_[0]
            interpretation = "Elastic" if abs(elasticity_coef) > 1 else "Inelastic"
            
            results['price_elasticity'] = {
                'coefficient': round(elasticity_coef, 4),
                'type': interpretation,
                'summary': f"A 1% price increase results in a {abs(elasticity_coef):.2f}% change in volume."
            }
            print(f"Elasticity calculated: {elasticity_coef:.4f} ({interpretation})")
            
            # Visualization: Price-Volume Relationship
            try:
                plt.figure(figsize=(12, 7))
                
                # Scatter plot
                plt.scatter(valid_data[price_col], valid_data[volume_col_temp],
                           alpha=0.5, s=50, color='#3498db', edgecolors='black', linewidth=0.5)
                
                # Fit line
                prices_sorted = np.sort(valid_data[price_col].values)
                volumes_pred = np.exp(model.predict(np.log(prices_sorted.reshape(-1, 1))))
                plt.plot(prices_sorted, volumes_pred, color='#e74c3c', linewidth=3, 
                        label=f'Elasticity: {elasticity_coef:.2f} ({interpretation})')
                
                plt.xlabel('Price (€)', fontsize=12, fontweight='bold')
                plt.ylabel('Volume (Units)', fontsize=12, fontweight='bold')
                plt.title('Price Elasticity Analysis', fontsize=14, fontweight='bold')
                plt.legend(fontsize=11, loc='best')
                plt.grid(True, alpha=0.3, linestyle='--')
                
                plt.tight_layout()
                save_plot('04_price_elasticity.png')
                plt.close()
            except Exception as e:
                print(f"   ⚠️ Could not generate elasticity visualization: {e}")
        else:
            print("Skipping Elasticity: Insufficient valid data points.")

    # ANALYSIS F: Contribution Margin Analysis
    cost_per_unit_col = mapping.get('unit_cost')

    if revenue_col and cost_per_unit_col and volume_col:
        print(f"Calculating Contribution Margin using {cost_per_unit_col}")
        
        df['total_variable_cost'] = df[cost_per_unit_col] * df[volume_col]
        df['contribution_margin'] = df[revenue_col] - df['total_variable_cost']
        df['margin_ratio'] = (df['contribution_margin'] / df[revenue_col].replace(0, 1)) * 100
        
        if dim_col:
            margin_by_dim = df.groupby(dim_col).agg({
                'contribution_margin': 'sum',
                'margin_ratio': 'mean',
                revenue_col: 'sum'
            }).rename(columns={revenue_col: 'total_revenue'})
            
            results['profitability_analysis'] = margin_by_dim.sort_values(by='contribution_margin', ascending=False)
            
            loss_making_segments = margin_by_dim[margin_by_dim['contribution_margin'] < 0].index.tolist()
            results['loss_making_segments'] = loss_making_segments
            
            # Visualization: Profitability Analysis
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
                
                # Left: Contribution Margin by Segment
                top_margin = margin_by_dim.sort_values('contribution_margin', ascending=False).head(10)
                colors1 = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_margin['contribution_margin']]
                bars1 = ax1.barh(range(len(top_margin)), top_margin['contribution_margin'],
                                color=colors1, edgecolor='black', linewidth=1.5)
                ax1.set_yticks(range(len(top_margin)))
                ax1.set_yticklabels(top_margin.index, fontsize=11)
                ax1.set_xlabel('Contribution Margin (€)', fontsize=12, fontweight='bold')
                ax1.set_title('Profitability by Segment (Top 10)', fontsize=14, fontweight='bold')
                ax1.axvline(0, color='black', linestyle='-', linewidth=1)
                ax1.grid(axis='x', alpha=0.3)
                
                # Right: Margin Ratio
                top_ratio = margin_by_dim.sort_values('margin_ratio', ascending=False).head(10)
                colors2 = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(top_ratio)))
                bars2 = ax2.bar(range(len(top_ratio)), top_ratio['margin_ratio'],
                               color=colors2, edgecolor='black', linewidth=1.5)
                ax2.set_xticks(range(len(top_ratio)))
                ax2.set_xticklabels(top_ratio.index, rotation=45, ha='right', fontsize=10)
                ax2.set_ylabel('Margin Ratio (%)', fontsize=12, fontweight='bold')
                ax2.set_title('Margin Ratio by Segment (Top 10)', fontsize=14, fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)
                ax2.axhline(0, color='black', linestyle='-', linewidth=1)
                
                plt.tight_layout()
                save_plot('05_profitability_analysis.png')
                plt.close()
            except Exception as e:
                print(f"   ⚠️ Could not generate profitability visualization: {e}")

    # ANALYSIS G: Customers K-Means
    # Get features for clustering with proper cleaning
    features_for_clustering = []
    
    monetary_val = get_single_col('monetary_value')
    volume_val = get_single_col('volume_metrics')
    
    if monetary_val:
        features_for_clustering.append(monetary_val)
    if volume_val:
        features_for_clustering.append(volume_val)
    
    # Remove duplicates and ensure they exist in df
    features_for_clustering = [col for col in features_for_clustering if col in df.columns]
    features_for_clustering = list(dict.fromkeys(features_for_clustering))  # Remove duplicates

    if len(features_for_clustering) >= 2:
        try:
            clustering_data = df[features_for_clustering].dropna()
            print(f"Performing K-Means clustering on features: {features_for_clustering}")
            cluster_profiles = auto_kmeans(clustering_data)
            results['customer_segments'] = cluster_profiles
        except Exception as e:
            print(f"   ⚠️ Clustering failed: {e}")
    
    # ANALYSIS H: Break-Even Analysis
    if 'contribution_margin' in df.columns and volume_col:
        estimated_fixed_costs = df[revenue_col].sum() * 0.20
        avg_unit_cm = df['contribution_margin'].sum() / df[volume_col].sum()
        
        if avg_unit_cm > 0:
            bep_units = estimated_fixed_costs / avg_unit_cm
            current_units = df[volume_col].sum()
            
            results['break_even'] = {
                'break_even_units': round(bep_units, 2),
                'current_units': current_units,
                'safety_margin_pct': round(((current_units - bep_units) / current_units) * 100, 2)
            }
            
            # Visualization: Break-Even Chart
            try:
                plt.figure(figsize=(12, 7))
                
                # Create units range
                units_range = np.linspace(0, current_units * 1.2, 100)
                total_revenue = units_range * (df[revenue_col].sum() / df[volume_col].sum())
                total_costs = estimated_fixed_costs + (units_range * (df['total_variable_cost'].sum() / df[volume_col].sum()))
                
                plt.plot(units_range, total_revenue, linewidth=3, color='#2ecc71', label='Total Revenue')
                plt.plot(units_range, total_costs, linewidth=3, color='#e74c3c', label='Total Costs')
                plt.axvline(bep_units, color='orange', linestyle='--', linewidth=2, label=f'Break-Even: {int(bep_units):,} units')
                plt.axvline(current_units, color='blue', linestyle='--', linewidth=2, label=f'Current: {int(current_units):,} units')
                
                # Fill profit area
                plt.fill_between(units_range, total_revenue, total_costs, 
                                where=(units_range > bep_units), alpha=0.3, color='green', label='Profit Zone')
                
                plt.xlabel('Units Sold', fontsize=12, fontweight='bold')
                plt.ylabel('Amount (€)', fontsize=12, fontweight='bold')
                plt.title(f'Break-Even Analysis (Safety Margin: {results["break_even"]["safety_margin_pct"]:.1f}%)', 
                         fontsize=14, fontweight='bold')
                plt.legend(fontsize=11, loc='upper left')
                plt.grid(True, alpha=0.3, linestyle='--')
                
                plt.tight_layout()
                save_plot('06_break_even_analysis.png')
                plt.close()
            except Exception as e:
                print(f"   ⚠️ Could not generate break-even visualization: {e}")

    # ANALYSIS I: Trend & Momentum Analysis
    var_types = identify_variables(df)
    date_cols = var_types.get('datetime', [])

    if date_cols and 'contribution_margin' in df.columns:
        date_col = date_cols[0]
        df[date_col] = pd.to_datetime(df[date_col])
        
        monthly_data = df.set_index(date_col).resample('ME').agg({
            revenue_col: 'sum',
            'contribution_margin': 'sum',
            volume_col: 'sum'
        })
        
        monthly_data['margin_pct'] = (monthly_data['contribution_margin'] / monthly_data[revenue_col].replace(0, 1)) * 100
        monthly_data['revenue_growth_mom'] = monthly_data[revenue_col].pct_change() * 100
        monthly_data['efficiency_change_mom'] = monthly_data['margin_pct'].diff()
        
        results['trends_analysis'] = monthly_data.dropna()
        
        # Visualization: Trends Analysis
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Top: Revenue Trend
            x = range(len(monthly_data))
            ax1.plot(x, monthly_data[revenue_col], marker='o', linewidth=3, 
                    markersize=8, color='#3498db', label='Revenue')
            ax1.fill_between(x, monthly_data[revenue_col], alpha=0.3, color='#3498db')
            ax1.set_ylabel('Revenue (€)', fontsize=12, fontweight='bold')
            ax1.set_title('Revenue Trend Over Time', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels([d.strftime('%Y-%m') for d in monthly_data.index], rotation=45, ha='right')
            ax1.legend(fontsize=11)
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # Bottom: Margin % Trend
            ax2.plot(x, monthly_data['margin_pct'], marker='s', linewidth=3, 
                    markersize=8, color='#2ecc71', label='Margin %')
            ax2.fill_between(x, monthly_data['margin_pct'], alpha=0.3, color='#2ecc71')
            ax2.set_xlabel('Period', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Margin (%)', fontsize=12, fontweight='bold')
            ax2.set_title('Profitability Trend Over Time', fontsize=14, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels([d.strftime('%Y-%m') for d in monthly_data.index], rotation=45, ha='right')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            save_plot('07_trends_analysis.png')
            plt.close()
        except Exception as e:
            print(f"   ⚠️ Could not generate trends visualization: {e}")
    else:
        print("Skipping Trend Analysis: No datetime variables identified.")  
    
    # ANALYSIS J: Price Optimization Strategy
    if 'price_elasticity' in results and 'profitability_analysis' in results:
        e = results['price_elasticity']['coefficient']
        if e < -1:
            optimal_markup_factor = e / (1 + e)
            results['pricing_strategy'] = {
                'recommendation': "Volume Penetration",
                'suggested_markup_multiplier': round(optimal_markup_factor, 2)
            }
        else:
            results['pricing_strategy'] = {
                'recommendation': "Premium Skimming",
                'action': "Test 5-10% price increase to capture consumer surplus."
            }
    
    # ANALYSIS K: Forecasting (Basic Linear Trend)
    target_info = identify_target_variable(df, user_objective)
    y_col = target_info.get('target_variable')

    if y_col:
        print(f"Generating Forecast for target: {y_col}")
        
        temp_df = df.dropna(subset=[y_col]).copy()
        temp_df['index_num'] = np.arange(len(temp_df))
        
        X = temp_df[['index_num']]
        y = temp_df[y_col]
        
        if len(X) > 5:
            model = LinearRegression()
            model.fit(X, y)
            
            future_indices = np.array([[len(X)], [len(X)+1], [len(X)+2]])
            preds = model.predict(future_indices)
            
            results['forecast'] = {
                'target_analyzed': y_col,
                'forecast_values': [float(p) for p in preds.tolist()],
                'trend': "Increasing" if preds[-1] > preds[0] else "Decreasing",
                'r2_score': round(model.score(X, y), 3)
            }
            
            # Visualization: Forecast
            try:
                plt.figure(figsize=(14, 7))
                
                # Actual data
                plt.plot(range(len(y)), y, 'o-', label='Actual Data',
                        linewidth=3, markersize=8, color='#3498db', alpha=0.8)
                
                # Forecast
                forecast_x = range(len(y), len(y) + 3)
                plt.plot(forecast_x, preds, 's--', label='3-Period Forecast',
                        color='#e74c3c', linewidth=3, markersize=10, alpha=0.8)
                
                # Forecast area
                plt.axvspan(len(y) - 0.5, len(y) + 2.5, alpha=0.1, color='orange')
                
                # Value labels
                for i, (x, pred) in enumerate(zip(forecast_x, preds)):
                    plt.text(x, pred + (y.max() - y.min()) * 0.03, f'{pred:.1f}',
                            ha='center', va='bottom', fontweight='bold', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                
                plt.xlabel('Time Period', fontsize=13, fontweight='bold')
                plt.ylabel(y_col.replace('_', ' ').title(), fontsize=13, fontweight='bold')
                plt.title(f'Business Forecast: {y_col.replace("_", " ").title()}\n(R² = {results["forecast"]["r2_score"]:.3f})',
                         fontsize=15, fontweight='bold', pad=20)
                plt.legend(loc='best', fontsize=12)
                plt.grid(True, alpha=0.3, linestyle='--')
                
                plt.tight_layout()
                save_plot('08_business_forecast.png')
                plt.close()
            except Exception as e:
                print(f"   ⚠️ Could not generate forecast visualization: {e}")

    print("Business Economics analysis completed.")
    
    return results


if __name__ == "__main__":
    file_path = os.path.join("data", "economy_test.csv")

    if os.path.exists(file_path):
        df_test = pd.read_csv(file_path)
        print(f"✅ Dataset caricato: {len(df_test)} righe")

        user_goal = "Analizza il profitto e suggerisci strategie di prezzo ottimali" 

        test_mapping = {
            'monetary_value': 'Price_Unit',      
            'volume_metrics': 'Quantity_Demanded',
            'business_dimension': 'Product_ID',
            'unit_cost': 'Unit_Variable_Cost',
            'price_metrics': 'Price_Unit'
        }

        try:
            results = analytics_business_economics(df_test, test_mapping, user_objective=user_goal,
                                                  save_plots=True, output_dir="test_business_plots")
            
            print("\n" + "="*40)
            print(f"🚀 TEST RESULTS SUMMARY (Goal: {user_goal})")
            print("="*40)
            
            for key, value in results.items():
                print(f"\n📊 {key.upper()}:")
                if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                    print(value.head())
                else:
                    print(value)
                    
        except Exception as e:
            print(f"❌ Errore durante l'esecuzione: {e}")
            import traceback
            traceback.print_exc()
            
    else:
        print(f"❌ Errore: Il file '{file_path}' non è stato trovato.")