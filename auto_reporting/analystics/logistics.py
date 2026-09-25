import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats
from datetime import datetime, timedelta

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from LLM.llm_output import identify_target_variable, needed_variables
from data_manipulation.read_data import identify_variables
from statistics_functions.clustering import auto_kmeans, auto_kmedoids

def analytics_logistics(df: pd.DataFrame, mapping: dict = None, user_objective: str = None, 
                        save_plots: bool = False, output_dir: str = "domain_plots"):
    """
    Perform comprehensive analytics for the Logistics domain.
    
    Analyses performed:
    - A: Warehouse Operations (Receiving, Storage, Picking, Shipping) - Analyzes process times and bottlenecks
    - B: Transportation Performance (Routes & ETA) - Evaluates delivery times, route efficiency, and SLA compliance
    - C: Fleet Management - Analyzes vehicle utilization, costs, and maintenance
    - D: Inventory Management - Evaluates stock levels, turnover rates, and obsolescence
    - E: Cost Analysis - Breaks down logistics costs (transport, storage, handling)
    - F: Productivity & Resource Efficiency - Measures operator and equipment efficiency
    - G: Supply Chain Visibility - Tracks shipment status and identifies delays
    - H: Reverse Logistics - Analyzes returns, recycling, and reverse flow efficiency
    - I: Route Optimization Analysis - Identifies optimal routes and delivery patterns
    - J: Service Level Agreement (SLA) Compliance - Monitors on-time delivery performance
    - K: Logistics Forecasting - Predicts future volumes, costs, and resource needs
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input logistics dataset
    mapping : dict, optional
        Column mapping for logistics roles
    user_objective : str, optional
        User's analysis objective
    save_plots : bool, default=False
        Whether to save plots to disk for report generation
    output_dir : str, default="logistics_analysis_plots"
        Directory to save plots if save_plots=True
    """
    print("Running comprehensive analysis for Logistics domain")
    
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
    
    # 1. Define logistics-specific roles for variable mapping
    logistics_roles = [
        "route_identifier",      # Route ID, Route Name, Lane
        "vehicle_identifier",    # Vehicle ID, Truck Number, Fleet ID
        "location_identifier",   # Warehouse ID, Hub, Depot
        "time_metric",          # Delivery Time, Process Time, Lead Time
        "volume_metric",        # Units Shipped, Pallets, Weight
        "cost_metric",          # Transportation Cost, Storage Cost
        "gps_coordinate",       # Latitude/Longitude, GPS Data
        "status_indicator",     # Delivery Status, Order Status
        "datetime_column",      # Timestamp, Delivery Date, Pickup Time
        "distance_metric",      # Distance, Miles, Kilometers
        "sla_metric"           # On-Time Delivery, SLA Compliance
    ]
    
    if mapping is None:
        mapping = needed_variables(df, logistics_roles)
    
    # 2. Helper function to extract single column with fallback
    def get_single_col(key):
        """Extract single column name from mapping, handling lists and comma-separated strings"""
        val = mapping.get(key)
        if not val: 
            return None
        if isinstance(val, list):
            val = val[0] if len(val) > 0 else None
        if isinstance(val, str) and "," in val:
            val = val.split(",")[0].strip()
        return val if val in df.columns else None
    
    # 3. Extract clean variable names with fallback warnings
    route_col = get_single_col('route_identifier')
    vehicle_col = get_single_col('vehicle_identifier')
    location_col = get_single_col('location_identifier')
    time_col = get_single_col('time_metric')
    volume_col = get_single_col('volume_metric')
    cost_col = get_single_col('cost_metric')
    gps_col = get_single_col('gps_coordinate')
    status_col = get_single_col('status_indicator')
    datetime_col = get_single_col('datetime_column')
    distance_col = get_single_col('distance_metric')
    sla_col = get_single_col('sla_metric')
    
    # Fallback warnings for missing critical columns
    missing_cols = []
    if not route_col and not vehicle_col:
        missing_cols.append('route_identifier or vehicle_identifier (Route/Vehicle)')
    if not time_col:
        missing_cols.append('time_metric (Delivery/Process Time)')
    if not volume_col:
        missing_cols.append('volume_metric (Shipment Volume)')
    
    if missing_cols:
        print(f"\n⚠️  WARNING: Missing columns detected:")
        for col in missing_cols:
            print(f"   - {col}")
        print("   Some analyses will be skipped.\n")
    
    results = {}
    
    # --- ANALYSIS A: Warehouse Operations (Receiving, Storage, Picking, Shipping) ---
    print("\n📦 A. Analyzing Warehouse Operations...")
    
    if time_col and status_col:
        # Analyze process times by operation stage
        warehouse_ops = df.groupby(status_col)[time_col].agg(['mean', 'median', 'std', 'count'])
        warehouse_ops.columns = ['avg_time', 'median_time', 'std_time', 'process_count']
        warehouse_ops = warehouse_ops.sort_values('avg_time', ascending=False)
        warehouse_ops.index.name = 'Operation_Stage'
        results['warehouse_operations'] = warehouse_ops
        
        # Identify bottlenecks (operations with longest times)
        if len(warehouse_ops) > 0:
            bottleneck_threshold = warehouse_ops['avg_time'].quantile(0.75)
            bottlenecks = warehouse_ops[warehouse_ops['avg_time'] > bottleneck_threshold]
            results['process_bottlenecks'] = bottlenecks.index.tolist()
            
            # Visualization: Warehouse process times
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Left: Average process time by stage
                colors = plt.cm.Oranges(np.linspace(0.4, 0.8, len(warehouse_ops)))
                bars1 = ax1.barh(range(len(warehouse_ops)), warehouse_ops['avg_time'], color=colors)
                ax1.set_yticks(range(len(warehouse_ops)))
                ax1.set_yticklabels(warehouse_ops.index, fontsize=11)
                ax1.set_xlabel('Average Process Time (hours)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Operation Stage', fontsize=12, fontweight='bold')
                ax1.set_title('Warehouse Process Times by Stage', fontsize=14, fontweight='bold')
                ax1.grid(axis='x', alpha=0.3)
                
                # Add value labels
                for i, bar in enumerate(bars1):
                    width = bar.get_width()
                    ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                            f'{width:.1f}h', ha='left', va='center', fontweight='bold')
                
                # Right: Process count distribution
                colors2 = plt.cm.Blues(np.linspace(0.4, 0.8, len(warehouse_ops)))
                bars2 = ax2.bar(range(len(warehouse_ops)), warehouse_ops['process_count'], color=colors2)
                ax2.set_xticks(range(len(warehouse_ops)))
                ax2.set_xticklabels(warehouse_ops.index, rotation=45, ha='right', fontsize=10)
                ax2.set_ylabel('Number of Operations', fontsize=12, fontweight='bold')
                ax2.set_title('Operation Volume by Stage', fontsize=14, fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for i, bar in enumerate(bars2):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                save_plot('01_warehouse_operations.png')
                plt.show()
            except Exception as e:
                print(f"   ⚠️ Could not generate warehouse operations visualization: {e}")
        
        print(f"   ✓ Identified {len(bottlenecks) if len(warehouse_ops) > 0 else 0} process bottlenecks")
    else:
        print("   ⚠️ Skipping warehouse operations analysis (time or status column not available)")
    
    # --- ANALYSIS B: Transportation Performance (Routes & ETA) ---
    print("\n🚚 B. Analyzing Transportation Performance...")
    
    if route_col and time_col:
        # Analyze delivery performance by route
        route_performance = df.groupby(route_col)[time_col].agg(['mean', 'median', 'std', 'count'])
        route_performance.columns = ['avg_delivery_time', 'median_delivery_time', 'std_dev', 'shipment_count']
        route_performance = route_performance.sort_values('avg_delivery_time', ascending=False)
        route_performance.index.name = 'Route'
        results['route_performance'] = route_performance
        
        # Calculate route efficiency score (lower time + less variance = better)
        if len(route_performance) > 0:
            route_performance['efficiency_score'] = (
                route_performance['avg_delivery_time'].max() - route_performance['avg_delivery_time']
            ) / route_performance['std_dev'].replace(0, 1)
            results['top_efficient_routes'] = route_performance.nlargest(5, 'efficiency_score')
            
            # Visualization: Route performance comparison
            try:
                plt.figure(figsize=(14, 7))
                
                # Take top 10 routes by volume
                top_routes = route_performance.nlargest(10, 'shipment_count')
                
                x = np.arange(len(top_routes))
                width = 0.35
                
                fig, ax = plt.subplots(figsize=(14, 7))
                bars1 = ax.bar(x - width/2, top_routes['avg_delivery_time'], width, 
                              label='Avg Delivery Time', color='#3498db', alpha=0.8)
                bars2 = ax.bar(x + width/2, top_routes['median_delivery_time'], width,
                              label='Median Delivery Time', color='#2ecc71', alpha=0.8)
                
                ax.set_xlabel('Route', fontsize=12, fontweight='bold')
                ax.set_ylabel('Delivery Time (hours)', fontsize=12, fontweight='bold')
                ax.set_title('Transportation Performance by Route (Top 10 by Volume)', 
                            fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(top_routes.index, rotation=45, ha='right')
                ax.legend(fontsize=11)
                ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                save_plot('02_route_performance.png')
                plt.show()
            except Exception as e:
                print(f"   ⚠️ Could not generate route performance visualization: {e}")
        
        print(f"   ✓ Analyzed {len(route_performance)} routes")
    
    elif vehicle_col and time_col:
        # Fallback: Analyze by vehicle if route not available
        vehicle_performance = df.groupby(vehicle_col)[time_col].agg(['mean', 'median', 'count'])
        vehicle_performance.columns = ['avg_delivery_time', 'median_delivery_time', 'trip_count']
        results['vehicle_performance'] = vehicle_performance.sort_values('avg_delivery_time')
        print(f"   ✓ Analyzed {len(vehicle_performance)} vehicles (route data not available)")
    else:
        print("   ⚠️ Skipping transportation performance analysis (route/vehicle or time column not available)")
    
    # --- ANALYSIS C: Fleet Management ---
    print("\n🚛 C. Analyzing Fleet Management...")
    
    if vehicle_col:
        # Vehicle utilization analysis
        vehicle_stats = df.groupby(vehicle_col).size().to_frame('trips')
        vehicle_stats.index.name = 'Vehicle'
        
        if time_col:
            vehicle_stats['total_time'] = df.groupby(vehicle_col)[time_col].sum()
            vehicle_stats['avg_trip_time'] = df.groupby(vehicle_col)[time_col].mean()
        
        if distance_col:
            vehicle_stats['total_distance'] = df.groupby(vehicle_col)[distance_col].sum()
            vehicle_stats['avg_distance_per_trip'] = df.groupby(vehicle_col)[distance_col].mean()
        
        if cost_col:
            vehicle_stats['total_cost'] = df.groupby(vehicle_col)[cost_col].sum()
            vehicle_stats['cost_per_trip'] = df.groupby(vehicle_col)[cost_col].mean()
        
        vehicle_stats = vehicle_stats.sort_values('trips', ascending=False)
        results['fleet_utilization'] = vehicle_stats
        
        # Identify underutilized vehicles (bottom 25%)
        if len(vehicle_stats) >= 4:
            utilization_threshold = vehicle_stats['trips'].quantile(0.25)
            underutilized = vehicle_stats[vehicle_stats['trips'] < utilization_threshold]
            results['underutilized_vehicles'] = underutilized.index.tolist()
            
            # Visualization: Fleet utilization
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
                
                # Left: Trip distribution
                top_vehicles = vehicle_stats.head(15)
                colors1 = plt.cm.Greens(np.linspace(0.4, 0.9, len(top_vehicles)))
                bars1 = ax1.bar(range(len(top_vehicles)), top_vehicles['trips'], color=colors1, edgecolor='black')
                ax1.set_xticks(range(len(top_vehicles)))
                ax1.set_xticklabels(top_vehicles.index, rotation=45, ha='right', fontsize=9)
                ax1.set_ylabel('Number of Trips', fontsize=12, fontweight='bold')
                ax1.set_title('Fleet Utilization (Top 15 Vehicles)', fontsize=14, fontweight='bold')
                ax1.grid(axis='y', alpha=0.3)
                
                # Add utilization threshold line
                ax1.axhline(utilization_threshold, color='red', linestyle='--', 
                           linewidth=2, label=f'Low Utilization Threshold ({int(utilization_threshold)} trips)')
                ax1.legend()
                
                # Right: Cost analysis (if available)
                if cost_col:
                    top_cost_vehicles = vehicle_stats.nlargest(15, 'total_cost')
                    colors2 = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_cost_vehicles)))
                    bars2 = ax2.bar(range(len(top_cost_vehicles)), 
                                   top_cost_vehicles['total_cost'], color=colors2, edgecolor='black')
                    ax2.set_xticks(range(len(top_cost_vehicles)))
                    ax2.set_xticklabels(top_cost_vehicles.index, rotation=45, ha='right', fontsize=9)
                    ax2.set_ylabel('Total Cost (€)', fontsize=12, fontweight='bold')
                    ax2.set_title('Fleet Costs (Top 15 Vehicles)', fontsize=14, fontweight='bold')
                    ax2.grid(axis='y', alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, 'Cost data not available', 
                            ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                    ax2.set_title('Fleet Costs', fontsize=14, fontweight='bold')
                
                plt.tight_layout()
                save_plot('03_fleet_management.png')
                plt.show()
            except Exception as e:
                print(f"   ⚠️ Could not generate fleet visualization: {e}")
        
        print(f"   ✓ Analyzed {len(vehicle_stats)} vehicles in fleet")
    else:
        print("   ⚠️ Skipping fleet management analysis (vehicle column not available)")
    
    # --- ANALYSIS D: Inventory Management ---
    print("\n📊 D. Analyzing Inventory Management...")
    
    if volume_col and location_col:
        # Inventory levels by location
        inventory_by_location = df.groupby(location_col)[volume_col].agg(['sum', 'mean', 'count'])
        inventory_by_location.columns = ['total_volume', 'avg_shipment_volume', 'shipment_count']
        inventory_by_location = inventory_by_location.sort_values('total_volume', ascending=False)
        inventory_by_location.index.name = 'Location'
        results['inventory_by_location'] = inventory_by_location
        
        # Calculate inventory turnover proxy (shipment count / avg volume)
        inventory_by_location['turnover_proxy'] = (
            inventory_by_location['shipment_count'] / 
            inventory_by_location['avg_shipment_volume'].replace(0, 1)
        )
        results['inventory_turnover'] = inventory_by_location[['turnover_proxy']].sort_values(
            'turnover_proxy', ascending=False
        )
        
        # Visualization: Inventory distribution
        try:
            plt.figure(figsize=(12, 8))
            colors = plt.cm.Set3(range(len(inventory_by_location)))
            explode = [0.05] * len(inventory_by_location)
            
            wedges, texts, autotexts = plt.pie(inventory_by_location['total_volume'], 
                                               labels=inventory_by_location.index,
                                               autopct='%1.1f%%',
                                               startangle=90,
                                               colors=colors,
                                               explode=explode,
                                               shadow=True,
                                               textprops={'fontsize': 11, 'fontweight': 'bold'})
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(12)
                autotext.set_fontweight('bold')
            
            plt.title('Inventory Distribution by Location', fontsize=15, fontweight='bold', pad=20)
            plt.axis('equal')
            plt.tight_layout()
            save_plot('04_inventory_distribution.png')
            plt.show()
        except Exception as e:
            print(f"   ⚠️ Could not generate inventory visualization: {e}")
        
        print(f"   ✓ Analyzed inventory across {len(inventory_by_location)} locations")
    else:
        print("   ⚠️ Skipping inventory management analysis (volume or location column not available)")
    
    # --- ANALYSIS E: Cost Analysis ---
    print("\n💰 E. Analyzing Logistics Costs...")
    
    if cost_col:
        # Overall cost statistics
        results['cost_overview'] = {
            'total_logistics_cost': float(round(df[cost_col].sum(), 2)),
            'avg_cost_per_shipment': float(round(df[cost_col].mean(), 2)),
            'median_cost': float(round(df[cost_col].median(), 2)),
            'cost_std_dev': float(round(df[cost_col].std(), 2))
        }
        
        # Cost breakdown by category (route, vehicle, or location)
        cost_breakdown_col = route_col or vehicle_col or location_col
        
        if cost_breakdown_col:
            cost_breakdown = df.groupby(cost_breakdown_col)[cost_col].agg(['sum', 'mean', 'count'])
            cost_breakdown.columns = ['total_cost', 'avg_cost', 'shipment_count']
            cost_breakdown = cost_breakdown.sort_values('total_cost', ascending=False)
            cost_breakdown.index.name = 'Category'
            results['cost_breakdown'] = cost_breakdown
            
            # Identify cost outliers (top 20%)
            cost_threshold = cost_breakdown['avg_cost'].quantile(0.80)
            high_cost_categories = cost_breakdown[cost_breakdown['avg_cost'] > cost_threshold]
            results['high_cost_categories'] = high_cost_categories.index.tolist()
            
            # Visualization: Cost analysis
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
                
                # Left: Total cost by category (top 10)
                top_cost = cost_breakdown.head(10)
                colors1 = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_cost)))
                bars1 = ax1.bar(range(len(top_cost)), top_cost['total_cost']/1000, 
                               color=colors1, edgecolor='black', linewidth=1.5)
                ax1.set_xticks(range(len(top_cost)))
                ax1.set_xticklabels(top_cost.index, rotation=45, ha='right', fontsize=10)
                ax1.set_ylabel('Total Cost (€K)', fontsize=12, fontweight='bold')
                ax1.set_title('Top 10 Cost Centers', fontsize=14, fontweight='bold')
                ax1.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for i, bar in enumerate(bars1):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'€{height:.1f}K', ha='center', va='bottom', fontweight='bold', fontsize=9)
                
                # Right: Average cost distribution (top 10)
                top_avg_cost = cost_breakdown.nlargest(10, 'avg_cost')
                colors2 = plt.cm.Oranges(np.linspace(0.4, 0.9, len(top_avg_cost)))
                bars2 = ax2.bar(range(len(top_avg_cost)), top_avg_cost['avg_cost'], 
                               color=colors2, edgecolor='black', linewidth=1.5)
                ax2.set_xticks(range(len(top_avg_cost)))
                ax2.set_xticklabels(top_avg_cost.index, rotation=45, ha='right', fontsize=10)
                ax2.set_ylabel('Average Cost per Shipment (€)', fontsize=12, fontweight='bold')
                ax2.set_title('Top 10 Highest Cost per Shipment', fontsize=14, fontweight='bold')
                ax2.grid(axis='y', alpha=0.3)
                ax2.axhline(cost_threshold, color='red', linestyle='--', linewidth=2,
                           label=f'High Cost Threshold (€{cost_threshold:.0f})')
                ax2.legend()
                
                # Add value labels
                for i, bar in enumerate(bars2):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'€{height:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
                
                plt.tight_layout()
                save_plot('05_cost_analysis.png')
                plt.show()
            except Exception as e:
                print(f"   ⚠️ Could not generate cost visualization: {e}")
        
        print(f"   ✓ Total logistics cost: €{results['cost_overview']['total_logistics_cost']:,.2f}")
    else:
        print("   ⚠️ Skipping cost analysis (cost column not available)")
    
    # --- ANALYSIS F: Productivity & Resource Efficiency ---
    print("\n⚡ F. Analyzing Productivity & Efficiency...")
    
    if volume_col and time_col:
        # Calculate productivity metric (volume per unit time)
        df['productivity'] = df[volume_col] / df[time_col].replace(0, 1)
        
        productivity_stats = {
            'avg_productivity': float(round(df['productivity'].mean(), 2)),
            'median_productivity': float(round(df['productivity'].median(), 2)),
            'productivity_std': float(round(df['productivity'].std(), 2))
        }
        results['productivity_overview'] = productivity_stats
        
        # Productivity by category
        productivity_col = vehicle_col or route_col or location_col
        if productivity_col:
            productivity_by_cat = df.groupby(productivity_col)['productivity'].agg(['mean', 'median', 'count'])
            productivity_by_cat.columns = ['avg_productivity', 'median_productivity', 'operation_count']
            productivity_by_cat = productivity_by_cat.sort_values('avg_productivity', ascending=False)
            results['productivity_by_category'] = productivity_by_cat
            
            # Identify low performers (bottom 25%)
            if len(productivity_by_cat) >= 4:
                productivity_threshold = productivity_by_cat['avg_productivity'].quantile(0.25)
                low_performers = productivity_by_cat[
                    productivity_by_cat['avg_productivity'] < productivity_threshold
                ]
                results['low_productivity_categories'] = low_performers.index.tolist()
        
        print(f"   ✓ Average productivity: {productivity_stats['avg_productivity']:.2f} units/hour")
    else:
        print("   ⚠️ Skipping productivity analysis (volume or time column not available)")
    
    # --- ANALYSIS G: Supply Chain Visibility & Tracking ---
    print("\n👁️ G. Analyzing Supply Chain Visibility...")
    
    if status_col:
        # Status distribution analysis
        status_distribution = df[status_col].value_counts()
        status_pct = (status_distribution / len(df) * 100).round(2)
        results['shipment_status_distribution'] = status_pct.to_dict()
        
        # Identify delayed or problematic statuses
        problem_keywords = ['delay', 'late', 'stuck', 'issue', 'problem', 'failed']
        problem_statuses = [
            status for status in status_distribution.index 
            if any(keyword in str(status).lower() for keyword in problem_keywords)
        ]
        
        if problem_statuses:
            problem_count = status_distribution[problem_statuses].sum()
            results['problematic_shipments'] = {
                'count': int(problem_count),
                'percentage': float(round(problem_count / len(df) * 100, 2)),
                'status_types': problem_statuses
            }
        
        # Visualization: Status distribution
        try:
            plt.figure(figsize=(12, 7))
            colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(status_distribution)))
            bars = plt.bar(range(len(status_distribution)), status_distribution.values, 
                          color=colors, edgecolor='black', linewidth=1.5)
            plt.xticks(range(len(status_distribution)), status_distribution.index, 
                      rotation=45, ha='right', fontsize=11)
            plt.ylabel('Number of Shipments', fontsize=12, fontweight='bold')
            plt.title('Supply Chain Visibility: Shipment Status Distribution', 
                     fontsize=14, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            
            # Add percentage labels
            for i, bar in enumerate(bars):
                height = bar.get_height()
                pct = status_pct.iloc[i]
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}\n({pct}%)', ha='center', va='bottom', 
                        fontweight='bold', fontsize=9)
            
            plt.tight_layout()
            save_plot('06_supply_chain_visibility.png')
            plt.show()
        except Exception as e:
            print(f"   ⚠️ Could not generate visibility visualization: {e}")
        
        print(f"   ✓ Tracking {len(status_distribution)} different status types")
    else:
        print("   ⚠️ Skipping visibility analysis (status column not available)")
    
    # --- ANALYSIS H: Reverse Logistics (Returns & Recycling) ---
    print("\n🔄 H. Analyzing Reverse Logistics...")
    
    if status_col:
        # Identify return-related statuses
        return_keywords = ['return', 'reverse', 'refund', 'reject', 'recycle']
        return_statuses = [
            status for status in df[status_col].unique() 
            if any(keyword in str(status).lower() for keyword in return_keywords)
        ]
        
        if return_statuses:
            returns_df = df[df[status_col].isin(return_statuses)]
            
            results['reverse_logistics'] = {
                'total_returns': int(len(returns_df)),
                'return_rate': float(round(len(returns_df) / len(df) * 100, 2)),
                'return_types': returns_df[status_col].value_counts().to_dict()
            }
            
            if cost_col:
                results['reverse_logistics']['total_return_cost'] = float(
                    round(returns_df[cost_col].sum(), 2)
                )
                results['reverse_logistics']['avg_return_cost'] = float(
                    round(returns_df[cost_col].mean(), 2)
                )
            
            print(f"   ✓ Return rate: {results['reverse_logistics']['return_rate']:.2f}%")
        else:
            print("   ℹ️ No return-related statuses found in data")
    else:
        print("   ⚠️ Skipping reverse logistics analysis (status column not available)")
    
    # --- ANALYSIS I: Route Optimization Analysis ---
    print("\n🗺️ I. Analyzing Route Optimization Opportunities...")
    
    if route_col and distance_col and time_col:
        # Calculate route efficiency (distance/time ratio)
        route_efficiency = df.groupby(route_col).agg({
            distance_col: 'mean',
            time_col: 'mean',
            volume_col: 'sum' if volume_col else 'count'
        })
        
        route_efficiency['speed'] = route_efficiency[distance_col] / route_efficiency[time_col].replace(0, 1)
        route_efficiency['volume_per_trip'] = route_efficiency[volume_col] / route_efficiency[time_col].replace(0, 1)
        route_efficiency = route_efficiency.sort_values('speed', ascending=False)
        
        results['route_efficiency_analysis'] = route_efficiency
        
        # Identify routes for optimization (slow routes with high volume)
        if len(route_efficiency) > 0:
            slow_threshold = route_efficiency['speed'].quantile(0.25)
            high_volume_threshold = route_efficiency[volume_col].quantile(0.75)
            
            optimization_candidates = route_efficiency[
                (route_efficiency['speed'] < slow_threshold) & 
                (route_efficiency[volume_col] > high_volume_threshold)
            ]
            
            if len(optimization_candidates) > 0:
                results['routes_for_optimization'] = optimization_candidates.index.tolist()
                print(f"   ✓ Identified {len(optimization_candidates)} routes for optimization")
            else:
                print("   ℹ️ No critical optimization opportunities identified")
    else:
        print("   ⚠️ Skipping route optimization analysis (route, distance, or time column not available)")
    
    # --- ANALYSIS J: SLA Compliance & On-Time Delivery ---
    print("\n⏱️ J. Analyzing SLA Compliance...")
    
    if sla_col:
        # SLA compliance rate
        sla_compliance = df[sla_col].value_counts(normalize=True) * 100
        results['sla_compliance_rate'] = sla_compliance.to_dict()
        
        # Overall on-time delivery rate (assuming binary or categorical)
        if df[sla_col].dtype in ['bool', 'int64']:
            on_time_rate = (df[sla_col].sum() / len(df) * 100)
            results['on_time_delivery_rate'] = float(round(on_time_rate, 2))
            print(f"   ✓ On-time delivery rate: {on_time_rate:.2f}%")
        else:
            # Categorical - look for positive indicators
            positive_keywords = ['on time', 'ontime', 'delivered', 'success', 'complete']
            on_time_statuses = [
                status for status in df[sla_col].unique() 
                if any(keyword in str(status).lower() for keyword in positive_keywords)
            ]
            if on_time_statuses:
                on_time_count = df[df[sla_col].isin(on_time_statuses)].shape[0]
                on_time_rate = (on_time_count / len(df) * 100)
                results['on_time_delivery_rate'] = float(round(on_time_rate, 2))
                print(f"   ✓ On-time delivery rate: {on_time_rate:.2f}%")
        
        # SLA compliance by route/vehicle
        sla_breakdown_col = route_col or vehicle_col
        if sla_breakdown_col and df[sla_col].dtype in ['bool', 'int64']:
            sla_by_category = df.groupby(sla_breakdown_col)[sla_col].agg(['sum', 'count'])
            sla_by_category['compliance_rate'] = (sla_by_category['sum'] / sla_by_category['count'] * 100).round(2)
            sla_by_category = sla_by_category.sort_values('compliance_rate', ascending=False)
            results['sla_by_category'] = sla_by_category
            
            # Visualization: SLA compliance
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
                
                # Left: Overall SLA distribution
                if isinstance(sla_compliance, pd.Series) and len(sla_compliance) <= 10:
                    colors1 = ['#2ecc71' if 'on' in str(idx).lower() or 'yes' in str(idx).lower() 
                              else '#e74c3c' for idx in sla_compliance.index]
                    bars1 = ax1.bar(range(len(sla_compliance)), sla_compliance.values, color=colors1)
                    ax1.set_xticks(range(len(sla_compliance)))
                    ax1.set_xticklabels(sla_compliance.index, rotation=45, ha='right')
                    ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
                    ax1.set_title('SLA Compliance Distribution', fontsize=14, fontweight='bold')
                    ax1.grid(axis='y', alpha=0.3)
                    
                    for i, bar in enumerate(bars1):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                # Right: Compliance by category (top 10)
                top_categories = sla_by_category.head(10)
                colors2 = plt.cm.RdYlGn(top_categories['compliance_rate'] / 100)
                bars2 = ax2.barh(range(len(top_categories)), top_categories['compliance_rate'], color=colors2)
                ax2.set_yticks(range(len(top_categories)))
                ax2.set_yticklabels(top_categories.index, fontsize=10)
                ax2.set_xlabel('Compliance Rate (%)', fontsize=12, fontweight='bold')
                ax2.set_title('SLA Compliance by Category (Top 10)', fontsize=14, fontweight='bold')
                ax2.grid(axis='x', alpha=0.3)
                ax2.axvline(80, color='orange', linestyle='--', linewidth=2, label='Target (80%)')
                ax2.legend()
                
                # Add value labels
                for i, bar in enumerate(bars2):
                    width = bar.get_width()
                    ax2.text(width + 1, bar.get_y() + bar.get_height()/2.,
                            f'{width:.1f}%', ha='left', va='center', fontweight='bold')
                
                plt.tight_layout()
                save_plot('07_sla_compliance.png')
                plt.show()
            except Exception as e:
                print(f"   ⚠️ Could not generate SLA visualization: {e}")
    else:
        print("   ⚠️ Skipping SLA analysis (SLA/on-time delivery column not available)")
    
    # --- ANALYSIS K: Logistics Forecasting ---
    print("\n🔮 K. Generating Logistics Forecasts...")
    
    # Identify variables for forecasting
    var_types = identify_variables(df)
    target_info = identify_target_variable(df, user_objective)
    y_col = target_info.get('target_variable')
    
    # Default to volume or cost if no target specified
    if not y_col or y_col not in df.columns:
        if volume_col:
            y_col = volume_col
        elif cost_col:
            y_col = cost_col
    
    if y_col and y_col in df.columns and np.issubdtype(df[y_col].dtype, np.number):
        temp_df = df.dropna(subset=[y_col]).copy()
        
        if len(temp_df) > 5:
            date_cols = var_types.get('datetime', [])
            
            if date_cols:
                # Time-based forecasting
                date_col = date_cols[0]
                temp_df[date_col] = pd.to_datetime(temp_df[date_col])
                temp_df = temp_df.sort_values(date_col)
                
                # Aggregate by day/week depending on data density
                temp_df['period'] = temp_df[date_col].dt.to_period('D')
                period_data = temp_df.groupby('period')[y_col].sum().reset_index()
                period_data['time_index'] = range(len(period_data))
                
                X = period_data[['time_index']].values
                y = period_data[y_col].values
            else:
                # Simple trend-based forecast
                temp_df['time_index'] = np.arange(len(temp_df))
                X = temp_df[['time_index']].values
                y = temp_df[y_col].values
            
            # Fit linear regression model
            model = LinearRegression()
            model.fit(X, y)
            
            # Forecast next 7 periods
            future_steps = np.array([[len(X) + i] for i in range(1, 8)])
            predictions = model.predict(future_steps)
            r2_score = model.score(X, y)
            
            results['logistics_forecast'] = {
                'target_metric': y_col,
                'forecast_next_7_periods': [float(round(p, 2)) for p in predictions],
                'trend': 'Increasing' if predictions[-1] > predictions[0] else 'Decreasing',
                'r2_score': float(round(r2_score, 3)),
                'trend_coefficient': float(round(model.coef_[0], 4)),
                'interpretation': f"Expected {'increase' if model.coef_[0] > 0 else 'decrease'} of {abs(model.coef_[0]):.2f} per period"
            }
            
            # Visualization: Forecast
            try:
                plt.figure(figsize=(14, 7))
                
                # Plot actual data
                plt.plot(range(len(y)), y, 'o-', label='Actual Data', 
                        linewidth=3, markersize=8, color='#3498db', alpha=0.8)
                
                # Plot forecast
                forecast_x = range(len(y), len(y) + 7)
                plt.plot(forecast_x, predictions, 's--', 
                        label='7-Period Forecast', color='#e74c3c', linewidth=3, markersize=10, alpha=0.8)
                
                # Add forecast area
                plt.axvspan(len(y) - 0.5, len(y) + 6.5, alpha=0.1, color='orange', label='Forecast Period')
                
                # Add value labels on forecast points
                for i, (x, pred) in enumerate(zip(forecast_x, predictions)):
                    if i % 2 == 0:  # Show every other label to avoid crowding
                        plt.text(x, pred + (y.max() - y.min()) * 0.03, f'{pred:.0f}',
                                ha='center', va='bottom', fontweight='bold', fontsize=10,
                                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                
                # Styling
                plt.xlabel('Time Period', fontsize=13, fontweight='bold')
                plt.ylabel(y_col.replace('_', ' ').title(), fontsize=13, fontweight='bold')
                plt.title(f'Logistics Forecast: {y_col.replace("_", " ").title()}\n(R² = {r2_score:.3f}, Trend: {results["logistics_forecast"]["trend"]})', 
                         fontsize=15, fontweight='bold', pad=20)
                plt.legend(loc='best', fontsize=12, framealpha=0.9)
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.tight_layout()
                save_plot('08_logistics_forecast.png')
                plt.show()
            except Exception as e:
                print(f"   ⚠️ Could not generate forecast visualization: {e}")
            
            print(f"   ✓ Forecast generated for {y_col} (R² = {r2_score:.3f})")
        else:
            print("   ⚠️ Insufficient data for forecasting (need at least 6 data points)")
    else:
        print("   ⚠️ Skipping forecasting (no suitable numeric column found)")
    
    # --- ANALYSIS L: Clustering Analysis (Optional) ---
    print("\n🔍 L. Performing Logistics Segmentation...")
    
    # Select features for clustering
    quantitative_features = []
    qualitative_features = []
    
    for col in [time_col, volume_col, cost_col, distance_col]:
        if col and col in df.columns and np.issubdtype(df[col].dtype, np.number):
            quantitative_features.append(col)
    
    for col in [route_col, vehicle_col, location_col, status_col]:
        if col and col in df.columns and not np.issubdtype(df[col].dtype, np.number):
            qualitative_features.append(col)
    
    # K-Means for quantitative data
    if len(quantitative_features) >= 2:
        kmeans_data = df[quantitative_features].dropna()
        
        if len(kmeans_data) >= 10:
            print(f"   → K-Means clustering on: {quantitative_features}")
            try:
                kmeans_profiles = auto_kmeans(kmeans_data)
                results['logistics_segments_kmeans'] = kmeans_profiles
                print(f"   ✓ K-Means completed: {len(kmeans_profiles)} clusters identified")
            except Exception as e:
                print(f"   ✗ K-Means failed: {e}")
    
    # K-Medoids for qualitative data
    if len(qualitative_features) >= 2:
        kmedoids_df = df[qualitative_features].copy()
        
        # Encode categorical columns
        for col in qualitative_features:
            if kmedoids_df[col].dtype == 'object' or kmedoids_df[col].dtype.name == 'category':
                kmedoids_df[col] = kmedoids_df[col].astype('category').cat.codes
        
        kmedoids_data = kmedoids_df.dropna()
        
        if len(kmedoids_data) >= 10:
            print(f"   → K-Medoids clustering on: {qualitative_features}")
            try:
                kmedoids_profiles = auto_kmedoids(kmedoids_data)
                results['logistics_segments_kmedoids'] = kmedoids_profiles
                print(f"   ✓ K-Medoids completed: {len(kmedoids_profiles)} clusters identified")
            except Exception as e:
                print(f"   ✗ K-Medoids failed: {e}")
    
    if len(quantitative_features) < 2 and len(qualitative_features) < 2:
        print("   ⚠️ Insufficient features for clustering analysis")
    
    print("\n" + "="*60)
    print("✅ Logistics Analytics completed successfully!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    """
    Test the logistics analytics function with sample data
    """
    print("\n" + "="*60)
    print("LOGISTICS ANALYTICS TEST MODE")
    print("="*60 + "\n")
    
    # Check if custom test data exists
    file_path = os.path.join("data", "logistics_test.csv")
    
    if os.path.exists(file_path):
        print(f"📂 Loading test data from: {file_path}")
        df_test = pd.read_csv(file_path)
        print(f"✅ Dataset loaded: {len(df_test)} rows, {len(df_test.columns)} columns\n")
        
        # Define test mapping (adjust based on your actual column names)
        test_mapping = {
            'route_identifier': 'Route_ID',
            'vehicle_identifier': 'Vehicle_ID',
            'location_identifier': 'Warehouse_ID',
            'time_metric': 'Delivery_Time_Hours',
            'volume_metric': 'Shipment_Volume',
            'cost_metric': 'Transportation_Cost',
            'status_indicator': 'Delivery_Status',
            'distance_metric': 'Distance_KM',
            'sla_metric': 'On_Time_Delivery'
        }
        
        user_goal = "Optimize logistics operations and reduce costs"
        
    else:
        print("📂 Test file not found. Creating sample data...\n")
        
        # Create comprehensive sample logistics dataset
        np.random.seed(42)
        n_shipments = 300
        
        routes = [f'Route_{i}' for i in range(1, 11)]
        vehicles = [f'Vehicle_{i}' for i in range(1, 21)]
        warehouses = ['Warehouse_A', 'Warehouse_B', 'Warehouse_C', 'Warehouse_D']
        statuses = ['Delivered', 'In Transit', 'Processing', 'Delayed', 'Returned']
        
        data = {
            'Shipment_ID': range(1, n_shipments + 1),
            'Route_ID': np.random.choice(routes, n_shipments),
            'Vehicle_ID': np.random.choice(vehicles, n_shipments),
            'Warehouse_ID': np.random.choice(warehouses, n_shipments),
            'Delivery_Time_Hours': np.random.gamma(shape=2, scale=3, size=n_shipments).round(1),
            'Shipment_Volume': np.random.randint(50, 1000, n_shipments),
            'Transportation_Cost': np.random.gamma(shape=5, scale=50, size=n_shipments).round(2),
            'Distance_KM': np.random.randint(10, 500, n_shipments),
            'Delivery_Status': np.random.choice(statuses, n_shipments, p=[0.70, 0.15, 0.08, 0.05, 0.02]),
            'On_Time_Delivery': np.random.choice([1, 0], n_shipments, p=[0.85, 0.15]),
            'Shipment_Date': pd.date_range(end='2024-12-31', periods=n_shipments, freq='D')
        }
        
        df_test = pd.DataFrame(data)
        print(f"✅ Sample dataset created: {len(df_test)} rows, {len(df_test.columns)} columns\n")
        
        # Mapping for sample data
        test_mapping = {
            'route_identifier': 'Route_ID',
            'vehicle_identifier': 'Vehicle_ID',
            'location_identifier': 'Warehouse_ID',
            'time_metric': 'Delivery_Time_Hours',
            'volume_metric': 'Shipment_Volume',
            'cost_metric': 'Transportation_Cost',
            'status_indicator': 'Delivery_Status',
            'datetime_column': 'Shipment_Date',
            'distance_metric': 'Distance_KM',
            'sla_metric': 'On_Time_Delivery'
        }
        
        user_goal = "Analyze logistics efficiency and forecast future demand"
    
    # Run the analytics
    try:
        print(f"🎯 Analysis Objective: {user_goal}\n")
        print("="*60 + "\n")
        
        results = analytics_logistics(df_test, mapping=test_mapping, 
                                     user_objective=user_goal,
                                     save_plots=True,  # Enable plot saving for testing
                                     output_dir="test_logistics_plots")
        
        # Display results summary
        print("\n" + "="*60)
        print("📋 RESULTS SUMMARY")
        print("="*60 + "\n")
        
        for key, value in results.items():
            print(f"\n{'─'*60}")
            print(f"📊 {key.upper().replace('_', ' ')}")
            print(f"{'─'*60}")
            
            if isinstance(value, (pd.DataFrame, pd.Series)):
                print(value.head(10))
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, (pd.DataFrame, pd.Series)):
                        print(f"\n{k}:")
                        print(v)
                    else:
                        print(f"  • {k}: {v}")
            else:
                print(value)
        
        print("\n" + "="*60)
        print("✅ Test completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()







