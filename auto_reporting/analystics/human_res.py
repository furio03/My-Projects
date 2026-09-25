import pandas as pd
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy import stats

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from LLM.llm_output import identify_target_variable, needed_variables
from data_manipulation.read_data import identify_variables
from statistics_functions.clustering import auto_kmeans, auto_kmedoids

def analytics_hr(df: pd.DataFrame, mapping: dict = None, user_objective: str = None, save_plots: bool = False, output_dir: str = "domain_plots"):
    """
    Perform analytics for the HR domain.

    Analyses performed:
    - A: Talent Acquisition & Recruiting - Analyzes recruitment sources, time-to-hire, cost-per-hire
    - B: Performance Management (9-Box Grid) - Maps talent using performance vs potential matrix
    - C: Retention & Turnover Analysis - Calculates turnover rates and identifies flight risk factors
    - D: Employee Engagement & Wellbeing - Measures satisfaction, absenteeism, and promotion rates
    - E: Training & Development (L&D) - Evaluates training effectiveness and skills gap impact
    - F: Workforce Planning (Demographics) - Analyzes workforce composition and future needs
    - G: Productivity & Costs - Evaluates labor costs and productivity metrics by department
    - H: Employee Segmentation - K-Means for quantitative metrics, K-Medoids for mixed/categorical data
    - I: Compensation Analysis - Analyzes salary equity and competitiveness
    - J: Attrition Risk Modeling - Identifies high-risk employees for retention strategies
    - K: HR Forecasting - Predicts future HR metrics (headcount, turnover, costs)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input HR dataset
    mapping : dict, optional
        Column mapping for HR roles
    user_objective : str, optional
        User's analysis objective
    save_plots : bool, default=False
        Whether to save plots to disk for report generation
    output_dir : str, default="hr_analysis_plots"
        Directory to save plots if save_plots=True
    """
    print("Running analysis for HR domain")
    
    # Create output directory for plots if needed
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Plots will be saved to: {output_dir}/")
    
    # Helper function to save plots
    def save_plot(filename):
        if save_plots:
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"   ✓ Plot saved: {filename}")
    
    # 1. Define HR-specific roles for variable mapping
    hr_roles = [
        "acquisition_metric",    # Source, Channel, or Time-to-hire
        "performance_status",    # Performance Rating (1-5 scale)
        "loyalty_metric",        # Tenure, Retention Status, or Flight Risk
        "engagement_volume",     # Satisfaction Score, Engagement Survey
        "conversion_status",     # Training Completion, Certification Status
        "monetary_value",        # Salary, Total Compensation
        "segment_col",           # Department, Business Unit
        "productivity_metric",   # Performance Score, Projects Completed
        "potential_metric"       # Potential Score for 9-box grid
    ]
    
    if mapping is None:
        mapping = needed_variables(df, hr_roles)

    # 2. Helper function to extract single column with fallback
    def get_single_col(key):
        val = mapping.get(key)
        if not val: 
            return None
        if isinstance(val, list):
            val = val[0] if len(val) > 0 else None
        if isinstance(val, str) and "," in val:
            val = val.split(",")[0].strip()
        return val if val in df.columns else None

    # 3. Extract clean variable names with fallback warnings
    recruiting_col  = get_single_col('acquisition_metric')
    performance_col = get_single_col('performance_status')
    retention_col   = get_single_col('loyalty_metric')
    engagement_col  = get_single_col('engagement_volume')
    training_col    = get_single_col('conversion_status')
    salary_col      = get_single_col('monetary_value')
    dept_col        = get_single_col('segment_col')
    productivity_col = get_single_col('productivity_metric')
    potential_col   = get_single_col('potential_metric')
    
    # Fallback warnings for missing critical columns
    missing_cols = []
    if not salary_col:
        missing_cols.append('monetary_value (Salary)')
    if not dept_col:
        missing_cols.append('segment_col (Department)')
    if not performance_col:
        missing_cols.append('performance_status (Performance Rating)')
    
    if missing_cols:
        print(f"\n⚠️  WARNING: Missing columns detected:")
        for col in missing_cols:
            print(f"   - {col}")
        print("   Some analyses will be skipped.\n")

    results = {}

    # --- ANALYSIS A: Talent Acquisition & Recruiting ---
    if recruiting_col:
        print(f"Analyzing recruitment sources using column: '{recruiting_col}'")
        
        # Debug: Show sample values
        print(f"   Sample values in {recruiting_col}: {df[recruiting_col].head(3).tolist()}")
        
        # Ensure we're grouping by the recruitment source column
        recruitment_stats = df.groupby(recruiting_col, dropna=False).size().to_frame('hiring_count')
        recruitment_stats = recruitment_stats.sort_values('hiring_count', ascending=False)
        recruitment_stats.index.name = 'Recruitment_Source'
        
        results['recruiting_efficiency'] = recruitment_stats
        
        if salary_col:
            cost_analysis = df.groupby(recruiting_col)[salary_col].agg(['mean', 'count'])
            cost_analysis.columns = ['avg_starting_salary', 'hires']
            results['cost_per_hire_by_source'] = cost_analysis.sort_values('hires', ascending=False)
            
            # Improved Visualization: Recruitment source effectiveness
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # Left chart: Number of hires by source
                colors = plt.cm.Set3(range(len(recruitment_stats)))
                ax1.barh(recruitment_stats.index, recruitment_stats['hiring_count'], color=colors)
                ax1.set_xlabel('Number of Hires', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Recruitment Source', fontsize=12, fontweight='bold')
                ax1.set_title('Hiring Volume by Source', fontsize=14, fontweight='bold')
                ax1.grid(axis='x', alpha=0.3)
                
                # Add value labels on bars
                for i, v in enumerate(recruitment_stats['hiring_count']):
                    ax1.text(v + 0.5, i, str(int(v)), va='center', fontweight='bold')
                
                # Right chart: Average salary by source
                cost_sorted = cost_analysis.sort_values('avg_starting_salary', ascending=False)
                colors2 = plt.cm.Pastel1(range(len(cost_sorted)))
                ax2.barh(cost_sorted.index, cost_sorted['avg_starting_salary'], color=colors2)
                ax2.set_xlabel('Average Starting Salary (€)', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Recruitment Source', fontsize=12, fontweight='bold')
                ax2.set_title('Cost-per-Hire by Source', fontsize=14, fontweight='bold')
                ax2.grid(axis='x', alpha=0.3)
                
                # Add value labels on bars
                for i, v in enumerate(cost_sorted['avg_starting_salary']):
                    ax2.text(v + 500, i, f'€{int(v):,}', va='center', fontweight='bold')
                
                plt.tight_layout()
                save_plot('01_recruitment_effectiveness.png')
                plt.show()
            except Exception as e:
                print(f"   ⚠️ Could not generate recruitment visualization: {e}")
        else:
            print("   ⚠️ Skipping recruitment cost analysis (salary column not available)")
    else:
        print("⚠️ Skipping Talent Acquisition analysis (recruitment source column not available)")

    # --- ANALYSIS B: Performance Management (9-Box Grid) ---
    if performance_col and (potential_col or engagement_col):
        # Use potential if available, otherwise engagement as proxy
        potential_proxy = potential_col if potential_col else engagement_col
        
        print(f"Generating 9-Box Grid: {performance_col} vs {potential_proxy}")
        
        # Create categorization
        perf_median = df[performance_col].median()
        pot_median = df[potential_proxy].median()
        
        def categorize_9box(row):
            perf = row[performance_col]
            pot = row[potential_proxy]
            if perf >= perf_median and pot >= pot_median:
                return 'Stars'
            elif perf >= perf_median and pot < pot_median:
                return 'Core Players'
            elif perf < perf_median and pot >= pot_median:
                return 'Emerging Talent'
            else:
                return 'Needs Development'
        
        df['talent_category'] = df.apply(categorize_9box, axis=1)
        results['talent_matrix_distribution'] = df['talent_category'].value_counts()
        
        # Improved Visualization: 9-Box Grid
        plt.figure(figsize=(12, 9))
        
        # Create color mapping for categories
        category_colors = {
            'Stars': '#2ecc71',  # Green
            'Core Players': '#3498db',  # Blue
            'Emerging Talent': '#f39c12',  # Orange
            'Needs Development': '#e74c3c'  # Red
        }
        
        # Plot each category with different color
        for category in df['talent_category'].unique():
            mask = df['talent_category'] == category
            plt.scatter(df[mask][performance_col], df[mask][potential_proxy], 
                       c=category_colors.get(category, 'gray'),
                       s=150, alpha=0.7, edgecolors='black', linewidth=1.5,
                       label=category)
        
        # Add quadrant lines
        plt.axhline(pot_median, color='black', linestyle='--', linewidth=2.5, alpha=0.8)
        plt.axvline(perf_median, color='black', linestyle='--', linewidth=2.5, alpha=0.8)
        
        # Add quadrant labels
        x_range = df[performance_col].max() - df[performance_col].min()
        y_range = df[potential_proxy].max() - df[potential_proxy].min()
        
        plt.text(perf_median + x_range*0.25, pot_median + y_range*0.35, 'STARS\n(High/High)', 
                ha='center', va='center', fontsize=13, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='#2ecc71', alpha=0.3))
        plt.text(perf_median + x_range*0.25, pot_median - y_range*0.35, 'CORE PLAYERS\n(High/Moderate)', 
                ha='center', va='center', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.3))
        plt.text(perf_median - x_range*0.25, pot_median + y_range*0.35, 'EMERGING TALENT\n(Moderate/High)', 
                ha='center', va='center', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#f39c12', alpha=0.3))
        plt.text(perf_median - x_range*0.25, pot_median - y_range*0.35, 'NEEDS DEVELOPMENT\n(Moderate/Moderate)', 
                ha='center', va='center', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.3))
        
        plt.xlabel('Performance Rating', fontsize=13, fontweight='bold')
        plt.ylabel('Potential Score', fontsize=13, fontweight='bold')
        plt.title('9-Box Talent Matrix: Performance vs Potential', fontsize=15, fontweight='bold', pad=20)
        plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
        plt.grid(True, alpha=0.2, linestyle=':')
        plt.tight_layout()
        plt.show()
        
        if dept_col:
            results['performance_by_dept'] = df.groupby(dept_col)[performance_col].describe()

    # --- ANALYSIS C: Retention & Turnover ---
    if retention_col:
        if df[retention_col].dtype in ['object', 'category', 'bool']:
            # Categorical turnover status
            turnover_rate = df[retention_col].value_counts(normalize=True) * 100
            results['turnover_rate'] = turnover_rate.to_dict()
            
            if dept_col:
                results['turnover_by_department'] = df.groupby(dept_col)[retention_col].apply(
                    lambda x: (x.value_counts(normalize=True) * 100).to_dict()
                )
        else:
            # Numeric tenure
            results['average_tenure'] = {
                'mean_years': round(df[retention_col].mean(), 2),
                'median_years': round(df[retention_col].median(), 2)
            }
            
            if dept_col:
                results['tenure_by_dept'] = df.groupby(dept_col)[retention_col].agg(['mean', 'median'])

    # --- ANALYSIS D: Employee Engagement & Wellbeing ---
    if engagement_col:
        results['engagement_stats'] = {
            'mean_engagement': round(df[engagement_col].mean(), 2),
            'median_engagement': round(df[engagement_col].median(), 2),
            'std_engagement': round(df[engagement_col].std(), 2)
        }
        
        if dept_col:
            engagement_by_dept = df.groupby(dept_col)[engagement_col].agg(['mean', 'median', 'count'])
            engagement_by_dept.columns = ['avg_engagement', 'median_engagement', 'employee_count']
            results['engagement_by_dept'] = engagement_by_dept.sort_values('avg_engagement', ascending=False)
            
            # Visualization: Engagement by department
            plt.figure(figsize=(12, 6))
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(engagement_by_dept)))
            bars = plt.bar(range(len(engagement_by_dept)), engagement_by_dept['avg_engagement'], 
                          color=colors, edgecolor='black', linewidth=1.5)
            plt.xticks(range(len(engagement_by_dept)), engagement_by_dept.index, 
                      rotation=45, ha='right', fontsize=11)
            plt.xlabel('Department', fontsize=12, fontweight='bold')
            plt.ylabel('Average Engagement Score', fontsize=12, fontweight='bold')
            plt.title('Employee Engagement by Department', fontsize=14, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels on top of bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            save_plot('03_engagement_by_department.png')
            plt.show()
        
        # Calculate engagement drivers (correlation without visualization)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 2 and engagement_col in numeric_cols:
            correlation_matrix = df[numeric_cols].corr()
            engagement_corr = correlation_matrix[engagement_col].drop(engagement_col).sort_values(ascending=False)
            results['engagement_drivers'] = engagement_corr.head(5).to_dict()

    # --- ANALYSIS E: Training & Development ---
    if training_col and performance_col:
        training_impact = df.groupby(training_col)[performance_col].agg(['mean', 'median', 'count'])
        training_impact.columns = ['avg_performance', 'median_performance', 'employee_count']
        results['training_impact_on_performance'] = training_impact
        
        # Statistical significance test
        trained = df[df[training_col] == 1][performance_col] if df[training_col].dtype in ['int64', 'bool'] else df[df[training_col].notna()][performance_col]
        untrained = df[df[training_col] == 0][performance_col] if df[training_col].dtype in ['int64', 'bool'] else df[df[training_col].isna()][performance_col]
        
        if len(trained) > 0 and len(untrained) > 0:
            t_stat, p_value = stats.ttest_ind(trained, untrained)
            results['training_statistical_test'] = {
                't_statistic': round(t_stat, 4),
                'p_value': round(p_value, 4),
                'significant': 'Yes' if p_value < 0.05 else 'No',
                'interpretation': f"Training {'significantly' if p_value < 0.05 else 'does not significantly'} impacts performance"
            }

    # --- ANALYSIS F: Workforce Planning ---
    var_types = identify_variables(df)
    results['workforce_overview'] = {
        'total_headcount': len(df),
        'departments': df[dept_col].nunique() if dept_col else "N/A",
        'avg_tenure': round(df[retention_col].mean(), 2) if retention_col and np.issubdtype(df[retention_col].dtype, np.number) else "N/A"
    }
    
    if dept_col:
        dept_distribution = df[dept_col].value_counts()
        results['headcount_by_department'] = dept_distribution.to_dict()
        
        # Improved Visualization: Workforce distribution
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set3(range(len(dept_distribution)))
        explode = [0.05] * len(dept_distribution)  # Slight separation for all slices
        
        wedges, texts, autotexts = plt.pie(dept_distribution.values, 
                                           labels=dept_distribution.index,
                                           autopct='%1.1f%%',
                                           startangle=90,
                                           colors=colors,
                                           explode=explode,
                                           shadow=True,
                                           textprops={'fontsize': 11, 'fontweight': 'bold'})
        
        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')
        
        plt.title('Workforce Distribution by Department', fontsize=15, fontweight='bold', pad=20)
        plt.axis('equal')
        plt.tight_layout()
        save_plot('04_workforce_distribution.png')
        plt.show()

    # --- ANALYSIS G: Productivity & Costs ---
    if salary_col:
        results['compensation_overview'] = {
            'total_payroll': round(df[salary_col].sum(), 2),
            'avg_salary': round(df[salary_col].mean(), 2),
            'median_salary': round(df[salary_col].median(), 2),
            'salary_std': round(df[salary_col].std(), 2)
        }
        
        if dept_col:
            labor_cost_by_dept = df.groupby(dept_col)[salary_col].agg(['sum', 'mean', 'median', 'count'])
            labor_cost_by_dept.columns = ['total_payroll', 'avg_salary', 'median_salary', 'employee_count']
            results['labor_cost_by_dept'] = labor_cost_by_dept.sort_values('total_payroll', ascending=False)
            
            # Improved Visualization: Labor cost distribution
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
            
            # Left chart: Total payroll
            colors1 = plt.cm.Reds(np.linspace(0.4, 0.8, len(labor_cost_by_dept)))
            bars1 = ax1.bar(range(len(labor_cost_by_dept)), 
                           labor_cost_by_dept['total_payroll'], 
                           color=colors1, edgecolor='black', linewidth=1.5)
            ax1.set_xticks(range(len(labor_cost_by_dept)))
            ax1.set_xticklabels(labor_cost_by_dept.index, rotation=45, ha='right', fontsize=11)
            ax1.set_title('Total Payroll by Department', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Department', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Total Payroll (€)', fontsize=12, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, bar in enumerate(bars1):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'€{int(height/1000)}K',
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            # Right chart: Average salary
            colors2 = plt.cm.Greens(np.linspace(0.4, 0.8, len(labor_cost_by_dept)))
            labor_sorted = labor_cost_by_dept.sort_values('avg_salary', ascending=False)
            bars2 = ax2.bar(range(len(labor_sorted)), 
                           labor_sorted['avg_salary'], 
                           color=colors2, edgecolor='black', linewidth=1.5)
            ax2.set_xticks(range(len(labor_sorted)))
            ax2.set_xticklabels(labor_sorted.index, rotation=45, ha='right', fontsize=11)
            ax2.set_title('Average Salary by Department', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Department', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Average Salary (€)', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for i, bar in enumerate(bars2):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'€{int(height):,}',
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            save_plot('05_labor_costs.png')
            plt.show()
    
    if productivity_col and dept_col:
        productivity_by_dept = df.groupby(dept_col)[productivity_col].agg(['mean', 'median', 'sum'])
        productivity_by_dept.columns = ['avg_productivity', 'median_productivity', 'total_output']
        results['productivity_by_dept'] = productivity_by_dept.sort_values('avg_productivity', ascending=False)

    # --- ANALYSIS H: Employee Segmentation (CORRECTED) ---
    print("\nPerforming Employee Segmentation...")
    
    # Separate quantitative and qualitative features
    quantitative_features = []
    qualitative_features = []
    
    # Check each potential clustering column
    for col in [salary_col, performance_col, engagement_col, retention_col, productivity_col]:
        if col and col in df.columns:
            if np.issubdtype(df[col].dtype, np.number):
                quantitative_features.append(col)
            else:
                qualitative_features.append(col)
    
    # Add categorical columns for K-Medoids
    if dept_col and dept_col not in qualitative_features:
        qualitative_features.append(dept_col)
    if recruiting_col and recruiting_col not in qualitative_features:
        qualitative_features.append(recruiting_col)
    
    # K-MEANS: Only for quantitative (continuous) data
    if len(quantitative_features) >= 2:
        kmeans_data = df[quantitative_features].dropna()
        
        if len(kmeans_data) >= 10:
            print(f"   → K-Means clustering on quantitative features: {quantitative_features}")
            try:
                kmeans_profiles = auto_kmeans(kmeans_data)
                results['employee_segments_kmeans'] = kmeans_profiles
                print(f"   ✓ K-Means completed: {len(kmeans_profiles)} clusters identified")
            except Exception as e:
                print(f"   ✗ K-Means failed: {e}")
        else:
            print(f"   ⚠ Insufficient data for K-Means (have {len(kmeans_data)}, need ≥10)")
    else:
        print(f"   ⚠ Not enough quantitative features for K-Means (have {len(quantitative_features)}, need ≥2)")
    
    # K-MEDOIDS: For categorical/mixed data
    if len(qualitative_features) >= 2:
        # Prepare data for K-Medoids (encode categorical variables)
        kmedoids_df = df[qualitative_features].copy()
        
        # Encode categorical columns to numeric
        for col in qualitative_features:
            if kmedoids_df[col].dtype == 'object' or kmedoids_df[col].dtype.name == 'category':
                kmedoids_df[col] = kmedoids_df[col].astype('category').cat.codes
        
        kmedoids_data = kmedoids_df.dropna()
        
        if len(kmedoids_data) >= 10:
            print(f"   → K-Medoids clustering on qualitative features: {qualitative_features}")
            try:
                kmedoids_profiles = auto_kmedoids(kmedoids_data)
                results['employee_segments_kmedoids'] = kmedoids_profiles
                print(f"   ✓ K-Medoids completed: {len(kmedoids_profiles)} clusters identified")
            except Exception as e:
                print(f"   ✗ K-Medoids failed: {e}")
        else:
            print(f"   ⚠ Insufficient data for K-Medoids (have {len(kmedoids_data)}, need ≥10)")
    else:
        print(f"   ⚠ Not enough qualitative features for K-Medoids (have {len(qualitative_features)}, need ≥2)")

    # --- ANALYSIS I: Compensation Analysis ---
    if salary_col and performance_col:
        df['pay_performance_ratio'] = df[salary_col] / (df[performance_col] + 1)
        
        if dept_col:
            pay_equity = df.groupby(dept_col).agg({
                salary_col: ['mean', 'std'],
                performance_col: 'mean',
                'pay_performance_ratio': 'mean'
            })
            pay_equity.columns = ['avg_salary', 'salary_std', 'avg_performance', 'pay_perf_ratio']
            results['compensation_equity'] = pay_equity
        
        salary_quartiles = df[salary_col].quantile([0.25, 0.5, 0.75])
        results['salary_distribution'] = {
            'Q1_25th_percentile': round(salary_quartiles[0.25], 2),
            'Q2_median': round(salary_quartiles[0.5], 2),
            'Q3_75th_percentile': round(salary_quartiles[0.75], 2),
            'IQR': round(salary_quartiles[0.75] - salary_quartiles[0.25], 2)
        }

    # --- ANALYSIS J: Attrition Risk Modeling ---
    risk_factors = []
    
    if engagement_col and np.issubdtype(df[engagement_col].dtype, np.number):
        df['engagement_risk'] = (df[engagement_col].max() - df[engagement_col]) / df[engagement_col].max()
        risk_factors.append('engagement_risk')
    
    if performance_col and np.issubdtype(df[performance_col].dtype, np.number):
        df['performance_risk'] = (df[performance_col].max() - df[performance_col]) / df[performance_col].max()
        risk_factors.append('performance_risk')
    
    if salary_col and np.issubdtype(df[salary_col].dtype, np.number):
        median_salary = df[salary_col].median()
        df['compensation_risk'] = np.where(df[salary_col] < median_salary, 1, 0)
        risk_factors.append('compensation_risk')
    
    if len(risk_factors) >= 2:
        df['attrition_risk_score'] = df[risk_factors].mean(axis=1) * 100
        
        df['risk_category'] = pd.cut(df['attrition_risk_score'], 
                                     bins=[0, 33, 66, 100],
                                     labels=['Low Risk', 'Medium Risk', 'High Risk'])
        
        risk_distribution = df['risk_category'].value_counts()
        results['attrition_risk_distribution'] = risk_distribution.to_dict()
        
        high_risk_threshold = df['attrition_risk_score'].quantile(0.75)
        high_risk_count = len(df[df['attrition_risk_score'] > high_risk_threshold])
        results['high_risk_employees'] = {
            'count': high_risk_count,
            'percentage': round((high_risk_count / len(df)) * 100, 2),
            'threshold_score': round(high_risk_threshold, 2)
        }

    # --- ANALYSIS K: HR Forecasting ---
    target_info = identify_target_variable(df, user_objective)
    y_col = target_info.get('target_variable')

    if not y_col or y_col not in df.columns:
        if salary_col:
            y_col = salary_col
        elif retention_col:
            y_col = retention_col

    if y_col and y_col in df.columns and np.issubdtype(df[y_col].dtype, np.number):
        temp_df = df.dropna(subset=[y_col]).copy()
        
        if len(temp_df) > 5:
            date_cols = var_types.get('datetime', [])
            
            if date_cols:
                date_col = date_cols[0]
                temp_df[date_col] = pd.to_datetime(temp_df[date_col])
                temp_df = temp_df.sort_values(date_col)
                temp_df['month'] = temp_df[date_col].dt.to_period('M')
                monthly_data = temp_df.groupby('month')[y_col].mean().reset_index()
                monthly_data['time_index'] = range(len(monthly_data))
                X = monthly_data[['time_index']].values
                y = monthly_data[y_col].values
            else:
                temp_df['time_index'] = np.arange(len(temp_df))
                X = temp_df[['time_index']].values
                y = temp_df[y_col].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            future_steps = np.array([[len(X)], [len(X)+1], [len(X)+2]])
            predictions = model.predict(future_steps)
            r2_score = model.score(X, y)
            
            results['hr_forecast'] = {
                'target_metric': y_col,
                'forecast_next_3_periods': [float(round(p, 2)) for p in predictions],  # Convert to Python float
                'trend': 'Increasing' if predictions[-1] > predictions[0] else 'Decreasing',
                'r2_score': float(round(r2_score, 3)),  # Convert to Python float
                'trend_coefficient': float(round(model.coef_[0], 4))  # Convert to Python float
            }
            
            # Improved Visualization: Forecast
            plt.figure(figsize=(14, 7))
            
            # Plot actual data
            plt.plot(range(len(y)), y, 'o-', label='Actual Data', 
                    linewidth=3, markersize=8, color='#3498db', alpha=0.8)
            
            # Plot forecast
            forecast_x = range(len(y), len(y) + 3)
            plt.plot(forecast_x, predictions, 's--', 
                    label='Forecast', color='#e74c3c', linewidth=3, markersize=10, alpha=0.8)
            
            # Add forecast area
            plt.axvspan(len(y) - 0.5, len(y) + 2.5, alpha=0.1, color='orange', label='Forecast Period')
            
            # Add value labels on forecast points
            for i, (x, pred) in enumerate(zip(forecast_x, predictions)):
                plt.text(x, pred + (y.max() - y.min()) * 0.03, f'{pred:.1f}',
                        ha='center', va='bottom', fontweight='bold', fontsize=11,
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            # Styling
            plt.xlabel('Time Period', fontsize=13, fontweight='bold')
            plt.ylabel(y_col.replace('_', ' ').title(), fontsize=13, fontweight='bold')
            plt.title(f'HR Forecast: {y_col.replace("_", " ").title()}\n(R² = {r2_score:.3f}, Trend: {results["hr_forecast"]["trend"]})', 
                     fontsize=15, fontweight='bold', pad=20)
            plt.legend(loc='best', fontsize=12, framealpha=0.9)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            save_plot('06_hr_forecast.png')
            plt.show()

    print("HR analysis completed.")
    
    return results


if __name__ == "__main__":
    file_path = os.path.join("data", "hr_test.csv")

    if os.path.exists(file_path):
        df_test = pd.read_csv(file_path)
        print(f"✅ Dataset loaded: {len(df_test)} rows")

        user_goal = "Analyze employee performance and predict retention"

        test_mapping = {
            'monetary_value': 'Salary',
            'performance_status': 'Performance_Rating',
            'engagement_volume': 'Engagement_Score',
            'segment_col': 'Department',
            'conversion_status': 'Training_Completed',
            'loyalty_metric': 'Tenure_Years',
            'acquisition_metric': 'Recruitment_Source',
            'productivity_metric': 'Projects_Completed',
            'potential_metric': 'Potential_Score'
        }

        try:
            results = analytics_hr(df_test, test_mapping, user_objective=user_goal)
            
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
            print(f"❌ Error: {e}")
            
    else:
        print(f"❌ File '{file_path}' not found.")
        print("Creating sample data for testing...")
        
        np.random.seed(42)
        n_employees = 150
        
        departments = ['IT', 'Sales', 'HR', 'Finance', 'Operations', 'Marketing']
        recruitment_sources = ['LinkedIn', 'Referral', 'Job Board', 'Campus', 'Agency']
        
        data = {
            'Employee_ID': range(1, n_employees + 1),
            'Department': np.random.choice(departments, n_employees),
            'Salary': np.random.normal(65000, 15000, n_employees).clip(35000, 120000).round(0),
            'Performance_Rating': np.random.choice([1, 2, 3, 4, 5], n_employees, p=[0.05, 0.15, 0.35, 0.30, 0.15]),
            'Engagement_Score': np.random.normal(75, 15, n_employees).clip(30, 100).round(0),
            'Tenure_Years': np.random.exponential(3, n_employees).clip(0.1, 20).round(1),
            'Training_Completed': np.random.choice([0, 1], n_employees, p=[0.3, 0.7]),
            'Recruitment_Source': np.random.choice(recruitment_sources, n_employees),
            'Projects_Completed': np.random.poisson(8, n_employees),
            'Potential_Score': np.random.normal(70, 20, n_employees).clip(20, 100).round(0)
        }
        
        df_test = pd.DataFrame(data)
        print(f"✅ Sample dataset created: {len(df_test)} rows")
        
        test_mapping = {
            'monetary_value': 'Salary',
            'performance_status': 'Performance_Rating',
            'engagement_volume': 'Engagement_Score',
            'segment_col': 'Department',
            'conversion_status': 'Training_Completed',
            'loyalty_metric': 'Tenure_Years',
            'acquisition_metric': 'Recruitment_Source',
            'productivity_metric': 'Projects_Completed',
            'potential_metric': 'Potential_Score'
        }
        
        user_goal = "Analyze employee performance and identify retention risks"
        
        try:
            results = analytics_hr(df_test, test_mapping, user_objective=user_goal)
            
            print("\n" + "="*40)
            print(f"🚀 TEST RESULTS SUMMARY")
            print("="*40)
            
            for key, value in results.items():
                print(f"\n📊 {key.upper()}:")
                if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                    print(value.head())
                else:
                    print(value)
                    
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()