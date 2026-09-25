import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from LLM.llm_output import identify_target_variable
from data_manipulation.read_data import identify_variables
from statistics_functions.clustering import auto_kmeans


def analytics_surveys(df: pd.DataFrame, mapping: dict = None, user_objective: str = None,
                     save_plots: bool = False, output_dir: str = "domain_plots") -> dict:
    """
    Perform comprehensive analytics on survey data.
    
    Analyses included:
    - A: Frequency & Top-Box Analysis - Response distributions and satisfaction metrics
    - B: Statistical Association - Chi-Square & Cramér's V for variable relationships
    - C: Multiple Correspondence Analysis (MCA) - Respondent segmentation with K-Means
    - D: Key Driver Analysis - Feature importance via Random Forest
    - E: Survey Health Dashboard - Overall satisfaction metrics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input survey dataset
    mapping : dict, optional
        Column mapping (not typically used for surveys)
    user_objective : str, optional
        User's analysis objective to identify target variable
    save_plots : bool, default=False
        Whether to save plots to disk
    output_dir : str, default="surveys_analysis_plots"
        Directory to save plots
    """
    print("Running comprehensive analysis for surveys")
    
    # Create output directory
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Plots will be saved to: {output_dir}/")
    
    def save_plot(filename):
        if save_plots:
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"   ✓ Plot saved: {filename}")
    
    results = {}
    
    # Identify variable types
    var_types = identify_variables(df)
    categorical_cols = var_types.get('categorical', [])
    numeric_cols = var_types.get('numeric', [])
    
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = list(set(categorical_cols + object_cols))
    
    # ANALYSIS A: Frequency & Top-Box
    print("\n📊 A. Calculating Frequency & Top-Box Analysis...")
    
    frequency_results = {}
    top_box_results = {}
    
    for col in categorical_cols[:20]:  # Limit to 20 to avoid too many plots
        if col in df.columns:
            freq_pct = df[col].value_counts(normalize=True) * 100
            frequency_results[col] = freq_pct.round(2).to_dict()
            
            unique_values = df[col].dropna().unique()
            
            try:
                numeric_values = pd.to_numeric(unique_values, errors='coerce')
                numeric_values = numeric_values[~np.isnan(numeric_values)]
                
                if len(numeric_values) >= 3:
                    min_val = int(numeric_values.min())
                    max_val = int(numeric_values.max())
                    
                    if max_val - min_val <= 10 and max_val <= 10:
                        top_2_values = sorted(numeric_values)[-2:]
                        col_numeric = pd.to_numeric(df[col], errors='coerce')
                        top_2_pct = col_numeric.isin(top_2_values).sum() / len(col_numeric.dropna()) * 100
                        
                        bottom_2_values = sorted(numeric_values)[:2]
                        bottom_2_pct = col_numeric.isin(bottom_2_values).sum() / len(col_numeric.dropna()) * 100
                        
                        top_box_results[col] = {
                            'scale_range': f"{min_val}-{max_val}",
                            'top_2_box_pct': round(top_2_pct, 2),
                            'bottom_2_box_pct': round(bottom_2_pct, 2),
                            'top_2_values': [int(v) for v in top_2_values],
                            'sentiment_ratio': round(top_2_pct / bottom_2_pct, 2) if bottom_2_pct > 0 else np.inf
                        }
            except:
                pass
    
    if frequency_results:
        freq_df = pd.DataFrame([
            {'Variable': k, 'Category': cat, 'Percentage': pct}
            for k, v in frequency_results.items()
            for cat, pct in v.items()
        ])
        results['frequency_distribution'] = freq_df
    
    if top_box_results:
        top_box_df = pd.DataFrame(top_box_results).T
        top_box_df.index.name = 'Variable'
        results['top_box_analysis'] = top_box_df.reset_index()
        
        # Visualization: Top-Box Comparison
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
            
            top_box_sorted = top_box_df.sort_values('top_2_box_pct', ascending=False).head(10)
            colors1 = plt.cm.Greens(np.linspace(0.4, 0.9, len(top_box_sorted)))
            bars1 = ax1.barh(range(len(top_box_sorted)), top_box_sorted['top_2_box_pct'],
                            color=colors1, edgecolor='black', linewidth=1.5)
            ax1.set_yticks(range(len(top_box_sorted)))
            ax1.set_yticklabels(top_box_sorted.index, fontsize=11)
            ax1.set_xlabel('Top-2 Box Score (%)', fontsize=12, fontweight='bold')
            ax1.set_title('Customer Satisfaction Scores (Top 10 Questions)', fontsize=14, fontweight='bold')
            ax1.axvline(50, color='orange', linestyle='--', linewidth=2, label='50% Threshold')
            ax1.legend(fontsize=10)
            ax1.grid(axis='x', alpha=0.3)
            
            for i, bar in enumerate(bars1):
                width = bar.get_width()
                ax1.text(width + 1, bar.get_y() + bar.get_height()/2.,
                        f'{width:.1f}%', ha='left', va='center', fontweight='bold')
            
            sentiment_sorted = top_box_df[top_box_df['sentiment_ratio'] != np.inf].sort_values('sentiment_ratio', ascending=False).head(10)
            colors2 = ['#2ecc71' if x > 1.5 else '#f39c12' if x > 0.8 else '#e74c3c' 
                      for x in sentiment_sorted['sentiment_ratio']]
            bars2 = ax2.barh(range(len(sentiment_sorted)), sentiment_sorted['sentiment_ratio'],
                            color=colors2, edgecolor='black', linewidth=1.5)
            ax2.set_yticks(range(len(sentiment_sorted)))
            ax2.set_yticklabels(sentiment_sorted.index, fontsize=11)
            ax2.set_xlabel('Sentiment Ratio (Positive/Negative)', fontsize=12, fontweight='bold')
            ax2.set_title('Response Sentiment Balance (Top 10)', fontsize=14, fontweight='bold')
            ax2.axvline(1, color='black', linestyle='-', linewidth=1)
            ax2.axvline(1.5, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Positive Threshold')
            ax2.legend(fontsize=10)
            ax2.grid(axis='x', alpha=0.3)
            
            for i, bar in enumerate(bars2):
                width = bar.get_width()
                ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                        f'{width:.2f}x', ha='left', va='center', fontweight='bold')
            
            plt.tight_layout()
            save_plot('01_satisfaction_scores.png')
            plt.show()
        except Exception as e:
            print(f"   ⚠️ Could not generate satisfaction visualization: {e}")
        
        print(f"   ✓ Top-Box Analysis completed for {len(top_box_results)} scale variables")
    
    # ANALYSIS B: Statistical Association
    print("\n🔗 B. Calculating Statistical Associations...")
    
    association_results = []
    
    if len(categorical_cols) >= 2:
        for i, col1 in enumerate(categorical_cols[:15]):
            for col2 in categorical_cols[i+1:15]:
                if col1 in df.columns and col2 in df.columns:
                    try:
                        contingency_table = pd.crosstab(df[col1], df[col2])
                        
                        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                            continue
                        
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        
                        n = contingency_table.sum().sum()
                        min_dim = min(contingency_table.shape[0] - 1, contingency_table.shape[1] - 1)
                        
                        if min_dim > 0 and n > 0:
                            cramers_v = np.sqrt(chi2 / (n * min_dim))
                        else:
                            cramers_v = 0
                        
                        if cramers_v >= 0.5:
                            strength = "Strong"
                        elif cramers_v >= 0.3:
                            strength = "Moderate"
                        elif cramers_v >= 0.1:
                            strength = "Weak"
                        else:
                            strength = "Negligible"
                        
                        association_results.append({
                            'Variable_1': col1,
                            'Variable_2': col2,
                            'Chi_Square': round(chi2, 4),
                            'p_value': round(p_value, 6),
                            'Cramers_V': round(cramers_v, 4),
                            'Association_Strength': strength,
                            'Significant': 'Yes' if p_value < 0.05 else 'No'
                        })
                    except:
                        continue
    
    if association_results:
        assoc_df = pd.DataFrame(association_results)
        assoc_df = assoc_df.sort_values(by='Cramers_V', ascending=False).reset_index(drop=True)
        results['statistical_association'] = assoc_df
        
        strong_assoc = assoc_df[assoc_df['Association_Strength'].isin(['Strong', 'Moderate'])]
        if len(strong_assoc) > 0:
            results['strong_associations'] = strong_assoc
            
            # Visualization
            try:
                top_associations = assoc_df.head(15)
                
                plt.figure(figsize=(14, 8))
                
                var_pairs = [f"{row['Variable_1'][:20]}\nvs\n{row['Variable_2'][:20]}" 
                            for _, row in top_associations.iterrows()]
                
                colors_map = {'Strong': '#e74c3c', 'Moderate': '#f39c12', 
                             'Weak': '#3498db', 'Negligible': '#95a5a6'}
                colors = [colors_map.get(row['Association_Strength'], '#95a5a6') 
                         for _, row in top_associations.iterrows()]
                
                bars = plt.barh(range(len(top_associations)), top_associations['Cramers_V'],
                               color=colors, edgecolor='black', linewidth=1.5)
                plt.yticks(range(len(top_associations)), var_pairs, fontsize=9)
                plt.xlabel("Cramér's V (Association Strength)", fontsize=12, fontweight='bold')
                plt.title('Variable Associations (Top 15 Relationships)', fontsize=14, fontweight='bold')
                plt.axvline(0.3, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Moderate')
                plt.axvline(0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Strong')
                plt.legend(fontsize=10)
                plt.grid(axis='x', alpha=0.3)
                
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                            f'{width:.3f}', ha='left', va='center', fontweight='bold', fontsize=9)
                
                plt.tight_layout()
                save_plot('02_variable_associations.png')
                plt.show()
            except Exception as e:
                print(f"   ⚠️ Could not generate association visualization: {e}")
            
            print(f"   ✓ Found {len(strong_assoc)} moderate/strong associations")
    
    # ANALYSIS C: MCA with K-Means
    print("\n👥 C. Performing Respondent Segmentation...")
    
    if len(categorical_cols) >= 2:
        try:
            mca_cols = [col for col in categorical_cols if col in df.columns][:10]
            
            if len(mca_cols) >= 2:
                df_encoded = pd.get_dummies(df[mca_cols], drop_first=False)
                df_encoded = df_encoded.loc[:, df_encoded.var() > 0]
                
                if df_encoded.shape[1] >= 2:
                    cluster_profiles = auto_kmeans(df_encoded)
                    results['mca_cluster_profiles'] = cluster_profiles
                    
                    n_clusters = len(cluster_profiles)
                    segment_summary = []
                    
                    for cluster_id in range(n_clusters):
                        cluster_data = cluster_profiles.loc[cluster_id]
                        count = int(cluster_data.get('Count', 0))
                        feature_values = cluster_data.drop('Count', errors='ignore')
                        top_features = feature_values.nlargest(5)
                        characteristics = [f.replace('_', ': ') for f in top_features.index.tolist()]
                        
                        segment_summary.append({
                            'Segment': f"Segment {cluster_id + 1}",
                            'Size': count,
                            'Top_Characteristics': ', '.join(characteristics[:3])
                        })
                    
                    results['respondent_segments'] = pd.DataFrame(segment_summary)
                    
                    # Visualization
                    try:
                        segment_df = results['respondent_segments']
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                        
                        colors = plt.cm.Set3(range(len(segment_df)))
                        explode = [0.05] * len(segment_df)
                        
                        wedges, texts, autotexts = ax1.pie(segment_df['Size'], 
                                                           labels=segment_df['Segment'],
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
                        
                        ax1.set_title('Respondent Segment Distribution', fontsize=14, fontweight='bold')
                        
                        bars = ax2.bar(range(len(segment_df)), segment_df['Size'],
                                      color=colors, edgecolor='black', linewidth=1.5)
                        ax2.set_xticks(range(len(segment_df)))
                        ax2.set_xticklabels(segment_df['Segment'], fontsize=11)
                        ax2.set_ylabel('Number of Respondents', fontsize=12, fontweight='bold')
                        ax2.set_title('Segment Sizes', fontsize=14, fontweight='bold')
                        ax2.grid(axis='y', alpha=0.3)
                        
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            ax2.text(bar.get_x() + bar.get_width()/2., height,
                                    f'{int(height):,}', ha='center', va='bottom', 
                                    fontweight='bold', fontsize=11)
                        
                        plt.tight_layout()
                        save_plot('03_respondent_segments.png')
                        plt.show()
                    except Exception as e:
                        print(f"   ⚠️ Could not generate segmentation visualization: {e}")
                    
                    print(f"   ✓ Identified {n_clusters} respondent segments")
        except Exception as e:
            print(f"   ⚠️ MCA analysis failed: {e}")
    
    # ANALYSIS D: Key Driver Analysis
    print("\n🎯 D. Performing Key Driver Analysis...")
    
    target_col = None
    if user_objective and user_objective.strip():
        target_info = identify_target_variable(df, user_objective)
        target_col = target_info.get('target_variable')
    
    if target_col and target_col in df.columns:
        try:
            feature_cols = [col for col in df.columns if col != target_col]
            X = df[feature_cols].copy()
            y = df[target_col].copy()
            
            valid_mask = ~y.isna()
            X = X[valid_mask]
            y = y[valid_mask]
            
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = X[col].fillna('Missing')
                    X[col] = le.fit_transform(X[col].astype(str))
                else:
                    X[col] = X[col].fillna(X[col].median())
            
            if y.dtype == 'object':
                le_target = LabelEncoder()
                y = le_target.fit_transform(y.astype(str))
            
            if len(X) > 10:
                rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                rf.fit(X, y)
                
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': rf.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                importance_df['Relative_Importance_Pct'] = (
                    importance_df['Importance'] / importance_df['Importance'].sum() * 100
                ).round(2)
                
                importance_df['Impact_Level'] = importance_df['Relative_Importance_Pct'].apply(
                    lambda x: 'High' if x >= 15 else ('Medium' if x >= 5 else 'Low')
                )
                
                results['key_drivers'] = importance_df.reset_index(drop=True)
                results['target_variable_analyzed'] = target_col
                
                top_drivers = importance_df.head(5)['Feature'].tolist()
                results['top_5_drivers'] = top_drivers
                
                # Visualization
                try:
                    top_10_drivers = importance_df.head(10)
                    
                    plt.figure(figsize=(12, 8))
                    
                    color_map = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#3498db'}
                    colors = [color_map.get(level, '#95a5a6') for level in top_10_drivers['Impact_Level']]
                    
                    bars = plt.barh(range(len(top_10_drivers)), top_10_drivers['Relative_Importance_Pct'],
                                   color=colors, edgecolor='black', linewidth=1.5)
                    plt.yticks(range(len(top_10_drivers)), top_10_drivers['Feature'], fontsize=11)
                    plt.xlabel('Relative Importance (%)', fontsize=12, fontweight='bold')
                    plt.title(f'Key Drivers of {target_col} (Top 10)', fontsize=14, fontweight='bold')
                    plt.grid(axis='x', alpha=0.3)
                    
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='#e74c3c', edgecolor='black', label='High (≥15%)'),
                        Patch(facecolor='#f39c12', edgecolor='black', label='Medium (5-15%)'),
                        Patch(facecolor='#3498db', edgecolor='black', label='Low (<5%)')
                    ]
                    plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
                    
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                                f'{width:.1f}%', ha='left', va='center', fontweight='bold')
                    
                    plt.tight_layout()
                    save_plot('04_key_drivers.png')
                    plt.show()
                except Exception as e:
                    print(f"   ⚠️ Could not generate key drivers visualization: {e}")
                
                print(f"   ✓ Top driver: {top_drivers[0]}")
        except Exception as e:
            print(f"   ⚠️ Key Driver Analysis failed: {e}")
    
    # ANALYSIS E: Survey Health
    print("\n💚 E. Calculating Survey Health Metrics...")
    
    if top_box_results:
        avg_top_box = np.mean([v['top_2_box_pct'] for v in top_box_results.values()])
        avg_sentiment_ratio = np.mean([v['sentiment_ratio'] for v in top_box_results.values() 
                                       if v['sentiment_ratio'] != np.inf])
        
        results['survey_health'] = {
            'average_top_2_box_pct': round(avg_top_box, 2),
            'average_sentiment_ratio': round(avg_sentiment_ratio, 2),
            'interpretation': 'Positive' if avg_sentiment_ratio > 1.5 else 
                            ('Neutral' if avg_sentiment_ratio > 0.8 else 'Negative')
        }
        
        # Visualization: Survey Health Dashboard
        try:
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])
            
            # Overall Health
            health_score = avg_top_box
            color = '#2ecc71' if health_score >= 70 else '#f39c12' if health_score >= 50 else '#e74c3c'
            
            ax1.barh([0], [health_score], height=0.5, color=color, edgecolor='black', linewidth=2)
            ax1.barh([0], [100-health_score], height=0.5, left=health_score, 
                    color='#ecf0f1', edgecolor='black', linewidth=2)
            ax1.set_xlim([0, 100])
            ax1.set_ylim([-0.5, 0.5])
            ax1.set_xlabel('Health Score (%)', fontsize=12, fontweight='bold')
            ax1.set_yticks([])
            ax1.set_title(f'Overall Survey Health: {health_score:.1f}%', fontsize=14, fontweight='bold')
            ax1.text(health_score/2, 0, f'{health_score:.1f}%', 
                    ha='center', va='center', fontsize=20, fontweight='bold', color='white')
            
            # Sentiment Ratio
            sentiment = avg_sentiment_ratio
            sentiment_color = '#2ecc71' if sentiment > 1.5 else '#f39c12' if sentiment > 0.8 else '#e74c3c'
            
            ax2.barh([0], [sentiment], height=0.5, color=sentiment_color, edgecolor='black', linewidth=2)
            ax2.set_xlim([0, max(3, sentiment * 1.2)])
            ax2.set_ylim([-0.5, 0.5])
            ax2.set_xlabel('Positive/Negative Ratio', fontsize=12, fontweight='bold')
            ax2.set_yticks([])
            ax2.set_title(f'Sentiment Balance: {sentiment:.2f}x', fontsize=14, fontweight='bold')
            ax2.axvline(1.5, color='green', linestyle='--', linewidth=2, alpha=0.5)
            ax2.text(sentiment/2, 0, f'{sentiment:.2f}x', 
                    ha='center', va='center', fontsize=20, fontweight='bold', color='white')
            
            # Score Distribution
            scores = [v['top_2_box_pct'] for v in top_box_results.values()]
            ax3.hist(scores, bins=10, color='#3498db', edgecolor='black', linewidth=1.5, alpha=0.7)
            ax3.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(scores):.1f}%')
            ax3.set_xlabel('Top-2 Box Score (%)', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Number of Questions', fontsize=12, fontweight='bold')
            ax3.set_title('Score Distribution', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=11)
            ax3.grid(axis='y', alpha=0.3)
            
            # Top vs Bottom
            if len(top_box_results) >= 5:
                items = list(top_box_results.items())
                top_5 = sorted(items, key=lambda x: x[1]['top_2_box_pct'], reverse=True)[:5]
                bottom_5 = sorted(items, key=lambda x: x[1]['top_2_box_pct'])[:5]
                
                labels = [item[0][:20] for item in top_5 + bottom_5]
                scores = [item[1]['top_2_box_pct'] for item in top_5 + bottom_5]
                colors_bar = ['#2ecc71']*5 + ['#e74c3c']*5
                
                bars = ax4.barh(range(len(labels)), scores, color=colors_bar, edgecolor='black', linewidth=1.5)
                ax4.set_yticks(range(len(labels)))
                ax4.set_yticklabels(labels, fontsize=9)
                ax4.set_xlabel('Score (%)', fontsize=12, fontweight='bold')
                ax4.set_title('Top 5 vs Bottom 5', fontsize=14, fontweight='bold')
                ax4.axvline(50, color='orange', linestyle='--', linewidth=2, alpha=0.5)
            
            plt.tight_layout()
            save_plot('05_survey_health_dashboard.png')
            plt.show()
        except Exception as e:
            print(f"   ⚠️ Could not generate health dashboard: {e}")
        
        print(f"   ✓ Survey Health: {results['survey_health']['interpretation']}")
    
    print("\n" + "="*60)
    print("✅ Survey Analytics completed successfully!")
    print("="*60)
    
    return results