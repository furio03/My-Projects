import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_path not in sys.path:
    sys.path.append(root_path)

from data_manipulation.read_data import identify_variables
from statistics_functions.clustering import auto_kmeans, auto_kmedoids
from LLM.llm_output import needed_variables


def analytics_insurance(df: pd.DataFrame, mapping: dict = None, user_objective: str = None,
                        save_plots: bool = False, output_dir: str = "domain_plots"):
    """
    Comprehensive insurance analytics covering all major business dimensions.

    Analyses performed:
    - A: Claims Analysis         - Frequency, severity, loss ratio by product/risk
    - B: Risk Scoring            - Customer risk profile segmentation and clustering
    - C: Churn Analysis          - Policy lapse and retention risk identification
    - D: Fraud Detection         - Anomaly flagging based on claim patterns
    - E: Premium vs Claims       - Profitability analysis and loss ratio monitoring
    - F: Underwriting KPIs       - Combined ratio, expense ratio, loss ratio
    - G: Customer Segmentation   - Portfolio clustering by risk/value
    - H: Temporal Trends         - Claims seasonality and premium growth
    - I: Product Performance     - Loss ratio and profitability by product line
    - J: Pricing & Forecasting   - Claim cost prediction and premium adequacy

    Parameters:
    -----------
    df : pd.DataFrame
        Input insurance dataset
    mapping : dict, optional
        Column mapping for insurance variables
    user_objective : str, optional
        User's analysis objective
    save_plots : bool, default=False
        Whether to save plots to disk
    output_dir : str, default="domain_plots"
        Directory to save plots
    """
    print("Running comprehensive insurance analytics")

    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Plots will be saved to: {output_dir}/")

    def save_plot(filename):
        if save_plots:
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"   ✓ Plot saved: {filename}")
        plt.close()

    # --- Variable Roles for Insurance ---
    insurance_roles = [
        'policy_id',       # Policy identifier
        'customer_id',     # Customer identifier
        'claim_amount',    # Amount paid for a claim
        'premium',         # Premium paid by the customer
        'claim_count',     # Number of claims
        'product_type',    # Type of insurance product (auto, life, health...)
        'risk_level',      # Risk category (low/medium/high)
        'duration',        # Policy duration in months/years
        'date',            # Date (policy start, claim date...)
        'status',          # Policy status (active, lapsed, cancelled)
        'age',             # Customer age
        'region',          # Geographic region
    ]

    if mapping is None:
        mapping = needed_variables(df, insurance_roles)

    def get_single_col(key):
        val = mapping.get(key)
        if not val:
            return None
        if isinstance(val, list):
            val = val[0] if len(val) > 0 else None
        if isinstance(val, str) and ',' in val:
            val = val.split(',')[0].strip()
        return val if isinstance(val, str) and val in df.columns else None

    # Extract mapped columns
    policy_col    = get_single_col('policy_id')
    customer_col  = get_single_col('customer_id')
    claim_col     = get_single_col('claim_amount')
    premium_col   = get_single_col('premium')
    claim_cnt_col = get_single_col('claim_count')
    product_col   = get_single_col('product_type')
    risk_col      = get_single_col('risk_level')
    duration_col  = get_single_col('duration')
    date_col      = get_single_col('date')
    status_col    = get_single_col('status')
    age_col       = get_single_col('age')
    region_col    = get_single_col('region')

    var_types    = identify_variables(df)
    numeric_cols = var_types.get('quantitative', [])

    results = {}

    # =========================================================================
    # ANALYSIS A: Claims Analysis — Frequency, Severity, Loss Ratio
    # =========================================================================
    print("\n📋 A. Analyzing Claims Frequency & Severity...")

    if claim_col and claim_col in numeric_cols:
        try:
            claims_data = pd.to_numeric(df[claim_col], errors='coerce').dropna()

            claim_stats = {
                'total_claims_value':    float(round(claims_data.sum(), 2)),
                'avg_claim_amount':      float(round(claims_data.mean(), 2)),
                'median_claim_amount':   float(round(claims_data.median(), 2)),
                'max_claim_amount':      float(round(claims_data.max(), 2)),
                'claim_severity_std':    float(round(claims_data.std(), 2)),
                'n_claims':              int(len(claims_data))
            }

            # Loss ratio = Total claims / Total premiums
            if premium_col and premium_col in numeric_cols:
                total_premium = pd.to_numeric(df[premium_col], errors='coerce').sum()
                if total_premium > 0:
                    claim_stats['loss_ratio_pct'] = float(round(
                        claims_data.sum() / total_premium * 100, 2))

            results['claims_statistics'] = claim_stats

            # Claims by product if available
            if product_col:
                product_claims = df.groupby(product_col)[claim_col].agg(['sum', 'mean', 'count'])
                product_claims.columns = ['total_claims', 'avg_claim', 'n_claims']
                if premium_col and premium_col in numeric_cols:
                    product_premium = df.groupby(product_col)[premium_col].sum()
                    product_claims['loss_ratio'] = (
                        product_claims['total_claims'] / product_premium.replace(0, 1) * 100
                    ).round(2)
                results['claims_by_product'] = product_claims.sort_values('total_claims', ascending=False)

            # Visualization
            try:
                n_cols = 2 if product_col and 'claims_by_product' in results else 1
                fig, axes = plt.subplots(1, 2, figsize=(16, 7))

                # Left: Claims distribution
                ax1 = axes[0]
                clipped = claims_data.clip(upper=claims_data.quantile(0.95))
                ax1.hist(clipped, bins=30, color='#e74c3c', edgecolor='black',
                         linewidth=1.5, alpha=0.7)
                ax1.axvline(claims_data.mean(), color='navy', linestyle='--',
                            linewidth=2, label=f'Mean: €{claims_data.mean():,.0f}')
                ax1.axvline(claims_data.median(), color='green', linestyle='--',
                            linewidth=2, label=f'Median: €{claims_data.median():,.0f}')
                ax1.set_xlabel('Claim Amount (€)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
                ax1.set_title('Claim Severity Distribution (95th pct clip)',
                              fontsize=14, fontweight='bold')
                ax1.legend(fontsize=10)
                ax1.grid(axis='y', alpha=0.3)

                # Right: Loss ratio or claims by product
                ax2 = axes[1]
                if product_col and 'claims_by_product' in results:
                    prod = results['claims_by_product'].head(10)
                    if 'loss_ratio' in prod.columns:
                        colors_lr = ['#2ecc71' if x < 70 else '#f39c12' if x < 100 else '#e74c3c'
                                     for x in prod['loss_ratio']]
                        bars = ax2.barh(range(len(prod)), prod['loss_ratio'],
                                        color=colors_lr, edgecolor='black', linewidth=1.5)
                        ax2.set_yticks(range(len(prod)))
                        ax2.set_yticklabels(prod.index, fontsize=10)
                        ax2.axvline(100, color='red', linestyle='--', linewidth=2,
                                    label='Break-even (100%)')
                        ax2.axvline(70, color='green', linestyle='--', linewidth=2,
                                    label='Target (70%)')
                        ax2.set_xlabel('Loss Ratio (%)', fontsize=12, fontweight='bold')
                        ax2.set_title('Loss Ratio by Product Line',
                                      fontsize=14, fontweight='bold')
                        ax2.legend(fontsize=10)
                        ax2.grid(axis='x', alpha=0.3)
                        for i, bar in enumerate(bars):
                            w = bar.get_width()
                            ax2.text(w + 1, bar.get_y() + bar.get_height() / 2.,
                                     f'{w:.1f}%', ha='left', va='center', fontweight='bold')
                    else:
                        bars = ax2.barh(range(len(prod)), prod['total_claims'],
                                        color='#e74c3c', edgecolor='black', linewidth=1.5)
                        ax2.set_yticks(range(len(prod)))
                        ax2.set_yticklabels(prod.index, fontsize=10)
                        ax2.set_xlabel('Total Claims (€)', fontsize=12, fontweight='bold')
                        ax2.set_title('Total Claims by Product Line',
                                      fontsize=14, fontweight='bold')
                        ax2.grid(axis='x', alpha=0.3)
                else:
                    # Show claims KPIs as text
                    ax2.axis('off')
                    ax2.set_title('Claims KPIs', fontsize=16, fontweight='bold', pad=20)
                    y_pos = 0.9
                    kpi_colors = plt.cm.Set3(range(len(claim_stats)))
                    for i, (k, v) in enumerate(claim_stats.items()):
                        label = k.replace('_', ' ').title()
                        text = f'{label}: €{v:,.2f}' if 'value' in k or 'amount' in k else f'{label}: {v}'
                        ax2.text(0.05, y_pos, text, fontsize=12, fontweight='bold',
                                 transform=ax2.transAxes,
                                 bbox=dict(boxstyle='round', facecolor=kpi_colors[i],
                                           alpha=0.7, edgecolor='black'))
                        y_pos -= 0.14

                plt.tight_layout()
                save_plot('01_claims_analysis.png')
            except Exception as e:
                print(f"   ⚠️ Claims visualization failed: {e}")
                plt.close()

            print(f"   ✓ Claims analyzed: Avg = €{claim_stats['avg_claim_amount']:,.2f}")
        except Exception as e:
            print(f"   ⚠️ Claims analysis failed: {e}")
    else:
        print("   ⚠️ Skipping claims analysis (claim_amount column not found)")

    # =========================================================================
    # ANALYSIS B: Risk Scoring — Customer Risk Profile Segmentation
    # =========================================================================
    print("\n🎯 B. Performing Risk Scoring & Segmentation...")

    risk_features = [c for c in [claim_col, premium_col, duration_col, age_col,
                                  claim_cnt_col]
                     if c and c in numeric_cols]

    if len(risk_features) >= 2:
        try:
            risk_df = df[risk_features].copy()
            for col in risk_features:
                risk_df[col] = pd.to_numeric(risk_df[col], errors='coerce')
            risk_df = risk_df.dropna()

            if len(risk_df) >= 10:
                scaler = StandardScaler()
                risk_scaled = scaler.fit_transform(risk_df)
                risk_scaled_df = pd.DataFrame(risk_scaled, columns=risk_features)

                cluster_profiles = auto_kmeans(risk_scaled_df)
                results['risk_segments'] = cluster_profiles

                # Reverse-map to original scale for interpretation
                n_clusters = len(cluster_profiles)
                risk_df_copy = risk_df.copy()
                risk_df_copy['risk_cluster'] = KMeans(
                    n_clusters=n_clusters, random_state=42, n_init=10
                ).fit_predict(risk_scaled)

                # Label clusters by avg claim amount (proxy for risk)
                if claim_col in risk_features:
                    cluster_means = risk_df_copy.groupby('risk_cluster')[claim_col].mean()
                    sorted_clusters = cluster_means.sort_values()
                    label_map = {c: lbl for c, lbl in
                                 zip(sorted_clusters.index,
                                     ['Low Risk', 'Medium Risk', 'High Risk'][:n_clusters])}
                    risk_df_copy['risk_label'] = risk_df_copy['risk_cluster'].map(label_map)
                    results['risk_label_distribution'] = (
                        risk_df_copy['risk_label'].value_counts().to_dict()
                    )

                # Visualization
                try:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

                    # Left: Cluster distribution
                    if 'risk_label_distribution' in results:
                        dist = results['risk_label_distribution']
                        risk_colors = {'Low Risk': '#2ecc71',
                                       'Medium Risk': '#f39c12',
                                       'High Risk': '#e74c3c'}
                        clr = [risk_colors.get(k, '#95a5a6') for k in dist.keys()]
                        bars1 = ax1.bar(range(len(dist)), list(dist.values()),
                                        color=clr, edgecolor='black', linewidth=1.5)
                        ax1.set_xticks(range(len(dist)))
                        ax1.set_xticklabels(list(dist.keys()), fontsize=12, fontweight='bold')
                        ax1.set_ylabel('Number of Policies', fontsize=12, fontweight='bold')
                        ax1.set_title('Risk Segment Distribution', fontsize=14, fontweight='bold')
                        ax1.grid(axis='y', alpha=0.3)
                        for bar in bars1:
                            h = bar.get_height()
                            ax1.text(bar.get_x() + bar.get_width() / 2., h,
                                     f'{int(h):,}', ha='center', va='bottom', fontweight='bold')

                    # Right: Scatter of two main features
                    f1, f2 = risk_features[0], risk_features[1]
                    scatter = ax2.scatter(risk_df_copy[f1], risk_df_copy[f2],
                                          c=risk_df_copy['risk_cluster'],
                                          cmap='RdYlGn_r', s=60, alpha=0.6,
                                          edgecolors='black', linewidth=0.5)
                    ax2.set_xlabel(f1.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                    ax2.set_ylabel(f2.replace('_', ' ').title(), fontsize=12, fontweight='bold')
                    ax2.set_title('Risk Clustering: Customer Portfolio',
                                  fontsize=14, fontweight='bold')
                    ax2.grid(True, alpha=0.3)
                    cbar = plt.colorbar(scatter, ax=ax2)
                    cbar.set_label('Risk Cluster', fontsize=10, fontweight='bold')

                    plt.tight_layout()
                    save_plot('02_risk_scoring.png')
                except Exception as e:
                    print(f"   ⚠️ Risk visualization failed: {e}")
                    plt.close()

                print(f"   ✓ Risk segmentation: {n_clusters} clusters identified")
        except Exception as e:
            print(f"   ⚠️ Risk scoring failed: {e}")
    else:
        print("   ⚠️ Skipping risk scoring (insufficient numeric features)")

    # =========================================================================
    # ANALYSIS C: Churn Analysis — Policy Lapse & Retention Risk
    # =========================================================================
    print("\n📉 C. Analyzing Churn & Policy Retention...")

    if status_col:
        try:
            status_dist = df[status_col].value_counts()
            results['policy_status_distribution'] = status_dist.to_dict()

            # Identify lapsed/cancelled policies as churn
            churn_keywords = ['lapsed', 'cancelled', 'canceled', 'terminated',
                              'inactive', 'expired', 'churned']
            active_keywords = ['active', 'in force', 'current', 'live']

            status_lower = df[status_col].astype(str).str.lower()
            churn_mask = status_lower.str.contains('|'.join(churn_keywords), na=False)
            active_mask = status_lower.str.contains('|'.join(active_keywords), na=False)

            n_churn = churn_mask.sum()
            n_total = len(df)
            churn_rate = round(n_churn / n_total * 100, 2)

            results['churn_metrics'] = {
                'total_policies':   n_total,
                'churned_policies': int(n_churn),
                'active_policies':  int(active_mask.sum()),
                'churn_rate_pct':   churn_rate,
                'retention_rate_pct': round(100 - churn_rate, 2)
            }

            # Churn by product
            if product_col:
                churn_by_product = pd.DataFrame({
                    'total': df.groupby(product_col).size(),
                    'churned': df[churn_mask].groupby(df.loc[churn_mask, product_col]).size()
                }).fillna(0)
                churn_by_product['churn_rate'] = (
                    churn_by_product['churned'] / churn_by_product['total'] * 100
                ).round(2)
                results['churn_by_product'] = churn_by_product.sort_values(
                    'churn_rate', ascending=False)

            # Visualization
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

                # Left: Status distribution pie
                colors_pie = plt.cm.Set3(range(len(status_dist)))
                wedges, texts, autotexts = ax1.pie(
                    status_dist.values, labels=status_dist.index,
                    autopct='%1.1f%%', colors=colors_pie, startangle=90,
                    shadow=True, textprops={'fontsize': 11, 'fontweight': 'bold'}
                )
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                ax1.set_title('Policy Status Distribution', fontsize=14, fontweight='bold')

                # Right: Churn rate by product
                ax2.axis('off')
                if product_col and 'churn_by_product' in results:
                    cbp = results['churn_by_product'].head(10)
                    colors_churn = ['#e74c3c' if x > 20 else '#f39c12' if x > 10 else '#2ecc71'
                                    for x in cbp['churn_rate']]
                    ax2.axis('on')
                    bars2 = ax2.bar(range(len(cbp)), cbp['churn_rate'],
                                    color=colors_churn, edgecolor='black', linewidth=1.5)
                    ax2.set_xticks(range(len(cbp)))
                    ax2.set_xticklabels(cbp.index, rotation=45, ha='right', fontsize=10)
                    ax2.set_ylabel('Churn Rate (%)', fontsize=12, fontweight='bold')
                    ax2.set_title('Churn Rate by Product Line', fontsize=14, fontweight='bold')
                    ax2.axhline(churn_rate, color='black', linestyle='--', linewidth=2,
                                label=f'Portfolio Avg: {churn_rate:.1f}%')
                    ax2.legend(fontsize=10)
                    ax2.grid(axis='y', alpha=0.3)
                    for bar in bars2:
                        h = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width() / 2., h,
                                 f'{h:.1f}%', ha='center', va='bottom', fontweight='bold')
                else:
                    # Show churn KPIs
                    metrics = results['churn_metrics']
                    y_pos = 0.85
                    colors_kpi = ['#e74c3c', '#f39c12', '#2ecc71', '#e74c3c', '#2ecc71']
                    for i, (k, v) in enumerate(metrics.items()):
                        ax2.text(0.05, y_pos, f"{k.replace('_', ' ').title()}: {v:,}",
                                 fontsize=13, fontweight='bold', transform=ax2.transAxes,
                                 bbox=dict(boxstyle='round', facecolor=colors_kpi[i % len(colors_kpi)],
                                           alpha=0.7))
                        y_pos -= 0.15
                    ax2.axis('off')
                    ax2.set_title('Churn KPIs', fontsize=14, fontweight='bold')

                plt.tight_layout()
                save_plot('03_churn_analysis.png')
            except Exception as e:
                print(f"   ⚠️ Churn visualization failed: {e}")
                plt.close()

            print(f"   ✓ Churn rate: {churn_rate:.1f}% | "
                  f"Retention: {results['churn_metrics']['retention_rate_pct']:.1f}%")
        except Exception as e:
            print(f"   ⚠️ Churn analysis failed: {e}")
    else:
        print("   ⚠️ Skipping churn analysis (status column not found)")

    # =========================================================================
    # ANALYSIS D: Fraud Detection — Anomaly Flagging
    # =========================================================================
    print("\n🚨 D. Detecting Potential Fraudulent Claims...")

    if claim_col and claim_col in numeric_cols:
        try:
            fraud_df = df.copy()
            claim_vals = pd.to_numeric(fraud_df[claim_col], errors='coerce')

            # Statistical anomaly detection
            Q1   = claim_vals.quantile(0.25)
            Q3   = claim_vals.quantile(0.75)
            IQR  = Q3 - Q1
            upper_fence = Q3 + 3.0 * IQR   # Extreme outlier threshold
            mean_c  = claim_vals.mean()
            std_c   = claim_vals.std()
            z_threshold = 3.0

            # Flag suspicious claims
            iqr_flag = claim_vals > upper_fence
            z_flag   = np.abs((claim_vals - mean_c) / (std_c + 1e-9)) > z_threshold

            fraud_df['iqr_suspect']   = iqr_flag
            fraud_df['z_suspect']     = z_flag
            fraud_df['fraud_suspect'] = iqr_flag | z_flag

            n_suspect = int(fraud_df['fraud_suspect'].sum())
            fraud_rate = round(n_suspect / len(fraud_df) * 100, 2)

            results['fraud_indicators'] = {
                'total_policies_analyzed': len(fraud_df),
                'suspect_claims': n_suspect,
                'fraud_suspect_rate_pct': fraud_rate,
                'iqr_flagged': int(iqr_flag.sum()),
                'z_score_flagged': int(z_flag.sum()),
                'upper_claim_threshold': float(round(upper_fence, 2)),
                'potential_exposure': float(round(
                    claim_vals[fraud_df['fraud_suspect']].sum(), 2))
            }

            # By product if available
            if product_col:
                fraud_by_prod = fraud_df.groupby(product_col)['fraud_suspect'].agg(
                    ['sum', 'count']).rename(columns={'sum': 'suspects', 'count': 'total'})
                fraud_by_prod['fraud_rate'] = (
                    fraud_by_prod['suspects'] / fraud_by_prod['total'] * 100).round(2)
                results['fraud_by_product'] = fraud_by_prod.sort_values(
                    'fraud_rate', ascending=False)

            # Visualization
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

                # Left: Claim distribution with fraud threshold
                bins = np.linspace(claim_vals.min(), claim_vals.quantile(0.99), 50)
                ax1.hist(claim_vals[~fraud_df['fraud_suspect']], bins=bins,
                         color='#3498db', alpha=0.7, label='Normal', edgecolor='black',
                         linewidth=0.8)
                ax1.hist(claim_vals[fraud_df['fraud_suspect']], bins=bins,
                         color='#e74c3c', alpha=0.9, label='Suspicious', edgecolor='black',
                         linewidth=0.8)
                ax1.axvline(upper_fence, color='orange', linestyle='--', linewidth=2.5,
                            label=f'IQR Threshold: €{upper_fence:,.0f}')
                ax1.set_xlabel('Claim Amount (€)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
                ax1.set_title('Claim Distribution: Normal vs Suspicious',
                              fontsize=14, fontweight='bold')
                ax1.legend(fontsize=10)
                ax1.grid(axis='y', alpha=0.3)

                # Right: Fraud by product or summary
                ax2.axis('off')
                if product_col and 'fraud_by_product' in results:
                    fbp = results['fraud_by_product'].head(10)
                    ax2.axis('on')
                    fraud_colors = ['#e74c3c' if x > 10 else '#f39c12' if x > 5 else '#2ecc71'
                                    for x in fbp['fraud_rate']]
                    bars2 = ax2.barh(range(len(fbp)), fbp['fraud_rate'],
                                     color=fraud_colors, edgecolor='black', linewidth=1.5)
                    ax2.set_yticks(range(len(fbp)))
                    ax2.set_yticklabels(fbp.index, fontsize=10)
                    ax2.set_xlabel('Fraud Suspect Rate (%)', fontsize=12, fontweight='bold')
                    ax2.set_title('Fraud Exposure by Product Line',
                                  fontsize=14, fontweight='bold')
                    ax2.grid(axis='x', alpha=0.3)
                    for i, bar in enumerate(bars2):
                        w = bar.get_width()
                        ax2.text(w + 0.2, bar.get_y() + bar.get_height() / 2.,
                                 f'{w:.1f}%', ha='left', va='center', fontweight='bold')
                else:
                    fi = results['fraud_indicators']
                    y_pos = 0.85
                    items = [(k.replace('_', ' ').title(), v) for k, v in fi.items()]
                    for i, (lbl, val) in enumerate(items):
                        color = '#e74c3c' if 'suspect' in lbl.lower() or 'exposure' in lbl.lower() else '#3498db'
                        text = f'{lbl}: €{val:,.2f}' if 'exposure' in lbl.lower() or 'threshold' in lbl.lower() else f'{lbl}: {val}'
                        ax2.text(0.05, y_pos, text, fontsize=11, fontweight='bold',
                                 transform=ax2.transAxes,
                                 bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
                        y_pos -= 0.13
                    ax2.set_title('Fraud Detection Summary', fontsize=14, fontweight='bold')

                plt.tight_layout()
                save_plot('04_fraud_detection.png')
            except Exception as e:
                print(f"   ⚠️ Fraud visualization failed: {e}")
                plt.close()

            print(f"   ✓ Fraud suspects: {n_suspect} ({fraud_rate:.1f}% of portfolio)")
        except Exception as e:
            print(f"   ⚠️ Fraud detection failed: {e}")
    else:
        print("   ⚠️ Skipping fraud detection (claim_amount not found)")

    # =========================================================================
    # ANALYSIS E: Premium vs Claims — Profitability & Loss Ratio
    # =========================================================================
    print("\n💰 E. Analyzing Premium vs Claims Profitability...")

    if premium_col and claim_col and premium_col in numeric_cols and claim_col in numeric_cols:
        try:
            temp_pnl = df[[premium_col, claim_col]].copy()
            temp_pnl[premium_col] = pd.to_numeric(temp_pnl[premium_col], errors='coerce')
            temp_pnl[claim_col]   = pd.to_numeric(temp_pnl[claim_col], errors='coerce')
            temp_pnl = temp_pnl.dropna()

            temp_pnl['underwriting_result'] = temp_pnl[premium_col] - temp_pnl[claim_col]
            temp_pnl['loss_ratio'] = (temp_pnl[claim_col] / temp_pnl[premium_col].replace(0, 1) * 100).round(2)
            temp_pnl['profitable'] = temp_pnl['underwriting_result'] > 0

            total_premium = temp_pnl[premium_col].sum()
            total_claims  = temp_pnl[claim_col].sum()
            overall_loss_ratio = total_claims / total_premium * 100 if total_premium > 0 else 0

            results['profitability'] = {
                'total_premiums_collected': float(round(total_premium, 2)),
                'total_claims_paid':        float(round(total_claims, 2)),
                'underwriting_profit':      float(round(total_premium - total_claims, 2)),
                'overall_loss_ratio_pct':   float(round(overall_loss_ratio, 2)),
                'profitable_policies_pct':  float(round(temp_pnl['profitable'].mean() * 100, 2)),
                'avg_underwriting_result':  float(round(temp_pnl['underwriting_result'].mean(), 2))
            }

            # Visualization
            try:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

                # Top-left: Underwriting result distribution
                uw = temp_pnl['underwriting_result']
                ax1.hist(uw.clip(uw.quantile(0.02), uw.quantile(0.98)),
                         bins=35, color='#3498db', edgecolor='black', linewidth=1.2, alpha=0.7)
                ax1.axvline(0, color='red', linestyle='-', linewidth=2.5, label='Break-even')
                ax1.axvline(uw.mean(), color='green', linestyle='--', linewidth=2,
                            label=f'Mean: €{uw.mean():,.0f}')
                ax1.set_xlabel('Underwriting Result (€)', fontsize=12, fontweight='bold')
                ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
                ax1.set_title('Underwriting Result Distribution', fontsize=14, fontweight='bold')
                ax1.legend(fontsize=10)
                ax1.grid(axis='y', alpha=0.3)

                # Top-right: Loss ratio distribution
                lr_clipped = temp_pnl['loss_ratio'].clip(0, 300)
                colors_hist = ['#e74c3c' if x > 100 else '#f39c12' if x > 70 else '#2ecc71'
                               for x in lr_clipped]
                ax2.hist(lr_clipped, bins=30, color='#9b59b6', edgecolor='black',
                         linewidth=1.2, alpha=0.7)
                ax2.axvline(100, color='red', linestyle='--', linewidth=2,
                            label='Break-even (100%)')
                ax2.axvline(70, color='green', linestyle='--', linewidth=2,
                            label='Target (70%)')
                ax2.axvline(overall_loss_ratio, color='navy', linestyle='-', linewidth=2.5,
                            label=f'Portfolio: {overall_loss_ratio:.1f}%')
                ax2.set_xlabel('Loss Ratio (%)', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
                ax2.set_title('Loss Ratio Distribution', fontsize=14, fontweight='bold')
                ax2.legend(fontsize=10)
                ax2.grid(axis='y', alpha=0.3)

                # Bottom-left: Premium vs Claims scatter
                sample = temp_pnl.sample(min(500, len(temp_pnl)), random_state=42)
                colors_scatter = ['#2ecc71' if x else '#e74c3c' for x in sample['profitable']]
                ax3.scatter(sample[premium_col], sample[claim_col],
                            c=colors_scatter, s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
                max_val = max(sample[premium_col].max(), sample[claim_col].max())
                ax3.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Break-even line')
                ax3.set_xlabel('Premium (€)', fontsize=12, fontweight='bold')
                ax3.set_ylabel('Claim Amount (€)', fontsize=12, fontweight='bold')
                ax3.set_title('Premium vs Claim: Green=Profit, Red=Loss',
                              fontsize=14, fontweight='bold')
                ax3.legend(fontsize=10)
                ax3.grid(True, alpha=0.3)

                # Bottom-right: Loss ratio by product
                if product_col and 'claims_by_product' in results and \
                        'loss_ratio' in results['claims_by_product'].columns:
                    prod_lr = results['claims_by_product']['loss_ratio'].head(10)
                    bar_colors = ['#2ecc71' if x < 70 else '#f39c12' if x < 100 else '#e74c3c'
                                  for x in prod_lr]
                    bars4 = ax4.bar(range(len(prod_lr)), prod_lr.values,
                                    color=bar_colors, edgecolor='black', linewidth=1.5)
                    ax4.set_xticks(range(len(prod_lr)))
                    ax4.set_xticklabels(prod_lr.index, rotation=45, ha='right', fontsize=10)
                    ax4.axhline(100, color='red', linestyle='--', linewidth=2)
                    ax4.axhline(70, color='green', linestyle='--', linewidth=2)
                    ax4.set_ylabel('Loss Ratio (%)', fontsize=12, fontweight='bold')
                    ax4.set_title('Loss Ratio by Product Line', fontsize=14, fontweight='bold')
                    ax4.grid(axis='y', alpha=0.3)
                    for bar in bars4:
                        h = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width() / 2., h,
                                 f'{h:.1f}%', ha='center', va='bottom', fontweight='bold')
                else:
                    ax4.axis('off')
                    ax4.set_title('Profitability KPIs', fontsize=14, fontweight='bold', pad=20)
                    y_pos = 0.85
                    kpi_colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']
                    for i, (k, v) in enumerate(results['profitability'].items()):
                        lbl = k.replace('_', ' ').title()
                        text = f'{lbl}: €{v:,.2f}' if 'premiums' in k or 'claims' in k or 'profit' in k or 'result' in k else f'{lbl}: {v}%'
                        ax4.text(0.05, y_pos, text, fontsize=12, fontweight='bold',
                                 transform=ax4.transAxes,
                                 bbox=dict(boxstyle='round',
                                           facecolor=kpi_colors[i % len(kpi_colors)],
                                           alpha=0.7))
                        y_pos -= 0.13

                plt.tight_layout()
                save_plot('05_profitability_analysis.png')
            except Exception as e:
                print(f"   ⚠️ Profitability visualization failed: {e}")
                plt.close()

            print(f"   ✓ Loss ratio: {overall_loss_ratio:.1f}% | "
                  f"Underwriting profit: €{results['profitability']['underwriting_profit']:,.0f}")
        except Exception as e:
            print(f"   ⚠️ Premium vs Claims analysis failed: {e}")
    else:
        print("   ⚠️ Skipping profitability analysis (premium and/or claim column not found)")

    # =========================================================================
    # ANALYSIS F: Underwriting KPIs
    # =========================================================================
    print("\n📊 F. Calculating Underwriting KPIs...")

    if premium_col and premium_col in numeric_cols:
        try:
            kpis = {}
            total_premium = pd.to_numeric(df[premium_col], errors='coerce').sum()
            kpis['total_gross_written_premium'] = float(round(total_premium, 2))
            kpis['avg_premium_per_policy'] = float(round(
                pd.to_numeric(df[premium_col], errors='coerce').mean(), 2))

            if claim_col and claim_col in numeric_cols:
                total_claims = pd.to_numeric(df[claim_col], errors='coerce').sum()
                kpis['loss_ratio_pct'] = float(round(total_claims / total_premium * 100, 2))

            if policy_col:
                kpis['total_policies'] = int(df[policy_col].nunique())
            if product_col:
                kpis['product_lines'] = int(df[product_col].nunique())
            if region_col:
                kpis['regions_covered'] = int(df[region_col].nunique())

            results['underwriting_kpis'] = kpis
            print(f"   ✓ KPIs: GWP = €{total_premium:,.0f}")
        except Exception as e:
            print(f"   ⚠️ KPI calculation failed: {e}")
    else:
        print("   ⚠️ Skipping KPIs (premium column not found)")

    # =========================================================================
    # ANALYSIS G: Temporal Trends — Claims Seasonality & Premium Growth
    # =========================================================================
    print("\n📅 G. Analyzing Temporal Trends...")

    if date_col and (claim_col or premium_col):
        try:
            metric_col = claim_col if claim_col and claim_col in numeric_cols else premium_col
            temp_time = df[[date_col, metric_col]].copy()
            temp_time[date_col]   = pd.to_datetime(temp_time[date_col], errors='coerce')
            temp_time[metric_col] = pd.to_numeric(temp_time[metric_col], errors='coerce')
            temp_time = temp_time.dropna()

            if len(temp_time) > 5:
                temp_time['month']       = temp_time[date_col].dt.month
                temp_time['month_name']  = temp_time[date_col].dt.strftime('%b')
                temp_time['year']        = temp_time[date_col].dt.year
                temp_time['period']      = temp_time[date_col].dt.to_period('M')

                monthly = temp_time.groupby('period')[metric_col].agg(['sum', 'count', 'mean'])
                monthly.columns = ['total', 'count', 'average']
                monthly['growth_pct'] = monthly['total'].pct_change() * 100

                monthly_by_month = temp_time.groupby('month')[metric_col].mean()

                results['temporal_trends'] = {
                    'period_analyzed':    f"{temp_time[date_col].min().strftime('%Y-%m')} to {temp_time[date_col].max().strftime('%Y-%m')}",
                    'peak_month':         int(monthly_by_month.idxmax()),
                    'avg_monthly_total':  float(round(monthly['total'].mean(), 2)),
                    'avg_growth_pct':     float(round(monthly['growth_pct'].mean(), 2))
                }

                # Visualization
                try:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

                    # Top: Monthly trend
                    x = range(len(monthly))
                    ax1.plot(x, monthly['total'], marker='o', linewidth=3,
                             markersize=8, color='#3498db', label='Monthly Total')
                    ax1.fill_between(x, monthly['total'], alpha=0.3, color='#3498db')
                    ax1.set_xticks(x)
                    ax1.set_xticklabels([str(p) for p in monthly.index],
                                        rotation=45, ha='right', fontsize=9)
                    ax1.set_ylabel(f'{metric_col.replace("_"," ").title()} (€)',
                                   fontsize=12, fontweight='bold')
                    ax1.set_title(f'Monthly Trend: {metric_col.replace("_"," ").title()}',
                                  fontsize=14, fontweight='bold')
                    ax1.legend(fontsize=11)
                    ax1.grid(True, alpha=0.3, linestyle='--')

                    # Bottom: Seasonality by month
                    month_labels = ['Jan','Feb','Mar','Apr','May','Jun',
                                    'Jul','Aug','Sep','Oct','Nov','Dec']
                    month_vals = [monthly_by_month.get(m, 0) for m in range(1, 13)]
                    peak_m = int(monthly_by_month.idxmax())
                    bar_colors = ['#e74c3c' if m == peak_m else '#3498db'
                                  for m in range(1, 13)]
                    bars2 = ax2.bar(range(12), month_vals, color=bar_colors,
                                    edgecolor='black', linewidth=1.5)
                    ax2.set_xticks(range(12))
                    ax2.set_xticklabels(month_labels, fontsize=11, fontweight='bold')
                    ax2.set_ylabel('Average Monthly Value (€)', fontsize=12, fontweight='bold')
                    ax2.set_title('Seasonality: Average by Month (Red = Peak)',
                                  fontsize=14, fontweight='bold')
                    ax2.grid(axis='y', alpha=0.3)
                    for bar in bars2:
                        h = bar.get_height()
                        if h > 0:
                            ax2.text(bar.get_x() + bar.get_width() / 2., h,
                                     f'€{h:,.0f}', ha='center', va='bottom',
                                     fontweight='bold', fontsize=8)

                    plt.tight_layout()
                    save_plot('06_temporal_trends.png')
                except Exception as e:
                    print(f"   ⚠️ Temporal visualization failed: {e}")
                    plt.close()

                print(f"   ✓ Peak month: {month_labels[results['temporal_trends']['peak_month']-1]}")
        except Exception as e:
            print(f"   ⚠️ Temporal trends failed: {e}")
    else:
        print("   ⚠️ Skipping temporal trends (date column not found)")

    # =========================================================================
    # ANALYSIS H: Product Performance — Loss Ratio & Portfolio Mix
    # =========================================================================
    print("\n🏷️  H. Analyzing Product Line Performance...")

    if product_col:
        try:
            product_agg = {'count': df.groupby(product_col).size()}

            if premium_col and premium_col in numeric_cols:
                product_agg['total_premium'] = pd.to_numeric(
                    df[premium_col], errors='coerce').groupby(df[product_col]).sum()

            if claim_col and claim_col in numeric_cols:
                product_agg['total_claims'] = pd.to_numeric(
                    df[claim_col], errors='coerce').groupby(df[product_col]).sum()

            product_perf = pd.DataFrame(product_agg)

            if 'total_premium' in product_perf.columns and 'total_claims' in product_perf.columns:
                product_perf['loss_ratio'] = (
                    product_perf['total_claims'] /
                    product_perf['total_premium'].replace(0, 1) * 100).round(2)
                product_perf['market_share_pct'] = (
                    product_perf['count'] / product_perf['count'].sum() * 100).round(2)

            results['product_performance'] = product_perf.sort_values(
                'total_premium' if 'total_premium' in product_perf.columns else 'count',
                ascending=False)

            # Visualization
            try:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

                # Top-left: Market share pie
                counts = product_perf['count']
                colors_pie = plt.cm.Set3(range(len(counts)))
                ax1.pie(counts, labels=counts.index, autopct='%1.1f%%',
                        colors=colors_pie, startangle=90, shadow=True,
                        textprops={'fontsize': 10, 'fontweight': 'bold'})
                ax1.set_title('Portfolio Mix by Product Line', fontsize=14, fontweight='bold')

                # Top-right: Premium by product
                if 'total_premium' in product_perf.columns:
                    prem_sorted = product_perf.sort_values('total_premium', ascending=False)
                    colors2 = plt.cm.Blues(np.linspace(0.4, 0.9, len(prem_sorted)))
                    bars2 = ax2.barh(range(len(prem_sorted)), prem_sorted['total_premium'],
                                     color=colors2, edgecolor='black', linewidth=1.5)
                    ax2.set_yticks(range(len(prem_sorted)))
                    ax2.set_yticklabels(prem_sorted.index, fontsize=11)
                    ax2.set_xlabel('Total Premium (€)', fontsize=12, fontweight='bold')
                    ax2.set_title('Gross Written Premium by Product',
                                  fontsize=14, fontweight='bold')
                    ax2.grid(axis='x', alpha=0.3)
                    for i, bar in enumerate(bars2):
                        w = bar.get_width()
                        ax2.text(w + w * 0.02, bar.get_y() + bar.get_height() / 2.,
                                 f'€{w:,.0f}', ha='left', va='center',
                                 fontweight='bold', fontsize=9)
                else:
                    ax2.axis('off')

                # Bottom-left: Loss ratio by product
                if 'loss_ratio' in product_perf.columns:
                    lr_sorted = product_perf.sort_values('loss_ratio', ascending=False)
                    bar_colors3 = ['#e74c3c' if x > 100 else '#f39c12' if x > 70 else '#2ecc71'
                                   for x in lr_sorted['loss_ratio']]
                    bars3 = ax3.bar(range(len(lr_sorted)), lr_sorted['loss_ratio'],
                                    color=bar_colors3, edgecolor='black', linewidth=1.5)
                    ax3.set_xticks(range(len(lr_sorted)))
                    ax3.set_xticklabels(lr_sorted.index, rotation=45, ha='right', fontsize=10)
                    ax3.axhline(100, color='red', linestyle='--', linewidth=2, label='Break-even')
                    ax3.axhline(70, color='green', linestyle='--', linewidth=2, label='Target 70%')
                    ax3.set_ylabel('Loss Ratio (%)', fontsize=12, fontweight='bold')
                    ax3.set_title('Loss Ratio by Product Line', fontsize=14, fontweight='bold')
                    ax3.legend(fontsize=10)
                    ax3.grid(axis='y', alpha=0.3)
                    for bar in bars3:
                        h = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width() / 2., h,
                                 f'{h:.1f}%', ha='center', va='bottom', fontweight='bold')
                else:
                    ax3.axis('off')

                # Bottom-right: Premium vs Claims comparison
                if 'total_premium' in product_perf.columns and 'total_claims' in product_perf.columns:
                    x_range = np.arange(len(product_perf))
                    width = 0.35
                    ax4.bar(x_range - width / 2, product_perf['total_premium'],
                            width, label='Premium', color='#2ecc71',
                            edgecolor='black', linewidth=1.5)
                    ax4.bar(x_range + width / 2, product_perf['total_claims'],
                            width, label='Claims', color='#e74c3c',
                            edgecolor='black', linewidth=1.5)
                    ax4.set_xticks(x_range)
                    ax4.set_xticklabels(product_perf.index, rotation=45, ha='right', fontsize=10)
                    ax4.set_ylabel('Amount (€)', fontsize=12, fontweight='bold')
                    ax4.set_title('Premium vs Claims by Product Line',
                                  fontsize=14, fontweight='bold')
                    ax4.legend(fontsize=11)
                    ax4.grid(axis='y', alpha=0.3)
                else:
                    ax4.axis('off')

                plt.tight_layout()
                save_plot('07_product_performance.png')
            except Exception as e:
                print(f"   ⚠️ Product visualization failed: {e}")
                plt.close()

            print(f"   ✓ {len(product_perf)} product lines analyzed")
        except Exception as e:
            print(f"   ⚠️ Product performance failed: {e}")
    else:
        print("   ⚠️ Skipping product performance (product_type column not found)")

    # =========================================================================
    # ANALYSIS I: Claim Cost Forecasting
    # =========================================================================
    print("\n🔮 I. Generating Claim Cost Forecast...")

    if date_col and claim_col and claim_col in numeric_cols:
        try:
            temp_fc = df[[date_col, claim_col]].copy()
            temp_fc[date_col]   = pd.to_datetime(temp_fc[date_col], errors='coerce')
            temp_fc[claim_col]  = pd.to_numeric(temp_fc[claim_col], errors='coerce')
            temp_fc = temp_fc.dropna()

            if len(temp_fc) > 7:
                temp_fc = temp_fc.sort_values(date_col)
                monthly_claims = temp_fc.groupby(
                    temp_fc[date_col].dt.to_period('M')
                )[claim_col].sum().reset_index()
                monthly_claims['time_index'] = range(len(monthly_claims))

                X = monthly_claims[['time_index']].values
                y = monthly_claims[claim_col].values

                model = LinearRegression()
                model.fit(X, y)

                future = np.array([[len(X) + i] for i in range(1, 7)])
                preds  = model.predict(future)
                r2     = model.score(X, y)

                results['claim_forecast'] = {
                    'forecast_6_months': [float(round(p, 2)) for p in preds],
                    'trend':             'Increasing' if preds[-1] > preds[0] else 'Decreasing',
                    'r2_score':          float(round(r2, 3)),
                    'monthly_trend':     float(round(float(model.coef_[0]), 2))
                }

                # Visualization
                try:
                    plt.figure(figsize=(14, 7))
                    plt.plot(range(len(y)), y, 'o-', linewidth=3, markersize=7,
                             color='#3498db', alpha=0.85, label='Actual Monthly Claims')
                    fx = range(len(y), len(y) + 6)
                    plt.plot(fx, preds, 's--', linewidth=3, markersize=10,
                             color='#e74c3c', alpha=0.85, label='6-Month Forecast')
                    plt.axvspan(len(y) - 0.5, len(y) + 5.5, alpha=0.1,
                                color='orange', label='Forecast Period')
                    for i, (xi, p) in enumerate(zip(fx, preds)):
                        if i % 2 == 0:
                            plt.text(xi, p + (y.max() - y.min()) * 0.04, f'€{p:,.0f}',
                                     ha='center', va='bottom', fontweight='bold', fontsize=10,
                                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                    plt.xlabel('Time Period (months)', fontsize=13, fontweight='bold')
                    plt.ylabel('Total Claims (€)', fontsize=13, fontweight='bold')
                    plt.title(f'Insurance Claim Cost Forecast – Next 6 Months\n'
                              f'Trend: {results["claim_forecast"]["trend"]} | R² = {r2:.3f}',
                              fontsize=15, fontweight='bold', pad=20)
                    plt.legend(loc='best', fontsize=12)
                    plt.grid(True, alpha=0.3, linestyle='--')
                    plt.tight_layout()
                    save_plot('08_claim_forecast.png')
                except Exception as e:
                    print(f"   ⚠️ Forecast visualization failed: {e}")
                    plt.close()

                print(f"   ✓ Forecast: {results['claim_forecast']['trend']} trend (R²={r2:.3f})")
        except Exception as e:
            print(f"   ⚠️ Claim forecasting failed: {e}")
    else:
        print("   ⚠️ Skipping forecast (date or claim column not found)")

    print("\n" + "=" * 60)
    print("✅ Insurance Analytics completed successfully!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("INSURANCE ANALYTICS — TEST MODE")
    print("=" * 60 + "\n")

    from datetime import datetime, timedelta

    np.random.seed(42)
    n = 800

    products = ['Auto', 'Home', 'Life', 'Health', 'Travel']
    risks    = ['Low', 'Medium', 'High']
    regions  = ['North', 'South', 'East', 'West', 'Central']
    statuses = ['Active', 'Lapsed', 'Cancelled', 'Active', 'Active']  # Skewed active

    start = datetime(2022, 1, 1)
    data  = []

    for i in range(n):
        product = np.random.choice(products)
        risk    = np.random.choice(risks, p=[0.5, 0.35, 0.15])

        # Premium varies by product and risk
        base_premium = {'Auto': 900, 'Home': 700, 'Life': 1200,
                        'Health': 1500, 'Travel': 300}[product]
        risk_mult    = {'Low': 0.8, 'Medium': 1.0, 'High': 1.4}[risk]
        premium      = round(base_premium * risk_mult * np.random.uniform(0.8, 1.2), 2)

        # Claims: only ~40% have claims
        has_claim    = np.random.random() < 0.4
        claim_amount = round(np.random.gamma(4, 300) * risk_mult, 2) if has_claim else 0.0

        age = np.random.randint(18, 75)
        days_offset  = np.random.randint(0, 700)
        claim_date   = start + timedelta(days=days_offset)

        data.append({
            'PolicyID':     f'POL{i+1:05d}',
            'CustomerID':   f'CUST{np.random.randint(1, 301):04d}',
            'Product':      product,
            'RiskCategory': risk,
            'Region':       np.random.choice(regions),
            'CustomerAge':  age,
            'Premium':      premium,
            'ClaimAmount':  claim_amount,
            'ClaimCount':   int(has_claim),
            'PolicyStatus': np.random.choice(statuses),
            'Duration_months': np.random.randint(1, 60),
            'Date':         claim_date.strftime('%Y-%m-%d')
        })

    df_test = pd.DataFrame(data)
    print(f"✅ Test dataset: {len(df_test)} policies\n")

    test_mapping = {
        'policy_id':    'PolicyID',
        'customer_id':  'CustomerID',
        'product_type': 'Product',
        'risk_level':   'RiskCategory',
        'region':       'Region',
        'age':          'CustomerAge',
        'premium':      'Premium',
        'claim_amount': 'ClaimAmount',
        'claim_count':  'ClaimCount',
        'status':       'PolicyStatus',
        'duration':     'Duration_months',
        'date':         'Date'
    }

    try:
        results = analytics_insurance(df_test, mapping=test_mapping,
                                      save_plots=True, output_dir="domain_plots")

        print("\n" + "=" * 60)
        print("📊 RESULTS SUMMARY")
        print("=" * 60)
        for key, value in results.items():
            print(f"\n{'─'*50}")
            print(f"📈 {key.upper().replace('_', ' ')}")
            print(f"{'─'*50}")
            if isinstance(value, (pd.DataFrame, pd.Series)):
                print(value.head(8))
            elif isinstance(value, dict):
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(value)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()