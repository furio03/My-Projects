import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from data_manipulation.read_data import identify_variables


def create_histogram(df: pd.DataFrame, save_path: str = "histogram.png", user_objective: str = None) -> str:
    """
    Creates and saves a histogram for a selected quantitative variable from the DataFrame.
    
    Args:
        df: DataFrame containing the data
        save_path: Path where to save the image (default: "histogram.png")
        user_objective: User's analysis objective to guide variable selection
    
    Returns:
        Name of the plotted variable
    """
    variables = identify_variables(df)
    quant_vars = variables['quantitative']
    
    if not quant_vars:
        print("No quantitative variables found for histogram")
        return None
    
    # Select variable based on user objective or default to second variable
    if user_objective and user_objective in quant_vars:
        var_name = user_objective
    else:
        var_name = quant_vars[1] if len(quant_vars) > 1 else quant_vars[0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    data = df[var_name].dropna()
    
    ax.hist(data, bins=30, color='#4CAF50', edgecolor='black', alpha=0.7)
    ax.set_xlabel(var_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Distribution of {var_name}', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    mean_val = data.mean()
    median_val = data.median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvline(median_val, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return var_name


def collect_domain_plots(domain_suffix: str) -> list:
    """
    Collect all PNG plots from the domain_plots directory.
    
    Args:
        domain_suffix: The domain name (not used anymore, kept for compatibility)
    
    Returns:
        list: Sorted list of plot file paths (absolute paths as strings)
    """
    domain_plots = []
    
    # Use absolute path to domain_plots directory
    # This ensures it works regardless of where the script is run from
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up from statistics_functions/ to project root
    plot_dir = Path(project_root) / "domain_plots"
    
    if plot_dir.exists() and plot_dir.is_dir():
        # Collect all PNG files
        png_files = sorted(plot_dir.glob("*.png"))
        if png_files:
            domain_plots = [str(p.absolute()) for p in png_files]
            print(f"   ✓ Found {len(png_files)} plots in domain_plots/")
        else:
            print(f"   ℹ️  domain_plots/ directory exists but is empty")
    else:
        print(f"   ℹ️  No domain_plots/ directory found at {plot_dir}")
    
    return domain_plots
    
    return domain_plots


def generate_plot_description_for_llm(domain_plots: list) -> str:
    """
    Generate a formatted text description of available plots for LLM prompt.
    
    Args:
        domain_plots: List of plot file paths
    
    Returns:
        str: Formatted description for LLM
    """
    if not domain_plots or len(domain_plots) == 0:
        return ""
    
    description = f"\n### DOMAIN-SPECIFIC VISUALIZATIONS AVAILABLE ###\n"
    description += f"The analysis has generated {len(domain_plots)} professional visualizations:\n\n"
    
    for i, plot_path in enumerate(domain_plots, 1):
        # Extract readable name from filename
        plot_name = Path(plot_path).stem
        # Remove numeric prefixes (01_, 02_, etc.)
        plot_name_clean = ''.join(c for c in plot_name if not c.isdigit()).strip('_- ')
        # Make title case and replace underscores
        plot_name_readable = plot_name_clean.replace('_', ' ').title()
        
        # Figure number (+2 because Figure 1 is histogram)
        figure_num = i + 2
        description += f"  Figure {figure_num}: {plot_name_readable}\n"
    
    description += f"\n**CRITICAL INSTRUCTION**: You MUST reference these figures throughout your analysis. "
    description += f"When discussing relevant topics, explicitly mention the figure number and explain what it reveals.\n"
    description += f"For example: 'As illustrated in Figure 3, the route performance analysis demonstrates...'\n\n"
    description += f"For each figure you reference, explain:\n"
    description += f"1. What pattern or trend is visible in the visualization\n"
    description += f"2. Why this matters for business decisions\n"
    description += f"3. What specific actions should be taken based on this visual evidence\n"
    description += f"\nWeave the figures naturally into your narrative - do NOT just list them at the end.\n"
    
    return description