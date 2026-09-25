import os
import inspect
import shutil
from data_manipulation.read_data import read, handle_Nans, correct_dtypes, identify_variables
from data_manipulation.privacy import protect_data
from analystics.data_analyis import basic_analytics_quant, basic_analytics_qual, basic_analytics_date
from LLM.report import analizza_dataset, analizza_outliers
from LLM.llm_output import recognise_domain, identify_target_variable
from analystics.agriculture import analytics_agriculture as an_agriculture
from analystics.business_economics import analytics_business_economics as an_business_economics
from analystics.demographic import analytics_demographic as an_demographic
from analystics.ecommerce import analytics_ecommerce as an_ecommerce
from analystics.education import analytics_education as an_education
from analystics.energy import analytics_energy as an_energy
from analystics.environment import analytics_environment as an_environment
from analystics.finance import analytics_finance as an_finance
from analystics.health import analytics_health as an_health
from analystics.human_res import analytics_hr as an_hr
from analystics.industrial import analytics_industrial as an_industrial
from analystics.insurance import analytics_insurance as an_insurance
from analystics.logistics import analytics_logistics as an_logistics
from analystics.macro_economics import analytics_macro_economics as an_macro_economics
from analystics.marketing import analytics_marketing as an_marketing
from analystics.politics import analytics_politics as an_politics
from analystics.real_estate import analytics_real_estate as an_real_estate
from analystics.retail import analytics_retail as an_retail
from analystics.scientific_research import analytics_scientific_research as an_scientific_research
from analystics.security import analytics_security as an_security
from analystics.social_media import analytics_social_media as an_social_media
from analystics.sports import analytics_sports as an_sports
from analystics.supply_chain import analytics_supply_chain as an_supply_chain
from analystics.surveys import analytics_surveys as an_surveys
from analystics.telecommunications import analytics_telecommunications as an_telecommunications
from analystics.tourism import analytics_tourism as an_tourism
from statistics_functions.clustering import auto_kmeans, auto_kmedoids
from statistics_functions.graphs import create_histogram, collect_domain_plots
from statistics_functions.chi_square import chi_square_test
from statistics_functions.outliers import handle_outliers

def run_analysis(file_path, user_objective: str = None):
    
    df = read(file_path).iloc[:500, 2:8]

    df = handle_Nans(df)
    df = correct_dtypes(df)
    dict_vars = identify_variables(df)

    print(df.info())
    print(df.head())
    print(dict_vars)

    # Identify target variable and language from user objective
    target_variable = None
    user_language = "en"
    
    if user_objective and user_objective.strip():
        target_info = identify_target_variable(df, user_objective)
        target_variable = target_info.get('target_variable')
        user_language = target_info.get('language', 'en')
        print(f"\nTarget variable identified: {target_variable}")
        print(f"User language: {user_language}")
    
    cleaning = handle_outliers(df, target_variable=target_variable)
    df_cleaned = cleaning['df_no_outliers']
    df_flagged = cleaning['df_flagged']
    df_outliers = cleaning['df_only_outliers']

    basic_quant, corr_matrix = basic_analytics_quant(df_cleaned, dict_vars)
    basic_qual = basic_analytics_qual(df_cleaned, dict_vars)
    basic_date = basic_analytics_date(df_cleaned, dict_vars)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    histogram_path = os.path.join(current_dir, "histogram.png")
    var_name = create_histogram(df, save_path=histogram_path)

    profiles = None
    ordinal_profiles = None

    if basic_quant is not None:
        try:
            profiles = auto_kmeans(df_cleaned[basic_quant.columns])
        except Exception as e:
            print(f"⚠️  K-Means clustering failed: {e}")

    if basic_qual is not None:
        try:
            ordinal_profiles = auto_kmedoids(df_cleaned[basic_qual.columns])
        except Exception as e:
            print(f"⚠️  K-Medoids clustering failed: {e}")

    chi_square_results = None
    if basic_qual is not None:
        try:
            chi_square_results = chi_square_test(df_cleaned[basic_qual.columns])
        except Exception as e:
            print(f"⚠️  Chi-Square test failed: {e}")

    print(basic_quant)
    print(basic_qual)
    print(basic_date)

    suffix = recognise_domain(df)
    if isinstance(suffix, str):
        suffix = suffix.strip().lower()
    print("\nRecognised domain:", suffix)
    
    func_name = f"an_{suffix}"
    analysis = globals().get(func_name)

    # Initialize domain plots
    domain_plots = []
    
    if analysis is None:
        print(f"⚠️ Warning: No analysis function found for '{suffix}'")
        analysis_titles = []
        results = {}
    else:
        # Check if analysis function supports save_plots parameter
        sig = inspect.signature(analysis)
        params = sig.parameters
        current_dir = os.path.dirname(os.path.abspath(__file__))
        domain_plots_dir = os.path.join(current_dir, "domain_plots")  # Absolute fixed path

        # BONUS: Auto-clean old domain plots before generating new ones
        if os.path.exists(domain_plots_dir):
            shutil.rmtree(domain_plots_dir)
            print("🗑️  Cleaned old plots")
        
        # Call analysis with save_plots if supported
        if 'save_plots' in params:
            print(f"   → Enabling plot generation for {suffix} domain")
            if 'output_dir' in params:
                results = analysis(df_cleaned, user_objective=user_objective, 
                                 save_plots=True, output_dir=domain_plots_dir)
            else:
                results = analysis(df_cleaned, user_objective=user_objective, 
                                 save_plots=True)
        else:
            results = analysis(df_cleaned, user_objective=user_objective)
        
        results = protect_data(results, domain=suffix)
        
        # Collect domain plots automatically using graphs.py
        domain_plots = collect_domain_plots(suffix)
        
        analysis_titles = [k.replace('_', ' ').upper() for k in results.keys()]
        print(f"\n✅ Domain '{suffix}' processed successfully.")
        print("Analyses found:", analysis_titles)

    # Generate report with domain plots
    report = analizza_dataset(basic_quant, basic_qual, basic_date, corr_matrix, df_cleaned, 
                             domain=results, graphs=histogram_path, 
                             domain_plots=domain_plots,  
                             user_objective=user_objective, user_language=user_language)
    
    print("")
    print("")
    print("Report LLM (Cleaned):")
    print(report)

    # Outliers analysis
    has_outliers_analysis = False
    basic_quant_outliers = None
    corr_matrix_outliers = None
    basic_qual_outliers = None
    basic_date_outliers = None
    profiles_outliers = None
    ordinal_profiles_outliers = None
    chi_square_results_outliers = None
    outliers_report = None
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    domain_plots_dir = os.path.join(current_dir, "domain_plots")  

    if len(df_outliers) > 30:
        has_outliers_analysis = True
        
        basic_quant_outliers, corr_matrix_outliers = basic_analytics_quant(df_outliers, dict_vars)
        basic_qual_outliers = basic_analytics_qual(df_outliers, dict_vars)
        basic_date_outliers = basic_analytics_date(df_outliers, dict_vars)
        
        if basic_quant_outliers is not None:
            try:
                profiles_outliers = auto_kmeans(df_outliers[basic_quant_outliers.columns])
            except Exception as e:
                print(f"⚠️  K-Means clustering (outliers) failed: {e}")

        if basic_qual_outliers is not None:
            try:
                ordinal_profiles_outliers = auto_kmedoids(df_outliers[basic_qual_outliers.columns])
            except Exception as e:
                print(f"⚠️  K-Medoids clustering (outliers) failed: {e}")

        if basic_qual_outliers is not None:
            try:
                chi_square_results_outliers = chi_square_test(df_outliers[basic_qual_outliers.columns])
            except Exception as e:
                print(f"⚠️  Chi-Square test (outliers) failed: {e}")
        
        results_outliers = {}
        if analysis is not None:
            print(f" Domain analysis execution '{suffix}' on OUTLIERS subset")
            
            sig = inspect.signature(analysis)
            params = sig.parameters
        
            outliers_plots_dir = os.path.join(domain_plots_dir, "outliers")
            if os.path.exists(outliers_plots_dir):
                shutil.rmtree(outliers_plots_dir)
            os.makedirs(outliers_plots_dir, exist_ok=True)

            if 'save_plots' in params:
                if 'output_dir' in params:
                    results_outliers = analysis(df_outliers, user_objective=user_objective, 
                                              save_plots=True, output_dir=outliers_plots_dir)
                else:
                    results_outliers = analysis(df_outliers, user_objective=user_objective, 
                                              save_plots=True)
            else:
                results_outliers = analysis(df_outliers, user_objective=user_objective)

            results_outliers = protect_data(results_outliers, domain=suffix)


        print("\n=== OUTLIERS ANALYSIS ===")
        print(basic_quant_outliers)
        print(basic_qual_outliers)
        print(basic_date_outliers)
        
        outliers_report = analizza_outliers(
            report, basic_quant_outliers, basic_qual_outliers, basic_date_outliers,
            corr_matrix_outliers, df_outliers, profiles_outliers, chi_square_results_outliers,
            domain_results=results_outliers, 
            user_language=user_language
        )
        
        print("\n")
        print("Report LLM (Outliers):")
        print(outliers_report)
        
        report = report + "\n\n=== OUTLIERS ANALYSIS ===\n\n" + outliers_report

    return {
        "df": df,
        "dict_vars": dict_vars,
        "basic_quant": basic_quant,
        "basic_qual": basic_qual,
        "report": report,
        "suffix": suffix,
        "corr_matrix": corr_matrix,
        "result_specific": results,
        "cluster_profiles": profiles,
        "kmedoids_profiles": ordinal_profiles,
        "ordinal_profiles": ordinal_profiles,
        "histogram_path": histogram_path, 
        "histogram_var": var_name,
        "chi_square_results": chi_square_results,
        "domain_plots": domain_plots,  
        "has_outliers_analysis": has_outliers_analysis,
        "basic_quant_outliers": basic_quant_outliers,
        "basic_qual_outliers": basic_qual_outliers,
        "corr_matrix_outliers": corr_matrix_outliers,
        "cluster_profiles_outliers": profiles_outliers,
        "kmedoids_profiles_outliers": ordinal_profiles_outliers,
        "ordinal_profiles_outliers": ordinal_profiles_outliers,
        "chi_square_results_outliers": chi_square_results_outliers,
        "domain_analysis_results": results
    }