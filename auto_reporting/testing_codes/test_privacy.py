"""
Test per la funzione protect_data con Global Differential Privacy
"""

from data_manipulation.privacy import protect_data, should_apply_dp

print("=" * 70)
print("TEST: Global Differential Privacy per Domini")
print("=" * 70)

# Test 1: Verifichiamo il filtro per ogni dominio
print("\n[TEST 1] Filtro di sensibilità per dominio:")
print("-" * 70)

domains_to_test = [
    'health', 'finance', 'hr', 'insurance', 'security',  # STRICT
    'marketing', 'ecommerce', 'business_economics',      # MODERATE
    'agriculture', 'scientific_research',                # LIGHT
    'unknown_domain'                                      # NONE
]

for domain in domains_to_test:
    apply_dp, epsilon, level = should_apply_dp(domain)
    status = "✓ DP" if apply_dp else "✗ NO DP"
    print(f"  {domain:25} → {status:10} (level={level:10}, ε={epsilon})")

# Test 2: Protezione dati Health (STRICT)
print("\n[TEST 2] Protezione HEALTH (STRICT - epsilon=0.5):")
print("-" * 70)

health_results = {
    'patient_count': 1000,
    'avg_age': 52.5,
    'avg_heart_rate': 78.3,
    'positive_cases': 120,
    'analysis_date': '2026-02-07'
}

print("Prima della protezione:")
for key, val in health_results.items():
    print(f"  {key}: {val}")

protected_health = protect_data(health_results, domain='health')

print("\nDopo la protezione:")
for key, val in protected_health.items():
    if isinstance(val, float):
        print(f"  {key}: {val:.2f}")
    else:
        print(f"  {key}: {val}")

# Test 3: Protezione dati Marketing (MODERATE)
print("\n[TEST 3] Protezione MARKETING (MODERATE - epsilon=1.0):")
print("-" * 70)

marketing_results = {
    'conversion_rate': 3.5,
    'avg_customer_value': 250.0,
    'click_through_rate': 2.1,
    'campaign_name': 'Winter Campaign'
}

print("Prima della protezione:")
for key, val in marketing_results.items():
    print(f"  {key}: {val}")

protected_marketing = protect_data(marketing_results, domain='marketing')

print("\nDopo la protezione:")
for key, val in protected_marketing.items():
    if isinstance(val, float):
        print(f"  {key}: {val:.2f}")
    else:
        print(f"  {key}: {val}")

# Test 4: Protezione dati Agriculture (LIGHT)
print("\n[TEST 4] Protezione AGRICULTURE (LIGHT - epsilon=2.0):")
print("-" * 70)

agriculture_results = {
    'yield_per_hectare': 8500.0,
    'moisture_level': 45.2,
    'crop_type': 'wheat'
}

print("Prima della protezione:")
for key, val in agriculture_results.items():
    print(f"  {key}: {val}")

protected_agriculture = protect_data(agriculture_results, domain='agriculture')

print("\nDopo la protezione:")
for key, val in protected_agriculture.items():
    if isinstance(val, float):
        print(f"  {key}: {val:.2f}")
    else:
        print(f"  {key}: {val}")

# Test 5: Dominio sconosciuto (no DP)
print("\n[TEST 5] Dominio sconosciuto (NO DP):")
print("-" * 70)

unknown_results = {
    'metric_1': 100.0,
    'metric_2': 200.0
}

print("Prima della protezione:")
for key, val in unknown_results.items():
    print(f"  {key}: {val}")

protected_unknown = protect_data(unknown_results, domain='unknown_domain')

print("\nDopo la protezione (dovrebbe essere identico):")
for key, val in protected_unknown.items():
    print(f"  {key}: {val}")

# Test 6: Override epsilon personalizzato
print("\n[TEST 6] Override con epsilon personalizzato (health con ε=2.0):")
print("-" * 70)

health_override = {
    'patients': 500,
    'avg_bpm': 72.0
}

protected_override = protect_data(health_override, domain='health', epsilon=2.0)

print("Risultati con epsilon=2.0 (più rumore, meno privacy):")
for key, val in protected_override.items():
    if isinstance(val, float):
        print(f"  {key}: {val:.2f}")
    else:
        print(f"  {key}: {val}")

print("\n" + "=" * 70)
print("✓ Test completati!")
print("=" * 70)
