# Privacy Integration Guide for Analytics

## Overview

Each domain analytics file should apply Differential Privacy to results before returning them to the reporting layer.

---

## Standard Template

```python
import pandas as pd
from data_manipulation.privacy import protect_data

def analytics_DOMAIN(df: pd.DataFrame, epsilon=None):
    """
    Perform analytics for the DOMAIN domain with privacy protection.

    Args:
        df (pd.DataFrame): Input data
        epsilon (float): Privacy budget (if None, uses domain default)

    Returns:
        dict: Results with differential privacy applied
    """
    print("Running analysis for DOMAIN")

    # === COMPUTE YOUR METRICS ===
    results = {
        'metric_1': value1,
        'metric_2': value2,
        'metric_3': value3,
        'description': 'Domain specific analysis'
    }

    # === APPLY PRIVACY (AUTOMATIC) ===
    protected_results = protect_data(results, domain='DOMAIN', epsilon=epsilon)

    return protected_results
```

---

## Step by Step

### 1. Import the privacy function

```python
from data_manipulation.privacy import protect_data
```

### 2. Build a `results` dictionary with your metrics

```python
results = {
    'mean': df['column'].mean(),
    'median': df['column'].median(),
    'count': len(df),
    'analysis_date': '2026-02-07'
}
```

### 3. Apply privacy (1 line)

```python
protected_results = protect_data(results, domain='finance', epsilon=None)
```

### 4. Return protected results

```python
return protected_results
```

---

## Domain -> Privacy Level (Automatic)

| Level | Epsilon | Domains |
|-------|---------|---------|
| STRICT | 3.0 | health, finance, hr, insurance, security |
| MODERATE | 5.0 | business_economics, demographics, ecommerce, education, energy, environment, industrial, logistics, macro_economics, marketing, politics, real_estate, retail, social_media, sports, supply_chain, surveys, telecommunications, tourism |
| LIGHT | 7.0 | agriculture, scientific_research |

---

## Metadata in the Result

After `protect_data()`, the dictionary may also contain metadata:

```python
results = protect_data(results, domain='health')
# Example:
{
    'metric_1': 49832.15,
    'metric_2': 48500.22,
    '_dp_applied': True,
    '_privacy_level': 'strict',
    '_epsilon': 2.0
}
```

### Use in LLM commentary:

```python
if results.get('_dp_applied'):
    privacy_comment = f"Data protected with privacy level={results['_privacy_level']}"
```

---

## Domain Examples

### Health

```python
results = {
    'patient_count': len(df),
    'avg_age': df['age'].mean(),
    'positive_cases': (df['status'] == 'positive').sum(),
}
protected = protect_data(results, domain='health')
```

### Finance

```python
results = {
    'total_revenue': df['amount'].sum(),
    'avg_transaction': df['amount'].mean(),
    'num_transactions': len(df),
}
protected = protect_data(results, domain='finance')
```

### Marketing

```python
results = {
    'conversion_rate': (df['converted'] == True).sum() / len(df) * 100,
    'avg_customer_value': df['spending'].mean(),
    'roi': df['roi'].mean(),
}
protected = protect_data(results, domain='marketing')
```

### Agriculture

```python
results = {
    'yield_per_hectare': df['yield'].mean(),
    'avg_moisture': df['moisture'].mean(),
    'crop_quality_score': df['quality'].mean(),
}
protected = protect_data(results, domain='agriculture')
```

---

## Custom Privacy Budget

If you need a custom epsilon:

```python
# Use epsilon=10.0 for less privacy (more precision)
protected = protect_data(results, domain='health', epsilon=10.0)

# Use epsilon=1.0 for more privacy (less precision)
protected = protect_data(results, domain='health', epsilon=1.0)
```

---

## Checklist for Each File

- [ ] Imported `protect_data` from `data_manipulation.privacy`
- [ ] Function returns a `results` dictionary with numeric metrics
- [ ] Applied `protect_data(results, domain='...')` before returning
- [ ] Verified output remains readable (not too noisy)
- [ ] Added optional `epsilon` parameter if override is needed

---

## Usage in Main

```python
from analystics.health import analytics_health
from analystics.finance import analytics_finance

# Analyses return already protected results
health_results = analytics_health(df_health)
finance_results = analytics_finance(df_finance)

# Results include privacy metadata
print(health_results['_privacy_level'])
print(finance_results['_epsilon'])
```

---

## FAQ

**Q: Doesn't this make data less precise?**
A: Yes, slightly. The tradeoff is intentional: strong privacy with still useful business insights.

**Q: Can I skip privacy?**
A: For unknown domains, yes: `protect_data` can return results unchanged. For known domains, privacy is applied automatically.

**Q: How should the LLM describe protected data?**
A: Use metadata such as `_dp_applied`, `_privacy_level`, and `_epsilon` in the narrative.

**Q: Can metrics become negative after noise?**
A: It can happen with stronger privacy settings. Handle it explicitly in the report interpretation.
