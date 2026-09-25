# Guida all'Integrazione della Privacy nelle Analisi

## Panoramica

Ogni file di analisi di dominio deve applicare **Differential Privacy** ai risultati prima di ritornarli per il report.

---

## Template Standard

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
    print(f"Running analysis for DOMAIN")
    
    # === CALCOLA LE TUE METRICHE ===
    results = {
        'metric_1': value1,
        'metric_2': value2,
        'metric_3': value3,
        'description': 'Domain specific analysis'
    }
    
    # === APPLICA PRIVACY (AUTOMATICO) ===
    protected_results = protect_data(results, domain='DOMAIN', epsilon=epsilon)
    
    return protected_results
```

---

## Passo per Passo

### 1. Importa la funzione privacy
```python
from data_manipulation.privacy import protect_data
```

### 2. Costruisci un dizionario `results` con i tuoi calcoli
```python
results = {
    'mean': df['column'].mean(),
    'median': df['column'].median(),
    'count': len(df),
    'analysis_date': '2026-02-07'
}
```

### 3. Applica privacy (1 riga!)
```python
protected_results = protect_data(results, domain='finance', epsilon=None)
```

### 4. Ritorna i risultati protetti
```python
return protected_results
```

---

## Dominio → Livello Privacy (Automatico)

| Livello | Epsilon | Domini |
|---------|---------|--------|
| **STRICT** | 3.0 | health, finance, hr, insurance, security |
| **MODERATE** | 5.0 | business_economics, demographics, ecommerce, education, energy, environment, industrial, logistics, macro_economics, marketing, politics, real_estate, retail, social_media, sports, supply_chain, surveys, telecommunications, tourism |
| **LIGHT** | 7.0 | agriculture, scientific_research |

---

## Metadati nel Risultato

Dopo `protect_data()`, il dizionario contiene anche:

```python
results = protect_data(results, domain='health')
# Risultato:
{
    'metric_1': 49832.15,              # ← Offuscato
    'metric_2': 48500.22,
    '_dp_applied': True,               # ← Privacy applicata?
    '_privacy_level': 'strict',        # ← Livello di privacy
    '_epsilon': 2.0                    # ← Budget usato
}
```

### Usa nei commenti LLM:
```python
if results.get('_dp_applied'):
    privacy_comment = f"Dati protetti con privacy level={results['_privacy_level']}"
```

---

## Esempi per Dominio

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

## Privacy Budget Personalizzato

Se vuoi un epsilon diverso:

```python
# Usa epsilon=10.0 per meno privacy (più precisione)
protected = protect_data(results, domain='health', epsilon=10.0)

# Usa epsilon=1.0 per più privacy (meno precisione)
protected = protect_data(results, domain='health', epsilon=1.0)
```

---

## Checklist per Ogni File

- [ ] Importato `protect_data` da `data_manipulation.privacy`
- [ ] Funzione ritorna un dizionario `results` con metriche numeriche
- [ ] Applicato `protect_data(results, domain='...')` prima di ritornare
- [ ] Testato che il risultato sia leggibile (non troppo rumore)
- [ ] Aggiunto parametro `epsilon` opzionale se vuoi override

---

## Uso nel Main

```python
from analystics.health import analytics_health
from analystics.finance import analytics_finance

# Le analisi ritornano risultati già protetti
health_results = analytics_health(df_health)
finance_results = analytics_finance(df_finance)

# I risultati contengono metadati di privacy
print(health_results['_privacy_level'])  # 'strict'
print(finance_results['_epsilon'])       # 2.0
```

---

## Domande Frequenti

**Q: Ma così i dati sono meno precisi!**
A: Sì, ma l'imprecisione è minima (~1-2% della media). È il compromesso privacy/precisione. L'azienda ottiene risultati usabili + privacy garantita.

**Q: Posso escludere la privacy?**
A: Per domini sconosciuti sì, ritorna `results` senza modifiche. Per dominio noto, `protect_data()` applica sempre.

**Q: L'LLM come commenta i dati protetti?**
A: Usa i metadati `_dp_applied`, `_privacy_level`, `_epsilon` nel report.

**Q: Le metriche possono diventare negative?**
A: Raramente con epsilon=2-5. Se succede, è una protezione più forte. Commenta nel report.
