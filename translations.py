# translations.py
# Multilingual support for English and Italian

translations = {
    "en": {
        # Page config
        "page_title": "Inverse Edge Lab",
        "login_title": "🔄 Inverse Edge Lab - Login",
        "app_title": "🔄 Inverse Edge Lab",
        "app_subtitle": "*Discover the hidden edge in your trading inversions*",
        
        # Authentication
        "username": "Username",
        "password": "Password",
        "login": "Login",
        "logout": "Logout",
        "invalid_credentials": "Invalid username or password",
        
        # Sidebar
        "simulation_controls": "⚙️ Simulation Controls",
        "bracket_pairs": "Bracket Pairs (TP, SL)",
        "add_bracket": "➕ Add Bracket",
        "current_brackets": "Current brackets:",
        "lock_rr": "Lock Risk-Reward Ratio",
        "rr_ratio": "R:R Ratio",
        "optimize_metric": "Optimize for metric:",
        "run_advanced_tests": "Run advanced significance tests",
        "run_simulation": "🚀 Run Simulation",
        
        # Advanced tests explanation
        "advanced_tests_title": "📊 What are Advanced Significance Tests?",
        "advanced_tests_desc": """**Advanced tests validate if your edge is statistically significant or just luck:**
        
• **T-Test**: Tests if average P/L is significantly different from zero
  - Null hypothesis: Mean P/L = 0 (no edge)
  - P-value < 0.05 suggests real edge (not random)

• **Bootstrap Confidence Interval (95% CI)**: 
  - Resamples your trades 1000 times to estimate uncertainty
  - If CI doesn't include zero, edge is likely real
  - Example: CI [5.2, 12.8] means 95% confident true edge is between 5.2-12.8 points

• **Jarque-Bera Normality Test**:
  - Tests if returns follow normal distribution
  - Important for risk models that assume normality
  - P-value < 0.05 indicates non-normal distribution (common in trading)

**Why use these?** They help distinguish genuine edge from lucky streaks.""",
        
        # File upload
        "upload_files": "Upload trade logs",
        "loaded_trades": "Loaded {count} trades from {files} files",
        "file_label": "📁 File: {name}",
        "original_columns": "Original columns: {cols}",
        "normalized_columns": "Normalized columns: {cols}",
        "missing_columns": "❌ Missing required columns: {cols}",
        "columns_found": "✅ Found all required columns!",
        "column_mapping": "Column mapping:",
        "data_preview": "📊 Data Preview",
        
        # Mathematical framework
        "math_framework_title": "📐 Mathematical Framework & Simulation Logic",
        "math_framework_content": """## How Inverse Trading Simulation Works

### 1. Core Concept
We simulate taking the **opposite position** of each trader with fixed risk-reward brackets.

**Key Insight**: When you bet against a trader:
- Their **favorable movement** (run-up) → Your **adverse movement** (potential loss)
- Their **unfavorable movement** (drawdown) → Your **favorable movement** (potential profit)

### 2. Mathematical Formulation

For each trade `i` in the dataset:
- `P/L_original[i]` = Original trader's profit/loss
- `RP[i]` = Run-up (maximum favorable excursion for trader)
- `DD[i]` = Drawdown (maximum adverse excursion for trader)
- `D/R[i]` = Flag indicating which came first (RP or DD)

**Inverse Position Calculation**:
```
For our inverse position with brackets (TP, SL):

IF D/R[i] == "RP" (trader saw profit first):
    IF RP[i] ≥ SL:  # Their profit hits our stop loss
        P/L_inverse[i] = -SL
    ELIF DD[i] ≥ TP:  # Their loss hits our take profit
        P/L_inverse[i] = +TP
    ELSE:  # Neither bracket hit
        P/L_inverse[i] = -P/L_original[i]
        
ELIF D/R[i] == "DD" (trader saw loss first):
    IF DD[i] ≥ TP:  # Their loss hits our take profit
        P/L_inverse[i] = +TP
    ELIF RP[i] ≥ SL:  # Their profit hits our stop loss
        P/L_inverse[i] = -SL
    ELSE:  # Neither bracket hit
        P/L_inverse[i] = -P/L_original[i]
```

### 3. Key Metrics Calculated

**Basic Metrics**:
- **Total P/L** = Σ P/L_inverse[i]
- **Hit Rate** = Count(P/L_inverse > 0) / Total Trades
- **Expectancy** = (Hit Rate × Avg Win) - ((1 - Hit Rate) × Avg Loss)
- **Sharpe Ratio** = Mean(Returns) / StdDev(Returns) × √252
- **Profit Factor** = Gross Profits / Gross Losses

**Risk Metrics**:
- **Maximum Drawdown** = Largest peak-to-trough decline in equity curve
- **MAR Ratio** = Total P/L / Max Drawdown

### 4. Why This Works

If traders consistently lose money (negative edge), then:
- Taking opposite positions should yield positive expectancy
- Fixed brackets (TP/SL) can optimize this edge by:
  - Limiting losses when traders occasionally win big
  - Capturing consistent profits from their frequent losses

### 5. Example Scenario

**Original Trade**: Trader loses 15 points
- Run-up: 5 points (RP)
- Drawdown: 20 points (DD)
- D/R Flag: "RP" (saw 5 point profit before 20 point loss)

**Our Inverse Trade** with TP=10, SL=10:
- Trader's 5 point run-up = Our 5 point drawdown (< 10 SL, continues)
- Trader's 20 point drawdown = Our 20 point run-up (> 10 TP, locked in)
- **Result**: We make +10 points (TP hit)

While simple inversion would gain +15, the bracket system:
- Protects against occasional large trader wins
- Provides consistent, controlled profits""",
        
        # Simulation
        "running_simulations": "Running simulations...",
        "simulation_results": "📈 Simulation Results",
        "how_inverse_works_title": "ℹ️ How Inverse Trading Simulation Works",
        "how_inverse_works": """**Inverse Trading Strategy**: We bet AGAINST each trader's position.

**Key Concepts**:
- When a trader goes LONG, we go SHORT (and vice versa)
- The trader's **run-up** (favorable movement) becomes our **adverse movement**
- The trader's **drawdown** (unfavorable movement) becomes our **favorable movement**

**Simulation Logic**:
1. For each trade, we take the opposite position
2. We apply fixed Take Profit (TP) and Stop Loss (SL) brackets
3. Based on the D/R flag, we know which movement happened first:
   - **RP (Run-up first)**: Trader saw profit first → We face loss first
   - **DD (Drawdown first)**: Trader saw loss first → We see profit first
4. We check if our TP or SL gets hit based on these movements

**Example**: If a trader lost 20 points, our inverse position would gain 20 points 
(unless limited by our TP bracket).""",
        
        # Tabs
        "basic_analysis_tab": "📊 Basic Analysis",
        "advanced_stats_tab": "🔬 Advanced Statistics",
        "basic_metrics_title": "### 📊 Basic Performance Metrics",
        "statistical_tests_title": "### 🔬 Statistical Significance Tests",
        
        # Results
        "best_bracket": "🎯 Best bracket: TP={tp}, SL={sl} (optimized for {metric}: {value:.2f})",
        "original_pl": "Original Traders' P/L",
        "simple_inverse_pl": "Simple Inverse P/L",
        "optimized_inverse_pl": "Optimized Inverse P/L",
        "visualizations": "### 📊 Visualizations",
        "pl_distribution": "P/L Distribution (TP={tp}, SL={sl})",
        "equity_curve": "Equity Curve (TP={tp}, SL={sl})",
        "trade_number": "Trade Number",
        "cumulative_pl": "Cumulative P/L",
        "frequency": "Frequency",
        "mean_label": "Mean: {value:.2f}",
        
        # Statistical significance
        "best_bracket_analysis": "### 📊 Best Bracket Statistical Analysis",
        "edge_significance": "**Edge Significance (T-Test)**",
        "highly_significant": "✅ Highly significant (p={p:.4f} < 0.01)",
        "significant": "⚠️ Significant (p={p:.4f} < 0.05)",
        "not_significant": "❌ Not significant (p={p:.4f} ≥ 0.05)",
        "very_strong_evidence": "Very strong evidence of real edge",
        "good_evidence": "Good evidence of real edge",
        "insufficient_evidence": "Insufficient evidence of edge (could be random)",
        "confidence_interval": "**95% Confidence Interval**",
        "true_edge_between": "True edge likely between: {ci}",
        "ci_excludes_zero": "✅ CI excludes zero - edge likely real",
        "ci_negative": "❌ CI entirely negative - inverse strategy losing",
        "ci_includes_zero": "⚠️ CI includes zero - edge uncertain",
        "interval_width": "**Interval width**: {width:.2f} points",
        "narrower_intervals": "Narrower intervals indicate more precise estimates.",
        "distribution_normality": "**Distribution Normality (Jarque-Bera Test)**",
        "non_normal_returns": "Returns are non-normal (p={p:.4f}). Consider using robust risk metrics.",
        "normal_returns": "Returns appear normally distributed (p={p:.4f})",
        
        # Bootstrap
        "bootstrap_title": "### 🎲 Bootstrap Analysis Details",
        "bootstrap_desc": """The bootstrap confidence interval was calculated by:
1. Resampling {count} trades with replacement
2. Calculating mean P/L for each resample
3. Repeating 1,000 times
4. Taking the 2.5th and 97.5th percentiles

This provides a robust estimate of uncertainty without assuming normal distribution.""",
        
        # Export
        "download_csv": "📥 Download Results CSV",
        
        # AI Assistant
        "ai_assistant": "🤖 AI Assistant",
        "ai_info": "Ask questions about your simulation results - powered by Perplexity AI!",
        "ai_prompt": "What would you like to know about the results?",
        "run_simulation_first": "Please run a simulation first so I can analyze your results!",
        
        # Column names
        "col_tp": "TP",
        "col_sl": "SL",
        "col_total_pl": "Total P/L",
        "col_avg_pl": "Avg P/L",
        "col_hit_rate": "Hit Rate",
        "col_payoff_ratio": "Payoff Ratio",
        "col_expectancy": "Expectancy",
        "col_sharpe": "Sharpe Ratio",
        "col_max_dd": "Max DD",
        "col_mar": "MAR",
        "col_profit_factor": "Profit Factor",
        "col_tstat": "T-stat",
        "col_pvalue": "P-value",
        "col_ci95": "CI 95%",
        "col_jb_pvalue": "JB p-value",
    },
    
    "it": {
        # Page config
        "page_title": "Laboratorio Edge Inverso",
        "login_title": "🔄 Laboratorio Edge Inverso - Accesso",
        "app_title": "🔄 Laboratorio Edge Inverso",
        "app_subtitle": "*Scopri l'edge nascosto nelle tue inversioni di trading*",
        
        # Authentication
        "username": "Nome utente",
        "password": "Password",
        "login": "Accedi",
        "logout": "Esci",
        "invalid_credentials": "Nome utente o password non validi",
        
        # Sidebar
        "simulation_controls": "⚙️ Controlli Simulazione",
        "bracket_pairs": "Coppie Bracket (TP, SL)",
        "add_bracket": "➕ Aggiungi Bracket",
        "current_brackets": "Bracket attuali:",
        "lock_rr": "Blocca Rapporto Rischio-Rendimento",
        "rr_ratio": "Rapporto R:R",
        "optimize_metric": "Ottimizza per metrica:",
        "run_advanced_tests": "Esegui test di significatività avanzati",
        "run_simulation": "🚀 Avvia Simulazione",
        
        # Advanced tests explanation
        "advanced_tests_title": "📊 Cosa sono i Test di Significatività Avanzati?",
        "advanced_tests_desc": """**I test avanzati verificano se il tuo edge è statisticamente significativo o solo fortuna:**
        
• **Test T**: Verifica se il P/L medio è significativamente diverso da zero
  - Ipotesi nulla: P/L medio = 0 (nessun edge)
  - P-value < 0.05 suggerisce edge reale (non casuale)

• **Intervallo di Confidenza Bootstrap (95% IC)**: 
  - Ricampiona le tue operazioni 1000 volte per stimare l'incertezza
  - Se l'IC non include zero, l'edge è probabilmente reale
  - Esempio: IC [5.2, 12.8] significa 95% di confidenza che il vero edge sia tra 5.2-12.8 punti

• **Test di Normalità Jarque-Bera**:
  - Verifica se i rendimenti seguono una distribuzione normale
  - Importante per modelli di rischio che assumono normalità
  - P-value < 0.05 indica distribuzione non normale (comune nel trading)

**Perché usarli?** Aiutano a distinguere un edge genuino da serie fortunate.""",
        
        # File upload
        "upload_files": "Carica log di trading",
        "loaded_trades": "Caricate {count} operazioni da {files} file",
        "file_label": "📁 File: {name}",
        "original_columns": "Colonne originali: {cols}",
        "normalized_columns": "Colonne normalizzate: {cols}",
        "missing_columns": "❌ Colonne mancanti richieste: {cols}",
        "columns_found": "✅ Trovate tutte le colonne richieste!",
        "column_mapping": "Mappatura colonne:",
        "data_preview": "📊 Anteprima Dati",
        
        # Mathematical framework
        "math_framework_title": "📐 Framework Matematico e Logica di Simulazione",
        "math_framework_content": """## Come Funziona la Simulazione di Trading Inverso

### 1. Concetto Base
Simuliamo l'apertura della **posizione opposta** di ogni trader con bracket rischio-rendimento fissi.

**Intuizione Chiave**: Quando scommetti contro un trader:
- Il suo **movimento favorevole** (run-up) → Il tuo **movimento avverso** (potenziale perdita)
- Il suo **movimento sfavorevole** (drawdown) → Il tuo **movimento favorevole** (potenziale profitto)

### 2. Formulazione Matematica

Per ogni operazione `i` nel dataset:
- `P/L_originale[i]` = Profitto/perdita del trader originale
- `RP[i]` = Run-up (massima escursione favorevole per il trader)
- `DD[i]` = Drawdown (massima escursione avversa per il trader)
- `D/R[i]` = Flag che indica cosa è venuto prima (RP o DD)

**Calcolo Posizione Inversa**:
```
Per la nostra posizione inversa con bracket (TP, SL):

SE D/R[i] == "RP" (il trader ha visto prima il profitto):
    SE RP[i] ≥ SL:  # Il suo profitto colpisce il nostro stop loss
        P/L_inverso[i] = -SL
    ALTRIMENTI SE DD[i] ≥ TP:  # La sua perdita colpisce il nostro take profit
        P/L_inverso[i] = +TP
    ALTRIMENTI:  # Nessun bracket colpito
        P/L_inverso[i] = -P/L_originale[i]
        
ALTRIMENTI SE D/R[i] == "DD" (il trader ha visto prima la perdita):
    SE DD[i] ≥ TP:  # La sua perdita colpisce il nostro take profit
        P/L_inverso[i] = +TP
    ALTRIMENTI SE RP[i] ≥ SL:  # Il suo profitto colpisce il nostro stop loss
        P/L_inverso[i] = -SL
    ALTRIMENTI:  # Nessun bracket colpito
        P/L_inverso[i] = -P/L_originale[i]
```

### 3. Metriche Chiave Calcolate

**Metriche Base**:
- **P/L Totale** = Σ P/L_inverso[i]
- **Hit Rate** = Conteggio(P/L_inverso > 0) / Operazioni Totali
- **Expectancy** = (Hit Rate × Media Vincite) - ((1 - Hit Rate) × Media Perdite)
- **Sharpe Ratio** = Media(Rendimenti) / DevStd(Rendimenti) × √252
- **Profit Factor** = Profitti Lordi / Perdite Lorde

**Metriche di Rischio**:
- **Drawdown Massimo** = Maggior declino da picco a valle nella curva equity
- **MAR Ratio** = P/L Totale / Drawdown Massimo

### 4. Perché Funziona

Se i trader perdono costantemente denaro (edge negativo), allora:
- Prendere posizioni opposte dovrebbe produrre expectancy positiva
- I bracket fissi (TP/SL) possono ottimizzare questo edge:
  - Limitando le perdite quando i trader occasionalmente vincono molto
  - Catturando profitti consistenti dalle loro frequenti perdite

### 5. Scenario di Esempio

**Operazione Originale**: Il trader perde 15 punti
- Run-up: 5 punti (RP)
- Drawdown: 20 punti (DD)
- Flag D/R: "RP" (ha visto 5 punti di profitto prima di 20 punti di perdita)

**La Nostra Operazione Inversa** con TP=10, SL=10:
- Run-up del trader di 5 punti = Nostro drawdown di 5 punti (< 10 SL, continua)
- Drawdown del trader di 20 punti = Nostro run-up di 20 punti (> 10 TP, bloccato)
- **Risultato**: Guadagniamo +10 punti (TP colpito)

Mentre l'inversione semplice guadagnerebbe +15, il sistema a bracket:
- Protegge contro occasionali grandi vincite del trader
- Fornisce profitti consistenti e controllati""",
        
        # Simulation
        "running_simulations": "Esecuzione simulazioni...",
        "simulation_results": "📈 Risultati Simulazione",
        "how_inverse_works_title": "ℹ️ Come Funziona la Simulazione di Trading Inverso",
        "how_inverse_works": """**Strategia di Trading Inverso**: Scommettiamo CONTRO la posizione di ogni trader.

**Concetti Chiave**:
- Quando un trader va LONG, noi andiamo SHORT (e viceversa)
- Il **run-up** del trader (movimento favorevole) diventa il nostro **movimento avverso**
- Il **drawdown** del trader (movimento sfavorevole) diventa il nostro **movimento favorevole**

**Logica di Simulazione**:
1. Per ogni operazione, prendiamo la posizione opposta
2. Applichiamo bracket fissi di Take Profit (TP) e Stop Loss (SL)
3. Basandoci sul flag D/R, sappiamo quale movimento è avvenuto prima:
   - **RP (Run-up prima)**: Il trader ha visto prima il profitto → Noi affrontiamo prima la perdita
   - **DD (Drawdown prima)**: Il trader ha visto prima la perdita → Noi vediamo prima il profitto
4. Verifichiamo se il nostro TP o SL viene colpito basandoci su questi movimenti

**Esempio**: Se un trader ha perso 20 punti, la nostra posizione inversa guadagnerebbe 20 punti 
(a meno che non sia limitata dal nostro bracket TP).""",
        
        # Tabs
        "basic_analysis_tab": "📊 Analisi Base",
        "advanced_stats_tab": "🔬 Statistiche Avanzate",
        "basic_metrics_title": "### 📊 Metriche di Performance Base",
        "statistical_tests_title": "### 🔬 Test di Significatività Statistica",
        
        # Results
        "best_bracket": "🎯 Miglior bracket: TP={tp}, SL={sl} (ottimizzato per {metric}: {value:.2f})",
        "original_pl": "P/L Trader Originali",
        "simple_inverse_pl": "P/L Inverso Semplice",
        "optimized_inverse_pl": "P/L Inverso Ottimizzato",
        "visualizations": "### 📊 Visualizzazioni",
        "pl_distribution": "Distribuzione P/L (TP={tp}, SL={sl})",
        "equity_curve": "Curva Equity (TP={tp}, SL={sl})",
        "trade_number": "Numero Operazione",
        "cumulative_pl": "P/L Cumulativo",
        "frequency": "Frequenza",
        "mean_label": "Media: {value:.2f}",
        
        # Statistical significance
        "best_bracket_analysis": "### 📊 Analisi Statistica del Miglior Bracket",
        "edge_significance": "**Significatività dell'Edge (Test T)**",
        "highly_significant": "✅ Altamente significativo (p={p:.4f} < 0.01)",
        "significant": "⚠️ Significativo (p={p:.4f} < 0.05)",
        "not_significant": "❌ Non significativo (p={p:.4f} ≥ 0.05)",
        "very_strong_evidence": "Evidenza molto forte di edge reale",
        "good_evidence": "Buona evidenza di edge reale",
        "insufficient_evidence": "Evidenza insufficiente di edge (potrebbe essere casuale)",
        "confidence_interval": "**Intervallo di Confidenza al 95%**",
        "true_edge_between": "Edge vero probabilmente tra: {ci}",
        "ci_excludes_zero": "✅ IC esclude zero - edge probabilmente reale",
        "ci_negative": "❌ IC interamente negativo - strategia inversa in perdita",
        "ci_includes_zero": "⚠️ IC include zero - edge incerto",
        "interval_width": "**Ampiezza intervallo**: {width:.2f} punti",
        "narrower_intervals": "Intervalli più stretti indicano stime più precise.",
        "distribution_normality": "**Normalità della Distribuzione (Test Jarque-Bera)**",
        "non_normal_returns": "I rendimenti sono non normali (p={p:.4f}). Considera metriche di rischio robuste.",
        "normal_returns": "I rendimenti appaiono distribuiti normalmente (p={p:.4f})",
        
        # Bootstrap
        "bootstrap_title": "### 🎲 Dettagli Analisi Bootstrap",
        "bootstrap_desc": """L'intervallo di confidenza bootstrap è stato calcolato:
1. Ricampionando {count} operazioni con reinserimento
2. Calcolando il P/L medio per ogni ricampionamento
3. Ripetendo 1.000 volte
4. Prendendo il 2.5° e 97.5° percentile

Questo fornisce una stima robusta dell'incertezza senza assumere distribuzione normale.""",
        
        # Export
        "download_csv": "📥 Scarica CSV Risultati",
        
        # AI Assistant
        "ai_assistant": "🤖 Assistente AI",
        "ai_info": "Fai domande sui tuoi risultati di simulazione - powered by Perplexity AI!",
        "ai_prompt": "Cosa vorresti sapere sui risultati?",
        "run_simulation_first": "Per favore esegui prima una simulazione così posso analizzare i tuoi risultati!",
        
        # Column names
        "col_tp": "TP",
        "col_sl": "SL",
        "col_total_pl": "P/L Totale",
        "col_avg_pl": "P/L Medio",
        "col_hit_rate": "Hit Rate",
        "col_payoff_ratio": "Rapporto Payoff",
        "col_expectancy": "Expectancy",
        "col_sharpe": "Sharpe Ratio",
        "col_max_dd": "DD Max",
        "col_mar": "MAR",
        "col_profit_factor": "Profit Factor",
        "col_tstat": "Stat-T",
        "col_pvalue": "P-value",
        "col_ci95": "IC 95%",
        "col_jb_pvalue": "JB p-value",
    }
}

def get_text(key, lang="en", **kwargs):
    """Get translated text with optional formatting"""
    text = translations.get(lang, {}).get(key, translations["en"].get(key, key))
    if kwargs:
        return text.format(**kwargs)
    return text