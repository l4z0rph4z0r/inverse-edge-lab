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
We simulate taking the **opposite position** of each trader with fixed risk-reward brackets, accounting for real trading costs.

**Key Insight**: When you bet against a trader:
- Their **favorable movement** (run-up) → Your **adverse movement** (potential loss)
- Their **unfavorable movement** (drawdown) → Your **favorable movement** (potential profit)

### 2. Mathematical Formulation

For each trade `i` in the dataset:
- `P/L_original[i]` = Original trader's profit/loss (in points)
- `RP[i]` = Run-up (maximum favorable excursion for trader)
- `DD[i]` = Drawdown (maximum adverse excursion for trader)
- `D/R[i]` = Flag indicating which came first (RP or DD)

**Step 1: Inverse Position Calculation (Gross P/L)**
```
For our inverse position with brackets (TP, SL):

IF D/R[i] == "RP" (trader saw profit first):
    IF RP[i] ≥ SL:  # Their profit hits our stop loss
        Gross_P/L[i] = -SL points
    ELIF DD[i] ≥ TP:  # Their loss hits our take profit
        Gross_P/L[i] = +TP points
    ELSE:  # Neither bracket hit
        Gross_P/L[i] = -P/L_original[i] points
        
ELIF D/R[i] == "DD" (trader saw loss first):
    IF DD[i] ≥ TP:  # Their loss hits our take profit
        Gross_P/L[i] = +TP points
    ELIF RP[i] ≥ SL:  # Their profit hits our stop loss
        Gross_P/L[i] = -SL points
    ELSE:  # Neither bracket hit
        Gross_P/L[i] = -P/L_original[i] points
```

**Step 2: Commission Calculation**
```
Commission_per_trade[i] = Commission_per_side × 2 × Contracts
                        = (Entry commission + Exit commission) × Contracts

Net_P/L[i] = Gross_P/L[i] - Commission_per_trade[i]
```

### 3. Complete Example with Commissions

**Setup**:
- Contract: 1 MNQ (Micro E-mini Nasdaq)
- Point value: $2 per point
- Commission: $2.25 per side ($4.50 round trip)
- Bracket: TP=20 points, SL=10 points

**Trade Example**:
1. Original trader loses 15 points
2. Our inverse position gains +15 points gross
3. But TP=20 not hit, so gross P/L = +15 points
4. Dollar gross P/L = 15 points × $2 = $30
5. Commission = $4.50 (round trip)
6. **Net P/L = $30 - $4.50 = $25.50 (12.75 points net)**

### 4. Impact on Key Metrics

All performance metrics use **Net P/L** after commissions:

**Basic Metrics**:
- **Total Net P/L** = Σ (Gross_P/L[i] - Commission[i])
- **Hit Rate** = Count(Net_P/L > 0) / Total Trades
- **Avg Win/Loss** = Based on net values after commission
- **Expectancy** = (Hit Rate × Avg Net Win) - ((1 - Hit Rate) × Avg Net Loss)
- **Sharpe Ratio** = Mean(Net Returns) / StdDev(Net Returns) × √252
- **Profit Factor** = Gross Net Profits / Gross Net Losses

**Commission Impact Analysis**:
- **Total Commission** = Commission_per_trade × Number of trades
- **Commission Impact %** = (Total Commission / |Gross P/L|) × 100
- **Break-even threshold** = Commission_per_trade in points

### 5. Why Commission Matters

**Critical for Small Edges**:
- A 5-point average win becomes 2.75 points after $4.50 commission
- High-frequency strategies are more impacted by commissions
- Win rate requirements increase to overcome commission drag

**Example Impact**:
- Without commission: 60% win rate, avg win/loss = 1:1 → Positive expectancy
- With $4.50 commission: Same setup might have negative expectancy
- Need higher win rate or better risk/reward to overcome costs

### 6. Optimization Considerations

The simulation finds the optimal TP/SL brackets considering:
1. **Gross edge** from inverse positions
2. **Commission drag** on every trade
3. **Net profitability** after all costs

This ensures your backtesting reflects real trading conditions and helps identify truly profitable configurations.""",
        
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
        "download_pdf": "📄 Download PDF Report",
        "report_date": "Report Generated",
        "best_config_title": "Best Configuration",
        "all_results_title": "Top 10 Results Summary",
        "advanced_stats_title": "Statistical Significance Tests",
        "charts_title": "Performance Charts",
        "top_configs_title": "Top 5 Configurations by Total P/L",
        "heatmap_title": "P/L Heatmap by TP/SL Configuration",
        
        # AI Assistant
        "ai_assistant": "🤖 AI Assistant",
        "ai_info": "Ask questions about your simulation results - powered by Perplexity AI!",
        "ai_prompt": "What would you like to know about the results?",
        "run_simulation_first": "Please run a simulation first so I can analyze your results!",
        "api_key_missing": "⚠️ AI Assistant API key not configured",
        
        # Commission settings
        "commission_settings": "💰 Commission Settings",
        "contract_size": "Contract Size",
        "contract_size_help": "Number of contracts per trade (e.g., 1 for MNQ)",
        "commission_per_side": "Commission per Side ($)",
        "commission_help": "Commission charged per contract per side (entry or exit)",
        "total_commission_info": "Total commission per round trip: ${total:.2f}",
        
        # Commission impact
        "commission_impact_title": "📊 Commission Impact Analysis",
        "gross_pl_metric": "Gross P/L",
        "total_commission_metric": "Total Commissions",
        "net_pl_metric": "Net P/L",
        "commission_impact_pct": "Commission Impact",
        "commission_impact_info": "Trading {contracts} contract(s) with ${commission_per_rt:.2f} commission per round trip across {total_trades} trades",
        
        # Point value
        "point_value": "Point Value ($)",
        "point_value_help": "Dollar value per point (e.g., $2 for MNQ, $5 for MES, $50 for ES)",
        
        # Column names
        "col_tp": "TP",
        "col_sl": "SL",
        "col_gross_pl": "Gross P/L",
        "col_commission": "Commission",
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
Simuliamo l'apertura della **posizione opposta** di ogni trader con bracket rischio-rendimento fissi, tenendo conto dei costi di trading reali.

**Intuizione Chiave**: Quando scommetti contro un trader:
- Il suo **movimento favorevole** (run-up) → Il tuo **movimento avverso** (potenziale perdita)
- Il suo **movimento sfavorevole** (drawdown) → Il tuo **movimento favorevole** (potenziale profitto)

### 2. Formulazione Matematica

Per ogni operazione `i` nel dataset:
- `P/L_originale[i]` = Profitto/perdita del trader originale (in punti)
- `RP[i]` = Run-up (massima escursione favorevole per il trader)
- `DD[i]` = Drawdown (massima escursione avversa per il trader)
- `D/R[i]` = Flag che indica cosa è venuto prima (RP o DD)

**Passo 1: Calcolo Posizione Inversa (P/L Lordo)**
```
Per la nostra posizione inversa con bracket (TP, SL):

SE D/R[i] == "RP" (il trader ha visto prima il profitto):
    SE RP[i] ≥ SL:  # Il suo profitto colpisce il nostro stop loss
        P/L_Lordo[i] = -SL punti
    ALTRIMENTI SE DD[i] ≥ TP:  # La sua perdita colpisce il nostro take profit
        P/L_Lordo[i] = +TP punti
    ALTRIMENTI:  # Nessun bracket colpito
        P/L_Lordo[i] = -P/L_originale[i] punti
        
ALTRIMENTI SE D/R[i] == "DD" (il trader ha visto prima la perdita):
    SE DD[i] ≥ TP:  # La sua perdita colpisce il nostro take profit
        P/L_Lordo[i] = +TP punti
    ALTRIMENTI SE RP[i] ≥ SL:  # Il suo profitto colpisce il nostro stop loss
        P/L_Lordo[i] = -SL punti
    ALTRIMENTI:  # Nessun bracket colpito
        P/L_Lordo[i] = -P/L_originale[i] punti
```

**Passo 2: Calcolo Commissioni**
```
Commissione_per_operazione[i] = Commissione_per_lato × 2 × Contratti
                               = (Commissione entrata + Commissione uscita) × Contratti

P/L_Netto[i] = P/L_Lordo[i] - Commissione_per_operazione[i]
```

### 3. Esempio Completo con Commissioni

**Configurazione**:
- Contratto: 1 MNQ (Micro E-mini Nasdaq)
- Valore per punto: $2 per punto
- Commissione: $2.25 per lato ($4.50 andata e ritorno)
- Bracket: TP=20 punti, SL=10 punti

**Esempio di Operazione**:
1. Il trader originale perde 15 punti
2. La nostra posizione inversa guadagna +15 punti lordi
3. Ma TP=20 non colpito, quindi P/L lordo = +15 punti
4. P/L lordo in dollari = 15 punti × $2 = $30
5. Commissione = $4.50 (andata e ritorno)
6. **P/L Netto = $30 - $4.50 = $25.50 (12.75 punti netti)**

### 4. Impatto sulle Metriche Chiave

Tutte le metriche di performance usano **P/L Netto** dopo le commissioni:

**Metriche Base**:
- **P/L Netto Totale** = Σ (P/L_Lordo[i] - Commissione[i])
- **Hit Rate** = Conteggio(P/L_Netto > 0) / Operazioni Totali
- **Media Vincite/Perdite** = Basata su valori netti dopo commissioni
- **Expectancy** = (Hit Rate × Media Vincite Nette) - ((1 - Hit Rate) × Media Perdite Nette)
- **Sharpe Ratio** = Media(Rendimenti Netti) / DevStd(Rendimenti Netti) × √252
- **Profit Factor** = Profitti Netti Lordi / Perdite Nette Lorde

**Analisi Impatto Commissioni**:
- **Commissione Totale** = Commissione_per_operazione × Numero di operazioni
- **Impatto Commissioni %** = (Commissione Totale / |P/L Lordo|) × 100
- **Soglia di pareggio** = Commissione_per_operazione in punti

### 5. Perché le Commissioni Contano

**Critiche per Edge Piccoli**:
- Una vincita media di 5 punti diventa 2.75 punti dopo $4.50 di commissione
- Le strategie ad alta frequenza sono più impattate dalle commissioni
- I requisiti di win rate aumentano per superare il peso delle commissioni

**Esempio di Impatto**:
- Senza commissioni: 60% win rate, rapporto win/loss = 1:1 → Expectancy positiva
- Con $4.50 commissione: Stessa configurazione potrebbe avere expectancy negativa
- Serve win rate più alto o miglior risk/reward per superare i costi

### 6. Considerazioni per l'Ottimizzazione

La simulazione trova i bracket TP/SL ottimali considerando:
1. **Edge lordo** dalle posizioni inverse
2. **Peso delle commissioni** su ogni operazione
3. **Profittabilità netta** dopo tutti i costi

Questo assicura che il tuo backtesting rifletta le condizioni di trading reali e aiuta a identificare configurazioni veramente profittevoli.""",
        
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
        "download_pdf": "📄 Scarica Report PDF",
        "report_date": "Report Generato",
        "best_config_title": "Migliore Configurazione",
        "all_results_title": "Riepilogo Top 10 Risultati",
        "advanced_stats_title": "Test di Significatività Statistica",
        "charts_title": "Grafici delle Prestazioni",
        "top_configs_title": "Top 5 Configurazioni per P/L Totale",
        "heatmap_title": "Heatmap P/L per Configurazione TP/SL",
        
        # AI Assistant
        "ai_assistant": "🤖 Assistente AI",
        "ai_info": "Fai domande sui tuoi risultati di simulazione - powered by Perplexity AI!",
        "ai_prompt": "Cosa vorresti sapere sui risultati?",
        "run_simulation_first": "Per favore esegui prima una simulazione così posso analizzare i tuoi risultati!",
        "api_key_missing": "⚠️ Chiave API Assistente AI non configurata",
        
        # Commission settings
        "commission_settings": "💰 Impostazioni Commissioni",
        "contract_size": "Dimensione Contratto",
        "contract_size_help": "Numero di contratti per operazione (es. 1 per MNQ)",
        "commission_per_side": "Commissione per Lato ($)",
        "commission_help": "Commissione addebitata per contratto per lato (entrata o uscita)",
        "total_commission_info": "Commissione totale andata e ritorno: ${total:.2f}",
        
        # Commission impact
        "commission_impact_title": "📊 Analisi Impatto Commissioni",
        "gross_pl_metric": "P/L Lordo",
        "total_commission_metric": "Commissioni Totali",
        "net_pl_metric": "P/L Netto",
        "commission_impact_pct": "Impatto Commissioni",
        "commission_impact_info": "Trading di {contracts} contratto/i con ${commission_per_rt:.2f} di commissione andata e ritorno su {total_trades} operazioni",
        
        # Point value
        "point_value": "Valore per Punto ($)",
        "point_value_help": "Valore in dollari per punto (es. $2 per MNQ, $5 per MES, $50 per ES)",
        
        # Column names
        "col_tp": "TP",
        "col_sl": "SL",
        "col_gross_pl": "P/L Lordo",
        "col_commission": "Commissione",
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