# Do Smarter LLMs Really Cost More?

### Testing whether benchmark performance predicts what AI models cost

**Short answer: not really.** Across 281 commercially available language models, benchmark scores explain only **16–24% of the variation in token price**, even with tuned non-linear models. LLM pricing looks driven far more by business strategy and product tiering than by measured capability.

*Team research project · UC San Diego, Data Science in Practice (COGS 108) · Fall 2025*

---

## Why it matters

Teams choosing an LLM usually assume that paying more buys a smarter model. If price tracked capability, the leaderboard would be a good shopping guide. If it doesn't, **cheaper models can match premium ones**, and model selection becomes a cost-optimization problem worth analyzing rather than a default to the most expensive option.

## Research question

> Can benchmark performance across reasoning, coding, mathematics and general knowledge reliably predict the token pricing of AI language models?

- **H₀:** benchmark scores have no significant relationship with token cost
- **H₁:** benchmark scores are significantly related to token cost

## Data

| | |
|---|---|
| **Source** | Public leaderboard from [Artificial Analysis](https://artificialanalysis.ai) (`data/00-raw/llm_performance_leaderboard.csv`) |
| **Coverage** | 325 models raw → **281 after cleaning** |
| **Features** | Humanity's Last Exam (HLE), MMLU-Pro, GPQA, LiveCodeBench, SciCode |
| **Target** | Blended price per 1M tokens (3:1 input-to-output), plus input and output prices |
| **Cleaning** | Sparse benchmark coverage, missing evaluations and extreme premium-tier price outliers |

## Approach

1. **Exploratory analysis:** distributions, pair plots and correlation structure across benchmarks and prices
2. **Feature engineering:** benchmark selection and preprocessing for modelling
3. **Predictive modelling** in scikit-learn: Support Vector Regression, Random Forest and Gradient Boosting, each tuned with `GridSearchCV` and **5-fold cross-validation**, scored on MSE, MAE and R²

## Results

| Model | R² | Mean absolute error ($ per 1M tokens) |
|---|---|---|
| Support Vector Regression (polynomial kernel) | ≈ 0.16 | ≈ 1.08 |
| **Random Forest** | **≈ 0.24** | **≈ 1.02** |
| Gradient Boosting | ≈ 0.21 | ≈ 1.04 |

**Key findings**
- **Price is mostly unexplained by capability.** The best model, a tuned Random Forest, explains under a quarter of price variance.
- **Benchmarks move together.** Scores are strongly correlated with each other, reflecting a shared "general ability" signal rather than independent skills.
- **Prices are heavily right-skewed.** Most models cost under $1 per million tokens; a small set of premium tiers sit far above.
- **High and low scorers are priced alike.** Plotted directly against price, benchmark accuracy shows little relationship to cost.

**Takeaway:** the benchmark-to-price gap means buyers can often get comparable benchmark performance for a fraction of the cost by choosing models on measured value instead of price tier.

## Limitations and next steps

- The data is a single leaderboard snapshot; prices and models change quickly.
- Benchmarks measure narrow skills, not latency, context length, reliability or brand, all of which plausibly drive price.
- **Next:** add provider, release date, context window and model size as features, and track the relationship over time.

## Repository structure

```
llm_cost_performance_analysis.ipynb   Full analysis: EDA, modelling and conclusions
data/00-raw/                          Raw leaderboard data
data/02-processed/                    Cleaned data written by the notebook
modules/get_data.py                   Helper for downloading data files
checkpoints/                          Earlier project stages (proposal, data, EDA)
requirements.txt                      Python dependencies
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook llm_cost_performance_analysis.ipynb
```

## Built with

`Python` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib` · `seaborn` · `Jupyter`
