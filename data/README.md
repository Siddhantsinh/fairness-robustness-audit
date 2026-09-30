# Data Card: UCI Adult Census Income

Educational audit dataset documentation for binary classification fairness and robustness analysis.

- **Dataset name and source:** UCI Adult Census Income dataset (1994 US Census Bureau database, extracted by Ronny Kohavi and Barry Becker). [UCI ML Repository #2](https://archive.ics.uci.edu/dataset/2/adult).
- **License and permitted use:** Creative Commons Attribution 4.0 International (CC BY 4.0). Permitted for academic, educational, and research use.
- **Download instructions:** Run the automated acquisition script:
  ```bash
  python scripts/download_data.py
  ```
  This downloads and cleans the complete 48,842-record dataset directly from the UCI archive and saves it to `data/raw.csv`.
- **Number of records and features:** 48,842 instances (combined 32,561 training and 16,281 test folds from the original partition), 14 predictors (6 numerical, 8 categorical), and 1 binary target.
- **Target definition:** `income` — binary classification of whether an individual's annual income exceeds $50,000 (`>50K` positive, `<=50K` negative).
- **Group attributes:**
  - `sex`: Binary classification in census records (`Male`, `Female`).
  - `race`: Five categorical groups (`White`, `Black`, `Asian-Pac-Islander`, `Amer-Indian-Eskimo`, `Other`).
- **Missing-value pattern:** Missing values (originally coded as `?`) exist in `workclass` (~5.7%), `occupation` (~5.8%), and `native_country` (~1.8%). Handled via categorical mode imputation during preprocessing.
- **Class distribution:** 
  - `<=50K`: 37,155 (76.1%)
  - `>50K`: 11,687 (23.9%)
  - Base rate imbalance: ~3.18 : 1.
- **Sampling process (if known):** Extracted from the 1994 Current Population Survey (CPS) by the US Census Bureau using filtering criteria: `AAGE > 16`, `AGI > 100`, `AFNLWGT > 1`, and `HRSWK > 0`.
- **Known limitations:**
  - 1994 historical snapshot; does not reflect modern wage distributions, inflation, or contemporary labor demographics.
  - Proxy for socio-economic status contains historical societal wage discrimination against women and minority demographics.
  - `fnlwgt` (final sampling weight) reflects demographic sampling strata rather than individual capability.
  - Race and gender categories are rigid historical census classifications and do not capture nuanced identities.
- **Potential harmful uses:** Direct deployment in automated credit allocation, loan underwriting, or resume screening that perpetuates historical pay disparities and reinforces demographic disadvantages.
- **Why this dataset is suitable for an educational audit:** It is the foundational benchmark in algorithmic fairness and AI ethics literature (supported natively across Fairlearn and AIF360). Its well-documented demographic disparities provide a reliable, reproducible testbed for evaluating fairness metrics, noise sensitivity, and mitigation strategies without exposing private PII.

---

## Harm-and-Group Table

| Question | Your decision |
|---|---|
| **What could a false positive cause?** | If used for means-tested benefits, an individual is incorrectly classified as high-income and denied critical financial aid or social subsidies. If used for credit limits, it could result in unaffordable debt. |
| **What could a false negative cause?** | An individual actually earning >$50K is misclassified as low-income, leading to unjust denial of loans, credit cards, premium housing, or employment recruitment filters. |
| **Which groups may be affected?** | Women (1994 base rate ~10.9% >50K vs ~30.4% for men) and Black/minority racial groups (base rate ~12.1% >50K vs ~25.4% for White). |
| **Which metric represents the concern?** | **Equal Opportunity / False Negative Rate (FNR) Disparity** (protecting qualified applicants from exclusion), **Equalized Odds Difference**, and **Demographic Parity Difference** (disparity in positive prediction rates). |
| **What context is missing from the data?** | Cost of living adjusted by geography, household wealth/assets (only taxable capital gains are noted), generational wealth, caretaking responsibilities, and systemic gender wage disparities. |
