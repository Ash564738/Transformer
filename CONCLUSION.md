CONCLUSION
1. Achieved Results
-	Data preprocessing pipeline: Built a 10–11 step cleaning pipeline on the EGAT DGA dataset (~4,561 rows, 26 columns), handling duplicate removal, LOC recovery, unit standardization, sign-error correction, outlier nullification, and grouped median imputation. 
- Labeling methodology: Computed both Duval Pentagon variants and selected Pentagon 1 as the default pentagon output based on the labeled-benchmark comparison; Pentagon 2 remains an explicit comparison LF. Snorkel combines configured LFs without manual weights and preserves abstention/conflict evidence.
- Feature engineering: Developed deterministic gas, ratio, percentage, metadata, and diagnostic features; scale-invariant representations are evaluated separately to quantify domain shift. 
-	Machine learning experimentation: Trained and compared four classifiers (Random Forest, XGBoost, SVM-RBF, Logistic Regression) across a 16-cell scheme-by-algorithm grid, including confusion matrix analysis and domain-gap diagnosis between EGAT and IEC TC10 datasets. 
- Fleet ranking: Implemented one row per transformer using IEEE C57.104-2019 Status 1/2/3 and an unweighted lexicographic evidence order. Historical trend, recurrence, mixed faults, and single-record sufficiency remain explicit fields; no arbitrary severity weights are used. 
- Backend system: Implemented a Flask backend with a read-only Text2SQL/RAG chatbot and dashboard APIs for transformer fleet analysis and diagnosis queries. 
-	Frontend dashboard: Built role-based dashboards (for maintenance engineers and instructors/model validators) using Next.js, with prioritized data visualizations for operational and analytical use cases. 
- Validation: Cross-tested traditional methods, combinations, weak-label models, and hybrids on train/development/locked-test splits using labeled DGA datasets. Operational severity accuracy is explicitly not claimed because the EGAT data has no independent severity ground truth.
2. Limitations
-	Dataset scale and diversity: The dataset, while cleaned and structured, is limited in size and may not fully represent all transformer types, oil types, or fault conditions encountered in real-world operation. 
-	Domain gap between datasets: Differences between the EGAT and IEC TC10 datasets (e.g., data distribution, labeling conventions) reduce model generalizability across different power system contexts. 
-	Label quality dependency: Since ground-truth fault labels are not directly available, the reliability of weak-supervision-generated labels depends on the accuracy of the underlying diagnostic rules (Duval Pentagon, IEC 60599). 
-	Model interpretability: Some ML classifiers (e.g., SVM, XGBoost) offer high accuracy but limited interpretability compared to traditional rule-based methods, which may affect engineer trust in automated recommendations. 
-	Limited real-time/field validation: The system has not yet been tested on live/real-time transformer monitoring data or validated with field engineers in an operational setting. 
-	Time constraints: The 8-week internship timeframe limited the depth of hyperparameter tuning, deep learning exploration, and extensive real-world deployment testing. 
-	Environment compatibility issues: Use of Python 3.14 introduced potential compatibility issues with certain ML libraries, requiring workarounds during development.