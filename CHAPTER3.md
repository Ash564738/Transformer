CHAPTER 3. IMPLEMENTATION AND RESULT
3.1. System Architecture
The system built during the internship follows a typical client-server monitoring architecture, designed to turn the offline research pipeline (data cleaning, labeling-method comparison, ML training, IEEE condition status, and evidence ranking) into a tool that maintenance engineers can use day to day. Figure 3.1 summarizes the complete pipeline, from the raw EGAT dataset to the final risk ranking and visualization.
 
Figure 3. 1 Overall pipeline of the monitoring system
Backend. The backend is implemented in Flask (Python) and is responsible for data access, running the selected weak-label/student pipeline, computing IEEE rule-derived status evidence and the history-aware fleet order, and exposing these results through a REST API consumed by the frontend.
	Frontend. The frontend is implemented in Next.js and renders the interactive parts of the dashboard: fleet-wide overview tables, per-transformer trend charts, the Top-N highest-risk ranking view, and the traditional-method comparison view, so that an engineer can move from a fleet-level summary down to a single transformer's gas history without leaving the browser.
	Weak-supervision layer. Between the traditional labeling functions and the ML layer sits a Snorkel LabelModel that combines votes and abstentions from IEC 60599, Rogers Ratio, Doernenburg, Duval Triangle 1, and both Duval Pentagon variants. It estimates latent-label probabilities without manual LF weights; disagreement and abstention remain visible for audit.
	Text2SQL chatbot. A Text2SQL / Retrieval-Augmented Generation chatbot is layered on top of the same database: a user's natural-language question is converted into a structured, read-only SQL query, the query is executed against the current dataset, and a language model turns the query result back into a plain-language answer. This keeps every answer grounded in the actual data rather than in free-form generation - an important property for a diagnostic tool that engineers rely on.
3.2. DGA Dataset and Data Preprocessing
3.2.1. Dataset Description
Two datasets are used throughout this research
Dataset	n samples	Characteristics	Role
EGAT DGA (raw)	4,561	34 columns, many transformers (CODETX), real periodic-monitoring data	Training + in-domain evaluation
EGAT 85% modeling	3,864 (534 units)	Split by CODETX group, seed = 42	ML training (GroupKFold 5-fold)
EGAT 15% holdout	697 (94 units)	Split by CODETX group, seed = 42	Operational IEEE-status/evidence ranking (not used for training)
IEC TC10 (167 raw)	167	117 inspection-confirmed cases + 50 Normal samples (typical value)	Transfer-domain evaluation
IEC TC10 (95 clean)	95	50 Normal + 22 imputed samples removed	Reference fault ground truth for diagnostic-method and transfer evaluation; no severity ground truth
	The 85%/15% split of the EGAT dataset is grouped by CODETX (the unique transformer identifier, seed = 42) so that samples from the same physical unit never appear on both sides of the split; this prevents an over-optimistic evaluation that could happen if two samples from the same transformer, taken close in time, ended up in both the modeling and the holdout sets. Within IEC TC10, the 95-sample "clean" version removes 50 non-inspected "Normal" samples drawn only from a typical-value table (not from real inspections) and 22 samples whose values had been imputed with a class-conditional mean (a source of information leakage), leaving only field-confirmed cases with the true label distribution D2 = 45, D1 = 23, T3 = 14, T1/T2 = 10, PD = 3.
3.2.2. Data Quality Assessment
Before any analysis, the raw Excel file "DGA of Main Tank only KT 11022026_09062026" (4,563 rows x 26 columns) was profiled and found to contain the following data-quality issues:
-	An entirely uninformative column ("Unnamed: 25"), blank in 4,562 of 4,563 rows.
-	2 fully duplicate rows (e.g. NBLKT1A, PDGKT1AA).
-	LOC (installation site) missing in 11 rows, covering 2 transformer units (NAKT1A, NAKT2A).
-	Inconsistent formatting of the KV (rated voltage) field, mixing "-" and "/" separators (e.g. "107-34.85").
-	Placeholder value "xxxx" used for the SER (serial number) field in some rows, plus rows missing SER entirely.
-	20 rows where Tested day was earlier than Sample Day - a logical impossibility, since the analysis date cannot precede the sampling date.
-	TEMP (oil temperature) stored as a messy text string (e.g. "59*"), with 335 rows missing a value entirely.
-	WATER (water content) similarly stored as a messy text string, with 32 rows missing.
-	The originally reported TCG (Total Combustible Gas) matched the sum of the 8 combustible gases in only 98.84% of rows.
-	YEAR_Energized (commissioning year) missing in 7 rows, all from the same manufacturer (OSAKA).
None of these issues is large on its own, but left unaddressed they would have introduced silent errors into every later step (feature engineering, labeling, and ML training) - which is why a dedicated, traceable cleaning pipeline was applied before any analysis.
3.2.3. Data Cleaning
The issues identified in Section 3.2.2 were resolved through an 11-step cleaning pipeline, executed in sequence and summarized in Table bellow.
#	Issue found	How it was handled	Result
1	"Unnamed: 25" almost entirely blank, carries no information	Dropped from the dataset	26 -> 25 columns
2	2 fully duplicate rows	drop_duplicates()	4,563 -> 4,561 rows
3	Several columns would be edited in later steps (LOC, KV, SER, YEAR_Energized, Sample Day, Tested day, TEMP, WATER, TCG)	Snapshot the ORIGINAL values (suffix "_raw") before editing, for traceability	9 "_raw" columns preserving the original
4	LOC missing in 11 rows (2 units: NAKT1A, NAKT2A)	Confirmed the rule CODETX = LOC + NAME holds for 4,550/4,561 rows -> inferred LOC = CODETX minus NAME	0 rows missing LOC
5	Inconsistent KV formatting (mixed "-" and "/")	Standardized: replace "-" with "/"	Consistent KV formatting (e.g. "107/34.85")
6	SER has placeholder value "xxxx"; some rows missing SER	Checked SER consistency within CODETX+MFG groups (0 conflicting groups) -> replaced "xxxx" with NaN -> filled from another valid SER in the same group -> groups missing entirely got a unique code "xxxx1, xxxx2..."	0 rows missing SER
7	20 rows violate date logic (Tested day < Sample Day)	Swapped the two values for the violating rows (assumed the dates were entered in reverse)	0 violations remaining
8	TEMP is a messy string (e.g. "59*"), 335 rows missing	Parsed via regex (extract number), cast to numeric, took absolute value; missing values filled with the CODETX+MFG group mean, remainder filled with the global mean	0 rows missing TEMP
9	WATER is a similarly messy string, 32 rows missing	Parsed via regex, cast to numeric, absolute value; missing values filled with the CODETX+MFG group mean	0 rows missing WATER
10	Original TCG may be inconsistent with measured combustible gases	Recomputed TDCG from available measured combustible gases and retained the source value in tdcg_raw	Derived TDCG is auditable; missing gas measurements are not fabricated
11	YEAR_Energized missing in 7 rows (OSAKA manufacturer only)	Filled with the average year of other OSAKA units	0 rows missing YEAR_Energized
Final result: 4,561 rows x 34 columns, with no remaining missing data in fields important for DGA analysis (only NB - a free-text lab note in Thai - is missing in 3,482/4,561 rows, which is optional and does not affect the main analysis). Every edited column keeps its original "_raw" version immediately before it (e.g. LOC_raw, LOC) so that any cleaning decision can be re-checked at any time. The output, DGA_cleaned.xlsx, is the report-facing base file used for every experiment reported in this chapter; CSV/Parquet are internal intermediate formats only.
3.3. Traditional DGA-Based Fault Diagnosis
The implemented diagnostic set includes IEC 60599, Rogers Ratio, Doernenburg, Duval Triangle 1, and Duval Pentagon 1/2. They are noisy labeling functions: disagreement, abstention, and mixed evidence are retained rather than silently converted into ground truth.
3.3.1. Key Gas Method
Before applying the four ratio/geometric methods, a simpler, complementary check is the Key Gas / TDCG method defined in IEEE C57.104: each gas produced by a specific fault mechanism (e.g. C2H2 as the key gas for high-energy arcing, CH4 and C2H4 for thermal faults, H2 for partial discharge and corona) is examined together with the Total Dissolved Combustible Gas (TDCG), and the transformer's overall condition is placed into one of four severity bands (Condition 1-4) using published concentration limits. In this project, the Key Gas / TDCG criteria are used only as a coarse, first-pass severity screen - a sanity check against which the more detailed ratio-based and graphical methods can be compared - rather than as a primary diagnostic method, since TDCG alone does not distinguish between fault types (for example it cannot separate a discharge fault from a thermal fault of similar total severity). The four ratio/geometric methods below remain the primary labeling methods evaluated in this study.
3.3.2. IEC 60599-Based Diagnosis
IEC 60599 computes three gas ratios (C2H2/C2H4, CH4/H2, C2H4/C2H6) for every EGAT sample and compares them against the published value ranges in Table 3.5 to classify the sample as PD/Normal, D1, D2, T1, T2 or T3. When no threshold combination matches - a structural gap in the lookup table rather than a data problem - the sample is labeled ND (Not Determined) and excluded from the ML training set for this scheme.
Label	C2H2/C2H4	CH4/H2	C2H4/C2H6
PD/Normal	NS	< 0.1	< 0.2
D1	1.0	0.1 - 0.5	> 1.0
D2	0.6 - 2.5	0.1 - 1.0	> 2.0
T1	NS	> 1.0	< 1.0
T2	< 0.1	> 1.0	1.0 - 4.0
T3	< 0.2	> 1.0	> 4.0
(NS = "not significant", i.e. that ratio does not constrain the label for this row of the table.) Rogers Ratio uses the same three ratios but reads them through the original Rogers lookup table; it is the only one of the four methods with an explicit "Normal" class, though its table has gaps and cannot express some fault types. Duval Triangle 1 and Duval Pentagon 1 are geometric-zoning methods (percentage composition mapped onto a triangle or a five-sided polygon) and therefore almost always resolve to a zone, giving them a much lower ND rate than the two threshold-lookup methods, as shown in Figure 3.3.
3.3.3. Traditional Diagnosis Results
 
Figure 3. 2 ND (Not-Determined) rate for the four methods
The four methods also disagree strongly on the shape of the resulting label distribution, summarized in Table.
Label	Duval Triangle 1	Duval Pentagon 1	IEC 60599	Rogers Ratio
Normal	-	-	-	411 (10.6%)
PD	1255 (32.5%)	35 (0.9%)	20 (0.5%)	56 (1.4%)
D1	23 (0.6%)	17 (0.4%)	2 (0.1%)	-
D2	19 (0.5%)	57 (1.5%)	8 (0.2%)	8 (0.2%)
DT	14 (0.4%)	-	-	-
S	-	2229 (57.7%)	-	-
T1	1208 (31.3%)	1097 (28.4%)	1169 (30.3%)	181 (4.7%)
T2	579 (15.0%)	193 (5.0%)	216 (5.6%)	184 (4.8%)
T3	464 (12.0%)	219 (5.7%)	61 (1.6%)	93 (2.4%)
ND	302 (7.8%)	17 (0.4%)	2388 (61.8%)	2931 (75.9%)
	Two patterns are worth flagging explicitly. First, Duval Triangle assigns 32.5% of samples to PD - suspected to be a "geometric mislabeling" artifact: when C2H2 = C2H4 = 0, the triangle coordinate degenerates to the CH4 = 100% corner and always falls into the PD zone, even for a perfectly healthy unit. Second, Duval Pentagon assigns 57.7% of samples to "S" (stray gassing, a mild/non-specific class), which dominates its distribution. Rogers Ratio is the only scheme with an explicit "Normal" concept, but pays for it with the highest ND rate (75.9%) of the four. These differences matter because the same raw data yields a materially different "ground truth" depending on which scheme is chosen to label it - a point that carries through directly into the machine-learning results in Section 3.4.
3.4. Machine Learning-Based Fault Diagnosis
Design: for each of the four schemes described in Section 3.3, ND samples are filtered out, and the remaining labels are used as pseudo ground truth to train four algorithms (Random Forest, XGBoost, SVM-RBF, Logistic Regression) on the ENTIRE EGAT 85% modeling set (n = 3,864, no fold splitting for the headline transfer numbers). Evaluation is then done via TRANSFER to IEC TC10 (95 inspection-confirmed fault cases), a dataset completely independent from the training data and labels, so that it reflects the model's true generalization ability rather than the circularity risk of evaluating on EGAT itself.
3.4.1. Label Generation
Each sample's label is taken directly from the corresponding scheme's output in Table 3.6. Because the ND rate differs sharply by scheme (Figure 3.3), the number of samples actually available for training differs sharply too: the training-set size for each scheme equals 3,864 minus that scheme's ND count on the 85% modeling set - Duval Triangle 1: 3,864 - 302 = 3,562; Duval Pentagon 1: 3,864 - 17 = 3,847; IEC 60599: 3,864 - 2,388 = 1,476; Rogers Ratio: 3,864 - 933 = 933 - which matches the "n train" column of Table 3.7 exactly and confirms that no additional sample loss occurred between labeling and training.
3.4.2. Feature Generation
The same 12 unified, scale-invariant features defined and justified used identically across all four labeling schemes throughout the ML experiments, ensuring a fair, apples-to-apples comparison in which no scheme receives a specialized feature set of its own.
3.4.3. Dataset Splitting
The EGAT dataset is split by CODETX group (seed = 42) into an 85% modeling set (3,864 samples, 534 transformer units) and a 15% holdout set (697 samples, 94 units); grouping by transformer prevents samples from the same physical unit from appearing on both sides, which would otherwise let the model "cheat" by recognizing a unit it has already seen rather than genuinely learning the fault pattern. The 85% modeling set is used with GroupKFold (5-fold, grouped by CODETX) during model development, but the headline results reported in Section 3.4.5 use models trained on the full 85% set and evaluated by TRANSFER to the independent IEC TC10 set (95 samples) rather than to a held-out fold of EGAT itself - a deliberately stricter test of generalization (Section 3.4.1). The 15% EGAT holdout is reserved separately and is not used for training or for the transfer numbers; it is used only for the Risk Score ranking.
3.4.4. Model Development
Four widely used classification algorithms were applied identically to each supported labeling scheme: Random Forest, XGBoost, Support Vector Machine with an RBF kernel, and Logistic Regression. Standard, consistent implementations were used so that performance differences reflect the labeling scheme rather than per-scheme tuning. Random Forest is used for the reference confusion-matrix detail; IEEE condition status and fleet ranking are separate rule/evidence outputs, not a model probability score.
3.4.5. Fault Classification Results
Table below reports the headline transfer results: macro-F1 and accuracy (strict / T1-T2-lenient, since TC10's true labels merge T1 and T2 into a single "T1/T2" class) for all 16 scheme x algorithm combinations, evaluated on the 95 IEC TC10 cases.
Scheme	Algorithm	n train	n TC10	macro-F1 TC10 (%)	Accuracy strict / lenient
Duval Triangle 1	Random Forest	3562	95	87.4	81.1% / 88.4%
Duval Triangle 1	XGBoost	3562	95	86.5	78.9% / 86.3%
Duval Triangle 1	SVM (RBF)	3562	95	66.0	50.5% / 56.8%
Duval Triangle 1	Logistic Regression	3562	95	85.4	75.8% / 83.2%
Duval Pentagon 1	Random Forest	3847	95	63.5	60.0% / 66.3%
Duval Pentagon 1	XGBoost	3847	95	66.6	56.8% / 63.2%
Duval Pentagon 1	SVM (RBF)	3847	95	71.5	65.3% / 71.6%
Duval Pentagon 1	Logistic Regression	3847	95	71.2	62.1% / 68.4%
IEC 60599	Random Forest	1476	95	75.2	71.6% / 80.0%
IEC 60599	XGBoost	1476	95	53.6	55.8% / 64.2%
IEC 60599	SVM (RBF)	1476	95	33.3	30.5% / 40.0%
IEC 60599	Logistic Regression	1476	95	64.8	51.6% / 58.9%
Rogers Ratio	Random Forest	933	95	52.5	61.1% / 66.3%
Rogers Ratio	XGBoost	933	95	50.9	58.9% / 65.3%
Rogers Ratio	SVM (RBF)	933	95	36.7	43.2% / 50.5%
Rogers Ratio	Logistic Regression	933	95	57.4	60.0% / 67.4%
 
Figure 3. 3 Grouped bar chart of macro-F1 by scheme x algorithm, transfer evaluation on IEC TC10 (n = 95).
Duval Triangle 1 is the scheme with the highest and most STABLE transfer results across all four algorithms (66-87%), especially with Random Forest, XGBoost and Logistic Regression (85-87%). In contrast, Rogers Ratio and IEC 60599 drop sharply on transfer (33-75%), much lower than their internal, in-domain cross-validation numbers on EGAT once suggested - a 25-48 point gap that is quantitative evidence of circularity (the ML features overlap with the label-defining formula for these two schemes), so a high in-domain score for Rogers/IEC 60599 should not be read as a "better" model.
	Table below summarizes the per-scheme accuracy of the reference model (Random Forest), whose per-class confusion matrices are used for the detailed discussion below.
Scheme	n (TC10)	Accuracy - Strict	Accuracy - Lenient (T1/T2 merged)
Duval Triangle 1	95	81.1%	88.4%
Duval Pentagon 1	95	60.0%	66.3%
IEC 60599	95	71.6%	80.0%
Rogers Ratio	95	61.1%	66.3%
-	Duval Triangle 1 achieves the highest transfer accuracy (81.1% strict / 88.4% lenient), matching its highest macro-F1 (87.4%) - reinforcing that it is the most trustworthy reference scheme among the four.
-	Rogers Ratio: 0 of 23 true D1 cases were predicted correctly (21 predicted as D2, 2 as T1) - not a model error but a STRUCTURAL LIMITATION of the scheme, since the Rogers lookup table has no ratio combination that produces a D1 label; a model trained on Rogers labels never "sees" the D1 class during training and structurally cannot predict it.
-	The most common confusion across all four schemes is D1 -> D2 (predicting a more severe fault than the true one); in a preventive-maintenance context this "over-warning" tendency is safer than the reverse, but should be accounted for when calibrating operational alert thresholds.
-	The "Lenient" metric (T1/T2 merged) improves accuracy by 6-8 points over "Strict" for every scheme, showing that part of the measured error is simply because TC10 is not detailed enough to distinguish T1 from T2, not a genuine model mistake.
	Taken together, these results are used to select the traditional method or combination on the development split, with Duval Pentagon 1 retained as the default pentagon output when its locked-test comparison supports that choice. No method is treated as ground truth for the unlabeled operational data; Random Forest is only a model baseline, while IEEE status supplies the documented condition class for ranking.
3.5. Transformer Condition Assessment and Ranking
The results of Section 3.4 give a per-sample fault label, but a maintenance team needs more than a label: it needs one comparable, history-aware row for each transformer. This section describes the IEEE evidence and the unweighted ranking order used to prioritize the fleet.
3.5.1. Condition Indicators
Three layers of indicators feed into the condition assessment: (1) measured gases and recomputed TCG/TDCG; (2) deterministic ratios, percentages, metadata, and diagnostic features; and (3) IEEE C57.104-2019 concentration, delta, and rate evidence. Fault labels and model posterior values remain diagnostic context and are not converted into a hand-weighted severity score.
3.5.2. Severity Scoring
Severity is the IEEE C57.104-2019 rule-derived Status 1/2/3 condition class (or INSUFFICIENT_DATA). No independent operational severity ground truth is available, so severity accuracy is not claimed.
Class	Normal / S / ND	PD / T1	T2	D1 / DT	T3	D2
Severity	0	1	2	3	4	5
Fault criticality (for example, D2 as high concern and stray gassing as lower concern) is source-backed qualitative context and a tie-break only; it never overrides IEEE status or acts as a numeric severity weight.
Fault type (label)	Meaning	Severity score	Used in scheme
Normal	Normal, no sign of fault	0	Rogers Ratio
S	Stray gassing / mild, non-specific fault	0	Duval Pentagon 1
ND	No Decision - insufficient conditions to classify (excluded from training)	0	All 4 schemes
PD	Partial Discharge	1	All 4 schemes
T1	Low thermal, < 300C	1	All 4 schemes
T1/T2	T1 and T2 merged (used only when matching against TC10's true label, which does not distinguish them)	1.5	TC10 comparison (Section 3.4.5)
T2	Medium thermal, 300-700C	2	All 4 schemes
D1	Low-energy discharge	3	Duval Triangle 1, Duval Pentagon 1, IEC 60599 (absent from Rogers Ratio)
DT	Mixed thermal + discharge	3	Duval Triangle 1
T3	High thermal, > 700C	4	All 4 schemes
D2	High-energy discharge / arcing	5	All 4 schemes
Rogers Ratio has no ratio combination that produces a D1 label (Section 3.3.2 and the D1 confusion-matrix result in Section 3.4.5), which is why the "Used in scheme" column for D1 excludes Rogers Ratio.
3.5.3. Ranking Method
The operational pipeline aggregates all records by transformer, selects the latest IEEE status as the primary key, and retains current concentration/rate/delta evidence, trigger counts, fault-context tie-breaks, historical maxima, recurrence, trend, and data sufficiency. Equal evidence vectors receive tied ranks. This is an ordinal maintenance order, not a failure probability or health score.
3.5.4. Ranking Results
 
Figure 3. 4 Top-20 highest-risk transformers on the EGAT holdout
The top 8 units (ranks #1-8) show strong consensus (D2/T3) across all four methods; from rank #9 onward the methods start to diverge - most clearly at #20 (PK2KT4A), where Duval Triangle says T3, Duval Pentagon says "S" (normal), IEC 60599 says T2, and Rogers says T1 for the very same sample.
 
Figure 3. 5 Top-20 highest-risk cases on IEC TC10 (95 cases)
3.5.5. Trend-Based Assessment
The EGAT dataset spans 2017-2024 with, in many cases, multiple samples per transformer (CODETX), which means the monitoring dashboard (Section 3.1) can plot each unit's gas ratios and IEEE status/evidence as a time series rather than a single snapshot, letting an engineer see whether a unit's condition is stable or worsening over successive samples. It should be stated plainly, however, that the headline quantitative results in Sections 3.4 and 3.5.4 are based on a single, latest sample per unit, and that a formal trend or gas-generation-rate model (for example, rate-of-change thresholds in the style of IEEE C57.104) was not part of the quantitative evaluation carried out in this report. Turning the dashboard's time-series view into a validated, rate-based early-warning indicator is left as a direction for future work.
3.5. Experimental Results and Discussion
3.5.1. ML Model Comparison
Across the 16 scheme x algorithm combinations evaluated by transfer to IEC TC10, Duval Triangle 1 is consistently the strongest and most stable labeling scheme, particularly when paired with Random Forest, XGBoost or Logistic Regression (85-87% macro-F1). Duval Pentagon 1 is more stable across algorithms than Rogers Ratio or IEC 60599 but caps out lower (64-72%). Rogers Ratio and IEC 60599 look competitive - even superior - when judged only on their own in-domain training data, but collapse on independent transfer (33-75%); this 25-48 point in-domain-to-transfer gap is quantitative evidence of circularity, since both schemes' labels are generated from the same gas ratios that are then reused as ML features. The practical recommendation is therefore to treat Duval Triangle 1 with Random Forest as the primary, most trustworthy combination, and to use the other three schemes as cross-checks rather than stand-alone diagnostic sources
3.5.2. Condition Ranking Results
The Top-20 risk rankings (Section 3.5.4) show that all four labeling schemes strongly agree on the most severe cases - both on the EGAT holdout (ranks #1-8) and on IEC TC10, where the top-20 highest-risk cases are almost entirely consensus D2 - which is reassuring for flagging the units that most urgently need attention. Agreement breaks down for moderate-severity cases (EGAT rank #9 onward), where the choice of labeling scheme can change a unit's predicted condition from T3 (Duval Triangle) to "Normal" (Duval Pentagon) for the exact same sample. This is why the dashboard's operational ranking (Section 3.1) uses a weak-supervision, multi-scheme consensus rather than committing to a single method: rank correlation between scheme pairs ranges from 0.21 to 0.90 depending on the pair, which is a large enough spread that the choice of scheme is a real operational decision, not just an academic footnote.
	Key findings.
-	Absolute-concentration features are the single largest cause (+38.0 F1 points) of the EGAT -> TC10 domain gap; switching to ratio/composition features is the key design decision, supported by both transfer-F1 and class-conditional KS evidence (Section 3.2.4).
-	The four labeling methods produce very different ND rates (0.4% -> 75.9%) and inconsistent outputs with each other (pairwise agreement rate 21.9%-88.9%) - the same raw data yields a different "ground truth" depending on which scheme is chosen (Section 3.3.3).
-	ML performance measured via TC10 transfer depends strongly on the scheme: Duval Triangle 1 is highest and most stable (87.4% macro-F1), while Rogers Ratio/IEC 60599 are much lower (52-75%) despite once reaching near-perfect scores (99-100%) when measured in-domain on EGAT - a 25-48 point gap that is quantitative evidence of circularity, so a high in-domain score should not be read as a "better" model (Section 3.4.5).
-	Because the operational ranking is an unweighted evidence order, rank differences reflect the selected diagnostic evidence and history rather than arbitrary probability/severity weights.
	Limitations.
-	Operational severity accuracy cannot be claimed: EGAT has no independent severity ground truth. IEEE status is a screening condition class and remains decision-support evidence.
-	The IEC TC10 evaluation set has only 95 field-confirmed cases, and its true labels merge T1 and T2 into a single class, which limits how finely the transfer results can be interpreted.
-	Rogers Ratio structurally cannot predict the D1 class (its lookup table has no combination that produces a D1 label), and Duval Triangle 1 shows a geometric-mislabeling artifact that pushes an unrealistic 32.5% of samples into the PD class.
-	Feature engineering reduced the domain gap on average, but made it slightly worse for the D1 class specifically (class-conditional KS 0.390 -> 0.422), which should be investigated further rather than treated as fully solved.
-	All four methods share a systematic blind spot: true D1 cases with very high gas concentration on IEC TC10 (e.g. F034, F035) are consistently mispredicted as D2 by every scheme.
## Methodology update

The current implementation does not use the former probability-weighted Risk Score; it uses IEEE Status 1/2/3 and an unweighted lexicographic evidence order instead.
or self-constructed fault-severity weights. Severity is the rule-derived IEEE
C57.104-2019 Status 1/2/3 condition class (with `INSUFFICIENT_DATA` when mandatory
evidence is unavailable). Fleet ranking is one row per transformer and uses an
unweighted lexicographic evidence order: current status, concentration/rate/delta
evidence, independent trigger counts, qualitative fault-context tie-break, and
historical evidence. The operational dataset has no independent severity ground
truth, so severity accuracy is not claimed. Traditional methods are noisy labeling
functions; their individual, combination, weak-label, and hybrid results are
evaluated only against the labeled benchmark splits.
