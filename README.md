## Credit Scoring Business Understanding

### Basel II, Risk Measurement, and Model Interpretability

The Basel II Capital Accord emphasizes accurate risk measurement, strong governance, and transparency in credit risk modeling. Financial institutions are required to demonstrate that their models are conceptually sound, well-documented, and auditable, as credit risk estimates directly influence capital requirements and lending decisions.

In this project, Basel II principles necessitate the use of an interpretable and explainable credit scoring model. The model must allow stakeholders, including risk managers and regulators, to understand how customer behavior translates into risk estimates. Clear documentation and explainability are essential to ensure regulatory compliance, enable effective model validation, and support ongoing monitoring and recalibration as customer behavior and market conditions evolve.

---

### Use of a Proxy Default Variable

A key challenge in this project is the absence of a direct default label. Since the buy-now-pay-later (BNPL) product is newly introduced, there is no historical loan repayment or default data available. However, supervised machine learning models require a target variable to learn risk patterns.

To address this, a proxy variable for credit risk is created using customer transactional behavior, particularly Recency, Frequency, and Monetary (RFM) characteristics. These behavioral indicators serve as a stand-in for default risk by capturing patterns associated with disengagement, declining spending, or irregular transaction activity.

Relying on a proxy introduces several business risks. The proxy may not perfectly represent true default behavior, leading to potential misclassification of customers. This can result in rejecting creditworthy customers or approving high-risk ones. Additionally, behavioral proxies may introduce bias if certain customer segments systematically exhibit different transaction patterns. These risks highlight the importance of conservative model usage, continuous performance monitoring, and future recalibration once actual repayment data becomes available.

---

### Trade-offs Between Interpretable and Complex Models

In a regulated financial environment, there is an inherent trade-off between model interpretability and predictive performance. Simple and interpretable models, such as logistic regression with Weight of Evidence (WoE), offer transparency, stability, and ease of explanation. These characteristics make them well-suited for regulatory review, internal governance, and long-term risk management, though they may struggle to capture complex non-linear relationships.

More complex models, such as gradient boosting or ensemble methods, often achieve higher predictive accuracy by modeling intricate interactions within the data. However, they introduce challenges related to explainability, validation, and regulatory acceptance. Their black-box nature can increase operational and compliance risk if decisions cannot be clearly justified.

Given these trade-offs, a balanced approach is required. Interpretable models provide a strong foundation for governance and trust, while more complex models can be evaluated as benchmarks to assess potential performance gains. Any deployment of advanced models must be accompanied by robust explainability techniques and strict model governance controls.
