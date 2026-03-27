# AI Data Attacks

---

Having established the critical role of data and the structure of the data pipeline, we now focus specifically on`AI data attacks`. This module explores the techniques adversaries use to compromise AI systems by targeting the data itself; either during the training phase or by manipulating the stored model artifacts.

Unlike`evasion attacks`(`manipulating inputs to fool a deployed model`) or privacy`attacks`(`extracting sensitive information from a model`), the attacks covered here`fundamentally undermine the model's integrity by corrupting its foundation`: the data it learns from or the format it's stored in.

![Diagram showing Attack Surface with objectives: Data Poisoning, Storage Tampering, Processing Manipulation, Model Poisoning, Deployment Injection, Retraining Exploitation, targeting AI Data Pipeline Stages: Data Collection, Storage, Data Processing, Analysis & Modeling, Deployment, Monitoring & Maintenance.](https://academy.hackthebox.com/storage/modules/302/pipeline_attacks.png)

Each stage of the data pipeline presents potential attack surfaces adversaries can exploit.

![Diagram showing Adversary injecting malicious data into Data Collection Stage, leading to Data Poisoning, resulting in Corrupted Training Set and Compromised AI Model.](https://academy.hackthebox.com/storage/modules/302/data_collection_attacks.png)

During`data collection`, the primary threat is initial`data poisoning`, where an attacker intentionally injects malicious data. This is a prime opportunity for introducing data intended for`label flipping`or`feature attacks`. In the e-commerce example, this could manifest as submitting fake positive reviews (poisoned features/labels) to boost a product's recommendations or reviews with specific keywords (potential backdoor triggers) designed to cause unintended behavior later. For the healthcare scenario, an attacker might subtly alter`DICOM`metadata during ingestion or manipulate clinical notes, potentially mislabeling samples or embedding subtle feature perturbations. If this poisoned data infiltrates the training set, it can corrupt the resulting model according to the attacker's goals.

![Diagram showing Adversary gaining unauthorized access to Storage System, leading to theft/tampering of datasets and models, and model replacement with Trojan, affecting AWS S3/Secure FS.](https://academy.hackthebox.com/storage/modules/302/stor_age_attacks.png)

The`storage`stage faces traditional data security threats alongside model-specific risks, particularly relevant for`model stenography`and`Trojan`attacks. Unauthorized access to the`AWS S3`data lake or the healthcare provider's secure storage could allow theft or tampering of training datasets, potentially modifying labels or features post-collection. Furthermore, stored model files (the`.pkl`recommendation model on`S3`, the`.pt`diagnostic model) are valuable targets. An attacker gaining write access could replace a legitimate model with a malicious one containing an embedded`trojan`or execute a`model stenography attack`by hiding code within the model file itself (leveraging insecure deserialization like`pickle.load()`), potentially compromising the`Flask`API server or the clinical system upon loading.

![Diagram showing Adversary manipulating processing logic in Data Processing Stage, leading to corrupted processed data and indirect downstream model impact.](https://academy.hackthebox.com/storage/modules/302/processing_attacks.png)

`Data processing`offers another avenue for manipulation, potentially facilitating`label flipping`or`feature attacks`even on initially clean data. If an attacker influences the cleaning, transformation, or feature engineering steps, they can corrupt data before modeling. Compromising the e-commerce platform's`Spark`job could lead to mislabeled review sentiments (`label flipping`), while manipulating the healthcare provider's`Python`scripts could introduce subtle errors into standardized images or extracted text features (`feature attacks`), impacting the downstream model.

![Diagram showing Adversary influencing Analysis & Modeling Stage with corrupted data, leading to a compromised model with biases and incorrect patterns.](https://academy.hackthebox.com/storage/modules/302/modeling_attacks.png)

The`analysis and modeling`stage is where the impact of`data poisoning`attacks introduced earlier becomes concrete. When the`AWS SageMaker`job trains the recommendation model on poisoned`Parquet`files containing flipped labels or perturbed features, or the`PyTorch`process trains the diagnostic CNN on data embedded with backdoor triggers, the resulting model learns the attacker's desired manipulations. It might learn incorrect patterns, exhibit biases, or contain hidden backdoors activated by specific inputs later.

![Diagram showing Adversary injecting malicious model file into Model Storage, leading to insecure loading in Production Environment, resulting in a compromised system.](https://academy.hackthebox.com/storage/modules/302/deployment_attacks.png)

During`deployment`, the integrity of the model artifact remains crucial, especially concerning`Trojan`and`model stenography`risks. If the mechanism loading the model from storage (`S3`, secure file system) into the production environment is insecure, an attacker could inject a malicious model file at this point, achieving the same trojan effect or code execution via stenography as compromising the storage layer directly.

![Diagram showing Adversary injecting malicious feedback into Retraining Pipeline, corrupting input and affecting Production Model, creating a retraining loop.](https://academy.hackthebox.com/storage/modules/302/monitoring_attacks.png)

Finally, the`monitoring and maintenance`stage, especially the common practice of`retraining`models, acts as a critical enabler for training data attacks like`online poisoning`. For example, the e-commerce platform's`Airflow`retraining pipeline is a prime target. Attackers could continuously submit manipulated data: perhaps subtly altering clickstream data (`feature attacks`), submitting misleading feedback to influence future labels (`label flipping`), or injecting data designed to skew model weights towards particular outcomes over time. This gradual corruption degrades recommendation quality or introduces biases without needing initial dataset access.

The impact of successful AI data attacks can be severe, ranging from subtly biased decision-making and degraded system performance to complete model compromise and potentially enabling broader system breaches through embedded trojans.

## Mapping Vulnerabilities to Security Frameworks

Leading security frameworks such as the`OWASP Top 10 for LLM Applications`, as highlighted in the "[Introduction to Red Teaming AI](https://enterprise.hackthebox.com/academy-lab/67497/preview/modules/294)" module, provides specific context for risks within the AI pipeline.

The major risk we are particularly focused on is`Data poisoning`, where attackers manipulate data during collection, processing, training, or feedback stages. This directly corresponds to`OWASP LLM03: Training Data Poisoning`.

Another relevant category of risk is the`AI Supply Chain`, addressed by`OWASP LLM05: Supply Chain Vulnerabilities`. This encompasses several related threats: compromising the integrity of third-party data sources, tampering with pre-trained model artifacts (like injecting trojans), or exploiting vulnerabilities in the software components and platforms that make up the pipeline infrastructure itself. While`LLM05`covers many infrastructure aspects tied to components, robust protection also demands adherence to general secure system design principles beyond specific LLM list items, preventing unauthorized access throughout the pipeline. Ultimately, recognizing how both`Training Data Poisoning`and`Supply Chain Vulnerabilities`manifest is key to understanding the vulnerabilities in the AI data and model lifecycle

Complementing OWASP's specific vulnerability focus,[Google's Secure AI Framework](https://saif.google/)(SAIF) provides a broader, lifecycle-oriented perspective. The data integrity issues identified map well onto SAIF's core elements.

![Flowchart showing Model Creation process: Data Sources to Data Filtering & Processing, Data Storage Infrastructure, Evaluation, Training & Tuning, and Model Storage Infrastructure.](https://academy.hackthebox.com/storage/modules/302/saif_data.png)

For instance, SAIF’s principles regarding`Secure Design`, securing`Data`components, and managing the`Secure Supply Chain`directly address the need to protect data throughout its lifecycle. Preventing`Data Poisoning`aligns with securing this`Data Supply Chain`and implementing rigorous`Security Testing`and validation during`Model`development, especially for data used in retraining. Likewise, maintaining model artifact integrity and preventing malicious code injection are central to SAIF’s`Secure Deployment`practices and verifying the`Secure Supply Chain`.

Finally, the challenge of monitoring for data or model manipulation, particularly within dynamic retraining loops, is covered by SAIF's emphasis on`Secure Monitoring & Response`.