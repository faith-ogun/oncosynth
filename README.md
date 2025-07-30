# 🧬 OncoSynth

**OncoSynth** is a command-line framework for generating structured synthetic lethality (SL) reports from gene pairs. It combines literature mining, drug data, and clinical trial insights using a multi-agent architecture built on [CrewAI](https://crewai.com/).

---

## 🚀 Features

- Multi-agent cancer data pipeline (PubMed, Open Targets, ClinicalTrials.gov)
- Outputs structured **markdown reports** per gene pair
- Confidence scoring and QA rejection safeguards
- Supports **interactive mode** and **batch mode**
- Plug-and-play: just need an OpenAI key and Entrez email

---

## 🛠 Installation

```bash
git clone https://github.com/faith-ogun/oncosynth.git
cd oncosynth
pip install -e .
````

This installs the `oncosynth` CLI globally.

---

## ⚙️ Setup

1. **Create your `.env` file:**

```bash
cp .env.example .env
```

2. **Edit `.env` with your API credentials:**

```
OPENAI_API_KEY=your-openai-key-here
ENTREZ_EMAIL=your.email@institute.edu
```

> Your email is required for PubMed API usage (NCBI Entrez).

---

## 📄 Input Format (Batch Mode)

You must provide a **CSV file with at least two columns**.

* The **first column** will be treated as the biomarker gene.
* The **second column** will be treated as the target gene.
* Column headers can be named anything (they’re auto-detected).

### Example:

```csv
GeneA,GeneB
MYC,CHEK1
BRCA1,ATR
CDK12,PARP1
```

---

## 🧪 Usage

### 🔹 Interactive Mode

Enter gene pairs manually via prompt:

```bash
oncosynth -i
```

---

### 🔹 Batch Mode

Run analysis on a full gene pair list from a CSV file:

```bash
oncosynth -b path/to/gene_pairs.csv
```

---

## 📁 Output

### Reports

* Successful reports are saved to:

  ```
  oncosynth/reports/
  ```

* Low-confidence or QA-rejected reports go to:

  ```
  oncosynth/low_confidence_reports/
  ```

### Logs

Logs for each agent’s output per gene pair are stored in:

```
oncosynth/logs/<BIOMARKER>_<TARGET>/
```

---

## 🧩 Architecture

OncoSynth runs a multi-agent system using:

* **SL Search Agent** → PubMed co-mention mining
* **Literature Agents** → Cancer relevance for both genes
* **Drug Agent** → Open Targets API integration
* **Trial Agent** → ClinicalTrials.gov search
* **Analyst + QA + Confidence + Writer** → Structured output and review

---

## 🧪 Coming Soon

* Optional `--dry-run` mode
* Custom scoring schemes
* Streamlit front-end

---

## 📄 License

MIT License. Built using CrewAI (Apache 2.0).

---

## 🙋‍♀️ Contact

Developed by Faith Ogundimu · Cancer Bioinformatics Researcher - PhD Candidate
🔗 [GitHub](https://github.com/faith-ogun)  
🔗 [LinkedIn](https://www.linkedin.com/in/faith-ogundimu)