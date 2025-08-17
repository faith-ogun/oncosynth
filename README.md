# 🧬 OncoSynth

**OncoSynth** is a command-line framework for generating structured synthetic lethality (SL) reports from gene pairs. It combines literature mining, drug data, and clinical trial insights using a multi-agent architecture built on [CrewAI](https://crewai.com/).

---

## 🚀 Features

- Multi-agent cancer data pipeline (PubMed, Open Targets, ClinicalTrials.gov)
- Human-focused literature searches using MeSH terms
- Deterministic confidence scoring (0-100) with automatic retry mechanism
- Outputs structured **markdown reports** per gene pair
- Supports **interactive mode** and **batch mode**
- Plug-and-play: just need an OpenAI key and Entrez email

---

## 🛠 Installation

```bash
git clone https://github.com/faith-ogun/oncosynth.git
cd oncosynth
pip install -e .
```

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
* Column headers can be named anything (they're auto-detected).

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

* All reports are saved to `oncosynth/reports/` with confidence scores in the header
* Reports include confidence interpretation (HIGH/MEDIUM/LOW) based on deterministic scoring

### Logs

Logs for each agent's output per gene pair are stored in:

```
oncosynth/logs/<BIOMARKER>_<TARGET>/
```

---

## 🎯 Confidence Scoring

OncoSynth uses a deterministic 100-point scoring system:

- **Direct SL Evidence** (40 points): Explicit synthetic lethality mentions and functional evidence
- **Druggability** (30 points): Open Targets tractability scores and known drugs
- **Clinical Evidence** (15 points): Active clinical trials for both genes
- **Cancer Relevance** (15 points): Cancer and ovarian cancer literature evidence

**Automatic Retry**: If a gene pair receives a score of 0, the system automatically retries up to 3 times before accepting the result.

---

## 🧩 Architecture

OncoSynth runs a multi-agent system using:

* **SL Search Agent** → PubMed co-mention mining with MeSH terms
* **Literature Agents** → Cancer relevance for both genes
* **Drug Agent** → Open Targets API integration
* **Trial Agent** → ClinicalTrials.gov search
* **Analyst + Confidence + Writer** → Structured output and scoring

---

## 🧪 Coming Soon

* Streamlit front-end

---

## 📄 License

MIT License. Built using CrewAI (Apache 2.0).

---

## 🙋‍♀️ Contact

Developed by Faith Ogundimu · Cancer Bioinformatics Researcher - PhD Candidate  
🔗 [GitHub](https://github.com/faith-ogun)  
🔗 [LinkedIn](https://www.linkedin.com/in/faith-ogundimu)