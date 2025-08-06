import os
import requests
from typing import Type, Optional, Dict, Any, Union, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field, PrivateAttr
from Bio import Entrez
from crewai import Agent, Task, Crew, Process, LLM
from langchain_openai import ChatOpenAI
from crewai.tools import BaseTool
import re
import json

# ---------------------------
# WARNING CONTROL
# ---------------------------
import warnings
import logging
from urllib3.connectionpool import log as urllib3_logger

warnings.filterwarnings('ignore')

# Suppress unclosed socket warnings from urllib3
logging.getLogger("urllib3").setLevel(logging.ERROR)
urllib3_logger.setLevel(logging.CRITICAL)

# ---------------------------
# ENVIRONMENT SETUP
# ---------------------------

env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if not os.path.exists(env_path):
    raise FileNotFoundError("Missing `.env`. Please copy `.env.example` and fill in credentials.")

load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")

if not OPENAI_API_KEY or not ENTREZ_EMAIL:
    raise EnvironmentError("OPENAI_API_KEY and ENTREZ_EMAIL must be set in .env.")

# Load required environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL")

# Set required environments for downstream usage
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
Entrez.email = ENTREZ_EMAIL

# ---------------------------
# LLM SETUP (GPT-3.5 Turbo)
# ---------------------------

llm = LLM(model="gpt-4o")

# ---------------------------
# HGNC to Ensembl
# ---------------------------

def load_hgnc_map(filepath: str) -> dict:
    import pandas as pd
    df = pd.read_csv(filepath, sep="\t", dtype=str)
    return dict(zip(df["symbol"], df["ensembl_gene_id"]))

# ---------------------------
# TOOL 1: SL-Specific Pair Search
# ---------------------------

class SLPairSearchInput(BaseModel):
    biomarker: str
    target: str

class SLPairSearchTool(BaseTool):
    name: str = "SL Pair PubMed Search Tool"
    description: str = "Searches PubMed for synthetic lethality between a biomarker-target gene pair."
    args_schema: Type[BaseModel] = SLPairSearchInput

    def _run(self, biomarker: str, target: str) -> str:
        queries = [
            f"{biomarker} AND {target} AND synthetic lethality",
            f"{biomarker} AND {target} AND synthetic lethal",
            f"{biomarker} AND {target}"
        ]

        try:
            all_results = []
            seen_pmids = set()

            for query in queries:
                handle = Entrez.esearch(db="pubmed", term=query, retmax=5)
                record = Entrez.read(handle)
                pmid_list = record["IdList"]

                if not pmid_list:
                    continue

                handle = Entrez.efetch(db="pubmed", id=",".join(pmid_list), retmode="xml")
                results = Entrez.read(handle)

                for article in results["PubmedArticle"]:
                    pmid = str(article["MedlineCitation"]["PMID"])
                    if pmid in seen_pmids:
                        continue
                    seen_pmids.add(pmid)

                    article_data = article["MedlineCitation"]["Article"]
                    title = article_data.get("ArticleTitle", "No title")
                    abstract = " ".join(article_data["Abstract"]["AbstractText"]) if "Abstract" in article_data else "No abstract"
                    
                    all_results.append({
                        "query": query,
                        "pmid": pmid,
                        "title": title.strip(),
                        "abstract": abstract.strip(),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
                    })

            if not all_results:
                return json.dumps({"results": [], "message": "No SL-specific PubMed results found."})

            return json.dumps({"results": all_results}, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Error accessing PubMed SL pair search: {str(e)}"})

# ---------------------------
# TOOL 2: Biomarker-only Cancer Context
# ---------------------------

class BiomarkerSearchInput(BaseModel):
    biomarker: str

class BiomarkerPubMedSearchTool(BaseTool):
    name: str = "Biomarker PubMed Search Tool"
    description: str = "Searches PubMed for cancer-related studies involving the biomarker gene"
    args_schema: Type[BaseModel] = BiomarkerSearchInput

    def _run(self, biomarker: str) -> str:
        queries = [
            f"{biomarker} AND cancer",
            f"{biomarker} AND ovarian cancer"
        ]

        try:
            all_results = []
            seen_pmids = set()

            for query in queries:
                handle = Entrez.esearch(db="pubmed", term=query, retmax=5)
                record = Entrez.read(handle)
                pmid_list = record["IdList"]

                if not pmid_list:
                    continue

                handle = Entrez.efetch(db="pubmed", id=",".join(pmid_list), retmode="xml")
                results = Entrez.read(handle)

                for article in results["PubmedArticle"]:
                    pmid = str(article["MedlineCitation"]["PMID"])
                    if pmid in seen_pmids:
                        continue
                    seen_pmids.add(pmid)

                    article_data = article["MedlineCitation"]["Article"]
                    title = article_data.get("ArticleTitle", "No title")
                    abstract = " ".join(article_data["Abstract"]["AbstractText"]) if "Abstract" in article_data else "No abstract"

                    all_results.append({
                        "query": query,
                        "pmid": pmid,
                        "title": title.strip(),
                        "abstract": abstract.strip(),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
                    })

            if not all_results:
                return json.dumps({"results": [], "message": f"No cancer-related results found for {biomarker}"})

            return json.dumps({"results": all_results}, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Error accessing PubMed for biomarker: {str(e)}"})

# ---------------------------
# TOOL 3: Target-only Cancer Context
# ---------------------------

class TargetSearchInput(BaseModel):
    target: str

class TargetPubMedSearchTool(BaseTool):
    name: str = "Target PubMed Search Tool"
    description: str = "Searches PubMed for cancer-related studies involving the target gene"
    args_schema: Type[BaseModel] = TargetSearchInput

    def _run(self, target: str) -> str:
        queries = [
            f"{target} AND cancer",
            f"{target} AND ovarian cancer"
        ]

        try:
            all_results = []
            seen_pmids = set()

            for query in queries:
                handle = Entrez.esearch(db="pubmed", term=query, retmax=5)
                record = Entrez.read(handle)
                pmid_list = record["IdList"]

                if not pmid_list:
                    continue

                handle = Entrez.efetch(db="pubmed", id=",".join(pmid_list), retmode="xml")
                results = Entrez.read(handle)

                for article in results["PubmedArticle"]:
                    pmid = str(article["MedlineCitation"]["PMID"])
                    if pmid in seen_pmids:
                        continue
                    seen_pmids.add(pmid)

                    article_data = article["MedlineCitation"]["Article"]
                    title = article_data.get("ArticleTitle", "No title")
                    abstract = " ".join(article_data["Abstract"]["AbstractText"]) if "Abstract" in article_data else "No abstract"

                    all_results.append({
                        "query": query,
                        "pmid": pmid,
                        "title": title.strip(),
                        "abstract": abstract.strip(),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"
                    })

            if not all_results:
                return json.dumps({"results": [], "message": f"No cancer-related results found for {target}"})

            return json.dumps({"results": all_results}, indent=2)

        except Exception as e:
            return json.dumps({"error": f"Error accessing PubMed for target: {str(e)}"})


# ---------------------------
# TOOL 4: Open Targets Search
# ---------------------------

class OpenTargetsInput(BaseModel):
    biomarker: str = Field(description="The biomarker gene symbol")
    target: str = Field(description="The target gene symbol")

class OpenTargetsTool(BaseTool):
    name: str = "Open Targets Tool"
    description: str = "Fetches drug info and tractability scores from Open Targets for both genes"
    args_schema: Type[BaseModel] = OpenTargetsInput

    _hgnc_map: dict = PrivateAttr()

    def __init__(self, hgnc_map: dict):
        super().__init__()
        self._hgnc_map = hgnc_map

    def symbol_to_ensembl(self, symbol: str) -> Optional[str]:
        return self._hgnc_map.get(symbol)

    def _run(self, biomarker: str, target: str) -> str:
        final_json = {}

        for gene_name in [biomarker, target]:
            ensembl_id = self.symbol_to_ensembl(gene_name)
            if not ensembl_id:
                final_json[gene_name.upper()] = {"ERROR": f"No Ensembl ID found for gene: {gene_name}"}
                continue

            query = """
            query target($ensemblId: String!) {
              target(ensemblId: $ensemblId) {
                id
                approvedSymbol
                biotype
                tractability {
                  label
                  modality
                  value
                }
                knownDrugs {
                  count
                  rows {
                    phase
                    status
                    urls {
                      url
                    }
                    disease {
                      name
                    }
                    drug {
                      name
                    }
                    mechanismOfAction
                  }
                }
              }
            }
            """

            variables = {"ensemblId": ensembl_id}
            url = "https://api.platform.opentargets.org/api/v4/graphql"

            try:
                res = requests.post(url, json={"query": query, "variables": variables})
                res.raise_for_status()
                response_data = res.json()

                if "errors" in response_data:
                    final_json[gene_name.upper()] = {"ERROR": response_data["errors"]}
                    continue

                target_data = response_data.get("data", {}).get("target")
                if not target_data:
                    final_json[gene_name.upper()] = {"ERROR": "No target data found"}
                    continue

                tractability = target_data.get("tractability", [])
                known_drugs = target_data.get("knownDrugs", {})
                drug_count = known_drugs.get("count", 0)
                drugs = known_drugs.get("rows", [])

                sm_scores = [t for t in tractability if t.get("modality") == "SM"]
                tract_labels = {t.get("label", ""): t.get("value", 10) for t in sm_scores}

                high_conf_labels = {"Approved Drug"}
                mid_conf_labels = {"Advanced Clinical", "Phase 1 Clinical", "Clinical Precedence"}
                low_conf_labels = {"Structure with Ligand", "High-Quality Ligand", "High-Quality Pocket", "Druggable Family"}

                # Only get labels where the value is True
                true_labels = {label for label, value in tract_labels.items() if value}

                if true_labels & high_conf_labels:
                    interpretation = "HIGH tractability (approved drug exists)"
                elif true_labels & mid_conf_labels:
                    interpretation = "MODERATE tractability (clinical-stage target)"
                elif true_labels & low_conf_labels:
                    interpretation = "LOW tractability (ligandable but no clinical-stage evidence)"
                else:
                    interpretation = "LOW tractability (no tractability evidence)"

                final_json[gene_name.upper()] = {
                    "TRACTABILITY_SCORES": {
                        "SMALL_MOLECULE": tract_labels,
                        "INTERPRETATION": interpretation
                    },
                    "KNOWN_DRUGS": drugs
                }

            except requests.RequestException as e:
                final_json[gene_name.upper()] = {"ERROR": f"HTTP error: {str(e)}"}
            except Exception as e:
                final_json[gene_name.upper()] = {"ERROR": f"Unexpected error: {str(e)}"}

        return json.dumps(final_json, indent=2)


# ---------------------------
# TOOL 5: ClinicalTrials.gov Search
# ---------------------------

class ClinicalTrialsInput(BaseModel):
    genes: List[str] = Field(description="List of gene symbols to search for in clinical trial records")

class ClinicalTrialsTool(BaseTool):
    name: str = "Clinical Trials Tool"
    description: str = "Searches ClinicalTrials.gov for trials involving one or more genes"
    args_schema: Type[BaseModel] = ClinicalTrialsInput

    def _run(self, genes: List[str]) -> str:
        merged_results = {}

        for gene in genes:
            res = requests.get(
                "https://clinicaltrials.gov/api/v2/studies",
                params={"query.term": gene, "pageSize": 10}
            )
            res.raise_for_status()
            studies = res.json().get("studies", [])

            all_results = []
            for study in studies:
                try:
                    ps = study["protocolSection"]
                    ident = ps["identificationModule"]
                    status = ps.get("statusModule", {})
                    conditions = ps.get("conditionsModule", {}).get("conditions", [])
                    design = ps.get("designModule", {})
                    nct_id = ident["nctId"]
                    title = ident.get("briefTitle", "No title")
                    condition_str = ", ".join(conditions) or "No condition listed"
                    phase = ", ".join(design.get("phases", [])) or "N/A"
                    trial_status = status.get("overallStatus", "Unknown")
                    trial_url = f"https://clinicaltrials.gov/study/{nct_id}"

                    all_results.append({
                        "gene": gene,
                        "nct_id": nct_id,
                        "title": title,
                        "phase": phase,
                        "status": trial_status,
                        "conditions": condition_str,
                        "url": trial_url
                    })
                except Exception as e:
                    all_results.append({
                        "gene": gene,
                        "error": f"Failed to parse study: {e}"
                    })

            merged_results[f"{gene.upper()} Trials"] = all_results

        return json.dumps(merged_results, indent=2)

# ---------------------------
# TOOL 6: Deterministic Scoring Tool
# ---------------------------
    
class ConfidenceScoringInput(BaseModel):
    biomarker: str
    target: str
    sl_results: Dict[str, Any]                
    biomarker_results: Dict[str, Any]          
    target_results: Dict[str, Any]           
    opentargets_results: Dict[str, Any] 
    trials_results: Dict[str, Any]                 

class DeterministicConfidenceTool(BaseTool):
    name: str = "Deterministic Confidence Scoring Tool"
    description: str = "Computes a deterministic, reproducible confidence score for synthetic lethality evidence between two genes"
    args_schema: Type[BaseModel] = ConfidenceScoringInput

    def _run(
    self,
    biomarker: str,
    target: str,
    sl_results: Dict[str, Any],
    biomarker_results: Dict[str, Any],
    target_results: Dict[str, Any],
    opentargets_results: Dict[str, Any],
    trials_results: Dict[str, Any]
    ) -> str:
    
        sl_data = sl_results
        biomarker_data = biomarker_results
        target_data = target_results
        trials_data = trials_results

        # DEBUG: Dump inputs to JSON file for inspection
        debug_dir = "oncosynth/logs/_debug_inputs"
        os.makedirs(debug_dir, exist_ok=True)

        with open(f"{debug_dir}/{biomarker}_{target}_inputs.json", "w") as f:
            json.dump({
                "biomarker": biomarker,
                "target": target,
                "sl_results": sl_data,
                "biomarker_results": biomarker_data,
                "target_results": target_data,
                "opentargets_results": opentargets_results,
                "trials_results": trials_data
            }, f, indent=2)

        total_score = 0
        breakdown = []
        breakdown.append(f"CONFIDENCE SCORE BREAKDOWN FOR {biomarker} – {target}")
        breakdown.append("=" * 60)

        # ----------------------------------------
        # 1. DIRECT SL EVIDENCE (40 points max)
        # ----------------------------------------
        sl_score = 0
        explicit_hit = False
        functional_hit = False

        # Process SL results from JSON
        sl_articles = sl_data.get("results", [])
        for article in sl_articles:
            title = article.get("title", "").lower()
            abstract = article.get("abstract", "").lower()
            combined_text = title + " " + abstract

            if any(kw in combined_text for kw in [
                "synthetic lethality", "synthetic lethal", "synthetically lethal", 
                "lethal sensitivity", "lethal interaction", "synergistic lethality"
            ]):
                explicit_hit = True

            if any(kw in combined_text for kw in [
                "inhibitor", "synergy", "synergistic", "knockdown", "knockout", 
                "essential", "dependency", "sensitization", "replication stress", 
                "dna damage", "cell cycle", "apoptosis"
            ]):
                functional_hit = True

        if explicit_hit:
            sl_score += 30
            breakdown.append("✅ Explicit SL mention found in PubMed results: 30 points")
        else:
            breakdown.append("❌ No explicit SL mention in PubMed results: 0 points")

        if functional_hit:
            sl_score += 10
            breakdown.append("✅ Functional perturbation evidence present: 10 points")
        else:
            breakdown.append("❌ No functional evidence terms found: 0 points")

        total_score += sl_score
        breakdown.append(f"→ SUBTOTAL: {sl_score}/40 points")

        # ----------------------------------------
        # 2. DRUGGABILITY (30 points max) 
        # ----------------------------------------
        drug_score = 0

        for gene in [biomarker, target]:
            gene_score = 0
            gene_data = opentargets_results.get(gene.upper(), {})

            tract = gene_data.get("TRACTABILITY_SCORES", {}).get("SMALL_MOLECULE", {})
            known_drugs = gene_data.get("KNOWN_DRUGS", [])

            # Count number of True flags
            tractability_keys = [
                "Approved Drug", "Advanced Clinical", "Phase 1 Clinical",
                "Structure with Ligand", "High-Quality Ligand",
                "High-Quality Pocket", "Med-Quality Pocket", "Druggable Family"
            ]
            true_flags = sum(1 for k in tractability_keys if tract.get(k) is True)

            if true_flags >= 6:
                gene_score = 15
                breakdown.append(f"✅ {gene}: 15 points - HIGH tractability ({true_flags}/8 flags)")
            elif 3 <= true_flags <= 5:
                gene_score = 10
                breakdown.append(f"⚠️ {gene}: 10 points - MODERATE tractability ({true_flags}/8 flags)")
            elif true_flags > 0 or known_drugs:
                gene_score = 5
                breakdown.append(f"⚠️ {gene}: 5 points - LOW tractability ({true_flags}/8 flags)")
            else:
                breakdown.append(f"❌ {gene}: 0 points - no tractability or drug evidence")

            drug_score += gene_score

        # ----------------------------------------
        # 3. CLINICAL EVIDENCE (15 points max)
        # ----------------------------------------
        clinical_score = 0
        
        # Check trials data for each gene
        biomarker_trials = trials_data.get(f"{biomarker.upper()} Trials", [])
        target_trials = trials_data.get(f"{target.upper()} Trials", [])

        biomarker_has_trials = len(biomarker_trials) > 0
        target_has_trials = len(target_trials) > 0

        if biomarker_has_trials:
            clinical_score += 7
            breakdown.append(f"✅ {biomarker} has clinical trials: 7 points")
        else:
            breakdown.append(f"❌ {biomarker} no trials: 0 points")

        if target_has_trials:
            clinical_score += 8
            breakdown.append(f"✅ {target} has clinical trials: 8 points")
        else:
            breakdown.append(f"❌ {target} no trials: 0 points")

        total_score += clinical_score
        breakdown.append(f"→ SUBTOTAL: {clinical_score}/15 points")

        # ----------------------------------------
        # 4. CANCER RELEVANCE (15 points max)
        # ----------------------------------------
        cancer_score = 0
        cancer_keywords = ["cancer", "tumor", "carcinoma", "oncology", "malignant", "lymphoma", "leukemia"]
        ovarian_keywords = ["ovarian cancer", "ovarian carcinoma", "ovarian tumor", "ovarian neoplasm"]

        # Check biomarker articles
        biomarker_articles = biomarker_data.get("results", [])
        biomarker_has_cancer = False
        biomarker_has_ovarian = False

        for article in biomarker_articles:
            combined_text = (article.get("title", "") + " " + article.get("abstract", "")).lower()
            if not biomarker_has_cancer and any(kw in combined_text for kw in cancer_keywords):
                biomarker_has_cancer = True
            if not biomarker_has_ovarian and any(kw in combined_text for kw in ovarian_keywords):
                biomarker_has_ovarian = True
            if biomarker_has_cancer and biomarker_has_ovarian:
                break

        # Check target articles
        target_articles = target_data.get("results", [])
        target_has_cancer = False
        target_has_ovarian = False

        for article in target_articles:
            combined_text = (article.get("title", "") + " " + article.get("abstract", "")).lower()

            if not target_has_cancer and any(kw in combined_text for kw in cancer_keywords):
                target_has_cancer = True

            if not target_has_ovarian and any(kw in combined_text for kw in ovarian_keywords):
                target_has_ovarian = True

            if target_has_cancer and target_has_ovarian:
                break
            

        if biomarker_has_cancer:
            cancer_score += 5
            breakdown.append(f"✅ {biomarker} cancer relevance: 5 points")
        else:
            breakdown.append(f"❌ {biomarker} no cancer relevance: 0 points")

        if target_has_cancer:
            cancer_score += 5
            breakdown.append(f"✅ {target} cancer relevance: 5 points")
        else:
            breakdown.append(f"❌ {target} no cancer relevance: 0 points")

        if biomarker_has_ovarian or target_has_ovarian:
            cancer_score += 5
            breakdown.append("✅ Ovarian cancer relevance: 5 points")
        else:
            breakdown.append("❌ No ovarian cancer relevance: 0 points")

        total_score += cancer_score
        breakdown.append(f"→ SUBTOTAL: {cancer_score}/15 points")

        # ----------------------------------------
        # Final score + interpretation
        # ----------------------------------------
        breakdown.append(f"\n{'=' * 60}")
        breakdown.append(f"FINAL CONFIDENCE SCORE: {total_score}/100")
        breakdown.append(f"{'=' * 60}")

        if total_score >= 70:
            breakdown.append("🎯 INTERPRETATION: HIGH CONFIDENCE")
        elif total_score >= 40:
            breakdown.append("⚠️ INTERPRETATION: MEDIUM CONFIDENCE")
        else:
            breakdown.append("❌ INTERPRETATION: LOW CONFIDENCE")

        return "\n".join(breakdown)

# ---------------------------
# Markdown REPORT WRITER
# ---------------------------

def write_markdown_report(biomarker, target, content, folder="oncosynth/reports"):
    filename = f"{folder}/{biomarker}_{target}_report.md"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    final_text = content.output if hasattr(content, "output") else str(content)

    with open(filename, "w") as f:
        f.write(f"# Synthetic Lethality Report: {biomarker} - {target}\n\n")
        f.write(final_text)

    print(f"✅ Markdown report saved to {filename}")

# ---------------------------
# LOGS 
# ---------------------------
    
def log_agent_output(biomarker, target, agent_name, content):
    filename = f"oncosynth/logs/{biomarker}_{target}/{biomarker}_{target}_{agent_name}.md"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(content.output if hasattr(content, "output") else str(content))

# ---------------------------
# CREW SETUP
# ---------------------------

def run_research(biomarker, target):

    sl_pair_tool = SLPairSearchTool()
    biomarker_tool = BiomarkerPubMedSearchTool()
    target_tool = TargetPubMedSearchTool()
    confidence_tool = DeterministicConfidenceTool()

    hgnc_map = load_hgnc_map(os.path.join(os.path.dirname(__file__), "assets", "gene_with_protein_product.txt"))
    opentargets_tool = OpenTargetsTool(hgnc_map)

    clinical_trials_tool = ClinicalTrialsTool()

    sl_pair_agent = Agent(
        role="SL PubMed Searcher",
        goal="Search PubMed for synthetic lethality between the biomarker and target gene pair",
        backstory="Specialist in extracting SL pair interactions from biomedical literature.",
        tools=[sl_pair_tool],
        allow_delegation=False,
        llm=None
    )

    biomarker_agent = Agent(
        role="Biomarker Literature Analyst",
        goal="Retrieve cancer-related PubMed studies about the biomarker gene",
        backstory="Expert in biomarker discovery through mining cancer-focused publications.",
        tools=[biomarker_tool],
        allow_delegation=False,
        llm=None
    )

    target_agent = Agent(
        role="Target Gene Literature Analyst",
        goal="Retrieve cancer-related PubMed studies about the target gene",
        backstory="Expert in evaluating potential therapeutic targets using literature evidence.",
        tools=[target_tool],
        allow_delegation=False,
        llm=None
    )

    opentargets = Agent(
    role="Drug Target Analyst",
    goal="Retrieve drug/inhibitor data for each gene from Open Targets",
    backstory="Expert in therapeutic targeting, mechanisms of action, and drug status evaluation.",
    tools=[opentargets_tool],
    allow_delegation=False,
    llm=None
    )

    trials = Agent(
    role="Clinical Trials Specialist",
    goal="Retrieve information about ongoing or past clinical trials for each gene",
    backstory="Specialist in mining ClinicalTrials.gov for drug development and biomarker translation data.",
    tools=[clinical_trials_tool],
    allow_delegation=False,
    llm=None
    )

    analyst = Agent(
        role="Biomedical Research Analyst",
        goal="Analyse and synthesise relevance of findings",
        backstory="Skilled at turning raw search data into structured biomedical insights with emphasis on cancer relevance.",
        verbose=True,
        llm=llm
    )

    writer = Agent(
        role="Scientific Technical Writer",
        goal="Write a clear, well-structured markdown report for clinicians and researchers.",
        backstory="Expert in translating dense biomedical content into readable, cited reports.",
        verbose=True,
        llm=llm
    )

    # Tasks
    sl_pair_task = Task(
    description=f"Use the SL PubMed Search Tool to search for synthetic lethal related studies involving {biomarker} and {target}.",
    expected_output="Raw JSON output from tool - do not summarize or reformat.",
    agent=sl_pair_agent
    )

    biomarker_task = Task(
    description=f"Use the Biomarker PubMed Search Tool to search for cancer-related studies involving {biomarker}.",
    expected_output="Raw JSON output from tool - do not summarize or reformat.",
    agent=biomarker_agent   
    )

    target_task = Task(
    description=f"Use the Target PubMed Search Tool to search for cancer-related studies involving {target}.",
    expected_output="Raw JSON output from tool - do not summarize or reformat.",
    agent=target_agent
    )

    drug_task = Task(
    description=f"Use the Open Targets Tool to search for drugs and tractability data for {biomarker} and {target}.",
    expected_output="Raw JSON output from Open Targets API - do not summarize or reformat.",
    agent=opentargets
    )

    clinical_task = Task(
    description=f"""
    Use the Clinical Trials Tool with input: genes=["{biomarker}", "{target}"].
    Return the raw JSON result. Do not summarize or reformat anything.
    """,
    expected_output="Raw JSON from tool with trials for both genes",
    agent=trials
    )

    analysis_task = Task(
    description=f"""
    You are analyzing ONLY the gene pair: **{biomarker} ({biomarker}) and {target} ({target})**.
    
    CRITICAL: You must ONLY discuss {biomarker} and {target}. Do not mention any other genes.
    
    Analyze the search results and assess their relevance to synthetic lethality between {biomarker} and {target} specifically.
    
    Focus on:
    - Whether {biomarker} and {target} show synthetic lethality
    - Cancer relevance of this specific pair
    - Therapeutic potential for {biomarker}-{target} interactions
    
    Do NOT analyze any other gene pairs. Only {biomarker} and {target}.
    """,
    expected_output=f"Analysis of {biomarker}-{target} synthetic lethality and therapeutic potential.",
    context=[sl_pair_task, biomarker_task, target_task, drug_task, clinical_task],
    agent=analyst
    )

    writing_task = Task(
        description=f"""
        Using only the search_task outputs, write a structured markdown report for the gene pair: **{biomarker} – {target}**.

        Mandatory report sections:
        1. **Background on Genes** – Summarise roles of both genes and their relevance to cancer, only if supported by the biomarker/target PubMed results.
        2. **SL Evidence (with PMIDs)** – Report abstracts that explicitly mention synthetic lethality between the pair. Cite with PMID and link.
        3. **Drug Targets (with Open Targets data)** – List known inhibitors, mechanism of action, approval status, and disease context.
        
        4. **Clinical Trials** – Summarise relevant trials involving either gene, with:
           - NCT ID
           - Trial Title (linked)
           - Phase
           - Status
           - Condition

        5. **Translational Potential (across cancers)** – If drugs or SL evidence exist in non-ovarian cancers, describe them briefly.
        6. **Conclusion** – Final summary and interpretation.

        7. **References** – List PMIDs using this format:
           - PMID: 12345678 – https://pubmed.ncbi.nlm.nih.gov/12345678  
             Title: Title of the study.

        Rules:
        - Only use real outputs from the SL, biomarker, and target PubMed tasks — do not fabricate PMIDs or titles.
        - Do not invent or infer SL if not clearly stated.
        - Do not fabricate drug or trial information.
        - If evidence is weak, say so explicitly. Do not add filler or boilerplate text.

        If the QA task result begins with `REJECTED:`, do not write a report. Return only the rejection message.

        Write in clean, clinical markdown.
        """,
        expected_output="A markdown-formatted report with real citations and structured sections.",
        context=[sl_pair_task, biomarker_task, target_task, drug_task, clinical_task],
        agent=writer
    )

    # Run crew
    crew = Crew(
        agents=[
            sl_pair_agent,
            biomarker_agent,
            target_agent,
            opentargets,
            trials,
            analyst,
            writer
        ],
        tasks=[
            sl_pair_task,
            biomarker_task,
            target_task,
            drug_task,
            clinical_task,
            analysis_task,
            writing_task
        ],
        verbose=True,
        process=Process.sequential
    )

    result = crew.kickoff()

# ---------------------------
# MANUAL CONFIDENCE SCORING 
# ---------------------------

    def extract_and_parse_json(task_output, name="unknown"):
        raw_str = task_output.raw if hasattr(task_output, 'raw') else str(task_output)
        debug_path = f"oncosynth/logs/_debug_inputs/{name}_raw.txt"
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        with open(debug_path, "w") as f:
            f.write(raw_str)

        try:
            # Fix bad unicode escapes like \u20 (incomplete)
            cleaned_str = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', raw_str)
            return json.loads(cleaned_str)
        except json.JSONDecodeError:
            # Try to extract inner JSON block if extra text surrounds it
            match = re.search(r'(\{.*\})', raw_str, re.DOTALL)
            if match:
                json_str = match.group(1)
                try:
                    cleaned_inner = re.sub(r'\\u(?![0-9a-fA-F]{4})', r'\\\\u', json_str)
                    return json.loads(cleaned_inner)
                except Exception as inner_e:
                    raise ValueError(f"⚠️ {name} JSON extraction failed (inner parse): {inner_e}")
            else:
                raise ValueError(f"⚠️ {name} JSON extraction failed: Could not find valid JSON in raw output.")

    sl_data = extract_and_parse_json(sl_pair_task.output, name="sl_pair")
    biomarker_data = extract_and_parse_json(biomarker_task.output, name="biomarker")
    target_data = extract_and_parse_json(target_task.output, name="target")
    opentargets_data = extract_and_parse_json(drug_task.output, name="opentargets")
    trials_data = extract_and_parse_json(clinical_task.output, name="clinicaltrials")

    try:
        confidence_result = confidence_tool._run(
            biomarker=biomarker,
            target=target,
            sl_results=sl_data,
            biomarker_results=biomarker_data,
            target_results=target_data,
            opentargets_results=opentargets_data,
            trials_results=trials_data
        )
        print("✅ Manual confidence scoring completed!")
    except Exception as e:
        confidence_result = "Error in confidence scoring"
        print(f"❌ Manual confidence scoring failed: {e}")

# ---------------------------
# LOGS
# ---------------------------
    log_agent_output(biomarker, target, "sl_pair_pubmed", sl_pair_task.output)
    log_agent_output(biomarker, target, "biomarker_pubmed", biomarker_task.output)
    log_agent_output(biomarker, target, "target_pubmed", target_task.output)
    log_agent_output(biomarker, target, "opentargets", drug_task.output)
    log_agent_output(biomarker, target, "clinicaltrials", clinical_task.output)
    log_agent_output(biomarker, target, "analysis", analysis_task.output)
    log_agent_output(biomarker, target, "confidence", type('obj', (object,), {'output': confidence_result})())
    log_agent_output(biomarker, target, "writer", result)

# ---------------------------
# Extract Confidence Score
# ---------------------------
    score_match = re.search(r"FINAL CONFIDENCE SCORE:\s*(\d+)", confidence_result)
    score = int(score_match.group(1)) if score_match else 0

# ---------------------------
# Build final report header
# ---------------------------
    if score < 50:
        header = f"⚠️ LOW CONFIDENCE REPORT ({score}/100)\n\n"
    else:
        header = f"✅ HIGH CONFIDENCE REPORT ({score}/100)\n\n"

# ---------------------------
# Save report (always to same folder)
# ---------------------------
    final_text = result.output if hasattr(result, "output") else str(result)
    final_report = header + final_text
    
    write_markdown_report(biomarker, target, final_report)
    print(f"📄 Report written for {biomarker}–{target} with DETERMINISTIC score {score}")