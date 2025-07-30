import os
import requests
from typing import Type, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field, PrivateAttr
from Bio import Entrez
from crewai import Agent, Task, Crew, Process, LLM
from langchain_openai import ChatOpenAI
from crewai.tools import BaseTool
import re

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
            all_entries = []
            seen_pmids = set()

            for query in queries:
                handle = Entrez.esearch(db="pubmed", term=query, retmax=10)
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
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"

                    all_entries.append(
                        f"Query: {query}\n- PMID: {pmid} – {url}\n  **Title**: {title.strip()}\n  **Abstract**: {abstract.strip()}"
                    )

            if not all_entries:
                return "No SL-specific PubMed results found."

            return "\n\n".join(all_entries)

        except Exception as e:
            return f"Error accessing PubMed SL pair search: {str(e)}"

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
            all_entries = []
            seen_pmids = set()

            for query in queries:
                handle = Entrez.esearch(db="pubmed", term=query, retmax=10)
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
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"

                    all_entries.append(
                        f"Query: {query}\n- PMID: {pmid} – {url}\n  **Title**: {title.strip()}\n  **Abstract**: {abstract.strip()}"
                    )

            if not all_entries:
                return f"No cancer-related results found for {biomarker}"

            return "\n\n".join(all_entries)

        except Exception as e:
            return f"Error accessing PubMed for biomarker: {str(e)}"

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
            all_entries = []
            seen_pmids = set()

            for query in queries:
                handle = Entrez.esearch(db="pubmed", term=query, retmax=10)
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
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}"

                    all_entries.append(
                        f"Query: {query}\n- PMID: {pmid} – {url}\n  **Title**: {title.strip()}\n  **Abstract**: {abstract.strip()}"
                    )

            if not all_entries:
                return f"No cancer-related results found for {target}"

            return "\n\n".join(all_entries)

        except Exception as e:
            return f"Error accessing PubMed for target: {str(e)}"


# ---------------------------
# TOOL: Open Targets Search
# ---------------------------

class OpenTargetsInput(BaseModel):
    target: str = Field(description="The target gene symbol")

class OpenTargetsTool(BaseTool):
    name: str = "Open Targets Tool"
    description: str = "Fetches drug info from Open Targets for the target gene"
    args_schema: Type[BaseModel] = OpenTargetsInput

    _hgnc_map: dict = PrivateAttr()

    def __init__(self, hgnc_map: dict):
        super().__init__()
        self._hgnc_map = hgnc_map  # ← use _hgnc_map, not hgnc_map

    def symbol_to_ensembl(self, symbol: str) -> Optional[str]:
        return self._hgnc_map.get(symbol)

    def _run(self, target: str) -> str:
        ensembl_id = self.symbol_to_ensembl(target)
        if not ensembl_id:
            return f"No Ensembl ID found for gene: {target}"

        query = """
        query KnownDrugsQuery($ensgId: String!, $cursor: String, $size: Int) {
          target(ensemblId: $ensgId) {
            knownDrugs(cursor: $cursor, size: $size) {
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

        variables = {
            "ensgId": ensembl_id,
            "cursor": None,
            "size": 20
        }

        url = "https://api.platform.opentargets.org/api/v4/graphql"
        res = requests.post(url, json={"query": query, "variables": variables})
        res.raise_for_status()
        drugs = res.json()["data"]["target"]["knownDrugs"]["rows"]

        if not drugs:
            return f"No drugs found for {target} ({ensembl_id})"

        output = [f"Drugs targeting {target} ({ensembl_id}):"]
        for d in drugs:
            drug_name = d["drug"]["name"]
            moa = d["mechanismOfAction"] or "N/A"
            disease = d["disease"]["name"]
            status = d["status"]
            phase = d["phase"]
            link = next((u["url"] for u in d["urls"] if u.get("url")), "N/A")
            output.append(
                f"- {drug_name}:\n"
                f"  • Phase: {phase}\n"
                f"  • Status: {status}\n"
                f"  • MoA: {moa}\n"
                f"  • Disease: {disease}\n"
                f"  • Link: {link}"
            )

        return "\n".join(output)

# ---------------------------
# TOOL: ClinicalTrials.gov Search
# ---------------------------

class ClinicalTrialsInput(BaseModel):
    gene: str = Field(description="Gene symbol to search for in clinical trial records")

class ClinicalTrialsTool(BaseTool):
    name: str = "Clinical Trials Tool"
    description: str = "Searches ClinicalTrials.gov for trials involving the gene"
    args_schema: Type[BaseModel] = ClinicalTrialsInput

    def _run(self, gene: str) -> str:
        res = requests.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params={"query.term": gene, "pageSize": 10}
        )
        res.raise_for_status()
        studies = res.json().get("studies", [])

        if not studies:
            return f"No clinical trials found for gene: {gene}."

        output = [f"Clinical trials mentioning {gene}:\n"]
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

                output.append(
                    f"- [{title}]({trial_url})\n"
                    f"  - **NCT ID**: {nct_id}\n"
                    f"  - **Phase**: {phase}\n"
                    f"  - **Status**: {trial_status}\n"
                    f"  - **Condition(s)**: {condition_str}"
                )
            except Exception as e:
                output.append(f"- Failed to parse study: {e}")

        return "\n\n".join(output)

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

    hgnc_map = load_hgnc_map(os.path.join(os.path.dirname(__file__), "assets", "gene_with_protein_product.txt"))
    opentargets_tool = OpenTargetsTool(hgnc_map)

    clinical_trials_tool = ClinicalTrialsTool()

    sl_pair_agent = Agent(
        role="SL PubMed Searcher",
        goal="Search PubMed for synthetic lethality between the biomarker and target gene pair",
        backstory="Specialist in extracting SL pair interactions from biomedical literature.",
        tools=[sl_pair_tool],
        allow_delegation=False
    )

    biomarker_agent = Agent(
        role="Biomarker Literature Analyst",
        goal="Retrieve cancer-related PubMed studies about the biomarker gene",
        backstory="Expert in biomarker discovery through mining cancer-focused publications.",
        tools=[biomarker_tool],
        allow_delegation=False
    )

    target_agent = Agent(
        role="Target Gene Literature Analyst",
        goal="Retrieve cancer-related PubMed studies about the target gene",
        backstory="Expert in evaluating potential therapeutic targets using literature evidence.",
        tools=[target_tool],
        allow_delegation=False
    )

    opentargets = Agent(
    role="Drug Target Analyst",
    goal="Retrieve drug/inhibitor data for each gene from Open Targets",
    backstory="Expert in therapeutic targeting, mechanisms of action, and drug status evaluation.",
    tools=[opentargets_tool],
    allow_delegation=False
    )

    trials = Agent(
    role="Clinical Trials Specialist",
    goal="Retrieve information about ongoing or past clinical trials for each gene",
    backstory="Specialist in mining ClinicalTrials.gov for drug development and biomarker translation data.",
    tools=[clinical_trials_tool],
    allow_delegation=False
    )

    analyst = Agent(
        role="Biomedical Research Analyst",
        goal="Analyse and synthesise relevance of findings",
        backstory="Skilled at turning raw search data into structured biomedical insights with emphasis on cancer relevance.",
        verbose=True,
        llm=llm
    )

    qa = Agent(
        role="Scientific QA Reviewer",
        goal="Ensure the report is focused, evidence-based, and cancer-relevant. If not, recommend further search.",
        backstory=(
            "You are a former editor at a top cancer research journal. You assess whether the findings are specific, actionable, and correctly sourced. "
            "If the output reads like generic filler, lacks citations, or overstates the evidence, you MUST reject the report and return feedback only."
        ),
        verbose=True,
        llm=llm
    )

    confidence_agent = Agent(
        role="Confidence Scoring Evaluator",
        goal="Score the strength of evidence for the SL interaction",
        backstory=(
            "You are a rigorous SL evidence evaluator. "
            "You score the strength of evidence based on peer-reviewed literature, drug info, cancer relevance, and quality of sources. "
            "You output a score from 0 to 100 and a justification for your score."
        ),
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
        description=f"""
        Search PubMed for **synthetic lethality** studies mentioning both **{biomarker}** and **{target}**.

        Focus strictly on co-mentions where synthetic lethality is explicitly discussed between the two genes.

        Include:
        - Up to 5 relevant abstracts.
        - PubMed IDs (PMIDs), titles, abstracts.
        - Direct PubMed links.

        This task is for SL-specific co-mentions, but you can mention general cancer, but NOT single-gene studies.
        """,
        expected_output="List of SL-specific PubMed abstracts for the gene pair with PMIDs and URLs.",
        agent=sl_pair_agent
    )

    biomarker_task = Task(
        description=f"""
        Search PubMed for **cancer-related studies** involving the biomarker gene: **{biomarker}**.

        Focus on:
        - General cancer relevance
        - Ovarian cancer specificity
        - Roles in gene regulation, DNA repair, tumour progression

        Include:
        - Up to 3 abstracts per query
        - PubMed IDs, titles, abstracts
        - Direct links

        This task should only return results related to the **biomarker**, not the target.
        """,
        expected_output=f"List of PubMed results focused on biomarker {biomarker}.",
        agent=biomarker_agent
    )

    target_task = Task(
        description=f"""
        Search PubMed for **cancer-related studies** involving the target gene: **{target}**.

        Focus on:
        - General cancer relevance
        - Ovarian cancer specificity
        - Roles in gene regulation, DNA repair, tumour progression

        Include:
        - Up to 3 abstracts per query
        - PubMed IDs, titles, abstracts
        - Direct links

        This task should only return results related to the **target**, not the biomarker.
        """,
        expected_output=f"List of PubMed results focused on target gene {target}.",
        agent=target_agent
    )

    drug_task = Task(
        description=f"""
        Use the Open Targets API to search for **drugs or inhibitors** associated with **{biomarker}** and **{target}**.

        Include:
        - Drug name, type, and approval status (e.g., Approved, Clinical Trial, Preclinical).
        - Disease context or cancer type targeted.
        - Mechanism of action (MoA) if available.
        - If no drug is found for a gene, state that clearly.

        Output must be structured by gene symbol.

        This task is strictly drug-focused. Do not include PubMed or trial data.
        """,
        expected_output="Structured drug data from Open Targets for each gene.",
        agent=opentargets
    )

    clinical_task = Task(
        description=f"""
        Search ClinicalTrials.gov for **active or past clinical trials** involving **{biomarker}** and **{target}**.

        Include for each trial:
        - Trial title
        - NCT ID
        - Trial phase (e.g. Phase I, II, III)
        - Status (e.g. Recruiting, Completed)
        - Condition/disease being studied
        - Direct link to trial

        Focus on **cancer-related** trials, especially those in **ovarian** cancer.

        This task is strictly ClinicalTrials.gov-focused. Do not include drug or PubMed data.
        """,
        expected_output="Structured list of relevant clinical trials involving either gene.",
        agent=trials
    )

    analysis_task = Task(
        description="Analyse the search results from all agents and assess their relevance to synthetic lethality, cancer context, and potential therapeutic value.",
        expected_output="A structured analysis of whether this gene pair is synthetically lethal and cancer-relevant, and whether any drug targets exist.",
        context=[sl_pair_task, biomarker_task, target_task, drug_task, clinical_task],
        agent=analyst
    )

    qa_task = Task(
        description="Review the analysis for clarity, depth, and relevance. If not strong, suggest more search or context refinement. If rejected, output must start with 'REJECTED: <reason>'.",
        expected_output="Approve or deny readiness for final writing. Suggest improvements if needed.",
        context=[analysis_task],
        agent=qa
    )

    confidence_task = Task(
    description=f"""
    You are evaluating the synthetic lethality (SL) strength of evidence for the gene pair **{biomarker} – {target}**.

    Score the pair using this rubric:

    1. **SL Evidence (PubMed) – 40 points**
       - Award **40/40** if any abstract in the SL PubMed search **explicitly** mentions "synthetic lethality" between {biomarker} and {target}.
       - Give 20–30 points for partial or indirect evidence (e.g. synergy, co-inhibition, pathway interaction).
       - Give 0–10 points if there's no clear relationship or relevance to SL.

    2. **Drug Evidence (Open Targets) – 25 points**
       - Award 20–25 if at least one drug is found with mechanism of action and disease context.
       - Give 10–15 for partial matches (e.g. drug but no disease).
       - Under 10 if only weak or exploratory drugs found.

    3. **Clinical Trials – 15 points**
       - Award 10–15 for active or past **cancer-related trials** involving either gene.
       - Partial credit for general studies or exploratory mentions.

    4. **Cancer-Relevant Literature (PubMed) – 20 points**
       - Award 15–20 if {biomarker} and/or {target} are mentioned in ovarian or other cancers.
       - Lower scores for weaker or tangential evidence.

    🔎 Use only the outputs of the SL PubMed search, Open Targets tool, ClinicalTrials.gov tool, and gene-specific PubMed tools — do not guess.

    Format your response exactly like this:

    ```
    Confidence Score: <score>/100
    SL Evidence (PubMed): <x>/40 – <justification>
    Drug Evidence (Open Targets): <x>/25 – <justification>
    Clinical Trials: <x>/15 – <justification>
    Cancer Literature (PubMed): <x>/20 – <justification>
    Reason: <Wrap-up summary>
    ```

    If no real evidence is found at all:
    ```
    Confidence Score: 10/100
    SL Evidence (PubMed): 0/40 – No evidence found
    Drug Evidence (Open Targets): 5/25 – No drug info
    Clinical Trials: 5/15 – No trials found
    Cancer Literature (PubMed): 0/20 – Gene not cancer-linked
    Reason: No meaningful data found to support SL, druggability or cancer relevance.
    ```
    """,
    expected_output="Structured score with component breakdown and rationale.",
    context=[
        sl_pair_task,
        biomarker_task,
        target_task,
        drug_task,
        clinical_task
    ],
    agent=confidence_agent
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
        context=[qa_task, sl_pair_task, biomarker_task, target_task, drug_task, clinical_task],
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
            qa,
            confidence_agent,
            writer
        ],
        tasks=[
            sl_pair_task,
            biomarker_task,
            target_task,
            drug_task,
            clinical_task,
            analysis_task,
            qa_task,
            confidence_task,
            writing_task
        ],
        verbose=True,
        process=Process.sequential
    )

    result = crew.kickoff()

    # Logs
    log_agent_output(biomarker, target, "sl_pair_pubmed", sl_pair_task.output)
    log_agent_output(biomarker, target, "biomarker_pubmed", biomarker_task.output)
    log_agent_output(biomarker, target, "target_pubmed", target_task.output)
    log_agent_output(biomarker, target, "opentargets", drug_task.output)
    log_agent_output(biomarker, target, "clinicaltrials", clinical_task.output)
    log_agent_output(biomarker, target, "analysis", analysis_task.output)
    log_agent_output(biomarker, target, "qa", qa_task.output)
    log_agent_output(biomarker, target, "confidence", confidence_task.output)
    log_agent_output(biomarker, target, "writer", result)

# ---------------------------
# Extract Confidence Score Safely
# ---------------------------
    conf_text = confidence_task.output.output if hasattr(confidence_task.output, "output") else str(confidence_task.output)

    # Extract score using regex for robustness
    score = 0
    score_match = re.search(r"Confidence Score:\s*(\d+)", conf_text)
    if score_match:
        score = int(score_match.group(1))

    # Check QA rejection
    qa_text = str(qa_task.output)
    rejected = qa_text.strip().startswith("REJECTED:")

# ---------------------------
# Save Report Based on Outcome
# ---------------------------
    if rejected:
        header = f"❌ QA REJECTED\n\n"
        fallback_text = header + (result.output if hasattr(result, "output") else str(result))
        write_markdown_report(biomarker, target, fallback_text, folder="oncosynth/low_confidence_reports")
        print("❌ QA rejected this report.")

    elif score < 50:
        header = f"⚠️ LOW CONFIDENCE REPORT ({score}/100)\n\n"
        fallback_text = header + (result.output if hasattr(result, "output") else str(result))
        write_markdown_report(biomarker, target, fallback_text, folder="oncosynth/low_confidence_reports")
        print(f"⚠️ Score too low ({score}/100) — saved to low confidence folder.")

    else:
        write_markdown_report(biomarker, target, result)