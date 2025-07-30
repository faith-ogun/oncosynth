from oncosynth.common import run_research

def run_research_batch(biomarker, target):
    run_research(biomarker, target)

# ---------------------------
# RUN BATCH MODE
# ---------------------------

if __name__ == "__main__":
    import pandas as pd

    input_csv = "agents/assets/clean_gene_pairs.csv"
    df = pd.read_csv(input_csv)

    for idx, row in df.iterrows():
        biomarker = str(row.get("Biomarker_HGNC", "")).strip().upper()
        target = str(row.get("TargetGene_HGNC", "")).strip().upper()

        if not biomarker or not target:
            print(f"Skipping row {idx+1} due to missing gene symbol(s)")
            continue

        print(f"\nRunning batch {idx+1}/{len(df)}: {biomarker} – {target}")
        try:
            run_research_batch(biomarker, target)
        except Exception as e:
            print(f"Failed for pair {biomarker}-{target}: {e}")
