import argparse
from oncosynth.batch import run_research_batch as run_batch
from oncosynth.interactive import run_research_interactive as run_interactive

def main():
    parser = argparse.ArgumentParser(
        description="OncoSynth: Synthetic lethality agent framework"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-i", "--interactive", action="store_true",
        help="Run OncoSynth in interactive mode (CLI prompts)"
    )
    group.add_argument(
        "-b", "--batch", metavar="CSV_FILE",
        help="Run OncoSynth in batch mode using a CSV file"
    )

    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    elif args.batch:
        import pandas as pd
        
        df = pd.read_csv(args.batch)

        # Validate: must have at least two columns
        if df.shape[1] < 2:
            raise ValueError("CSV must contain at least two columns for biomarker and target.")

        # Auto-detect columns
        biomarker_col, target_col = df.columns[:2]

        for idx, row in df.iterrows():
            biomarker = str(row[biomarker_col]).strip().upper()
            target = str(row[target_col]).strip().upper()

            if not biomarker or not target:
                print(f"Skipping row {idx+1} due to missing gene symbol(s)")
                continue

            print(f"\nRunning batch {idx+1}/{len(df)}: {biomarker} – {target}")
            try:
                run_batch(biomarker, target)
            except Exception as e:
                print(f"❌ Failed for pair {biomarker}-{target}: {e}")

