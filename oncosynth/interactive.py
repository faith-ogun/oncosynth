from oncosynth.common import run_research

def run_research_interactive():
    print("🔬 SL Report Generator")
    biomarker = input("Enter biomarker gene symbol (e.g. MYC): ").strip().upper()
    target = input("Enter target gene symbol (e.g. CHEK1): ").strip().upper()
    run_research(biomarker, target)

# ---------------------------
# RUN INTERACTIVE MODE
# ---------------------------

if __name__ == "__main__":
    run_research_interactive()
