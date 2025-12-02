import pandas as pd
from pathlib import Path

def df_to_latex(df: pd.DataFrame, out_path: str, caption: str = "", label: str = ""):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tex = df.to_latex(index=False, escape=True, caption=caption, label=label)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex)
