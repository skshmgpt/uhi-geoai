"""Build the IEEE two-column version of docs/paper-draft.md.

Usage: python3 docs/ieee/build.py   (needs pandoc and tectonic; writes docs/ieee/paper_ieee.tex and docs/paper_ieee.pdf)
"""
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent
N_REFS = 25

TABLE_CAPTIONS = [
    "Coverage of the pipeline stages by prior work",
    "Datasets",
    "Implementation components",
    "Evaluation protocol",
    "Segmentation on 2024 test tiles (weak labels), mean ± SD over 3 seeds (%)",
    "mIoU (%) by test year; 2025 imagery was never seen in training",
    "LST of our inversion against USGS L2 ST\\_B10 (2023 / 2024 / 2025)",
    "Masked hotspot counts after clustering and the 0.5 ha cut",
    "Top-20 hotspot stability (E5)",
    "Cooling model, ten-fold spatial cross-validation on 2 km blocks (2023 / 2024 / 2025)",
]
ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]


def sec(m):
    major, minor = int(m.group(2)), m.group(4)
    out = ROMAN[major] + (f"-{chr(64 + int(minor))}" if minor else "")
    return f"{m.group(1)}\u00a0{out}"


def md_to_tex(md, extra=()):
    return subprocess.run(["pandoc", "-f", "markdown", "-t", "latex", *extra], input=md, text=True,
                          capture_output=True, check=True).stdout


def cites(text):
    def repl(m):
        nums = [int(x) for x in re.split(r",\s*", m.group(1))]
        if all(1 <= x <= N_REFS for x in nums):
            return "\\cite{" + ",".join(f"b{x}" for x in nums) + "}"
        return m.group(0)
    return re.sub(r"\\\[(\d+(?:,\s*\d+)*)\\\]", repl, text)


def main():
    src = (DOCS / "paper-draft.md").read_text()
    title = src.splitlines()[0].lstrip("# ").strip()
    abstract = src.split("## Abstract", 1)[1].split("**Keywords:**", 1)[0].strip()
    keywords = src.split("**Keywords:**", 1)[1].split("\n", 1)[0].strip()
    body = src.split("## 1. Introduction", 1)[1].split("## References", 1)[0]
    body = "## 1. Introduction" + body
    refs = re.findall(r"^\d+\. (.+)$", src.split("## References", 1)[1], flags=re.M)
    assert len(refs) == N_REFS, len(refs)

    disclosure = re.search(r"\*\*Disclosure\.\*\* (.+)", body).group(1)
    body = re.sub(r"\n\*\*Disclosure\.\*\* .+\n", "\n", body)

    # Equations: numbered, scaled to the column when long.
    def eq(m):
        parts = [p.strip() for p in re.split(r",?\s*\\q?quad\s*", m.group(1).strip()) if p.strip()]
        fit = lambda p: p if len(p) < 110 else "\\resizebox{0.95\\columnwidth}{!}{$\\displaystyle " + p + "$}"
        e = fit(parts[0]) if len(parts) == 1 else "\\begin{gathered}\n" + " \\\\\n".join(fit(p) for p in parts) + "\n\\end{gathered}"
        return "\\begin{equation}\n" + e + "\n\\end{equation}"
    body = re.sub(r"```latex\n(.*?)\n```", eq, body, flags=re.S)

    # Cross-references to IEEE numbering.
    body = re.sub(r"Sections (\d)–(\d)", lambda m: f"Sections {ROMAN[int(m.group(1))]}–{ROMAN[int(m.group(2))]}", body)
    body = re.sub(r"(Section|§)\s?((\d))(?:\.(\d))?", lambda m: sec(m).replace("§\u00a0", "Section\u00a0"), body)
    body = re.sub(r"\(Figure (\d)\)", lambda m: "(Fig.\u00a0\\ref{fig" + m.group(1) + "})", body)
    body = re.sub(r"^(#+) \d+(\.\d+)?\.? ", r"\1 ", body, flags=re.M)
    body = cites(body)

    meta = "---\ntablecaptions:\n" + "".join(f"  - '{c}'\n" for c in TABLE_CAPTIONS) + "---\n"
    body_tex = md_to_tex(meta + body, ["--shift-heading-level-by=-1", f"--lua-filter={HERE / 'tables.lua'}"])
    bib = "\n".join(f"\\bibitem{{b{i}}} " + md_to_tex(r).strip() for i, r in enumerate(refs, 1))

    disclosure = re.sub(r"(Section|§)\s?((\d))(?:\.(\d))?", lambda m: sec(m).replace("§\u00a0", "Section\u00a0"), disclosure)
    disclosure_tex = md_to_tex(disclosure).strip()

    tex = rf"""\documentclass[conference]{{IEEEtran}}
\usepackage{{fontspec}}
\setmainfont{{Times New Roman}}
\setmonofont{{Menlo}}[Scale=0.85]
\usepackage{{amsmath,amssymb}}
\usepackage{{newunicodechar}}
\newunicodechar{{₀}}{{\textsubscript{{0}}}}\newunicodechar{{₁}}{{\textsubscript{{1}}}}\newunicodechar{{₂}}{{\textsubscript{{2}}}}
\newunicodechar{{₃}}{{\textsubscript{{3}}}}\newunicodechar{{₄}}{{\textsubscript{{4}}}}
\newunicodechar{{⁻}}{{\textsuperscript{{$-$}}}}\newunicodechar{{⁴}}{{\textsuperscript{{4}}}}\newunicodechar{{∈}}{{$\in$}}
\usepackage{{graphicx,booktabs,tabularx,array}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{xurl}}
\usepackage{{cite}}
\providecommand{{\tightlist}}{{\setlength{{\itemsep}}{{0pt}}\setlength{{\parskip}}{{0pt}}}}
\newcommand{{\passthrough}}[1]{{#1}}
\begin{{document}}
\title{{{md_to_tex(title).strip()}}}
\author{{\IEEEauthorblockN{{Saksham Gupta}}
\IEEEauthorblockA{{\textit{{Affiliation to be added}}}}}}
\maketitle
\begin{{abstract}}
{md_to_tex(cites(abstract)).strip()}
\end{{abstract}}
\begin{{IEEEkeywords}}
{keywords}
\end{{IEEEkeywords}}

{body_tex}
\section*{{AI Use Disclosure}}
{disclosure_tex}

\begin{{thebibliography}}{{00}}
{bib}
\end{{thebibliography}}
\end{{document}}
"""
    (HERE / "paper_ieee.tex").write_text(tex)
    subprocess.run(["tectonic", "-X", "compile", "--keep-logs", "paper_ieee.tex"], cwd=HERE, check=True)
    shutil.copy(HERE / "paper_ieee.pdf", DOCS / "paper_ieee.pdf")


if __name__ == "__main__":
    main()
