"""Build the TWC manuscript version of the paper.

The source file carries both versions behind an \\iffull switch and is written
for the one-column reading layout. This script leaves that file untouched: it
copies it to a scratch directory, flips the switch to \\fullfalse and the class
to IEEEtran 10pt/twocolumn/journal, compiles, and drops the result next to the
source as `manuscript.pdf`.

    python "Research Paper/build_manuscript.py"          # manuscript (13-page target)
    python "Research Paper/build_manuscript.py" --full   # the complete version

Prints the page count and the four numbers worth watching after every build:
fatal errors, undefined references or citations, LaTeX warnings, overfull boxes.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "HQC-HAC for Optimizing Sum-Rate with IRS-Assisted.tex")
BIB = os.path.join(HERE, "reference.bib")
ASSET_DIRS = ("figures", "images")

ONECOL = r"\documentclass[12pt,onecolumn]{IEEEtran}"
TWOCOL = r"\documentclass[10pt,twocolumn,journal]{IEEEtran}"

MIKTEX = r"C:\Users\aacl\AppData\Local\Programs\MiKTeX\miktex\bin\x64"


def _wsl_path(win):
    """C:\a\b -> /mnt/c/a/b, so a WSL run can reach a Windows MiKTeX."""
    if len(win) > 2 and win[1] == ":":
        return "/mnt/" + win[0].lower() + win[2:].replace("\\", "/")
    return win


def under_wsl():
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def tools():
    """(pdflatex, bibtex, using_windows_exe).

    WSL has no TeX of its own here, so a WSL run borrows the Windows MiKTeX
    through /mnt/c. That works, but a Windows executable cannot see a Linux-only
    directory such as /tmp, which is why the scratch directory is created beside
    the source rather than in the system temp.
    """
    p, b = shutil.which("pdflatex"), shutil.which("bibtex")
    if p and b:
        return p, b, False
    for d in (MIKTEX, _wsl_path(MIKTEX)):
        pe, be = os.path.join(d, "pdflatex.exe"), os.path.join(d, "bibtex.exe")
        if os.path.isfile(pe) and os.path.isfile(be):
            return pe, be, True
    sys.exit("cannot find pdflatex/bibtex on PATH, in %s, or at %s\n"
             "  (WSL has no TeX installed; the Windows MiKTeX is used instead)"
             % (MIKTEX, _wsl_path(MIKTEX)))


def build(full, out_name):
    pdflatex, bibtex, win_exe = tools()
    # keep the scratch directory beside the source: a Windows pdflatex launched
    # from WSL cannot reach /tmp, and this path is reachable from both sides
    work = tempfile.mkdtemp(prefix=".build_", dir=HERE if win_exe and under_wsl()
                            else None)
    stem = "paper"

    src = open(SRC, encoding="utf-8", newline="").read()
    if not full:
        lines = src.split("\n")
        hits = 0
        for i, l in enumerate(lines):
            if l.strip() == "\\fulltrue":          # the switch, not the comment
                lines[i], hits = "\\fullfalse", hits + 1
        if hits != 1:
            sys.exit("expected exactly one bare \\fulltrue line, found %d" % hits)
        src = "\n".join(lines)
        if ONECOL not in src:
            sys.exit("documentclass line not found; update ONECOL in this script")
        src = src.replace(ONECOL, TWOCOL)

    open(os.path.join(work, stem + ".tex"), "w", encoding="utf-8", newline="").write(src)
    shutil.copy(BIB, work)
    for d in ASSET_DIRS:
        s = os.path.join(HERE, d)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(work, d))

    quiet = {"cwd": work, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    subprocess.call([pdflatex, "-interaction=nonstopmode", stem + ".tex"], **quiet)
    subprocess.call([bibtex, stem], **quiet)
    # three passes: adding a citation renumbers the list, which moves labels and
    # needs one more resolution round than the usual two
    for _ in range(3):
        subprocess.call([pdflatex, "-interaction=nonstopmode", stem + ".tex"], **quiet)

    log = open(os.path.join(work, stem + ".log"), encoding="utf-8",
               errors="replace").read()
    pages = re.search(r"on %s\.pdf \((\d+) pages" % stem, log)
    n = lambda p: len(re.findall(p, log, re.I))
    print("%-11s %s pages   ERR=%d  undef=%d  warn=%d  overfull=%d"
          % (out_name.replace(".pdf", "").upper(),
             pages.group(1) if pages else "?",
             len(re.findall(r"(?m)^! ", log)),
             n(r"undefined (reference|citation)"),
             len(re.findall(r"LaTeX Warning", log)),
             n(r"Overfull .hbox")))
    for m in re.findall(r"(?m)^! .*|Reference `[^']+' .*undefined|"
                        r"Citation `[^']+' .*undefined", log)[:6]:
        print("   ", m.strip())

    made = os.path.join(work, stem + ".pdf")
    if not os.path.isfile(made):
        sys.exit("no PDF produced; scratch dir kept at " + work)
    dst = os.path.join(HERE, out_name)
    shutil.copy(made, dst)
    # keep the .aux: it is the only reliable record of which floats and sections
    # THIS version actually typeset, which \iffull makes impossible to read off
    # the source by inspection
    shutil.copy(os.path.join(work, stem + ".aux"),
                os.path.join(HERE, out_name.replace(".pdf", ".aux")))
    shutil.rmtree(work, ignore_errors=True)
    print("    ->", dst)


if __name__ == "__main__":
    if "--full" in sys.argv:
        build(True, "completed-version.pdf")
    else:
        build(False, "manuscript.pdf")
