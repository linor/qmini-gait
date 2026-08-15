# Papers

**These two PDFs are not in the repo — drop them in here.** They live in the
Claude Desktop "BDX" project knowledge, which the packaging step could not
reach (it is outside the sandbox mount). Nothing else is missing.

Save them with exactly these names, which is what `docs/PAPERS.md` and
`CLAUDE.md` refer to:

| save as | paper |
|---|---|
| `stylized-gaits-2024.pdf` | Hopkins, Wiedebach, Cesare, Bishop, Knoop & Bächer, *Interactive Design of Stylized Walking Gaits for Robotic Characters*, ACM TOG 43(4), 2024 — 15 pp. |
| `bipedal-character-2024.pdf` | Grandia, Knoop, Hopkins, Wiedebach, Bishop, Pickles, Müller & Bächer, *Design and Control of a Bipedal Robotic Character*, RSS 2024 — 14 pp. |

They are excluded by `.gitignore` — published PDFs generally should not go into
a git repo. Remove that line if you would rather they were tracked.

## Why this matters more than usual here

`CLAUDE.md` and the source comments cite the first paper constantly, by
section, table, figure and equation number — "Sec. 5.3 Eqs 8–10", "Tab. 2
function curves", "Fig. 5 reverse foot lock". Without the PDF present, Claude
Code reads those as opaque labels and cannot check the implementation against
what the paper actually says.
