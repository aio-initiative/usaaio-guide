"""Compile the original TikZ diagrams used by the site.

Requires pdflatex and PyMuPDF. Outputs are committed for static hosting.
"""
from pathlib import Path
import subprocess, tempfile, shutil
import fitz

root = Path(__file__).resolve().parents[2]
out = root / 'public/assets/diagrams'
out.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    result = subprocess.run(['/Library/TeX/texbin/pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory',tmp,str(Path(__file__).with_name('ui-diagrams.tex'))],stdout=subprocess.PIPE,text=True)
    if result.returncode:
        raise RuntimeError(result.stdout[-3500:])
    doc = fitz.open(Path(tmp)/'ui-diagrams.pdf')
    for page,name in zip(doc,['descent','linear-algebra','calculus','probability','classification','graphs','attention']):
        rects = [d['rect'] for d in page.get_drawings()] + [fitz.Rect(b[:4]) for b in page.get_text('blocks')]
        box = fitz.Rect(rects[0])
        for rect in rects[1:]: box |= rect
        box += (-8,-8,8,8)
        page.set_cropbox(box & page.rect)
        (out/(name+'.svg')).write_text(page.get_svg_image(text_as_path=True))
print('Built seven TikZ SVGs.')
