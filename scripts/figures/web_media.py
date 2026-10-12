"""Curated, freely-licensed diagrams from Wikimedia Commons.

Some concepts are architecture diagrams, and matplotlib draws those badly —
the ResNet figure it produced plotted invented numbers as if they were the
published degradation result. For those, a canonical diagram beats anything
generated here.

Every entry below was opened and looked at before being listed, and every
licence was read off the Commons API rather than assumed. Automated image
search is not safe to trust unreviewed: the same searches turned up a nude
photograph for "ROI pooling" and a corporate logo for "VAE".

Run:  python3 scripts/figures/web_media.py          # download + install
      python3 scripts/figures/web_media.py --check  # verify only
"""
import io
import json
import pathlib
import sys
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
MEDIA = ROOT / "assets" / "module-media" / "web"
CREDITS = ROOT / "content" / "media-credits.json"
UA = {"User-Agent": "aioinitiative-guide/1.0 (https://www.aioinitiative.com; educational figures)"}

C = "https://commons.wikimedia.org/wiki/File:"
U = "https://upload.wikimedia.org/wikipedia/commons"

# slug -> where it goes and what it is.
#   heading: inserted after the first paragraph under this "## " heading
MEDIA_SET = [
    dict(slug="cv4-residual", module="CV4", heading="## Hierarchical Vision Architectures",
         alt="A residual block adds the input back to the output, so the layers only have to learn the difference",
         title="ResNet block.svg", author="Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola",
         lic="CC BY-SA 4.0", lic_url="https://creativecommons.org/licenses/by-sa/4.0",
         src=f"{U}/b/b3/ResNet_block.svg", page=f"{C}ResNet_block.svg", width=1200),

    dict(slug="tf2-families", module="TF2", heading="## Transformer Architecture Families",
         alt="In an encoder, self-attention lets every token attend to every other token",
         title="Encoder self-attention, block diagram.png", author="dvgodoy",
         lic="CC BY 4.0", lic_url="https://creativecommons.org/licenses/by/4.0",
         src=f"{U}/5/58/Encoder_self-attention%2C_block_diagram.png",
         page=f"{C}Encoder_self-attention,_block_diagram.png", width=1400),

    dict(slug="tf3-families", module="TF3", heading="## Transformer Architecture Families",
         alt="Causal masking stops a decoder token from attending to anything after it",
         title="Decoder self-attention with causal masking, detailed diagram.png", author="dvgodoy",
         lic="CC BY 4.0", lic_url="https://creativecommons.org/licenses/by/4.0",
         src=f"{U}/7/72/Decoder_self-attention_with_causal_masking%2C_detailed_diagram.png",
         page=f"{C}Decoder_self-attention_with_causal_masking,_detailed_diagram.png", width=1400),

    dict(slug="tf4-families", module="TF4", heading="## Transformer Architecture Families",
         alt="The full encoder-decoder transformer, with cross-attention joining the two stacks",
         title="The-Transformer-model-architecture.png", author="Yuening Jia",
         lic="CC BY-SA 3.0", lic_url="https://creativecommons.org/licenses/by-sa/3.0",
         src=f"{U}/8/8f/The-Transformer-model-architecture.png",
         page=f"{C}The-Transformer-model-architecture.png", width=900),

    dict(slug="rp1-latent", module="RP1", heading="## Autoencoders",
         alt="An autoencoder squeezes the input through a narrow code and reconstructs it",
         title="Autoencoder structure.png", author="Chervinskii",
         lic="CC BY-SA 4.0", lic_url="https://creativecommons.org/licenses/by-sa/4.0",
         src=f"{U}/2/28/Autoencoder_structure.png", page=f"{C}Autoencoder_structure.png", width=1200),

    dict(slug="rp3-latent", module="RP3", heading="## Variational Autoencoders",
         alt="The reparameterization trick moves the sampling step off the gradient path",
         title="Reparameterized Variational Autoencoder.png", author="EugenioTL",
         lic="CC BY-SA 4.0", lic_url="https://creativecommons.org/licenses/by-sa/4.0",
         src=f"{U}/1/11/Reparameterized_Variational_Autoencoder.png",
         page=f"{C}Reparameterized_Variational_Autoencoder.png", width=1200),

    dict(slug="nl2-bottleneck", module="NL2", heading="## LSTM and GRU",
         alt="The LSTM cell: a cell state that is added to rather than rewritten, controlled by gates",
         title="The LSTM cell.png", author="Guillaume Chevalier",
         lic="CC BY 4.0", lic_url="https://creativecommons.org/licenses/by/4.0",
         src=f"{U}/3/3b/The_LSTM_cell.png", page=f"{C}The_LSTM_cell.png", width=1200),

    dict(slug="cv6-patches", module="CV6", heading="## The architecture",
         alt="A ViT splits the image into patches, embeds each one, and feeds the sequence to a transformer encoder",
         title="Vision Transformer.svg", author="Aston Zhang, Zachary C. Lipton, Mu Li, Alexander J. Smola",
         lic="CC BY-SA 4.0", lic_url="https://creativecommons.org/licenses/by-sa/4.0",
         src=f"{U}/7/77/Vision_Transformer.svg", page=f"{C}Vision_Transformer.svg", width=1200),
]

BY_SLUG = {e["slug"]: e for e in MEDIA_SET}


def fetch(entry):
    """Ask the API for a render at our width.

    Building thumbnail URLs by hand is not allowed any more — upload.wikimedia
    rejects widths outside its own list with a 400. The API picks a legal size
    and renders SVG to PNG on the way.
    """
    api = ("https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=imageinfo"
           f"&iiprop=url&iiurlwidth={entry['width']}"
           f"&titles=File:{urllib.parse.quote(entry['title'])}")
    d = json.load(urllib.request.urlopen(urllib.request.Request(api, headers=UA), timeout=40))
    page = next(iter(d["query"]["pages"].values()))
    info = page["imageinfo"][0]
    url = info.get("thumburl") or info["url"]
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=40).read()


def install(data, slug):
    """Flatten any transparency onto white so the figure reads on the light page."""
    from PIL import Image
    im = Image.open(io.BytesIO(data))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        im = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im)
    im = im.convert("RGB")
    path = MEDIA / f"{slug}.png"
    im.save(path, optimize=True)
    return path, im.size


def main():
    check = "--check" in sys.argv
    MEDIA.mkdir(parents=True, exist_ok=True)
    credits = {}
    for e in MEDIA_SET:
        credits[e["slug"]] = {k: e[k] for k in
                              ("title", "author", "lic", "lic_url", "page", "module")}
        if check:
            ok = (MEDIA / f"{e['slug']}.png").exists()
            print(f"  {'ok     ' if ok else 'MISSING'} {e['slug']}")
            continue
        _, size = install(fetch(e), e["slug"])
        print(f"  {e['slug']:16s} {size[0]}x{size[1]:<5} {e['lic']:<12} {e['title'][:42]}")
    if not check:
        CREDITS.write_text(json.dumps(credits, indent=2) + "\n")
        print(f"\n  {len(credits)} credits -> {CREDITS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
