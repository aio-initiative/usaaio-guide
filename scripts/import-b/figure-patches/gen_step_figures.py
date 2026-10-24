"""Patch for B's `gen_step_figures.py`: no step numbers in the labels.

The step modules were merged into five-part tracks, so "Step 2" and "Step 4"
no longer name anything on the site. Two labels drawn inside figures still
did. This renames them and leaves every other line of the generator as B
wrote it:

- `vision-patch-embedding`: "Step 2's map at stride 3" -> "vertical map at stride 3"
- `generative-reverse-step`: "one jump, Step 4" -> "one jump"
"""

RENAMES = {
    "Step 2's map at stride 3": "vertical map at stride 3",
    "one jump, Step 4": "one jump",
}


def apply(module):
    original = module.label

    def label(ax, x, y, text, *args, **kwargs):
        return original(ax, x, y, RENAMES.get(text, text), *args, **kwargs)

    # The figure functions look `label` up in the module's globals at call
    # time, so replacing it there renames the text without copying them.
    module.label = label
