# AIO Initiative Topic Template

Use this template when adding a module under `content/`.

Create two files when the module has practice material:

- A `.md` lesson in the correct stage folder, named `<stage>.<n>-<CODE>-<slug>.md`.
- A matching `.problems.json` file in `content/problemBank/` if the module needs structured practice metadata.

## Module Template

The site parses these headings by name, so use them exactly and in this order.

```markdown
# 2.8 · RP2 — Topic Title Here

> **Goal:** one sentence saying what you will be able to do. Becomes the summary.
> **Stage:** Round 2

---

## Learning Outcomes

After this module you should be able to:

1. State each outcome as a task, not a topic.

---

## Overview

What the topic is, where it sits, and why a contest cares.

## Prerequisites

- [1.5 PS1](../1_Round_1/1.5-PS1-probability.md): what you need from it.

## Theory

Definitions, derivations, proofs. Use `### 1. Sub-topic` headings. Inline math
is `$...$`; display math is a `$$...$$` block.

## Intuition

The picture behind the algebra.

## From-Scratch Implementation

NumPy, no library shortcuts.

## Library Usage

The idiomatic scikit-learn / PyTorch version.

## Worked Example

Small numbers, carried through by hand.

## Further Practice

Extensions worth attempting.

## Practice Problems

**Problem 1 (topic).**
The question.

<details><summary>Answer</summary>

The full solution.
</details>
```

## Problem Bank Template

```json
{
  "MODULE_ID": "module-id-here",
  "practice": [
    {
      "uniqueId": "source-topic-number",
      "name": "Problem Name",
      "url": "https://example.com",
      "source": "USAAIO",
      "difficulty": "Normal",
      "isStarred": false,
      "tags": ["Topic", "Skill"],
      "solutionMetadata": {
        "kind": "external"
      }
    }
  ]
}
```

## Style Rules

- Begin with a concrete situation the learner can picture. Let a limitation or a question in that example motivate the next mathematical idea.
- Explain what an operation does before presenting its general formula. Introduce symbols close to where they are used, and carry a small example through the calculation.
- Give each paragraph a clear job. Expand compressed causal claims into the missing steps, but do not restate the same conclusion in the overview, theory, intuition, and worked example.
- Use the Intuition section to connect or reinterpret ideas already developed. Intuition belongs throughout the lesson, not only in that section.
- Replace vague emphasis such as "that distinction matters" with the consequence: what prediction, calculation, or design choice changes?
- Keep the voice conversational through clear reasoning and natural transitions. Do not manufacture informality with slang, filler questions, or exaggerated claims.
- Keep formal derivations, but annotate new equations with the role of their terms: what is being averaged, which direction an update moves, or how a sign changes the result. A label such as "regularizer" alone is not an explanation.
- Carry motivating examples into the general formula. Show the intermediate recurrence or calculation before naming the theorem or equation; a concrete opening should not be followed by an unrelated formal jump.
- Avoid stock announcements such as "three reasons", "this deserves its own section", "for free", or "the whole reason". Explain the actual operation, saving, or limitation in the surrounding context.
- Check theory, worked solutions, and code comments as well as openings. Replace compressed fragments with complete reasoning without weakening the mathematics or adding repeated summaries.
- Preserve assumptions and mathematical substance. Simplifying prose must not turn a tendency into a guarantee or delete the derivation a learner needs.
- Use reproducible numerical examples. Attribute empirical results to a source and setting; do not present invented benchmark numbers as observations.
- Give practice questions open outcomes when experiments could go either way. Ask learners to measure and explain results, rather than to confirm a promised improvement.
- Keep module IDs lowercase and hyphenated; the id is the slug part of the filename.
- Add the module id to a chapter in the `CHAPTERS` table in `scripts/build-content.mjs`.
- Never hand-edit `content/ordering.js`, `content/catalog.js`, or `content/lessons.js` — they are generated.
- Run `npm run build:content && npm run validate` before submitting changes.
