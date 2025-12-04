"""
pdf_proofread_agent.py

End-to-end pipeline to:
- Take a PDF
- For each page: send a PNG snapshot to OpenAI
- Let the model return a structured list of typos via function-calling
- Resolve each typo’s position in the original PDF using PyMuPDF
- Add sticky-note comments at the correct locations
- Save an annotated copy of the full PDF

This is set up for copy-editing: typos, missing periods, malformed
references, punctuation spacing, etc. Not for major content rewrites.
"""

import os
import io
import json
import base64
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from dotenv import load_dotenv; load_dotenv()
import fitz  # PyMuPDF
from PIL import Image
from openai import OpenAI  # pip install openai>=1.0.0

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------

OPENAI_MODEL = "gpt-5.1"  # or gpt-4.1, gpt-4.1-preview, etc.
DPI = 200                      # PNG rendering resolution

client = OpenAI()

# --------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------

@dataclass
class TypoIssue:
    """Single typo / copy-edit issue returned by the model."""
    search_text: str          # text to search for on the PDF page
    suggested_replacement: str
    comment: str              # human-readable explanation
    severity: str             # e.g. "typo", "punctuation", "style", "reference"
    # Optional: extra context for disambiguation
    context_before: Optional[str] = None
    context_after: Optional[str] = None


@dataclass
class PositionedTypo:
    """Typo with resolved position on the PDF page."""
    page_index: int
    issue: TypoIssue
    rect: fitz.Rect           # bounding box where annotation should go


# --------------------------------------------------------------------
# Utility: PDF → PNG per page
# --------------------------------------------------------------------

def pdf_to_page_pngs(pdf_path: str, out_dir: str, dpi: int = DPI) -> List[str]:
    """
    Render each page of the PDF to a PNG, save to out_dir.
    Returns list of PNG paths in page index order (0-based).
    """
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    png_paths = []

    for i, page in enumerate(doc):
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 pt = 1 inch
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = os.path.join(out_dir, f"page_{i+1:04d}.png")
        pix.save(out_path)
        png_paths.append(out_path)

    return png_paths


def encode_image_to_base64_png(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# --------------------------------------------------------------------
# OpenAI tools / function-calling schema
# --------------------------------------------------------------------

def get_tools_spec() -> List[Dict[str, Any]]:
    """
    Tool (function) schema for the model to report typos in a structured way.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "report_typos",
                "description": (
                    "Return a list of typos, punctuation errors, malformed "
                    "references, capitalization errors, and similar copy-editing "
                    "issues you see on this single PDF page image. "
                    "If there are no issues, return an empty list."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "issues": {
                            "type": "array",
                            "description": (
                                "List of issues found on this page. "
                                "Sometimes this may be empty."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "search_text": {
                                        "type": "string",
                                        "description": (
                                            "A short exact snippet of text that appears "
                                            "in the PDF and uniquely identifies the location "
                                            "of the issue on THIS page. "
                                            "Prefer 3–8 consecutive words that are unlikely "
                                            "to appear elsewhere on the page. "
                                            "This will be used with a text search in the real PDF."
                                        ),
                                    },
                                    "suggested_replacement": {
                                        "type": "string",
                                        "description": (
                                            "The corrected version of the problematic text. "
                                            "Include the surrounding words only if truly necessary."
                                        ),
                                    },
                                    "comment": {
                                        "type": "string",
                                        "description": (
                                            "Brief explanation of what is wrong and why this "
                                            "change is suggested. For example: "
                                            "'Remove extra space before comma', "
                                            "'Missing period at end of sentence', "
                                            "'Malformed reference style', etc."
                                        ),
                                    },
                                    "severity": {
                                        "type": "string",
                                        "enum": [
                                            "typo",
                                            "punctuation",
                                            "grammar",
                                            "reference",
                                            "formatting",
                                            "style",
                                        ],
                                        "description": "Rough category of the issue.",
                                    },
                                    "context_before": {
                                        "type": "string",
                                        "description": (
                                            "OPTIONAL: up to ~10 words immediately before search_text, "
                                            "to help disambiguate if the search text appears multiple times."
                                        ),
                                    },
                                    "context_after": {
                                        "type": "string",
                                        "description": (
                                            "OPTIONAL: up to ~10 words immediately after search_text."
                                        ),
                                    },
                                },
                                "required": [
                                    "search_text",
                                    "suggested_replacement",
                                    "comment",
                                    "severity",
                                ],
                            },
                        }
                    },
                    "required": ["issues"],
                },
            },
        }
    ]


# --------------------------------------------------------------------
# OpenAI call for a single page
# --------------------------------------------------------------------

def call_model_for_page_typos(
    png_path: str,
    page_index: int,
    model: str = OPENAI_MODEL,
) -> List[TypoIssue]:
    """
    Send a page PNG to the model. Model must respond by calling report_typos.
    Returns a list of TypoIssue objects (may be empty).
    """
    tools = get_tools_spec()
    image_b64 = encode_image_to_base64_png(png_path)
    image_url = f"data:image/png;base64,{image_b64}"

    system_prompt = (
        "You are a meticulous copy editor working on a scientific PDF. "
        "You are given ONE page at a time as an image. "
        "Your job is to find only real, objective issues like:\n"
        "- misspellings, typos\n"
        "- incorrect or missing punctuation\n"
        "- malformed references or citations\n"
        "- obvious grammar mistakes\n"
        "- inconsistent spacing (e.g., space before comma, double spaces)\n\n"
        "Do NOT propose stylistic rewrites or content changes. "
        "If the page has no issues, call report_typos with issues=[]. "
        "Return only objective issues you are confident about.\n"
        f"This is page index {page_index} (0-based indexing)."
    )

    user_prompt = (
        "Inspect this page for typos / copy-editing issues. "
        "Respond only by calling the function report_typos with a list of issues, "
        "or an empty list if there are none."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            },
        ],
        tools=tools,
        reasoning={ "effort": "medium" },
        tool_choice={"type": "function", "function": {"name": "report_typos"}},
    )

    choice = response.choices[0]
    if not choice.message.tool_calls:
        # Model failed to follow the tool protocol; treat as no issues.
        return []

    tool_call = choice.message.tool_calls[0]
    if tool_call.function.name != "report_typos":
        return []

    args = json.loads(tool_call.function.arguments)
    issues_raw = args.get("issues", [])
    issues: List[TypoIssue] = []

    for item in issues_raw:
        issues.append(
            TypoIssue(
                search_text=item["search_text"],
                suggested_replacement=item["suggested_replacement"],
                comment=item["comment"],
                severity=item["severity"],
                context_before=item.get("context_before"),
                context_after=item.get("context_after"),
            )
        )
    return issues


# --------------------------------------------------------------------
# Position resolution: search in PDF with PyMuPDF
# --------------------------------------------------------------------

def resolve_positions_for_page(
    page: fitz.Page,
    issues: List[TypoIssue],
    max_ambiguous: int = 3,
) -> Tuple[List[PositionedTypo], List[TypoIssue]]:
    """
    For each TypoIssue, try to resolve a unique fitz.Rect on the page
    using page.search_for(search_text).

    Returns (resolved, unresolved) where:
      - resolved: list of PositionedTypo
      - unresolved: issues with 0 or >1 matches

    You can optionally feed unresolved back to the model for 1 refinement pass.
    """
    resolved: List[PositionedTypo] = []
    unresolved: List[TypoIssue] = []

    for issue in issues:
        matches = page.search_for(issue.search_text)
        if len(matches) == 1:
            r = matches[0]
            # Place comment slightly left and vertically centered
            x = r.x0 - 15
            y = (r.y0 + r.y1) / 2
            rect = fitz.Rect(x, y, x + 1, y + 1)  # point-like; fitz uses center for text_annot
            resolved.append(
                PositionedTypo(
                    page_index=page.number,
                    issue=issue,
                    rect=rect,
                )
            )
        else:
            # Either 0 or many matches; treat as unresolved
            unresolved.append(issue)

        # Simple protection: don't explode on pages with super-generic strings
        if len(unresolved) > max_ambiguous:
            # Bail out and leave the rest unresolved for this page.
            # Could be refined with iteration logic.
            break

    return resolved, unresolved


# --------------------------------------------------------------------
# (Optional) one refinement round with the model
# --------------------------------------------------------------------

def refine_unresolved_with_model(
    png_path: str,
    page_index: int,
    unresolved: List[TypoIssue],
    model: str = OPENAI_MODEL,
) -> List[TypoIssue]:
    """
    Given unresolved issues (0 or too many matches), ask the model once
    to produce an improved, fully-resolvable list of issues.

    Strategy:
      - Tell the model explicitly which search_texts failed
      - Ask it to propose a COMPLETE new list of issues for this page
        (not incremental), with search_text phrases that are more
        uniquely identifiable.
    """
    if not unresolved:
        return []

    tools = get_tools_spec()
    image_b64 = encode_image_to_base64_png(png_path)
    image_url = f"data:image/png;base64,{image_b64}"

    failed_descriptions = []
    for issue in unresolved:
        failed_descriptions.append(
            {
                "search_text": issue.search_text,
                "comment": issue.comment,
            }
        )

    system_prompt = (
        "You previously tried to identify typos on this page, but some of the "
        "search_text snippets were either not found or not unique when we "
        "searched the underlying PDF text.\n\n"
        "You will see the problematic search_text values and their comments. "
        "You must now produce a *complete, fresh* list of issues for this page, "
        "again using report_typos. This time, choose search_text values that are "
        "longer and more distinctive (e.g., 3–8 words that clearly identify a "
        "single place on the page).\n\n"
        "IMPORTANT: You must return the full list of issues for the page "
        "(including those that were already fine before), not only the unresolved ones."
    )

    user_prompt = (
        "Here is the page again. The following search_text values were ambiguous "
        "or not found when we searched the actual PDF text:\n\n"
        f"{json.dumps(failed_descriptions, indent=2)}\n\n"
        "Please re-generate a *complete* issues list for this page, using more "
        "unique search_text snippets. Again, respond only by calling report_typos."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            },
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "report_typos"}},
    )

    choice = response.choices[0]
    if not choice.message.tool_calls:
        return []

    tool_call = choice.message.tool_calls[0]
    if tool_call.function.name != "report_typos":
        return []

    args = json.loads(tool_call.function.arguments)
    issues_raw = args.get("issues", [])
    refined: List[TypoIssue] = []

    for item in issues_raw:
        refined.append(
            TypoIssue(
                search_text=item["search_text"],
                suggested_replacement=item["suggested_replacement"],
                comment=item["comment"],
                severity=item["severity"],
                context_before=item.get("context_before"),
                context_after=item.get("context_after"),
            )
        )
    return refined


# --------------------------------------------------------------------
# Annotate full PDF with positioned typos
# --------------------------------------------------------------------

def apply_annotations_to_pdf(
    pdf_path: str,
    output_path: str,
    positioned_typos: List[PositionedTypo],
) -> None:
    """
    Given a PDF and a list of PositionedTypo (i.e., we know page + rect),
    add sticky-note comments at the specified positions and save output_path.
    """
    doc = fitz.open(pdf_path)
    issues_by_page: Dict[int, List[PositionedTypo]] = {}

    for pt in positioned_typos:
        issues_by_page.setdefault(pt.page_index, []).append(pt)

    for page_index, pts in issues_by_page.items():
        page = doc[page_index]
        for pt in pts:
            # PyMuPDF text_annot uses the point as the center of the icon.
            center = (pt.rect.x0, pt.rect.y0)
            page.add_text_annot(center, pt.issue.comment)

    doc.save(output_path, deflate=True)


# --------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------

def proofread_pdf(
    pdf_path: str,
    output_path: str,
    tmp_dir: str,
    model: str = OPENAI_MODEL,
    use_refinement_round: bool = True,
) -> None:
    """
    Top-level orchestration:
    - Render PDF to PNG pages
    - Per page: call model, resolve positions, optionally refine once
    - Apply annotations to a copy of the original PDF
    """
    png_dir = os.path.join(tmp_dir, "pages")
    os.makedirs(tmp_dir, exist_ok=True)
    png_paths = pdf_to_page_pngs(pdf_path, png_dir, dpi=DPI)

    doc = fitz.open(pdf_path)
    all_positioned: List[PositionedTypo] = []

    for page_index, png_path in enumerate(png_paths):
        print(f"Processing page {page_index + 1}/{len(png_paths)}...")
        issues = call_model_for_page_typos(png_path, page_index, model=model)

        if not issues:
            continue  # no typos on this page

        page = doc[page_index]
        resolved, unresolved = resolve_positions_for_page(page, issues)

        if unresolved and use_refinement_round:
            refined_issues = refine_unresolved_with_model(
                png_path, page_index, unresolved, model=model
            )
            if refined_issues:
                # Try resolving again with refined suggestions (one extra pass only)
                resolved2, unresolved2 = resolve_positions_for_page(page, refined_issues)
                # We ignore unresolved2 at this point per your spec
                resolved = resolved2

        all_positioned.extend(resolved)

    # Now write annotations into a copy of the original PDF
    apply_annotations_to_pdf(pdf_path, output_path, all_positioned)
    print(f"Done. Annotated PDF written to: {output_path}")


# --------------------------------------------------------------------
# Example usage (CLI-style)
# --------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Proofread a PDF with OpenAI + PyMuPDF.")
    parser.add_argument("pdf_path", help="Path to input PDF")
    parser.add_argument(
        "-o", "--output",
        help="Path to annotated output PDF",
        default="annotated_output.pdf",
    )
    parser.add_argument(
        "--tmp-dir",
        help="Temporary working directory",
        default=".tmp_pdf_proofread",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable second refinement round if matches are ambiguous.",
    )

    args = parser.parse_args()

    proofread_pdf(
        pdf_path=args.pdf_path,
        output_path=args.output,
        tmp_dir=args.tmp_dir,
        use_refinement_round=not args.no_refine,
    )
