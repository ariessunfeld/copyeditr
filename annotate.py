#!/usr/bin/env python
"""
auto-copyedit v6
-----------------
- Clean 50/50 page segmentation
- Corrected annotation positioning (no more downward/right drift)
- Full-page text context for LLM
- Responses API with tool-calling
- Exact + fuzzy PDF text matching
- Per-page annotation placement with tagging
- Extensive debugging output
"""

import os
from dotenv import load_dotenv; load_dotenv()
import io
import json
import base64
import argparse
import difflib
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import fitz  # PyMuPDF
from PIL import Image
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENAI_MODEL_DEFAULT = "gpt-4.1"
DPI_DEFAULT = 200
FUZZY_MIN_SCORE = 0.80

# Annotation offset tuning (empirically correct for Preview + Acrobat)
ANNOT_X_OFFSET = -8      # slightly left of the matched text
ANNOT_Y_OFFSET = -2      # slightly up to counteract downward drift

client = OpenAI()

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TypoIssue:
    search_text: str
    suggested_replacement: str
    comment: str
    severity: str
    context_before: Optional[str] = None
    context_after: Optional[str] = None

@dataclass
class PositionedTypo:
    page_index: int
    issue: TypoIssue
    rect: fitz.Rect
    match_type: str   # "exact" or "fuzzy"

@dataclass
class WordInfo:
    text: str
    norm: str
    x0: float
    y0: float
    x1: float
    y1: float
    block: int
    line: int
    word_no: int

# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    if not s:
        return ""
    ligatures = {
        "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff",
        "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st"
    }
    for k,v in ligatures.items():
        s = s.replace(k, v)
    quotes = {
        "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
        "‘": "'", "’": "'", "‚": "'", "′": "'",
    }
    for k,v in quotes.items():
        s = s.replace(k, v)
    dashes = {"–": "-", "—": "-", "‒": "-"}
    for k,v in dashes.items():
        s = s.replace(k, v)
    s = s.replace("\u00a0", " ")
    s = s.replace("\t"," ").replace("\r"," ").replace("\n"," ")
    s = s.lower()
    while "  " in s:
        s = s.replace("  "," ")
    return s.strip()

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_system_prompt(page_index: int, slice_label: str) -> str:
    return (
        # "You are a meticulous scientific copy editor. You are shown ONE slice of a PDF page "
        # "(top or bottom half). You are ALSO given the full extracted text of the entire PDF page "
        # "as PDF_TEXT.\n\n"
        # "CRITICAL RULES:\n"
        # "1) Only report issues VISIBLY PRESENT in this slice image.\n"
        # "2) For each issue, choose search_text by COPYING EXACT WORDING from PDF_TEXT.\n"
        # "   Use 3–8 consecutive words that uniquely identify the issue.\n"
        # "3) If the slice shows no issues, call report_typos with issues=[].\n\n"
        # f"This is page index {page_index}, slice='{slice_label}'."
        f"""You are a meticulous scientific copy editor. You are shown ONE slice of a PDF page as an image 
(top or bottom half). You are ALSO given PDF_TEXT, a raw extracted version of the full page's text.

CRITICAL PRIORITY RULES (read carefully):

1) **The image is the ground truth.**
   You MUST judge correctness, formatting, typography, symbols, spacing, and visual quality 
   **based ONLY on what you see in the image.**
   If the image looks correct, you MUST NOT suggest a change—even if the extracted PDF_TEXT 
   contains strange characters, missing italics, broken Unicode, or odd spacing.

2) **PDF_TEXT is ONLY for locating text.**
   PDF_TEXT exists solely to allow you to copy exact substrings for `search_text`.
   Do NOT use PDF_TEXT to decide whether something is incorrect.
   Do NOT flag issues that arise from PDF_TEXT extraction artifacts (e.g., bad Unicode, lost italics, 
   math symbol corruption, hyperlink artifacts, odd spacing, broken ligatures).

3) **Only report issues that are visually evident in the slice image.**
   If the image shows correct formatting, punctuation, math notation, italics, special characters, 
   or symbols—even if PDF_TEXT differs—you MUST treat the image as correct.

4) **STRICTLY FORBIDDEN issue types (do NOT report these):**
   - Problems that appear only in PDF_TEXT but not in the image.
   - Missing italics, boldface, or formatting that appears correct in the image.
   - Unicode or encoding "mistakes" only visible in PDF_TEXT.
   - Math formula “corrections” unless the actual printed formula in the image is wrong.
   - Complaints about hyperlink formatting, reference anchors, or equation citations 
     that appear visually normal in the image.
   - Stylistic preferences not clearly incorrect.

5) **Allowed issues:**
   Only objective, visually-confirmed errors in the slice image:
   - clear misspellings visible in the image
   - punctuation mistakes visible in the image
   - spacing issues visible in the image (e.g., missing space, doubled space)
   - clear grammar errors visible in the image
   - malformed references or citations visible in the image
   - obvious math/notation mistakes that appear wrong *in the printed image*

6) **search_text must be copied EXACTLY from PDF_TEXT.**
   Use the image to determine the error, but copy the exact textual snippet from PDF_TEXT 
   for the search anchor.

7) If there are no issues visible in the image, return `issues=[]`.

This is page index {page_index}, slice='{slice_label}'.
"""
    )

def build_user_prompt() -> str:
    return (
        # "Inspect ONLY THIS SLICE for copy-editing issues.\n"
        # "Use PDF_TEXT solely as a source of exact text to copy.\n\n"
        # "Return ONLY by calling report_typos()."
        """You will examine ONLY the visible portion of the PDF page slice shown in the image.  
You also receive PDF_TEXT (the entire page text) ONLY for the purpose of copying exact 
text when constructing `search_text`.

Instructions:

- Judge correctness ONLY by what you SEE in the slice image.
- Ignore ANY apparent errors that occur only in PDF_TEXT.
- If the image looks correct, treat it as correct.
- Do NOT suggest changes based on PDF_TEXT formatting, Unicode, italics, math symbols, spacing, etc.
- Report only errors that are visually present in the slice image.
- Use PDF_TEXT solely to copy exact text for `search_text` (3–8 consecutive words).
- Respond ONLY by calling `report_typos`.
- If no issues are visible, return `issues=[]`.

"""
    )

def build_refine_system_prompt() -> str:
    return (
        # "You previously attempted to provide issues for this slice, but some search_text values "
        # "did not match PDF_TEXT. Now produce a COMPLETE NEW LIST with corrected search_text copied "
        # "literally from PDF_TEXT."
        """You previously suggested issues for this page slice, but some `search_text` strings did not match 
the extracted PDF_TEXT.

IMPORTANT CLARIFICATION:

- You must continue to identify errors **only if they are visually present in the slice image**.
- You must IGNORE any issues that are visible only in PDF_TEXT (text extraction artifacts, 
  broken characters, lost italics, mangled math, hyperlink text, etc.).
- The image is ALWAYS the source of truth for correctness.
- PDF_TEXT is ONLY for copying exact text substrings for `search_text`.

Now produce a COMPLETE NEW ISSUE LIST for this slice.  
Use visually-confirmed errors ONLY and choose distinctive search_text snippets copied EXACTLY 
from PDF_TEXT.
Return ONLY by calling report_typos.
"""
    )

def build_refine_user_prompt(unresolved: List[TypoIssue]) -> str:
    arr = [{"search_text": u.search_text, "comment": u.comment} for u in unresolved]
    return (
        "These search_text values were ambiguous or not found:\n"
        f"{json.dumps(arr, indent=2)}\n\n"
        "Please regenerate a FULL issue list with corrected search_text "
        "copied exactly from PDF_TEXT."
    )

# ---------------------------------------------------------------------------
# Tools spec for Responses API
# ---------------------------------------------------------------------------

def get_tools_spec():
    return [
        {
            "type": "function",
            "name": "report_typos",
            "description": "Return issues found in this slice; search_text MUST come verbatim from PDF_TEXT.",
            "parameters": {
                "type": "object",
                "properties": {
                    "issues": {
                        "type": "array",
                        "items": {
                            "type":"object",
                            "properties": {
                                "search_text":{"type":"string"},
                                "suggested_replacement":{"type":"string"},
                                "comment":{"type":"string"},
                                "severity":{"type":"string"},
                                "context_before":{"type":"string"},
                                "context_after":{"type":"string"}
                            },
                            "required":["search_text","suggested_replacement","comment","severity"]
                        }
                    }
                },
                "required":["issues"]
            }
        }
    ]

# ---------------------------------------------------------------------------
# Page split (clean 50/50)
# ---------------------------------------------------------------------------

def render_page_halves(page: fitz.Page, page_index: int, out_dir: str, dpi: int):
    """
    Clean 50/50 split:
      top half = [0, 50%]
      bottom half = [50%, 100%]
    """
    os.makedirs(out_dir, exist_ok=True)
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    W,H = img.size
    mid = H//2

    top = img.crop((0,0,W,mid))
    bot = img.crop((0,mid,W,H))

    top_path = os.path.join(out_dir,f"page_{page_index:04d}_top.png")
    bot_path = os.path.join(out_dir,f"page_{page_index:04d}_bot.png")

    top.save(top_path)
    bot.save(bot_path)

    return [
        {"path": top_path, "label":"top"},
        {"path": bot_path, "label":"bottom"},
    ]

# ---------------------------------------------------------------------------
# LLM interactions
# ---------------------------------------------------------------------------

def _encode_image_b64(path):
    with open(path,"rb") as f:
        return base64.b64encode(f.read()).decode("ascii")

def _extract_report_typos_call(resp):
    if not resp.output: return None
    blocks=[b for b in resp.output
            if getattr(b,"type",None)=="function_call"
            and getattr(b,"name",None)=="report_typos"]
    if not blocks: return None
    call=blocks[0]
    try:
        return json.loads(call.arguments)
    except:
        return None

def call_model_for_page_slice(
    png_path: str,
    page_index: int,
    slice_label: str,
    page_text: str,
    model: str
) -> List[TypoIssue]:

    tools = get_tools_spec()
    img_b64 = _encode_image_b64(png_path)
    img_url = f"data:image/png;base64,{img_b64}"

    system_prompt = build_system_prompt(page_index, slice_label)
    user_prompt = build_user_prompt()

    pdf_text_block = (
        "PDF_TEXT (full extracted text for this page):\n\n"
        f"{page_text}\n\n"
        f"Remember: Only report issues visible in the {slice_label} half."
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role":"system","content":[{"type":"input_text","text":system_prompt}]},
            {
                "role":"user",
                "content":[
                    {"type":"input_text","text":pdf_text_block},
                    {"type":"input_text","text":user_prompt},
                    {"type":"input_image","image_url":img_url},
                ],
            },
        ],
        tools=tools,
        tool_choice={"type":"function","name":"report_typos"},
    )

    print(f"\nDEBUG: response.output blocks for page {page_index} slice '{slice_label}':")
    for i,b in enumerate(resp.output):
        print(f"  [{i}] type={getattr(b,'type',None)}, name={getattr(b,'name',None)}")

    args = _extract_report_typos_call(resp)
    if not args:
        print("DEBUG: No tool call.")
        return []

    issues=[]
    for raw in args.get("issues",[]):
        issues.append(TypoIssue(
            search_text=raw["search_text"],
            suggested_replacement=raw["suggested_replacement"],
            comment=raw["comment"],
            severity=raw["severity"],
            context_before=raw.get("context_before"),
            context_after=raw.get("context_after"),
        ))

    print(f"DEBUG: model reported {len(issues)} issues for slice '{slice_label}'.")
    return issues

def refine_unresolved_with_model(
    png_path, page_index, slice_label, page_text, unresolved, model
):
    if not unresolved: return []
    tools = get_tools_spec()

    img_b64 = _encode_image_b64(png_path)
    img_url = f"data:image/png;base64,{img_b64}"

    sys = build_refine_system_prompt()
    usr = build_refine_user_prompt(unresolved)
    pdf_text_block = (
        "PDF_TEXT (full extracted text for this page):\n\n"
        f"{page_text}\n\n"
        f"Only report issues visible in the {slice_label} half."
    )

    resp = client.responses.create(
        model=model,
        input=[
            {"role":"system","content":[{"type":"input_text","text":sys}]},
            {
                "role":"user",
                "content":[
                    {"type":"input_text","text":pdf_text_block},
                    {"type":"input_text","text":usr},
                    {"type":"input_image","image_url":img_url},
                ],
            },
        ],
        tools=tools,
        tool_choice={"type":"function","name":"report_typos"}
    )

    args=_extract_report_typos_call(resp)
    if not args:
        print("DEBUG(refine): No function call found.")
        return []

    refined=[]
    for raw in args.get("issues",[]):
        refined.append(TypoIssue(
            search_text=raw["search_text"],
            suggested_replacement=raw["suggested_replacement"],
            comment=raw["comment"],
            severity=raw["severity"],
            context_before=raw.get("context_before"),
            context_after=raw.get("context_after"),
        ))
    print(f"DEBUG(refine): model returned {len(refined)} refined issues.")
    return refined

# ---------------------------------------------------------------------------
# Word index + fuzzy
# ---------------------------------------------------------------------------

def build_page_word_index(page: fitz.Page)->List[WordInfo]:
    out=[]
    words=page.get_text("words")
    for (x0,y0,x1,y1,word,block,line,wno) in words:
        norm=normalize_text(word)
        if not norm: continue
        out.append(WordInfo(word,norm,x0,y0,x1,y1,block,line,wno))
    return out

def fuzzy_match(issue: TypoIssue, words: List[WordInfo]):
    q=normalize_text(issue.search_text)
    if not q: return None
    qwords=q.split()
    if not qwords: return None

    L=len(words)
    N=len(qwords)
    spans=[max(1,N-1), N, N+1]

    best_score=-1
    best=None
    best_text=""

    for start in range(L):
        for length in spans:
            end=start+length
            if end>L: continue
            cand=" ".join(w.norm for w in words[start:end])
            score=difflib.SequenceMatcher(None,cand,q).ratio()
            if score>best_score:
                best_score=score
                best=(start,end)
                best_text=cand

    if best_score < FUZZY_MIN_SCORE:
        print(f"DEBUG(fuzzy): FAIL '{issue.search_text[:60]}' score={best_score:.2f}")
        return None

    s,e=best
    xs0=min(w.x0 for w in words[s:e])
    ys0=min(w.y0 for w in words[s:e])
    xs1=max(w.x1 for w in words[s:e])
    ys1=max(w.y1 for w in words[s:e])
    rect=fitz.Rect(xs0,ys0,xs1,ys1)

    print(f"DEBUG(fuzzy): OK '{issue.search_text[:60]}' score={best_score:.2f} cand='{best_text[:80]}'")
    return rect, best_text

# ---------------------------------------------------------------------------
# Match resolution
# ---------------------------------------------------------------------------

def resolve_positions_for_page(page, issues, word_index):
    resolved=[]
    unresolved=[]

    print(f"\n=== DEBUG: resolving issues on page {page.number} ===")

    for i,issue in enumerate(issues):
        st=issue.search_text
        print(f"\nDEBUG: issue[{i}] search_text='{st[:80]}'")

        matches = page.search_for(st)
        print(f"DEBUG:   exact matches={len(matches)}")

        match_type="none"
        rect=None

        if len(matches)==1:
            rect=matches[0]
            match_type="exact"
            print(f"DEBUG:   exact rect={rect}")
        else:
            fm=fuzzy_match(issue, word_index)
            if fm:
                rect,_=fm
                match_type="fuzzy"
                print(f"DEBUG:   fuzzy rect={rect}")

        if not rect:
            print("DEBUG:   UNRESOLVED")
            unresolved.append(issue)
            continue

        # Annotation placement: slightly left/up for stable placement
        ax = rect.x0 + ANNOT_X_OFFSET
        ay = rect.y0 + ANNOT_Y_OFFSET
        annot_rect = fitz.Rect(ax, ay, ax+1, ay+1)
        print(f"DEBUG:   annotation at {annot_rect} (match_type={match_type})")

        resolved.append(PositionedTypo(page.number, issue, annot_rect, match_type))

    print(f"DEBUG: resolved={len(resolved)}, unresolved={len(unresolved)}")
    return resolved, unresolved

# ---------------------------------------------------------------------------
# Apply annotations
# ---------------------------------------------------------------------------

def apply_annotations(pdf_path, output_path, items):
    print("\n=== DEBUG: Applying annotations ===")
    doc=fitz.open(pdf_path)

    by_page={}
    for pt in items:
        by_page.setdefault(pt.page_index,[]).append(pt)

    for pg,pts in by_page.items():
        page=doc[pg]
        print(f"DEBUG: Page {pg}: {len(pts)} annotations")
        for pt in pts:
            c=f"[{pt.match_type}] {pt.issue.comment}"
            print(f"  → annot at {pt.rect} on page {pg}")
            try: page.add_text_annot((pt.rect.x0,pt.rect.y0), c)
            except Exception as e: print("ERROR annot:",e)

    doc.save(output_path, deflate=True)
    print(f"Saved annotated PDF → {output_path}")

# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

def proofread_pdf(pdf_path, output_path, tmp_dir, model, dpi, use_refine=True):
    os.makedirs(tmp_dir, exist_ok=True)
    halves_dir=os.path.join(tmp_dir,"halves")
    os.makedirs(halves_dir, exist_ok=True)

    doc=fitz.open(pdf_path)
    all_annots=[]

    for page_index,page in enumerate(doc):
        print(f"\n\n=== Processing page {page_index+1}/{len(doc)} ===")

        page_text=page.get_text("text")
        halves=render_page_halves(page,page_index,halves_dir,dpi)

        page_issues=[]
        for h in halves:
            issues=call_model_for_page_slice(
                h["path"], page_index, h["label"], page_text, model
            )
            page_issues.extend(issues)

        print(f"DEBUG: total raw issues={len(page_issues)}")
        if not page_issues: continue

        widx=build_page_word_index(page)
        resolved, unresolved = resolve_positions_for_page(page, page_issues, widx)

        if unresolved and use_refine:
            # refine on top half only (arbitrary design choice)
            top=halves[0]
            refined = refine_unresolved_with_model(
                top["path"], page_index, top["label"], page_text, unresolved, model
            )
            if refined:
                resolved2,_ = resolve_positions_for_page(page,refined,widx)
                resolved.extend(resolved2)

        all_annots.extend(resolved)

    print(f"\n=== Final: {len(all_annots)} annotations to apply ===")
    apply_annotations(pdf_path, output_path, all_annots)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p=argparse.ArgumentParser()
    p.add_argument("pdf_path")
    p.add_argument("-o","--output",default="annotated_output.pdf")
    p.add_argument("--tmp-dir",default=".tmp_auto_copyedit")
    p.add_argument("--model",default=OPENAI_MODEL_DEFAULT)
    p.add_argument("--dpi",type=int,default=DPI_DEFAULT)
    p.add_argument("--no-refine",action="store_true")
    args=p.parse_args()

    proofread_pdf(
        args.pdf_path,
        args.output,
        args.tmp_dir,
        args.model,
        args.dpi,
        use_refine=not args.no_refine
    )

if __name__=="__main__":
    main()
