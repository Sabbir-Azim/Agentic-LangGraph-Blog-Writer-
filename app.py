from __future__ import annotations

import hashlib
import html
import json
import operator
import os
import re
import textwrap
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
import streamlit as st

# Fix SSL issue by removing invalid SSL_CERT_DIR and SSL_CERT_FILE if set
os.environ.pop("SSL_CERT_DIR", None)
os.environ.pop("SSL_CERT_FILE", None)

load_dotenv(dotenv_path=Path(".env"))

OUTPUT_DIR = Path("outputs")
ASSET_DIR = OUTPUT_DIR / "assets"
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
IMAGE_BLOCK_START = "<!-- blog-images:start -->"
IMAGE_BLOCK_END = "<!-- blog-images:end -->"

# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="3-6 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(..., description="Target word count for this section (120-550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # keep if Tavily provides; DO NOT rely on it
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class VisualAssetSpec(BaseModel):
    title: str = Field(..., description="Short title for the generated visual asset.")
    subtitle: str = Field(..., description="One sentence explaining what the visual shows.")
    key_points: List[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description="Three to five concise labels for the visual flow or infographic.",
    )
    callouts: List[str] = Field(
        default_factory=list,
        max_length=3,
        description="Optional short callouts to place below the visual.",
    )
    caption: str = Field(..., description="A polished caption for the blog.")
    alt_text: str = Field(..., description="Accessible alt text for the generated image.")


class GeneratedVisualAsset(BaseModel):
    saved_path: str
    markdown_path: str
    alt_text: str
    caption: str


class State(TypedDict):
    topic: str
    image_markdown: str
    visual_asset: Optional[GeneratedVisualAsset]

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)
    final: str
    saved_file: str


def safe_markdown_filename(title: str) -> str:
    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", title).strip()
    safe_name = re.sub(r"\s+", " ", safe_name)
    safe_name = safe_name.rstrip(". ")
    return f"{safe_name or 'generated_blog'}.md"


def safe_asset_stem(value: str) -> str:
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._")
    return safe_name[:70] or "blog-image"


def save_markdown_file(filename: str, contents: str) -> str:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / filename
    output_path.write_text(contents, encoding="utf-8")
    return str(output_path)


def _svg_text_lines(text: str, width: int) -> List[str]:
    return textwrap.wrap(text.strip(), width=width) or [""]


def _svg_tspan(lines: List[str], x: int, y: int, size: int, color: str, weight: int = 500) -> str:
    escaped_lines = [html.escape(line) for line in lines]
    tspans = []
    for index, line in enumerate(escaped_lines):
        dy = 0 if index == 0 else int(size * 1.25)
        tspans.append(f'<tspan x="{x}" dy="{dy}">{line}</tspan>')
    return (
        f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
        f'font-family="Inter, Segoe UI, Arial, sans-serif" font-weight="{weight}">'
        f'{"".join(tspans)}</text>'
    )


def fallback_visual_spec(topic: str, plan: Optional[Plan]) -> VisualAssetSpec:
    task_titles = [task.title for task in (plan.tasks if plan else [])][:4]
    key_points = task_titles or [
        "Understand the core idea",
        "Map the implementation path",
        "Evaluate tradeoffs",
        "Apply practical next steps",
    ]

    return VisualAssetSpec(
        title=f"{topic}: visual overview",
        subtitle="A compact visual summary generated from the blog outline.",
        key_points=key_points[:5],
        callouts=["Planner", "Research", "Writer"],
        caption="Generated visual summary for the article.",
        alt_text=f"Infographic summarizing {topic}.",
    )


def render_visual_asset(topic: str, spec: VisualAssetSpec) -> GeneratedVisualAsset:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    asset_payload = spec.model_dump()
    digest = hashlib.sha1(json.dumps(asset_payload, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    filename = f"{safe_asset_stem(topic.lower())[:48]}-{digest}.svg"
    saved_path = ASSET_DIR / filename

    width = 1200
    height = 720
    card_y = 246
    card_h = 190
    margin = 72
    gap = 22
    points = spec.key_points[:5]
    card_w = int((width - (margin * 2) - (gap * (len(points) - 1))) / max(len(points), 1))
    palette = ["#5d7cff", "#26d6a4", "#f9c74f", "#f97068", "#9b5de5"]

    cards = []
    for index, point in enumerate(points):
        x = margin + index * (card_w + gap)
        color = palette[index % len(palette)]
        cards.append(
            f"""
            <g>
                <rect x="{x}" y="{card_y}" width="{card_w}" height="{card_h}" rx="18" fill="#141824" stroke="#2a3143" stroke-width="2"/>
                <circle cx="{x + 34}" cy="{card_y + 38}" r="18" fill="{color}"/>
                <text x="{x + 34}" y="{card_y + 45}" text-anchor="middle" fill="#07100d" font-size="18" font-family="Inter, Segoe UI, Arial, sans-serif" font-weight="800">{index + 1}</text>
                {_svg_tspan(_svg_text_lines(point, 18), x + 26, card_y + 92, 22, "#f4f6fb", 700)}
            </g>
            """
        )

    callout_text = " | ".join(spec.callouts[:3]) if spec.callouts else "Concept map | Key takeaways | Practical lens"

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(spec.alt_text)}">
    <rect width="{width}" height="{height}" fill="#07080c"/>
    <rect x="36" y="36" width="{width - 72}" height="{height - 72}" rx="28" fill="#0f1117" stroke="#252b3a" stroke-width="2"/>
    <rect x="72" y="72" width="230" height="46" rx="12" fill="#161b28" stroke="#2a3143"/>
    <text x="96" y="102" fill="#9aa3b2" font-size="17" font-family="Inter, Segoe UI, Arial, sans-serif" font-weight="700">GENERATED VISUAL</text>
    {_svg_tspan(_svg_text_lines(spec.title, 38), 72, 178, 42, "#f4f6fb", 800)}
    {_svg_tspan(_svg_text_lines(spec.subtitle, 86), 72, 210, 21, "#c7cedd", 500)}
    {"".join(cards)}
    <rect x="72" y="512" width="{width - 144}" height="82" rx="18" fill="#101722" stroke="#263044"/>
    <text x="100" y="562" fill="#26d6a4" font-size="22" font-family="Inter, Segoe UI, Arial, sans-serif" font-weight="800">Key lens</text>
    {_svg_tspan(_svg_text_lines(callout_text, 80), 220, 562, 22, "#f4f6fb", 600)}
    <text x="72" y="650" fill="#8e98aa" font-size="18" font-family="Inter, Segoe UI, Arial, sans-serif">{html.escape(spec.caption)}</text>
</svg>"""

    saved_path.write_text(svg, encoding="utf-8")

    return GeneratedVisualAsset(
        saved_path=str(saved_path),
        markdown_path=f"assets/{filename}",
        alt_text=spec.alt_text,
        caption=spec.caption,
    )


def build_image_markdown(asset: Optional[GeneratedVisualAsset]) -> str:
    if not asset:
        return ""

    blocks = [IMAGE_BLOCK_START, "## Visual References"]
    blocks.append(f"![{asset.alt_text}]({asset.markdown_path})\n\n*{asset.caption}*")
    blocks.append(IMAGE_BLOCK_END)
    return "\n\n".join(blocks)


def strip_image_markdown(markdown_text: str) -> str:
    pattern = rf"{re.escape(IMAGE_BLOCK_START)}.*?{re.escape(IMAGE_BLOCK_END)}\s*"
    return re.sub(pattern, "", markdown_text, flags=re.DOTALL).strip()

# -----------------------------
# 2) LLM
# -----------------------------
llm = ChatOpenAI(model=MODEL_NAME)


VISUAL_ASSET_SYSTEM = """You are a visual design agent for a technical blog generator.

Create a concise infographic plan that can be rendered as a graphical SVG image.

Rules:
- Base the visual only on the blog topic, outline, and provided evidence summary.
- Prefer a flow, lifecycle, comparison, or concept map that helps readers understand the article.
- Keep every label short enough to fit in a small card.
- Do not mention that the visual is AI-generated.
- Output must strictly match the VisualAssetSpec schema.
"""

# -----------------------------
# 3) Router (decide upfront)
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3-10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""

def router_node(state: State) -> dict:
    
    topic = state["topic"]
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {topic}"),
        ]
    )

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily) 
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({"query": query})

    normalized: List[dict] = []
    for r in results or []:
        normalized.append(
            {
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            }
        )
    return normalized


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:

    # take the first 10 queries from state
    queries = (state.get("queries", []) or [])
    max_results = 6

    raw_results: List[dict] = []

    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=max_results))

    if not raw_results:
        return {"evidence": []}

    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=f"Raw results:\n{raw_results}"),
        ]
    )

    # Deduplicate by URL
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {"evidence": list(dedup.values())}

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5-9 sections (tasks) suitable for the topic and audience.
- Each task must include:
  1) goal (1 sentence)
  2) 3-6 bullets that are concrete, specific, and non-overlapping
  3) target word count (120-550)

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
    and includes only what can be supported.
Output must strictly match the Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    planner = llm.with_structured_output(Plan)

    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )

    return {"plan": plan}

# -----------------------------
# 6) Visual asset agent
# -----------------------------
def visual_asset_node(state: State) -> dict:
    plan = state["plan"]
    evidence = state.get("evidence", [])

    try:
        visual_designer = llm.with_structured_output(VisualAssetSpec)
        spec = visual_designer.invoke(
            [
                SystemMessage(content=VISUAL_ASSET_SYSTEM),
                HumanMessage(
                    content=(
                        f"Topic: {state['topic']}\n"
                        f"Blog title: {plan.blog_title}\n"
                        f"Audience: {plan.audience}\n"
                        f"Blog kind: {plan.blog_kind}\n\n"
                        f"Outline tasks:\n{[task.model_dump() for task in plan.tasks]}\n\n"
                        f"Evidence summary:\n{[item.model_dump() for item in evidence[:8]]}"
                    )
                ),
            ]
        )
    except Exception:
        spec = fallback_visual_spec(state["topic"], plan)

    asset = render_visual_asset(state["topic"], spec)
    image_markdown = build_image_markdown(asset)

    return {"visual_asset": asset, "image_markdown": image_markdown}

# -----------------------------
# 7) Fanout
# -----------------------------
def fanout(state: State):
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 8) Worker (write one section)
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (+/-15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""

def worker_node(payload: dict) -> dict:
    
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    section_md = llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}

# -----------------------------
# 9) Reducer (merge + save)
# -----------------------------
def reducer_node(state: State) -> dict:

    plan = state["plan"]

    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    image_markdown = state.get("image_markdown", "").strip()
    parts = [f"# {plan.blog_title}", image_markdown, body]
    final_md = "\n\n".join(part for part in parts if part).strip() + "\n"

    filename = safe_markdown_filename(plan.blog_title)
    saved_file = save_markdown_file(filename, final_md)

    return {"final": final_md, "saved_file": saved_file}

# -----------------------------
# 10) Build graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("visual_asset", visual_asset_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_node)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_edge("orchestrator", "visual_asset")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("visual_asset", "reducer")
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app_graph = g.compile()

# -----------------------------
# 11) Runner
# -----------------------------
def run(topic: str):
    out = app_graph.invoke(
        {
            "topic": topic,
            "image_markdown": "",
            "visual_asset": None,
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "sections": [],
            "final": "",
            "saved_file": "",
        }
    )

    return out

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Multimodal Blog Agent",
    page_icon="BA",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --bg: #07080c;
            --panel: #0f1117;
            --panel-2: #141824;
            --border: rgba(255,255,255,.08);
            --text: #f4f6fb;
            --muted: #9aa3b2;
            --soft: #c7cedd;
            --accent: #7c5cff;
            --accent-2: #26d6a4;
            --danger: #ff6b6b;
            --shadow: 0 24px 80px rgba(0,0,0,.45);
        }

        .stApp {
            background: linear-gradient(180deg, #07080c 0%, #090b11 46%, #07080c 100%);
            color: var(--text);
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1040px;
            padding-top: 4.2rem;
            padding-bottom: 5rem;
        }

        #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
            visibility: hidden;
            height: 0;
        }

        h1, h2, h3, p, li, label, span, div {
            color: var(--text);
        }

        .hero {
            border: 1px solid var(--border);
            background: linear-gradient(145deg, rgba(15,17,23,.92), rgba(20,24,36,.82));
            box-shadow: var(--shadow);
            border-radius: 8px;
            padding: 1.6rem;
            margin-bottom: 1.2rem;
            position: relative;
            overflow: hidden;
        }

        .hero:before {
            content: none;
        }

        .hero-inner {
            position: relative;
            z-index: 1;
        }

        .brand {
            display: inline-flex;
            align-items: center;
            gap: .5rem;
            color: var(--soft);
            font-size: .78rem;
            letter-spacing: .14em;
            text-transform: uppercase;
            border: 1px solid var(--border);
            background: rgba(255,255,255,.04);
            border-radius: 8px;
            padding: .48rem .78rem;
            margin-bottom: 1.2rem;
        }

        .hero h1 {
            font-size: 2.4rem;
            line-height: 1.08;
            letter-spacing: 0;
            margin: 0 0 .75rem 0;
            max-width: 760px;
        }

        .hero p {
            color: var(--muted);
            font-size: 1.06rem;
            line-height: 1.75;
            max-width: 660px;
            margin: 0;
        }

        .input-card, .result-card {
            border: 1px solid var(--border);
            background: rgba(15,17,23,.82);
            box-shadow: 0 18px 60px rgba(0,0,0,.30);
            border-radius: 8px;
            padding: 1.25rem;
        }

        .result-card {
            margin-top: 1rem;
            padding: 1.4rem;
        }

        .small-muted {
            color: var(--muted);
            font-size: .92rem;
            line-height: 1.7;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: .42rem;
            border: 1px solid var(--border);
            background: rgba(255,255,255,.04);
            border-radius: 8px;
            padding: .42rem .75rem;
            color: var(--soft);
            font-size: .82rem;
            margin-bottom: .8rem;
        }

        .stTextArea textarea {
            background: #0b0d13 !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            min-height: 132px;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.02);
        }

        .stTextArea textarea:focus {
            border-color: rgba(124,92,255,.70) !important;
            box-shadow: 0 0 0 4px rgba(124,92,255,.16) !important;
        }

        .stButton > button {
            width: 100%;
            border: 0;
            border-radius: 8px;
            padding: .86rem 1rem;
            color: #ffffff;
            font-weight: 800;
            background: linear-gradient(135deg, #7c5cff 0%, #5d7cff 100%);
            box-shadow: 0 18px 42px rgba(124,92,255,.28);
            transition: all .18s ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 22px 55px rgba(124,92,255,.38);
            color: #ffffff;
        }

        .stDownloadButton > button {
            width: 100%;
            border-radius: 8px;
            padding: .82rem 1rem;
            color: #07100d;
            font-weight: 800;
            background: linear-gradient(135deg, #26d6a4, #b6f7d4);
            border: none;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: .55rem;
            border-bottom: 1px solid var(--border);
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,.04);
            border-radius: 8px;
            border: 1px solid var(--border);
            padding: .52rem .9rem;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(124,92,255,.22);
            border-color: rgba(124,92,255,.55);
        }

        .stMarkdown, .stMarkdown p, .stMarkdown li {
            color: var(--soft);
        }

        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: var(--text);
        }

        .element-container:has(.stAlert) {
            margin-top: .8rem;
        }

        div[data-testid="stCodeBlock"] {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def _safe_text(value) -> str:
    return html.escape(str(value or ""))


if "result" not in st.session_state:
    st.session_state.result = None

if "last_topic" not in st.session_state:
    st.session_state.last_topic = "State of Multimodal LLMs in 2026"

st.markdown(
    """
    <div class="hero">
        <div class="hero-inner">
            <div class="brand">Agentic Blog Writer</div>
            <h1>Generate technical blogs with built-in visuals.</h1>
            <p>
                Enter a topic. The graph writes the article and creates a graphical visual for the blog.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

input_col, _ = st.columns([1.4, .8], gap="large")

with input_col:
    st.markdown("<div class='status-pill'>Start with a topic</div>", unsafe_allow_html=True)
    topic = st.text_area(
        "Topic",
        value=st.session_state.last_topic,
        placeholder="Example: State of Multimodal LLMs in 2026",
        label_visibility="collapsed",
    )
    generate = st.button("Generate", use_container_width=True)

if generate:
    clean_topic = topic.strip()

    if not clean_topic:
        st.warning("Please enter a topic first.")
    else:
        st.session_state.last_topic = clean_topic
        status = st.empty()

        try:
            with st.spinner("Writing your blog..."):
                result = run(clean_topic)

            status.success("Done.")
            st.session_state.result = result

        except Exception as exc:
            status.error("Something went wrong. Please check your setup and try again.")
            st.exception(exc)

result = st.session_state.result

if result:
    final_md = result.get("final", "") or ""
    saved_file = result.get("saved_file", "generated_blog.md") or "generated_blog.md"
    visual_asset = result.get("visual_asset")
    if isinstance(visual_asset, dict):
        visual_asset = GeneratedVisualAsset(**visual_asset)
    plan = result.get("plan")

    title = _safe_text(getattr(plan, "blog_title", "Your blog") if plan else "Your blog")

    st.markdown(
        f"""
        <div class="result-card">
            <div class="status-pill">Ready</div>
            <h2 style="margin-top:0; letter-spacing:-.04em;">{title}</h2>
            <p class="small-muted">Preview the article, inspect the generated visual, or download it as Markdown. Saved to {_safe_text(saved_file)}.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab_article, tab_visual, tab_markdown, tab_download = st.tabs(["Article", "Generated Visual", "Markdown", "Download"])

    with tab_article:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        if visual_asset:
            st.image(visual_asset.saved_path, caption=visual_asset.caption, use_container_width=True)
        if final_md:
            st.markdown(strip_image_markdown(final_md))
        else:
            st.info("No article was returned.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_visual:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        if visual_asset:
            st.image(visual_asset.saved_path, caption=visual_asset.caption, use_container_width=True)
            st.markdown(f"**Alt text:** {_safe_text(visual_asset.alt_text)}")
            st.markdown(f"**Asset path:** `{visual_asset.saved_path}`")
        else:
            st.info("No visual asset was generated for this run.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_markdown:
        st.code(final_md, language="markdown")

    with tab_download:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.download_button(
            label="Download Markdown",
            data=final_md,
            file_name=Path(saved_file).name,
            mime="text/markdown",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
