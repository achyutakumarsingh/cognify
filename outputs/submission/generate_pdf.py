#!/usr/bin/env python3
"""
generate_pdf.py
Compiles the Markdown solution document into a professional, publication-quality
PDF file for the Indo-Swiss Hackathon submission.

Requirements:
- reportlab (installed)
- Pillow (installed)
"""
from __future__ import annotations
import os
import re
from pathlib import Path
from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import Image as RLImage
from reportlab.pdfgen import canvas

# ─── File Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MD_PATH = SCRIPT_DIR / "1_SOLUTION_DOCUMENT.md"
PDF_PATH = SCRIPT_DIR / "Supply_Chain_Decision_Intelligence_Solution.pdf"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"

# ─── Two-Pass Canvas for Page Numbering and Headers/Footers ──────────────────
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        # Cover page (Page 1) doesn't get headers or footers
        if self._pageNumber == 1:
            self.saveState()
            # Draw primary color band at the top
            self.setFillColor(colors.HexColor("#1e1b4b"))  # Deep Navy/Indigo
            self.rect(0, 770, 612, 22, fill=True, stroke=False)
            # Draw accent color bar below it
            self.setFillColor(colors.HexColor("#4f46e5"))  # Indigo Blue
            self.rect(0, 765, 612, 5, fill=True, stroke=False)
            self.restoreState()
            return

        self.saveState()
        
        # 1. Header (Top of Page)
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#1e1b4b"))  # Primary
        self.drawString(54, 745, "SUPPLY CHAIN DECISION INTELLIGENCE")
        
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))  # Slate Gray
        self.drawRightString(558, 745, "SOLUTION & TECHNICAL WHITE PAPER")
        
        # Header Line Divider
        self.setStrokeColor(colors.HexColor("#cbd5e1"))  # Slate-300
        self.setLineWidth(0.5)
        self.line(54, 737, 558, 737)
        
        # 2. Footer (Bottom of Page)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748b"))
        self.drawString(54, 40, "Confidential - Indo-Swiss Hackathon Team Submission")
        
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 40, page_text)
        
        # Footer Line Divider
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 52, 558, 52)
        
        self.restoreState()

# ─── Markdown Inline Formatting ──────────────────────────────────────────────
def markdown_to_html(text: str) -> str:
    """Converts standard markdown bold, italic, code, and links to ReportLab HTML tags."""
    # Escape HTML special chars (excluding brackets which we use for links)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # Re-allow basic tags if we need them, but since we are converting, we convert markdown:
    # 1. Bold: **text** -> <b>text</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # 2. Italic: *text* -> <i>text</i>
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    # 3. Inline code: `code` -> Courier tag
    text = re.sub(r'`(.*?)`', r'<font face="Courier" size="9" color="#0f172a"><b>\1</b></font>', text)
    # 4. Links: [text](url) -> anchor tag
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<font color="#4f46e5"><u><a href="\2">\1</a></u></font>', text)
    
    return text

# ─── Flowable Helpers ────────────────────────────────────────────────────────
def make_hr(color_hex: str = "#cbd5e1", thickness: float = 0.5, space_after: float = 12) -> Table:
    """Creates a full-width line separator."""
    t = Table([['']], colWidths=[504])
    t.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), thickness, colors.HexColor(color_hex)),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    return t

def get_image_flowable(path: Path, max_width: float = 460) -> tuple[RLImage, Spacer] | tuple[None, None]:
    """Proportionally resizes an image and wraps it in a ReportLab Flowable."""
    if not path.exists():
        print(f"Warning: Image file not found at {path}")
        return None, None
    try:
        img = PILImage.open(path)
        w, h = img.size
        aspect = h / w
        width = min(max_width, w)
        height = width * aspect
        
        rl_img = RLImage(str(path), width=width, height=height)
        rl_img.hAlign = 'CENTER'
        return rl_img, Spacer(1, 4)
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None, None

def get_section_image(stage_name: str) -> tuple[Path | None, str | None]:
    """Maps stage name to its respective plot file and caption."""
    mapping = {
        "Stage 3: Forecasting (XGBoost)": (
            PLOTS_DIR / "actual_vs_predicted.png",
            "Figure 1: XGBoost Tweedie Point Forecast vs. Actual Walmart Sales (M5 Dataset)"
        ),
        "Stage 4: Uncertainty Quantification": (
            PLOTS_DIR / "quantile_vs_conformal_comparison.png",
            "Figure 2: Prediction Interval Comparison: Pinball Loss Quantiles vs. Split Conformal Bounds"
        ),
        "Stage 5: Calibration Evaluation": (
            PLOTS_DIR / "calibration_curves.png",
            "Figure 3: Split Conformal Calibration Curve showing nominal vs. empirical coverage"
        ),
        "Stage 6: Risk Engine": (
            PLOTS_DIR / "risk_level_dist.png",
            "Figure 4: Portfolio Risk Segmentation (Low, Medium, High) computed by the Risk Engine"
        ),
        "Stage 7: Inventory Simulation": (
            PLOTS_DIR / "cost_breakdown_comparison.png",
            "Figure 5: Operational Cost Comparison: Point-Forecast Ordering vs. Conformal-Risk Safety Stock"
        ),
        "Dashboard & User Workflow": (
            PLOTS_DIR / "risk_distribution.png",
            "Figure 6: Executive Dashboard visualization showing average risk scores across product stores"
        )
    }
    return mapping.get(stage_name, (None, None))

# ─── Main PDF Generation ─────────────────────────────────────────────────────
def generate_pdf():
    print("Reading and parsing Markdown document...")
    if not MD_PATH.exists():
        raise FileNotFoundError(f"Source markdown document not found at {MD_PATH}")
        
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_content = f.read()

    # Parse document structure
    lines = md_content.splitlines()
    blocks = []
    current_para = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_para:
                blocks.append(('para', ' '.join(current_para)))
                current_para = []
            continue
            
        if stripped.startswith('# '):
            if current_para:
                blocks.append(('para', ' '.join(current_para)))
                current_para = []
            blocks.append(('h1', stripped[2:]))
        elif stripped.startswith('## '):
            if current_para:
                blocks.append(('para', ' '.join(current_para)))
                current_para = []
            blocks.append(('h2', stripped[3:]))
        elif stripped.startswith('### '):
            if current_para:
                blocks.append(('para', ' '.join(current_para)))
                current_para = []
            blocks.append(('h3', stripped[4:]))
        elif stripped in ('---', '***'):
            if current_para:
                blocks.append(('para', ' '.join(current_para)))
                current_para = []
            blocks.append(('hr', ''))
        elif stripped.startswith('- ') or stripped.startswith('* '):
            if current_para:
                blocks.append(('para', ' '.join(current_para)))
                current_para = []
            blocks.append(('bullet', stripped[2:]))
        elif re.match(r'^\d+\.\s+', stripped):
            if current_para:
                blocks.append(('para', ' '.join(current_para)))
                current_para = []
            match = re.match(r'^(\d+)\.\s+(.*)', stripped)
            num = match.group(1)
            content = match.group(2)
            blocks.append(('numbered', (num, content)))
        else:
            current_para.append(stripped)

    if current_para:
        blocks.append(('para', ' '.join(current_para)))

    # Set up styling
    styles = getSampleStyleSheet()
    
    # Document colors
    primary_color = colors.HexColor("#1e1b4b")  # Deep Navy
    secondary_color = colors.HexColor("#4f46e5")  # Indigo Blue
    body_text_color = colors.HexColor("#1e293b")  # slate-800
    caption_color = colors.HexColor("#64748b")  # slate-500
    
    title_style = ParagraphStyle(
        'CoverTitleStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=30,
        leading=36,
        textColor=primary_color,
        spaceAfter=15
    )

    subtitle_style = ParagraphStyle(
        'CoverSubtitleStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=13,
        leading=17,
        textColor=secondary_color,
        spaceAfter=50
    )

    meta_label_style = ParagraphStyle(
        'MetaLabelStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=primary_color
    )
    
    meta_val_style = ParagraphStyle(
        'MetaValStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=body_text_color
    )

    h1_style = ParagraphStyle(
        'Heading1Style',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceBefore=22,
        spaceAfter=10,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2Style',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=primary_color,
        spaceBefore=18,
        spaceAfter=8,
        keepWithNext=True
    )

    h3_style = ParagraphStyle(
        'Heading3Style',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=secondary_color,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14.5,
        textColor=body_text_color,
        spaceAfter=8
    )

    list_style = ParagraphStyle(
        'ListStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=body_text_color,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=5
    )

    caption_style = ParagraphStyle(
        'CaptionStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=11,
        textColor=caption_color,
        alignment=1,  # Center
        spaceBefore=5,
        spaceAfter=15
    )

    # Initialize layout
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=75,
        bottomMargin=75
    )

    story = []

    # ─── COVER PAGE ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 120))
    
    # Tag Badge
    story.append(Paragraph(
        "<font color='#4f46e5'><b>INDO-SWISS HACKATHON 2026 &nbsp;|&nbsp; ROUND 2 SUBMISSION</b></font>",
        ParagraphStyle('Badge', parent=styles['Normal'], fontSize=9, leading=11, spaceAfter=15)
    ))
    
    # Title & Subtitle
    story.append(Paragraph("Supply Chain Decision Intelligence", title_style))
    story.append(Paragraph("Calibrated Uncertainty Quantification for Supply Chain Risk Triage", subtitle_style))
    
    story.append(make_hr(color_hex="#4f46e5", thickness=2, space_after=40))
    story.append(Spacer(1, 20))
    
    # Metadata Table
    meta_data = [
        [Paragraph("Category:", meta_label_style), Paragraph("Solution & Technical White Paper", meta_val_style)],
        [Paragraph("Authors:", meta_label_style), Paragraph("Indo-Swiss Hackathon Team", meta_val_style)],
        [Paragraph("Date:", meta_label_style), Paragraph("June 30, 2026", meta_val_style)],
        [Paragraph("Version:", meta_label_style), Paragraph("1.0.0", meta_val_style)],
        [Paragraph("Live Dashboard:", meta_label_style), Paragraph("<u><font color='#4f46e5'><a href='https://cognifyaiprediction.streamlit.app/'>cognifyaiprediction.streamlit.app</a></font></u>", meta_val_style)],
    ]
    t_meta = Table(meta_data, colWidths=[110, 394])
    t_meta.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_meta)
    
    story.append(PageBreak())

    # ─── MAIN CONTENT ────────────────────────────────────────────────────────
    # Define which H2 headings force a page break for clean formatting
    page_break_headings = {
        "1. Problem Understanding": False,  # Follows cover page, starts on pg 2
        "2. Proposed Solution": False,
        "3. Technical Approach": True,      # Starts on pg 3
        "4. Prototype Design": True,        # Starts on pg 5
        "5. Feasibility Analysis": True,
        "6. Expected Impact": True,
        "7. Future Scope": False,
    }

    first_h2 = True
    
    for block_type, content in blocks:
        if block_type == 'h1':
            # Skip repeating main title
            continue
            
        elif block_type == 'h2':
            h2_text = content.strip()
            # Force page break if specified
            if page_break_headings.get(h2_text, False):
                story.append(PageBreak())
            
            p = Paragraph(markdown_to_html(h2_text), h2_style)
            hr = make_hr(color_hex="#4f46e5", thickness=1.2)
            story.append(KeepTogether([p, Spacer(1, 4), hr, Spacer(1, 8)]))
            
        elif block_type == 'h3':
            h3_text = content.strip()
            story.append(Paragraph(markdown_to_html(h3_text), h3_style))
            
        elif block_type == 'para':
            para_text = content.strip()
            story.append(Paragraph(markdown_to_html(para_text), body_style))
            
        elif block_type == 'bullet':
            bullet_text = content.strip()
            formatted = f"<font color='#4f46e5'>&bull;</font>&nbsp;&nbsp;{markdown_to_html(bullet_text)}"
            story.append(Paragraph(formatted, list_style))
            
        elif block_type == 'numbered':
            num, num_text = content
            num_text = num_text.strip()
            formatted = f"<font color='#4f46e5'><b>{num}.</b></font>&nbsp;&nbsp;{markdown_to_html(num_text)}"
            story.append(Paragraph(formatted, list_style))
            
        elif block_type == 'hr':
            story.append(make_hr(color_hex="#cbd5e1", thickness=0.5, space_after=12))
            story.append(Spacer(1, 6))

        # Check if we should insert a plot image after a heading/section
        if block_type in ('h3', 'para', 'bullet'):
            # Check if this content matches any of the stages
            stage_match = None
            if block_type == 'h3':
                stage_match = content.strip()
            elif block_type == 'para' and content.startswith("We constructed a rich feature space"):
                # Feature engineering stage is finished, we don't have a plot for it, but check others
                pass
            
            if stage_match:
                img_path, caption = get_section_image(stage_match)
                if img_path and caption:
                    img_flow, spacer = get_image_flowable(img_path)
                    if img_flow:
                        story.append(Spacer(1, 6))
                        story.append(KeepTogether([
                            img_flow,
                            spacer,
                            Paragraph(caption, caption_style)
                        ]))

    # Build the document
    print(f"Saving compiled PDF to {PDF_PATH}...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print("PDF successfully generated!")

if __name__ == "__main__":
    generate_pdf()
