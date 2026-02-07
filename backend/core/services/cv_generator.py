import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Dict
from jinja2 import Environment, FileSystemLoader
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ---------------------------------------------------------
# HELPER: LaTeX Environment Setup
# ---------------------------------------------------------
def escape_latex(text: str) -> str:
    if not text: return ""
    chars = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
    }
    return "".join(chars.get(c, c) for c in str(text))

def create_latex_env(template_dir: str):
    env = Environment(
        loader=FileSystemLoader(template_dir),
        block_start_string='\\BLOCK{', block_end_string='}',
        variable_start_string='\\VAR{', variable_end_string='}',
        comment_start_string='\\#{', comment_end_string='}',
        trim_blocks=True, autoescape=False,
    )
    env.filters['latex_escape'] = escape_latex 
    return env

# ---------------------------------------------------------
# PDF Generator (LaTeX)
# ---------------------------------------------------------
class PDFGenerator:
    LATEX_PATH = "/Library/TeX/texbin/pdflatex" # Change if on Linux/Windows

    def __init__(self, template_dir: str):
        self.env = create_latex_env(template_dir)

    def render_cv(self, context: dict, section_order: List[str] = None, section_titles: Dict[str, str] = None) -> bytes:
        # 1. Defaults (Added 'summary' to top)
        if not section_order:
            section_order = ["summary", "education", "skills", "projects", "experience", "hobbies"]
        
        # 2. Inject order and titles into context
        context['section_order'] = [s.lower() for s in section_order]
        context['section_titles'] = section_titles or {}

        # 3. Render
        template = self.env.get_template('cv_template.tex')
        tex_content = template.render(**context)
        return self._compile_tex(tex_content)

    def _compile_tex(self, tex_content: str) -> bytes:
        with tempfile.TemporaryDirectory() as temp_dir:
            tex_path = Path(temp_dir) / "resume.tex"
            tex_path.write_text(tex_content, encoding='utf-8')

            for _ in range(2):
                subprocess.run(
                    [self.LATEX_PATH, "-interaction=nonstopmode", "resume.tex"],
                    cwd=temp_dir,
                    stdout=subprocess.DEVNULL,
                    check=True
                )

            pdf_path = Path(temp_dir) / "resume.pdf"
            return pdf_path.read_bytes()

# ---------------------------------------------------------
# Word Generator (Docx)
# ---------------------------------------------------------
class WordGenerator:
    
    def create_docx(self, cv_data: dict, skill_groups: dict, section_order: List[str] = None, section_titles: Dict[str, str] = None) -> Path:
        doc = Document()
        self._setup_page_layout(doc)
        self._add_header(doc, cv_data)

        # 1. Defaults (Added 'summary' to top)
        if not section_order:
            section_order = ["summary", "education", "skills", "projects", "experience", "hobbies"]
        
        titles = section_titles or {}
        
        # Define default text for fallbacks
        default_titles = {
            "summary": "Professional Summary",
            "education": "Education",
            "skills": "Technical Skills",
            "projects": "Academic & Research Projects",
            "experience": "Experience",
            "hobbies": "Interests & Hobbies"
        }

        # 2. Dispatcher Mapping
        renderers = {
            "summary": lambda: self._add_summary(doc, cv_data, titles.get('summary', default_titles['summary'])),
            "education": lambda: self._add_education(doc, cv_data, titles.get('education', default_titles['education'])),
            "skills": lambda: self._add_skills(doc, skill_groups, titles.get('skills', default_titles['skills'])),
            "projects": lambda: self._add_projects(doc, cv_data, titles.get('projects', default_titles['projects'])),
            "experience": lambda: self._add_experience(doc, cv_data, titles.get('experience', default_titles['experience'])),
            "hobbies": lambda: self._add_hobbies(doc, cv_data, titles.get('hobbies', default_titles['hobbies'])),
        }

        # 3. Dynamic Execution
        for section in section_order:
            section_key = section.lower()
            if section_key in renderers:
                renderers[section_key]()

        # Save to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            doc.save(tmp.name)
            return Path(tmp.name)

    # --- Internal Helpers ---

    def _setup_page_layout(self, doc):
        for section in doc.sections:
            section.top_margin = Inches(0.5)
            section.bottom_margin = Inches(0.5)
            section.left_margin = Inches(0.5)
            section.right_margin = Inches(0.5)
        style = doc.styles['Normal']
        style.font.name = 'Times New Roman'
        style.font.size = Pt(11)

    def _add_section_header(self, doc, title):
        p = doc.add_paragraph()
        run = p.add_run(str(title).upper()) 
        run.font.size = Pt(14)
        run.bold = True
        
        pPr = p._p.get_or_add_pPr()
        pBdr = OxmlElement('w:pBdr')
        bottom = OxmlElement('w:bottom')
        bottom.set(qn('w:val'), 'single')
        bottom.set(qn('w:sz'), '6')
        bottom.set(qn('w:space'), '1')
        bottom.set(qn('w:color'), 'auto')
        pBdr.append(bottom)
        pPr.append(pBdr)
        
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(2)

    def _add_header(self, doc, cv_data):
        header = doc.add_paragraph()
        header.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        name = f"{cv_data.get('first_name', '')} {cv_data.get('last_name', '').upper()}"
        name_run = header.add_run(name)
        name_run.font.size = Pt(24)
        name_run.bold = True
        
        info = cv_data.get('contact_info', {})
        parts = []
        for k, v in info.items():
            if v: parts.append(v)
            
        contact_line = " | ".join(parts)
        contact_run = header.add_run(f"\n{contact_line}")
        contact_run.font.size = Pt(10)

    # --- SECTION RENDERERS ---

    def _add_summary(self, doc, cv_data, title):
        if not cv_data.get('summary'): return
        self._add_section_header(doc, title)
        p = doc.add_paragraph(cv_data['summary'])
        p.paragraph_format.space_after = Pt(4)

    def _add_education(self, doc, cv_data, title):
        if not cv_data.get('education'): return
        self._add_section_header(doc, title)
        
        for edu in cv_data['education']:
            table = doc.add_table(rows=2, cols=2)
            table.autofit = True
            
            r1 = table.rows[0]
            r1.cells[0].text = edu.get('institution', '')
            r1.cells[0].paragraphs[0].runs[0].bold = True
            
            r2 = table.rows[1]
            r2.cells[0].text = f"{edu.get('degree', '')} {edu.get('field', '')}"
            r2.cells[0].paragraphs[0].runs[0].italic = True
            r2.cells[1].text = f"{edu.get('start_date', '')} -- {edu.get('end_date', '')}"
            r2.cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT

            for ach in edu.get('achievements', []):
                bull = doc.add_paragraph(ach['text'], style='List Bullet')
                bull.paragraph_format.left_indent = Inches(0.25)
                bull.paragraph_format.space_after = Pt(0)

    def _add_skills(self, doc, skill_groups, title):
        if not skill_groups: return
        self._add_section_header(doc, title)
        
        for cat, skills in skill_groups.items():
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(0)
            t = p.add_run(f"{cat}: ")
            t.bold = True
            p.add_run(", ".join(skills))

    def _add_projects(self, doc, cv_data, title):
        if not cv_data.get('projects'): return
        self._add_section_header(doc, title)
        
        for proj in cv_data['projects']:
            table = doc.add_table(rows=1, cols=2)
            table.autofit = False 
            table.columns[0].width = Inches(6.0)
            table.columns[1].width = Inches(1.5)
            
            r1 = table.rows[0]
            title_text = f"{proj.get('title')} | {proj.get('related_context', '')}"
            r1.cells[0].text = title_text
            r1.cells[0].paragraphs[0].runs[0].bold = True
            r1.cells[1].text = proj.get('dates', 'Present')
            r1.cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT

            for ach in proj.get('achievements', []):
                bull = doc.add_paragraph(ach['text'], style='List Bullet')
                bull.paragraph_format.left_indent = Inches(0.25)
                bull.paragraph_format.space_after = Pt(0)

    def _add_experience(self, doc, cv_data, title):
        if not cv_data.get('experiences'): return
        self._add_section_header(doc, title)
        
        for exp in cv_data['experiences']:
            table = doc.add_table(rows=2, cols=2)
            table.autofit = True
            
            r1 = table.rows[0]
            r1.cells[0].text = exp.get('title', '')
            r1.cells[0].paragraphs[0].runs[0].bold = True
            r1.cells[1].text = f"{exp.get('start_date','')} -- {exp.get('end_date','')}"
            r1.cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.RIGHT
            
            r2 = table.rows[1]
            r2.cells[0].text = exp.get('company', '')
            r2.cells[0].paragraphs[0].runs[0].italic = True
            
            for ach in exp.get('achievements', []):
                bull = doc.add_paragraph(ach['text'], style='List Bullet')
                bull.paragraph_format.left_indent = Inches(0.25)
                bull.paragraph_format.space_after = Pt(0)

    def _add_hobbies(self, doc, cv_data, title):
        if not cv_data.get('hobbies'): return
        self._add_section_header(doc, title)
        
        for hobby in cv_data['hobbies']:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(hobby.get('name', ''))
            run.bold = True
            
            desc = hobby.get('description')
            if desc: p.add_run(f": {desc}")

            for ach in hobby.get('achievements', []):
                bull = doc.add_paragraph(ach['text'], style='List Bullet')
                bull.paragraph_format.left_indent = Inches(0.25)
                bull.paragraph_format.space_after = Pt(0)