import re
from typing import List, Literal, Optional, Dict

# Optional spaCy import
try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    spacy = None
    SPACY_AVAILABLE = False



import torch
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from backend.core.models import JobDescriptionFeature

class Segmentor:
    """
    Hardened Job Description Segmentor
    - Bullet-proof inconsistent bullet handling
    - ✅ Conjunction splitting DISABLED
    - POS-aware short fragment attachment
    - Header detection
    - Inline dash, pipe, unicode bullets
    """

    # ------------------------------
    # Regex / constants
    # ------------------------------

    BULLET_MARKER_RE = re.compile(
        r'^\s*([•●▪◦·◆►‣◉▹▫︎▪︎]|\d{1,3}[\.\)\-]|[-–—>→▪])\s*'
    )

    INLINE_BULLET_RE = re.compile(
        r'(?:(?<=\s)|(?<=^)|(?<=[\(\[\;:—–|]))'
        r'([•●▪◦·◆►‣◉▹▫︎▪︎]|\d{1,3}[\.\)\-]|[-–—>→▪])\s*'
    )

    PIPE_BULLET_RE = re.compile(r"\s*\|\s*")

    EM_DASH_SPLIT = re.compile(r"\s*[;:—–]\s*")

    HEADER_CLEAN_RE = re.compile(r"^[\s\-\•\●\*>\d\)\(:#=]+")

    HEADER_REGEX = re.compile(
        r"""^\s*(?:
            about\s+the\s+role|
            about\s+this\s+role|
            role\s+overview|
            role\s+summary|
            position\s+summary|
            job\s+summary|
            job\s+overview|
            overview|
            summary|
            introduction|
            about\s+the\s+team|
            about\s+the\s+company|
            about\s+us|
            company\s+overview|
            company\s+profile|
            organisation\s+overview|
            who\s+we\s+are|
            who\s+is\s+|
            our\s+mission|
            our\s+vision|
            our\s+values|
            mission\s+and\s+values|
            company\s+culture|
            culture|
            responsibilities|
            key\s+responsibilities|
            primary\s+responsibilities|
            main\s+responsibilities|
            essential\s+responsibilities|
            duties|
            job\s+duties|
            essential\s+duties|
            key\s+duties|
            tasks|
            key\s+tasks|
            role\s+responsibilities|
            day[-\s]*to[-\s]*day|
            what\s+you(?:'|’)?ll\s+do|
            what\s+you\s+will\s+do|
            what\s+you\s+will\s+be\s+doing|
            your\s+impact|
            what\s+we\s+expect|
            expectations|
            scope\s+of\s+work|
            requirements|
            job\s+requirements|
            minimum\s+requirements|
            basic\s+requirements|
            mandatory\s+requirements|
            essential\s+requirements|
            required|
            must[-\s]*have|
            what\s+you\s+bring|
            eligibility\s+requirements|
            eligibility|
            work\s+authorization|
            work\s+authorisation|
            security\s+clearance|
            qualifications|
            required\s+qualifications|
            preferred\s+qualifications|
            desired\s+qualifications|
            qualification\s+requirements|
            education|
            required\s+education|
            experience|
            relevant\s+experience|
            professional\s+experience|
            background|
            certifications|
            licences|
            licenses|
            skills|
            required\s+skills|
            preferred\s+skills|
            desired\s+skills|
            technical\s+skills|
            soft\s+skills|
            competencies|
            key\s+competencies|
            abilities|
            core\s+skills|
            benefits|
            perks|
            total\s+rewards|
            compensation|
            salary|
            pay|
            pay\s+range|
            wage|
            bonus|
            bonuses|
            equity|
            stock\s+options|
            rewards|
            remuneration|
            compensation\s+and\s+benefits|
            job\s+details|
            position\s+details|
            employment\s+details|
            job\s+type|
            employment\s+type|
            contract\s+type|
            schedule|
            working\s+hours|
            hours|
            shift|
            location|
            work\s+location|
            work\s+arrangement|
            remote|
            hybrid|
            on[-\s]*site|
            travel\s+requirements|
            relocation|
            physical\s+requirements|
            physical\s+demands|
            working\s+conditions|
            work\s+environment|
            reporting\s+structure|
            reports\s+to|
            supervisory\s+responsibilities|
            why\s+us|
            why\s+join\s+us|
            why\s+you(?:'|’)?ll\s+love\s+working\s+here|
            who\s+you\s+are|
            ideal\s+candidate|
            candidate\s+profile|
            about\s+you|
            what\s+we\s+are\s+looking\s+for|
            additional\s+information|
            other\s+information|
            more\s+information|
            notes|
            disclaimer|
            diversity\s+statement|
            diversity\s+and\s+inclusion|
            equal\s+opportunity|
            eeo|
            legal\s+disclaimer|
            equal\s+employment\s+opportunity|
            how\s+to\s+apply|
            application\s+process|
            next\s+steps|
            application\s+instructions|
            recruitment\s+process|
            hiring\s+process|
            contact\s+information|
            contact
        )[:\s]*$""",
        re.IGNORECASE | re.VERBOSE,
    )

 # ------------------------------
    # Constructor / spaCy pipeline
    # ------------------------------
    def __init__(self):
        self.nlp = self._create_spacy_pipeline()

    def _create_spacy_pipeline(self):
        if not SPACY_AVAILABLE:
            raise RuntimeError("spaCy is required for this segmentor")

        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner"])
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp
        except Exception:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            return nlp

    # ------------------------------
    # Normalization
    # ------------------------------
    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return "\n".join(ln.strip() for ln in text.split("\n")).strip()

    # ------------------------------
    # Header detection & block splitting
    # ------------------------------
    def is_header(self, line: str) -> bool:
        clean = self.HEADER_CLEAN_RE.sub("", line).strip().lower()
        return bool(self.HEADER_REGEX.match(clean))

    def split_into_blocks(self, text: str):
        lines = text.split("\n")
        blocks = []
        cur = {"header": None, "lines": []}

        for ln in lines:
            if self.is_header(ln):
                if cur["lines"]:
                    blocks.append(cur)
                cur = {"header": ln.strip().rstrip(":").strip(), "lines": []}
            else:
                cur["lines"].append(ln)

        if cur["lines"]:
            blocks.append(cur)

        return blocks

    # ------------------------------
    # Bullet splitting
    # ------------------------------
    def split_inline_bullets(self, line: str) -> List[str]:
        if "|" in line:
            parts = [p.strip() for p in self.PIPE_BULLET_RE.split(line) if p.strip()]
            if len(parts) > 1:
                return parts

        parts = []
        pos = 0
        for m in self.INLINE_BULLET_RE.finditer(line):
            start, end = m.span()
            if start > pos:
                chunk = line[pos:start].strip()
                if chunk:
                    parts.append(chunk)
            pos = end

        trailing = line[pos:].strip()
        if trailing:
            parts.append(trailing)

        if len(parts) == 1 and re.search(r"-[A-Za-z]{3,}", parts[0]):
            exploded = re.split(r"\s*(?=-[A-Za-z])", parts[0])
            return [self.BULLET_MARKER_RE.sub("", p).strip() for p in exploded if p.strip()]

        return parts

    def split_bullets(self, lines: List[str]) -> List[str]:
        out = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue

            while self.BULLET_MARKER_RE.match(ln):
                ln = self.BULLET_MARKER_RE.sub("", ln).strip()

            if self.INLINE_BULLET_RE.search(ln):
                out.extend(self.split_inline_bullets(ln))
            else:
                out.append(ln)

        return out

    # ------------------------------
    # Sentence splitting
    # ------------------------------
    def split_sentences(self, texts: List[str]) -> List[List[str]]:
        docs = self.nlp.pipe(texts)
        out = []
        for doc in docs:
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
            out.append(sents if sents else [doc.text.strip()])
        return out

    # ------------------------------
    # ✅ Atomic split (NO CONJUNCTION SPLITTING)
    # ------------------------------
    def atomic_split(self, seg: str) -> List[str]:
        return [p.strip() for p in self.EM_DASH_SPLIT.split(seg) if p.strip()]

    # ------------------------------
    # Short fragment attachment
    # ------------------------------
    def attach_short_fragments(self, segments: List[str], min_words: int = 3) -> List[str]:
        out = []
        for seg in segments:
            wc = len(seg.split())
            doc = self.nlp(seg)
            first_pos = doc[0].pos_ if doc else None
            has_verb = any(tok.pos_ == "VERB" for tok in doc)

            if wc >= min_words:
                out.append(seg)
            elif first_pos in ("ADP", "CCONJ", "PART", "DET", "ADV"):
                if out:
                    out[-1] += " " + seg
                else:
                    out.append(seg)
            elif wc <= 2 and not has_verb and out:
                out[-1] += " " + seg
            else:
                out.append(seg)

        return out

    # ------------------------------
    # Cleaning
    # ------------------------------
    def _clean_segment_text(self, s: str) -> str:
        s = re.sub(r'^[\-\•\●\*>\:\s]+', '', s)
        s = re.sub(r'\s{2,}', ' ', s)
        return s.rstrip(" ,;:").strip()

    # ------------------------------
    # ✅ Public API (HEADER PREPENDING ENABLED)
    # ------------------------------
    def segment(self, text: str) -> List[str]:
        text = self.normalize(text)

        text = re.sub(r'(?m)^\s*([-–—])(?=[A-Za-z])', r'\1 ', text)

        blocks = self.split_into_blocks(text)
        if not blocks:
            blocks = [{"header": None, "lines": [text]}]

        final_segments = []

        for block in blocks:
            header = block["header"]
            header_prefix = f"[{header}] " if header else ""

            bullet_chunks = self.split_bullets(block["lines"])
            sentence_lists = self.split_sentences(bullet_chunks)

            for lst in sentence_lists:
                for sent in lst:
                    atoms = self.atomic_split(sent)
                    atoms = self.attach_short_fragments(atoms, min_words=3)

                    for a in atoms:
                        cleaned = self._clean_segment_text(a)
                        if cleaned:
                            final_segments.append(header_prefix + cleaned)

        # Attach ultra-micro fragments
        merged = []
        for seg in final_segments:
            if len(seg.split()) < 2 and merged:
                merged[-1] = merged[-1] + " " + seg
            else:
                merged.append(seg)

        return merged





class JobDescriptionFeatureExtractor:
    """
    End-to-end pipeline:
    - Loads model + tokenizer
    - Segments job description
    - Runs multi-label classification
    - Returns structured JobDescriptionFeature objects
    """

    TAXONOMY = [
        "responsibility",
        "hard_skill",
        "soft_skill",
        "experience",
        "qualification",
        "requirement",
        "nice_to_have",
        "employer_mission",
        "employer_culture",
        "role_value",
        "benefit",
        "other",
    ]

    def __init__(
        self,
        threshold: float = 0.9,
        top_k: Optional[int] = None,
        max_length: int = 128,
    ):
        
        self.segmentor = Segmentor()
        self.threshold = threshold
        self.top_k = top_k
        self.max_length = max_length

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Attach already-loaded model + tokenizer.
        (Prevents tight coupling to HF/torch loading logic)
        
        """
        model_name = "microsoft/deberta-v3-small"

        class DebertaMultiLabel(nn.Module):
            def __init__(self, model_name, num_classes):
                super().__init__()
                self.backbone = AutoModel.from_pretrained(
                    model_name, 
                    trust_remote_code=False,
                    local_files_only=False,   # allows download if available
                    use_safetensors=False     # avoid safetensors conversion errors
                )
                
                hidden_size = self.backbone.config.hidden_size
                self.classifier = nn.Linear(hidden_size, num_classes)
                self.config = self.backbone.config
            def forward(self, input_ids, attention_mask):
                out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
                cls = out.last_hidden_state[:, 0]  # CLS token
                logits = self.classifier(cls)
                return logits

        checkpoint = torch.load("backend/core/job_model/checkpoints/model_best.pt", map_location="cpu")

        best_model = DebertaMultiLabel(model_name, len(self.TAXONOMY))

        best_model.load_state_dict(checkpoint['model_state'])

        self.device = torch.device("cpu")


        tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.tokenizer = tokenizer
        self.model = best_model

        self.model.to(self.device)


    def _predict_segment(self, segment_text: str):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded via load_model()")

        enc = self.tokenizer(
            segment_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.sigmoid(logits).squeeze(0)

        if self.top_k is not None:
            top_indices = torch.topk(probs, k=self.top_k).indices.cpu().tolist()
            predicted_labels = top_indices
        else:
            predicted_labels = (probs >= self.threshold).nonzero(as_tuple=True)[0].cpu().tolist()
            if len(predicted_labels) == 0:
                predicted_labels = [probs.argmax().item()]

        return predicted_labels, probs.cpu().tolist()

    # -----------------------------
    # ✅ Public Extraction API
    # -----------------------------

    def extract(self, job_text: str) -> List[JobDescriptionFeature]:
        """
        Full end-to-end extraction:
        JD → Segments → Model → Structured Features
        """

        segments = self.segmentor.segment(job_text)

        features: List[JobDescriptionFeature] = []

        for segment in segments:
            predicted_indices, probs = self._predict_segment(segment)

            # print(f"Segment: {segment}: Predicted Indices: {predicted_indices}, Probs: {probs}")

        

            predicted_taxonomy_labels = [
                self.TAXONOMY[i] for i in predicted_indices
            ]

            for label in predicted_taxonomy_labels:
                feature = JobDescriptionFeature(
                    type=label,
                    description=segment,
                )

                features.append(feature)

        return features




import pandas as pd
import re

# --------------------------------------------------------
# 0. Imports from your provided modules
# --------------------------------------------------------

# --- Header Regex ---
HEADER_CLEAN_RE = re.compile(r"^[\s\-\•\●\*>\d\)\(\.]+")

CATEGORY_PATTERNS = {
    # --- RESPONSIBILITIES / DUTIES ---
    "responsibility": re.compile(
        r"(responsibilit(y|ies)|duties|tasks|key\s+tasks|key\s+responsibilities|"
        r"primary\s+responsibilities|main\s+responsibilities|essential\s+duties|"
        r"job\s+duties|role\s+responsibilities|what\s+you('|’)?ll\s+do|"
        r"what\s+you\s+will\s+do|what\s+you\s+will\s+be\s+doing|day[-\s]*to[-\s]*day|"
        r"scope\s+of\s+work|expectations|your\s+impact)",
        re.IGNORECASE
    ),

    # --- QUALIFICATIONS ---
    "qualification": re.compile(
        r"(qualifications?|required\s+qualifications|desired\s+qualifications|"
        r"preferred\s+qualifications|qualification\s+requirements|education|"
        r"required\s+education|background|certifications?|licenses?|licences?)",
        re.IGNORECASE
    ),

    # --- EXPERIENCE (priority before requirement) ---
    "experience": re.compile(
        r"(experience|relevant\s+experience|professional\s+experience|"
        r"prior\s+experience|work\s+experience)",
        re.IGNORECASE
    ),

    # --- HARD SKILLS (priority before requirement) ---
    "hard_skill": re.compile(
        r"(skills?|required\s+skills|preferred\s+skills|technical\s+skills|"
        r"desired\s+skills|core\s+skills|competenc(y|ies)|competencies|"
        r"key\s+competencies|abilities|hard\s+skills?)",
        re.IGNORECASE
    ),

    # --- SOFT SKILLS (priority before requirement) ---
    "soft_skill": re.compile(
        r"(soft\s+skills|communication\s+skills|interpersonal\s+skills|"
        r"leadership\s+skills|people\s+skills|collaboration|teamwork|"
        r"problem[-\s]*solving|adaptability|time\s+management)",
        re.IGNORECASE
    ),

    # --- OPTIONAL REQUIREMENTS ---
    "nice_to_have": re.compile(
        r"(preferred|nice\s+to\s+have|desired|bonus\s+points|"
        r"good\s+to\s+have|preferred\s+qualifications|preferred\s+skills)",
        re.IGNORECASE
    ),

    # --- BENEFITS / COMPENSATION ---
    "benefit": re.compile(
        r"(benefits?|perks|total\s+rewards|compensation|salary|pay|bonus|"
        r"bonuses|equity|stock\s+options|wage|rewards|remuneration|"
        r"compensation\s+and\s+benefits|pay\s+range)",
        re.IGNORECASE
    ),

    # --- EMPLOYER MISSION ---
    "employer_mission": re.compile(
        r"(our\s+mission|mission|vision|purpose|our\s+vision|"
        r"making\s+an\s+impact|impact\s+statement)",
        re.IGNORECASE
    ),

    # --- COMPANY CULTURE ---
    "employer_culture": re.compile(
        r"(company\s+culture|culture|values|our\s+values|about\s+us|"
        r"who\s+we\s+are|about\s+the\s+team|who\s+you\s+are|team\s+culture|"
        r"why\s+you('|’)?ll\s+love\s+working\s+here)",
        re.IGNORECASE
    ),

    # --- VALUE / SELLING THE ROLE ---
    "role_value": re.compile(
        r"(career\s+growth|opportunity|growth\s+opportunities|"
        r"ownership|influence|why\s+join\s+us|why\s+us|"
        r"what\s+we\s+offer\s+you)",
        re.IGNORECASE
    ),

    # --- GENERIC REQUIREMENTS (lowest priority among real labels) ---
    "requirement": re.compile(
        r"(requirements?|job\s+requirements|minimum\s+requirements|"
        r"basic\s+requirements|mandatory\s+requirements|essential\s+requirements|"
        r"must[-\s]*have|eligibility|eligibility\s+requirements|authorization|"
        r"authorisation|work\s+authorization|security\s+clearance)",
        re.IGNORECASE
    ),
}


# --- Pattern-based + Context-based signals ---
PATTERNS = {
    "experience": [
        r"\b\d+\+?\s*(years|yrs)\s*(of\s+)?experience\b",
        r"\bminimum\s+of\s+\d+\s+years\b",
        r"\b\d+\s*(years|yrs)\s+working in\b",
        r"\b\d+\s*(years|yrs)\s+in\s+\w+",
    ],
    "qualification": [
        r"\b(Bachelor|Master|PhD|BA|BS|MBA|B\.Sc|M\.Sc|Doctorate)\b",
        r"\bdegree in\b",
        r"\bcertified in\b",
        r"\bformal education in\b",
    ],
    "certification": [
        r"\b(PMP|CFA|CPA|AWS|Azure|GCP|Cisco|ITIL|Scrum Master)\b",
        r"\bcertified\b",
    ],
    "benefit": [
        r"\$\d+[\d,]*|\$\d+k|salary range|bonus|equity|paid leave|health insurance|401k|stock options",
    ],
    "soft_skill": [
        r"ability to .*? (lead|collaborate|communicate|present|coordinate|manage|influence)",
        r"strong (communication|teamwork|leadership|interpersonal|problem-solving) skills",
        r"(excellent|outstanding) (communication|teamwork|leadership|collaboration) skills",
    ],
    "responsibility": [
        r"\bresponsible for\b",
        r"\bown\b",
        r"\bdrive\b",
        r"\bdeliver\b",
        r"\bmaintain\b",
        r"\bimplement\b",
        r"\bensure\b",
        r"\bcoordinate\b",
    ],
    "role_value": [
        r"\bopportunity to\b",
        r"\bcareer growth\b",
        r"\bexposure to\b",
        r"\binfluence\b",
        r"\bownership\b",
        r"\bdevelop skills\b",
    ],
    "employer_culture": [
        r"\bculture\b",
        r"\binclusive\b",
        r"\bdiverse\b",
        r"\bvalues\b",
        r"\bmission\b",
        r"\bvision\b",
        r"\bteam environment\b",
        r"\bworkplace\b",
    ],
}

# --- 2️⃣ Context / Content-Based Heuristics (vectorized) ---
CONTEXT_TRIGGERS = {
    "soft_skill_context": ["lead","collaborate","communicate","manage","coordinate",
                           "influence","negotiate","mentor","coach","facilitate"],
    "hard_skill_context": ["experience with","proficient in","knowledge of",
                           "familiar with","skilled in","expert in","hands-on experience"],
    "responsibility_context": ["responsible for","own","drive","deliver","maintain","implement",
                               "ensure","coordinate","manage","oversee","execute","supervise"],
    "role_value_context": ["opportunity to","career growth","exposure to","influence",
                           "ownership","develop skills","advance","promotion","impact"],
    "employer_culture_context": ["culture","inclusive","diverse","values","mission","vision",
                                 "team environment","workplace","collaborative","supportive","community"]
}

def apply_vectorized_signals(df: pd.DataFrame) -> pd.DataFrame:
    text_lower = df['segment_text'].str.lower()

    # Pattern-based
    for cat, patterns in PATTERNS.items():
        df[f"signal_{cat}"] = text_lower.str.contains("|".join(patterns), regex=True, na=False)

    # Context-based
    for cat, triggers in CONTEXT_TRIGGERS.items():
        df[f"signal_{cat}"] = text_lower.str.contains("|".join(map(re.escape, triggers)),
                                                      regex=True, na=False)
    return df

# --- Lexicon regex ---
LEXICONS = {
    "responsibility": [
        "lead", "manage", "oversee", "coordinate", "execute", "develop",
        "implement", "own", "drive", "deliver", "support", "maintain",
        "supervise", "mentor", "train", "build", "design", "plan",
        "manage projects", "manage stakeholders", "conduct", "perform",
        "create", "establish", "operate", "run", "optimize", "monitor",
        "report", "analyze", "improve", "guide", "direct", "facilitate",
        "negotiate", "liaise", "research", "evaluate", "assemble",
        "manage team", "lead team", "coordinate efforts", "provide guidance",
        "ensure compliance", "deliver outcomes", "shape strategy",
        "execute roadmap", "own roadmap", "drive initiatives", "take ownership",
        "handle"
    ],

    "hard_skill": [
        "python", "sql", "java", "javascript", "typescript", "c++",
        "c#", "go", "rust", "ruby", "php", "html", "css", "react",
        "angular", "vue", "node.js", "django", "flask", "spring",
        "tensorflow", "pytorch", "keras", "scikit-learn", "pandas",
        "numpy", "aws", "azure", "gcp", "docker", "kubernetes",
        "linux", "git", "jira", "airflow", "terraform", "ansible",
        "bash", "shell scripting", "etl", "api design", "microservices",
        "data modeling", "tableau", "power bi", "sap", "salesforce",
        "spark", "hadoop", "rest api"
    ],

    "soft_skill": [
        "communication", "verbal communication", "written communication",
        "interpersonal", "teamwork", "collaboration", "leadership",
        "problem solving", "critical thinking", "adaptability",
        "time management", "organization", "conflict resolution",
        "presentation skills", "public speaking", "negotiation",
        "decision making", "creativity", "initiative", "ownership",
        "attention to detail", "stakeholder management",
        "relationship building", "active listening", "empathy",
        "work ethic", "accountability", "self-motivation", "coaching",
        "mentoring", "influence", "cross-functional collaboration",
        "strategic thinking", "analytical thinking", "multitasking",
        "prioritization", "flexibility", "independence", "proactive",
        "resourcefulness", "interpersonal skills", "emotional intelligence",
        "customer focus", "professionalism", "dependability", "resilience",
        "openness to feedback", "collaborative mindset", "results oriented",
        "clarity in communication"
    ],

    "experience": [
        "years of experience", "experience with", "experience in",
        "hands-on experience", "worked with", "background in",
        "industry experience", "professional experience", "relevant experience",
        "proven experience", "track record", "practical experience",
        "career experience", "past experience", "prior experience",
        "work history", "work experience", "field experience",
        "subject matter expertise", "expertise in", "familiarity with",
        "skilled in", "competent in", "knowledge of", "applied experience",
        "functional experience", "extensive experience", "solid experience",
        "deep experience", "previously worked", "experience managing",
        "experience leading", "experience building", "experience delivering",
        "experience developing", "experience designing", "experience implementing",
        "experience troubleshooting", "experience supporting", "experience maintaining",
        "experience optimizing", "experience coordinating", "experience analyzing",
        "experience documenting", "experience monitoring", "experience planning",
        "experience reporting", "experience improving", "experience in a role",
        "multi-year experience", "minimum of"
    ],

    "qualification": [
        "bachelor", "bachelor’s degree", "master", "master’s degree", "phd",
        "mba", "ba", "bs", "ms", "degree", "college degree", "university degree",
        "related field", "certification", "certificate", "certified", "licensed",
        "license", "credential", "qualification", "educational background",
        "formal education", "academic background", "degree in", "professional certification",
        "industry certification", "pmp", "csm", "cissp", "aws certification",
        "azure certification", "gcp certification", "associate degree",
        "vocational", "technical training", "training program", "graduate degree",
        "postgraduate degree", "accredited", "recognized qualification",
        "diploma", "apprenticeship", "licence", "certified professional",
        "certified engineer", "certified specialist", "degree required",
        "educational requirement", "qualification requirement", "degree preferred",
        "academic qualification", "higher education"
    ],

    "requirement": [
        "must have", "required", "requirement", "mandatory", "minimum",
        "at least", "need to", "must be able to", "must demonstrate",
        "must possess", "must understand", "must show", "must meet",
        "must comply", "must hold", "ability to", "ability for",
        "ability in", "ability to perform", "ability to execute",
        "ability to deliver", "ability to manage", "ability to lead",
        "ability to analyze", "ability to solve", "proficiency", "proficient in",
        "competency", "must meet qualifications", "basic qualifications",
        "minimum qualifications", "required qualifications", "critical requirements",
        "essential requirements", "essential criteria", "job requirements",
        "eligibility", "eligible to work", "ability to work",
        "authorization required", "authorization to work", "work authorization",
        "background check required", "clearance required", "drug test required",
        "compliance requirement", "core requirement", "qualification requirement",
        "must have experience", "required skills", "must be"
    ],

    "nice_to_have": [
        "nice to have", "preferred", "preferred skills", "preferred qualifications",
        "bonus points", "good to have", "desirable", "ideal candidate",
        "additional skills", "additional experience", "preferred experience",
        "preferred background", "preferred education", "advantageous",
        "plus", "considered a plus", "added advantage", "strongly preferred",
        "recommended", "helpful", "asset", "nice bonus", "not required",
        "optional", "preferred tools", "preferred technologies",
        "preferred certifications", "preferred degree", "preferred training",
        "preferred knowledge", "preferred expertise", "extra skills",
        "extra experience", "extra qualification", "would be great",
        "would be nice", "nice extra", "optional skill", "optional experience",
        "desired skill", "desired experience", "desired background",
        "desired qualifications", "bonus experience", "bonus skill",
        "extra knowledge", "value-added skill", "value-add experience",
        "beneficial", "preferable"
    ],

    "employer_mission": [
        "mission", "our mission", "vision", "our vision", "purpose", "making an impact",
        "impact", "social impact", "sustainability", "sustainable", "climate",
        "public good", "nonprofit mission", "charitable", "impact-driven",
        "mission-driven", "purpose-driven", "meaningful work", "change the world",
        "company mission", "impact statement", "serve customers", "customer impact",
        "social responsibility", "corporate responsibility", "csr", "ethical",
        "values driven", "impact-focused", "positive impact", "community impact",
        "improve lives", "transform", "create change", "advancing", "mission focused",
        "mission-aligned", "why we exist", "our purpose", "what we do", "visionary",
        "doing good", "making a difference", "impactful work", "purposeful",
        "social value", "public service", "mission oriented", "global mission"
    ],

    "employer_culture": [
        "company culture", "culture", "our values", "values", "about us",
        "who we are", "team culture", "collaborative", "inclusive", "diversity",
        "diverse", "belonging", "friendly", "fast-paced", "startup culture",
        "entrepreneurial", "innovative", "open culture", "transparent",
        "work-life balance", "flexible", "supportive", "team oriented",
        "family friendly", "people first", "high performing", "customer centric",
        "employee-centric", "flat organization", "fun", "inclusive environment",
        "respectful", "culture of learning", "continuous learning", "mentorship culture",
        "peer support", "celebrate success", "collaboration-first", "team-first",
        "community", "office culture", "remote-friendly", "hybrid-friendly",
        "results-driven", "mission-driven culture", "values-aligned", "culture matters",
        "policies", "careers", "human-centered"
    ],

    "role_value": [
        "career growth", "growth opportunities", "opportunity", "ownership",
        "influence", "visibility", "impactful", "stretch assignment", "advancement",
        "career development", "promotion potential", "fast track", "mentorship",
        "leadership opportunity", "shape strategy", "make an impact",
        "high visibility", "drive change", "cross-functional exposure",
        "meaningful impact", "professional growth", "skills growth", "learning",
        "development program", "career path", "expand your role", "influence product",
        "build something", "take ownership", "lead initiatives", "high-impact role",
        "strategic role", "exposure to executives", "growth-oriented", "accelerated growth",
        "career advancement", "significant responsibility", "opportunity to innovate",
        "hands-on ownership", "empowerment", "autonomy", "decision-making authority",
        "shape the future", "be a key player", "high-responsibility", "impact on customers",
        "unique opportunity", "role with influence", "work with leaders"
    ],

    "benefit": [
        "salary", "pay", "compensation", "health insurance", "medical insurance",
        "dental insurance", "vision insurance", "paid time off", "pto", "vacation",
        "sick leave", "parental leave", "maternity leave", "paternity leave",
        "401k", "retirement plan", "bonus", "bonuses", "equity", "stock options",
        "rsu", "wellness program", "gym membership", "flexible schedule", "remote work",
        "hybrid work", "transportation allowance", "meal allowance", "learning stipend",
        "education reimbursement", "tuition reimbursement", "signing bonus",
        "performance bonus", "annual bonus", "commission", "incentives", "perks",
        "employee discount", "life insurance", "disability insurance",
        "relocation assistance", "work-life balance", "flexible hours",
        "paid holidays", "company laptop", "company phone", "professional development",
        "career growth support", "mental health support", "childcare support"
    ],

    "other": [
        "apply", "how to apply", "apply online", "contact", "email us", "submit resume",
        "submit cv", "job id", "job reference", "EOE", "equal opportunity",
        "equal opportunity employer", "visa sponsorship", "sponsorship available",
        "work authorization", "work permit", "background check", "criminal background",
        "drug test", "start date", "immediate start", "contract", "temporary",
        "part-time", "full-time", "internship", "intern", "consultant", "freelance",
        "contractor", "permanent", "on-site", "on site", "remote", "hybrid",
        "shift", "night shift", "day shift", "hours", "overtime", "travel required",
        "occasional travel", "must relocate", "relocation", "confidential", "non-disclosure",
        "nda", "applicant", "recruiter contact"
    ]
}

def build_regex_from_list(values):
    values_sorted = sorted(set(values), key=lambda s: -len(s))
    pattern = r"\b(?:" + "|".join(re.escape(v) for v in values_sorted) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)

LEXICON_REGEX = {label: build_regex_from_list(words)
                 for label, words in LEXICONS.items()}

PRIORITY_ORDER = [
    "hard_skill","soft_skill","experience","qualification","responsibility",
    "role_value","employer_mission","employer_culture","benefit",
    "nice_to_have","requirement","other"
]

def lexicon_label_segment(text: str):
    if not isinstance(text, str) or not text.strip():
        return "other", 0.0, {}

    match_counts = {}
    match_positions = {}

    for label, regex in LEXICON_REGEX.items():
        found = set()
        first_pos = None
        for m in regex.finditer(text):
            found.add(m.group(0).lower())
            if first_pos is None:
                first_pos = m.start()
        if found:
            match_counts[label] = len(found)
            match_positions[label] = first_pos

    if not match_counts:
        return "other", 0.0, {}

    # Highest count
    max_count = max(match_counts.values())
    candidates = [k for k,v in match_counts.items() if v == max_count]

    # Priority rule
    for p in PRIORITY_ORDER:
        if p in candidates:
            chosen = p
            break

    # Tie-break by earliest appearance
    chosen = min(candidates, key=lambda c: match_positions.get(c, 10**9))

    confidence = min(1.0, match_counts.get(chosen, 0) / 2)
    return chosen, confidence, match_counts


def apply_lexicon_labeling(df: pd.DataFrame):
    labels = df["segment_text"].apply(lambda t: lexicon_label_segment(t))
    df["lex_label"] = labels.apply(lambda x: x[0])
    df["lex_confidence"] = labels.apply(lambda x: x[1])
    df["lex_match_counts"] = labels.apply(lambda x: x[2])
    return df


TAXONOMY_LABELS = [
    "responsibility",
    "hard_skill",
    "soft_skill",
    "experience",
    "qualification",
    "requirement",
    "nice_to_have",
    "employer_mission",
    "employer_culture",
    "role_value",
    "benefit",
    "other",
]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_multilabel_probabilities(row):
    """
    Produces a MULTI-LABEL probability for each taxonomy label.
    Each label is scored independently.
    """

    logits = {label: -2.0 for label in TAXONOMY_LABELS}

    # --- 1. Header evidence ---
    header_label = row.get("header_label")
    if header_label in logits:
        logits[header_label] += 2.5

    # --- 2. Lexicon evidence ---
    lex_label = row.get("lex_label")
    lex_conf = float(row.get("lex_confidence", 0.0))
    lex_counts = row.get("lex_match_counts", {})

    if lex_label in logits:
        logits[lex_label] += 2.0 * lex_conf

    for label, count in lex_counts.items():
        if label in logits:
            logits[label] += 0.4 * count

    # --- 3. Pattern/context signals ---
    for col in row.index:
        if col.startswith("signal_") and bool(row[col]):
            label = col.replace("signal_", "")
            if label in logits:
                logits[label] += 1.0

    # --- 4. Convert logits → probability ---
    probs = {label: float(sigmoid(logits[label])) for label in TAXONOMY_LABELS}
    return probs



# --------------------------------------------------------
# 1. WeakJobDescriptionFeatureExtractor
# --------------------------------------------------------

class WeakJobDescriptionFeatureExtractor:

    def __init__(self, threshold: float = 0.8):
        """
        segmentor_cls -> your Segmentor() class
        """
        self.segmentor_cls = Segmentor()
        self.threshold = threshold

    def extract(self, job_description: str) -> pd.DataFrame:
        """
        Returns DataFrame with:
        - segment_text
        - header_label
        - signal_* columns
        - lex_label, lex_confidence, lex_match_counts
        - final_weak_label
        """
        # --- A. SEGMENTATION (only allowed method)
        segmentor = self.segmentor_cls
        segments = segmentor.segment(job_description)

        df = pd.DataFrame({"segment_text": segments})

        # --- B. HEADER DETECTION
        df["header_label"] = df["segment_text"].apply(self._classify_header)

        # --- C. PATTERN + CONTEXT SIGNALS
        df = apply_vectorized_signals(df)

        # --- D. LEXICON LABELS
        df = apply_lexicon_labeling(df)

        # --- E. MERGE WEAK LABELS (priority rules)
        df["final_weak_label"] = df.apply(self._merge_labels, axis=1)

        return df

    # --------------------------------------------------------
    # Header classification
    # --------------------------------------------------------
    def _classify_header(self, text: str):
        cleaned = HEADER_CLEAN_RE.sub("", text).strip().lower()

        for cat, regex in CATEGORY_PATTERNS.items():
            if regex.search(cleaned):
                return cat
        return None

    # --------------------------------------------------------
    # Label merging logic
    # --------------------------------------------------------
    def _merge_labels(self, row):
        """
        Priority order:
           1. Header label (if present)
           2. Lexicon label (if lex_confidence > 0)
           3. Strong pattern/context signals (if any)
           4. Default to 'other'
        """

        # 1. Header override
        if row["header_label"]:
            return row["header_label"]

        # 2. Lexicon high-confidence
        if row["lex_confidence"] >= 0.75:
            return row["lex_label"]

        # 3. Pattern/context signals
        signal_cols = [c for c in row.index if c.startswith("signal_")]
        positives = [c.replace("signal_", "") for c in signal_cols if row[c]]

        if positives:
            # Use highest-priority positive label
            for label in PRIORITY_ORDER:
                if label in positives:
                    return label

        # 4. Lexicon low-confidence fallback
        if row["lex_confidence"] > 0:
            return row["lex_label"]

        return "other"
    
    # --------------------------------------------------------
    # NEW: ENTITY GENERATION STEP
    # --------------------------------------------------------
    def extract_entities(self, job_description: str) -> List[JobDescriptionFeature]:
        """
        Produces a list of JobDescriptionFeature objects based on thresholded multilabel probabilities.
        """
        df = self.extract(job_description)

        df["label_probabilities"] = df.apply(compute_multilabel_probabilities, axis=1)

        entities = []

        for _, row in df.iterrows():
            text = row["segment_text"]
            probs: Dict[str, float] = row["label_probabilities"]

            # find ALL labels ≥ threshold
            labels_above_threshold = [
                label for label, p in probs.items() if p >= self.threshold
            ]

            if labels_above_threshold:
                for label in labels_above_threshold:
                    entities.append(JobDescriptionFeature(type=label, description=text))
            else:
                # default fallback
                entities.append(JobDescriptionFeature(type="other", description=text))

        return entities




