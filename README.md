# Job App Assistant: AI-Augmented Application Suite

**A Local-First, Privacy-Centric "Career CRM" for High-Volume, High-Quality Job Applications.**

This platform ingests unstructured data (CVs, Job Descriptions) and transforms them into a structured **Knowledge Graph**. It uses a hybrid AI engine to perform forensic gap analysis, automate tailoring, and generate evidence-backed cover letters without hallucination.

---

## System Architecture

The system is designed with a **Service-Oriented Architecture (SOA)** on the backend and a component-driven React frontend.

### **1. The Core "Brain" (AI & Logic Layer)**
The application does not rely on simple LLM prompts. It uses a sophisticated pipeline:
* **Inference Engine (`backend/core/inferer.py`):**
    * **Hybrid Search:** Uses **TF-IDF** (Exact keyword frequency) + **DeBERTa Embeddings** (Semantic meaning) to match Job Requirements to CV Skills/Experience.
    * **Atomic Evidence:** Deconstructs a CV into "atoms" (e.g., a single bullet point) while preserving its **Lineage** (e.g., *Match found in Bullet 3 -> Project A -> Experience B*).
* **LLM Manager (`backend/core/llm_manager.py`):**
    * Manages a local **Llama 3** instance (via `llama.cpp`) for privacy.
    * **JSON Repair Protocol:** Includes a robust regex-based repair engine to fix malformed JSON output from local models, ensuring API stability.
* **Forensics Engine (`backend/core/forensics.py`):**
    * Calculates a **"Fit Score"** by assigning weighted values to requirements (e.g., *Critical: 1.5x*, *Nice-to-have: 0.5x*).
    * Generates a "RoleCase" report identifying critical gaps vs. verified strengths.

### **2. The Persistence Layer (`backend/core/registry.py`)**
* **Pattern:** Repository/Registry Pattern.
* **Current State:** **TinyDB** (JSON file-based) for easy local setup.
* **Future-Proofing:** The `Registry` class completely abstracts data access. To migrate to **MongoDB**, only the methods inside `Registry` need to be swapped; the rest of the app remains untouched.
* **Object Models (`backend/core/models.py`):**
    * **`MappingPair`:** The atomic link between a Job Requirement and a CV Item. Stores the "Strength" score and "Meta" evidence.
    * **`DerivedCV`:** A "Snapshot" of the Base CV tailored for a specific job.
    * **`CoverLetter`:** Structured as `Paragraphs` -> `Ideas` -> `Evidence`.

### **3. The API Layer (`backend/routes/`)**
Built with **FastAPI**. Key endpoints include:
* **`POST /forensics/generate`:** The "One-Click" analysis. Chains `Inference` -> `DB Save` -> `Calculation`.
* **`POST /mapping/{id}/infer`:** Supports **"Tuning Modes"** (e.g., `super_eager`, `picky_mode`) to adjust AI sensitivity.
* **`POST /coverletter/{id}/autofill`:** Implements the **"Ownership Engine"**. It builds a letter structure around user-provided points while filling gaps with AI suggestions, ensuring user content is never overwritten.
* **`POST /job/upsert`:** The entry point for the Chrome Extension to dump scraped job data.

---

## 💻 Frontend Ecosystem (`frontend/src/`)

A React + Vite application using standard routing (`App.jsx`).

### **Workflow & Key Components**
1.  **Job Library (`/jobs`):**
    * **Component:** `JobLibrary.jsx`.
    * **Function:** Lists all scraped or manually added jobs. Entry point for the "RoleCase" analysis.
2.  **Application Tracker (`/applications`):**
    * **Component:** `AppTrackerPage.jsx`.
    * **Function:** Kanban/List view of active applications.
3.  **Deep Dive Workspace (`/application/:id`):**
    * The central hub connecting three specialized studios:
    * **Mapping Manager (`MappingManager.jsx`):** A triage board to review/reject AI-suggested matches.
    * **Tailored CV (`TailoredCVManager.jsx`):** A WYSIWYG editor to customize the CV snapshot.
    * **Doc Studio (`SupportingDocStudio.jsx`):** A block-based editor for Cover Letters using the "Ownership Engine".
4.  **Forensic Analysis (`RoleCaseView.jsx`):**
    * Visualizes the "Fit Score" and "Critical Gaps." Allows users to manually link evidence to missing requirements.

---

## Extensions & Integrations

### **RoleCase Scraper (Chrome Extension)**
* **Location:** `kimathinyota/rolecasejobscrapeextension`
* **Function:** Scrapes job details (Title, Company, Description) from web pages.
* **Integration:** Sends a `POST` request to the backend's `/job/upsert` endpoint to instantly add the job to the database.

---

## Developer Guide

### **Environment Setup**
1.  **Backend:**
    * Python 3.10+
    * Install dependencies: `pip install -r requirements.txt`
    * **Model Setup:** Place your `.gguf` Llama model in `backend/data/models/`.
    * Run: `python run.py` (Starts FastAPI on port 8000).
2.  **Frontend:**
    * Node.js 18+
    * Install: `npm install`
    * Run: `npm run dev` (Starts Vite on port 5173).

### **Migration to Production (MongoDB)**
To support multiple users:
1.  Replace `backend/core/database.py` with a PyMongo implementation.
2.  Update `backend/core/registry.py`:
    * Rewrite `_get`, `_insert`, `_update` methods to use MongoDB collections instead of TinyDB tables.
    * Ensure `_get_nested_entity` uses MongoDB array filters or aggregation pipelines for performance.
3.  Implement JWT Authentication middleware in `backend/main.py`.

---