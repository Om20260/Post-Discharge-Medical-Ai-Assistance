Post Discharge Medical AI AssistantA production-ready proof-of-concept multi-agent AI system for post-discharge patient care management. It features RAG-based medical knowledge retrieval from a comprehensive clinical textbook, intelligent agent routing with LangGraph, and patient data management.🚨 Medical DisclaimerThis software is provided for educational and demonstration purposes only. It is NOT intended to be used for actual medical diagnosis, treatment, or patient care. The AI-generated responses should not be considered as professional medical advice. Always consult qualified healthcare professionals for medical advice, diagnosis, and treatment. In case of emergency, call your local emergency services immediately. The developers and contributors of this software assume no liability for any medical decisions made based on the use of this software. ✨ Key FeaturesMulti-Agent Architecture: A Receptionist Agent identifies the patient and retrieves their data, while a Clinical AI Agent handles medical questions. Orchestration is managed by LangGraph.Advanced RAG Implementation: Answers are generated using Retrieval-Augmented Generation (RAG) from a comprehensive clinical nephrology PDF, providing thousands of detailed, evidence-based text chunks.Patient Data Management: Includes 27 diverse, dummy patient discharge reports covering various nephrology conditions.Web Search Fallback: Integrated with Tavily (with a DuckDuckGo fallback) for recent medical information not covered in the static knowledge base.Comprehensive Logging: Detailed system logs (system.log) and structured interaction logs (interactions.jsonl) for full observability.Modern Web Interface: A clean, responsive chat interface built with Streamlit, showing agent transitions and system status.Production-Ready Backend: A high-performance async API built with FastAPI, featuring auto-generated documentation.🏗️ System ArchitectureThe system uses a Streamlit frontend that communicates via REST API to a FastAPI backend. The backend hosts the LangGraph multi-agent system, which routes tasks between the Receptionist and Clinical AI agents. These agents utilize tools to access a SQLite Patient Database, a ChromaDB RAG system, and a Web Search tool.
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Frontend                     │
│              (http://localhost:8501)                    │
└────────────────────┬────────────────────────────────────┘
                     │ REST API
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Backend                        │
│              (http://localhost:8000)                    │
│                                                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │         LangGraph Multi-Agent System              │ │
│  │                                                   │ │
│  │  ┌─────────────────┐      ┌──────────────────┐  │ │
│  │  │  Receptionist   │─────▶│  Clinical AI     │  │ │
│  │  │     Agent       │      │     Agent        │  │ │
│  │  └─────────────────┘      └──────────────────┘  │ │
│  │         │                         │              │ │
│  └─────────┼─────────────────────────┼──────────────┘ │
│            │                         │                │
│            ▼                         ▼                │
│  ┌──────────────────┐    ┌──────────────────────┐   │
│  │ Patient Database │    │   RAG System         │   │
│  │   (SQLite)       │    │  (ChromaDB)          │   │
│  └──────────────────┘    └──────────────────────┘   │
│                                  │                   │
│                                  ▼                   │
│                          ┌──────────────────┐       │
│                          │   Web Search     │       │
│                          │ (Tavily/DDG)     │       │
│                          └──────────────────┘       │
└─────────────────────────────────────────────────────┘
🛠️ Technologies UsedComponentTechnologyMulti-Agent FrameworkLangGraphAgent FrameworkLangChainVector DatabaseChromaDBEmbeddingsSentence-TransformersDatabaseSQLite + SQLAlchemyBackendFastAPIFrontendStreamlitWeb SearchTavily + DuckDuckGoLoggingLoguru🚀 Quick Start (5 Minutes)1. Install DependenciesBashpip install -r requirements.txt
2. Setup EnvironmentFirst, copy the example environment file:Bash# Windows
copy .env.example .env

# macOS/Linux
cp .env.example .env
Next, edit the new .env file and add your OPENAI_API_KEY.OPENAI_API_KEY=sk-your-key-here
3. Initialize SystemRun the setup scripts to create the patient database and the vector store.Bash# Setup patient database
python scripts/setup_database.py

# Setup vector database (may take 5-15 mins for PDF processing)
python scripts/setup_vector_db.py
4. Start BackendIn your first terminal:Bashcd backend
uvicorn main:app --reload --port 8000
5. Start FrontendOpen a new terminal:Bashcd frontend
streamlit run app.py
6. Open BrowserVisit http://localhost:8501 to use the application.🏃 Windows Quick SetupFor Windows users, an automated setup script is available:Bash# 1. Run automated setup
setup.bat

# 2. Edit .env with your API key

# 3. Start backend
start_backend.bat

# 4. Start frontend (in new window)
start_frontend.bat
💬 Sample WorkflowInitial InteractionUser: Hello
Receptionist: Hello! I'm your post-discharge care assistant. 
              What's your name?

User: My name is John Smith
Receptionist: [Uses patient_data_retrieval tool]
              Hi John! I found your discharge report from January 15th 
              for Chronic Kidney Disease Stage 3. How are you feeling today?
Medical Query RoutingUser: I'm having swelling in my legs. Should I be worried?
Receptionist: This sounds like a medical concern. Let me connect you 
              with our Clinical AI Agent.

Clinical Agent: [Uses RAG over nephrology reference]
                Based on your CKD diagnosis and nephrology guidelines, 
                leg swelling can indicate fluid retention...
                
                [Provides detailed medical information with citations]
                
                ⚕️ This is educational information only. Please consult 
                your healthcare provider for personalized advice.
Web Search FallbackUser: What's the latest research on SGLT2 inhibitors for kidney disease?
Clinical Agent: [Recognizes need for current information]
                [Uses web_search tool]
                

📁 Project Structurepost-discharge-ai-assistant/
├── backend/
│   ├── agents/
│   │   ├── receptionist_agent.py    # Patient interaction agent
│   │   ├── clinical_agent.py        # Medical Q&A agent
│   │   └── agent_graph.py           # LangGraph orchestration
│   ├── database/
│   │   ├── models.py                # SQLAlchemy models
│   │   └── database.py              # Database manager
│   ├── rag/
│   │   └── vector_store.py          # ChromaDB integration
│   ├── tools/
│   │   ├── patient_retrieval.py     # Patient data tool
│   │   ├── rag_tool.py              # RAG search tool
│   │   └── web_search.py            # Web search tool
│   ├── utils/
│   │   └── logger.py                # Comprehensive logging
│   └── main.py                      # FastAPI application
├── frontend/
│   └── app.py                       # Streamlit interface
├── data/
│   ├── patient_reports.json         # 27 patient records
│   ├── nephrology_reference.txt     # Fallback medical reference
│   └── ...
├── knowledge base for RAG/
│   └── comprehensive-clinical-nephrology.pdf # Primary knowledge base
├── scripts/
│   ├── setup_database.py            # Database initialization
│   └── setup_vector_db.py           # Vector DB initialization
├── logs/                          # (Generated, gitignored)
├── docs/
│   ├── architecture_justification.md # Technical decisions
│   └── demo_guide.md                # Demo walkthrough
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment template
├── .gitignore                       
├── INSTALLATION.md                  # Detailed installation guide
├── QUICKSTART.md                    # This guide
├── LICENSE                          
└── PROJECT_SUMMARY.md    

📚 Available DocumentationINSTALLATION.md: A detailed, step-by-step setup guide for all operating systems.: A high-level overview of the project, features, and deliverables.: Details on how the comprehensive nephrology PDF is used for RAG.: A complete walkthrough script for demonstrating the application's capabilities.: In-depth explanations for all technical and design decisions.API Docs: Once the backend is running, visit http://localhost:8000/docs for the auto-generated FastAPI (Swagger) documentation.🐛 TroubleshootingIssue: "Module not found" errorsSolution: Ensure your virtual environment is activated and run pip install --upgrade -r requirements.txt.Issue: "OpenAI API key not found"Solution: Check that you copied .env.example to .env (not .env.txt) and that your API key is correctly pasted inside.Issue: "Port 8000 already in use"Solution: Find and stop the process using port 8000, or restart the backend on a different port (e.g., uvicorn main:app --reload --port 8001).Issue: "Database is empty" / "Patient not found"Solution: Run the database setup script: python scripts/setup_database.py.Issue: "Vector store is empty" / Slow medical answersSolution: Run the vector DB setup script: python scripts/setup_vector_db.py. Note this can take 5-15 minutes.
