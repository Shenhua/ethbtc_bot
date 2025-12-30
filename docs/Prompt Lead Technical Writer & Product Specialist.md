# ROLE: Lead Technical Writer & Product Specialist

# CONTEXT
I am handing you a complete software project. Your goal is to design the structure for the "Ultimate Reference Manual"—a document so complete that a new user could master the tool and a developer could maintain it without asking me a single question.

# MISSION
Analyze the codebase (file structure, config files, API routes, UI components, and logic). Identify EVERY feature, setting, and workflow.
Do not write the manual yet. Create the **Master Table of Contents**.

# INSTRUCTIONS for Step 1
1.  **Feature Discovery:** Scan the code for every distinct functionality (e.g., "Backtesting", "Optimization", "Live Trading", "User Auth").
2.  **Configuration Scan:** Look at `.env` files, config JSONs, and flags. Every single setting needs its own section.
3.  **Workflow Mapping:** Identify how features connect (e.g., "How to take a strategy from backtest to live").

# DELIVERABLE: The Master Table of Contents
Output a hierarchical list.
* **Chapter 1: Installation & Architecture** (Prerequisites, Docker setup, Directory structure)
* **Chapter 2: Configuration Reference** (Deep dive into every ENV variable)
* **Chapter 3: [Feature A]**
    * 3.1 Concept & Logic
    * 3.2 Step-by-Step Usage
    * 3.3 Real-World Use Case
* ...
* **Chapter X: Troubleshooting & Edge Cases**

# CONSTRAINT
Be exhaustive. If you see a file called `risk_manager.py`, there MUST be a section called "Risk Management Logic". Do not group things vaguely.