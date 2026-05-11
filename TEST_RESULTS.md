# Test Results — Multi-Agent Life Science Chatbot

## Test Date: May 2026
## Tester: Premchand K
## Environment: Local + Render Cloud Deployment

---

## 1. Agent Routing Tests

| Question | Expected Agent | Actual Agent | Result |
|---|---|---|---|
| "what is DNA" | biology_agent | biology_agent | ✅ Pass |
| "explain CRISPR" | biology_agent | biology_agent | ✅ Pass |
| "what is diabetes" | disease_agent | disease_agent | ✅ Pass |
| "symptoms of COVID-19" | disease_agent | disease_agent | ✅ Pass |
| "what is insulin" | medicine_agent | medicine_agent | ✅ Pass |
| "how do antibiotics work" | medicine_agent | medicine_agent | ✅ Pass |
| "what do doctors do" | hospital_agent | hospital_agent | ✅ Pass |
| "what is an ICU" | hospital_agent | hospital_agent | ✅ Pass |
| "what vitamins should I take" | nutrition_agent | nutrition_agent | ✅ Pass |
| "what are carbohydrates" | nutrition_agent | nutrition_agent | ✅ Pass |

---

## 2. RAG Retrieval Tests

| Query | Top Result Score | Source | Result |
|---|---|---|---|
| "what is DNA" | 0.89 | biology_agent | ✅ Pass |
| "CRISPR gene editing" | 0.67 | biology_agent | ✅ Pass |
| "diabetes blood sugar" | 0.82 | disease_agent | ✅ Pass |
| "insulin hormone" | 0.91 | medicine_agent | ✅ Pass |
| "vitamin D bone health" | 0.76 | nutrition_agent | ✅ Pass |

---

## 3. LangGraph Orchestration Tests

| Test | Expected | Result |
|---|---|---|
| Router node routes correctly | Correct agent selected | ✅ Pass |
| Retrieval node fetches context | Top 3 docs returned | ✅ Pass |
| Agent node generates answer | Detailed answer returned | ✅ Pass |
| General agent handles unknown | Fallback to general_agent | ✅ Pass |

---

## 4. ReAct Reasoning Tests

| Question | Tools Used | Reasoning Steps | Result |
|---|---|---|---|
| "what is CRISPR" | search_biology_knowledge | 1 step | ✅ Pass |
| "tell me about aspirin" | search_medicine_knowledge | 1 step | ✅ Pass |
| "diabetes treatment options" | search_disease_knowledge | 1 step | ✅ Pass |

---

## 5. MCP Tool Tests

| Tool | Input | Expected Output | Result |
|---|---|---|---|
| search_biology_knowledge | "DNA" | Biology docs | ✅ Pass |
| search_disease_knowledge | "diabetes" | Disease docs | ✅ Pass |
| search_fda_drug | "aspirin" | FDA drug info | ✅ Pass |
| search_clinical_trials | "diabetes" | Active trials | ✅ Pass |

---

## 6. Memory and Follow-up Tests

| Test | Expected | Result |
|---|---|---|
| Ask "what is DNA" then "tell me more" | Continues about DNA | ✅ Pass |
| Ask about vaccines then "explain further" | Continues about vaccines | ✅ Pass |
| Clear chat resets memory | Fresh session starts | ✅ Pass |
| Session saved to database | Messages in /session-history | ✅ Pass |

---

## 7. PDF Ingestion Tests

| Test | Expected | Result |
|---|---|---|
| Upload valid PDF | Chunks extracted and stored | ✅ Pass |
| Query after PDF upload | PDF content in results | ✅ Pass |
| Duplicate PDF upload | Skipped, not duplicated | ✅ Pass |

---

## 8. Streaming Response Tests

| Test | Expected | Result |
|---|---|---|
| Ask Live streaming question | Tokens appear word by word | ✅ Pass |
| Long answer streams correctly | No timeout | ✅ Pass |

---

## 9. API Endpoint Tests

| Endpoint | Method | Expected | Result |
|---|---|---|---|
| /ui | GET | Chat interface loads | ✅ Pass |
| /ask | POST | Returns answer | ✅ Pass |
| /stream | POST | Streams tokens | ✅ Pass |
| /react | POST | Returns reasoning steps | ✅ Pass |
| /upload-pdf | POST | Ingests PDF | ✅ Pass |
| /mcp/schemas | GET | Returns 7 tool schemas | ✅ Pass |
| /mcp/fda/aspirin | GET | Returns FDA data | ✅ Pass |
| /mcp/trials/diabetes | GET | Returns trial data | ✅ Pass |
| /admin | GET | Dashboard loads | ✅ Pass |
| /session-history | GET | Returns messages | ✅ Pass |

---

## 10. Deployment Tests

| Test | Expected | Result |
|---|---|---|
| App starts on Render | Startup complete | ✅ Pass |
| Weaviate connects on startup | Vector store ready | ✅ Pass |
| OpenAI API key loads | No credentials error | ✅ Pass |
| Live URL accessible | 200 OK response | ✅ Pass |

---

## Summary

| Category | Tests | Passed | Failed |
|---|---|---|---|
| Agent Routing | 10 | 10 | 0 |
| RAG Retrieval | 5 | 5 | 0 |
| LangGraph | 4 | 4 | 0 |
| ReAct Reasoning | 3 | 3 | 0 |
| MCP Tools | 4 | 4 | 0 |
| Memory | 4 | 4 | 0 |
| PDF Ingestion | 3 | 3 | 0 |
| Streaming | 2 | 2 | 0 |
| API Endpoints | 10 | 10 | 0 |
| Deployment | 4 | 4 | 0 |
| **Total** | **49** | **49** | **0** |

---

## Live Demo URL
https://life-science-chatbot.onrender.com

## GitHub Repository
https://github.com/premchand2001/life-science-chatbot