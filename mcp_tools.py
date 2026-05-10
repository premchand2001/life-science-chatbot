# mcp_tools.py
# MCP (Model Context Protocol) Tool Schemas and Implementations

import requests
from langchain_core.tools import tool
from vector_store import chroma_search

# ── MCP Tool Schemas ──────────────────────────────────────────
# These define the structure of each tool following MCP standards

MCP_TOOL_SCHEMAS = {
    "search_biology_knowledge": {
        "name": "search_biology_knowledge",
        "description": "Search the biology knowledge base for information about DNA, cells, genes, proteins, CRISPR, mutations, and other biology topics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The biology question or topic to search for"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 3
                }
            },
            "required": ["query"]
        },
        "output_schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "score": {"type": "number"},
                    "source": {"type": "string"}
                }
            }
        }
    },

    "search_disease_knowledge": {
        "name": "search_disease_knowledge",
        "description": "Search the disease knowledge base for information about diabetes, cancer, COVID-19, hypertension, infections, and other diseases.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The disease or condition to search for"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 3
                }
            },
            "required": ["query"]
        },
        "output_schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "score": {"type": "number"},
                    "source": {"type": "string"}
                }
            }
        }
    },

    "search_medicine_knowledge": {
        "name": "search_medicine_knowledge",
        "description": "Search the medicine knowledge base for information about drugs, treatments, vaccines, and medications.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The medicine or treatment to search for"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 3
                }
            },
            "required": ["query"]
        },
        "output_schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "score": {"type": "number"},
                    "source": {"type": "string"}
                }
            }
        }
    },

    "search_fda_drug": {
        "name": "search_fda_drug",
        "description": "Search the OpenFDA database for real drug information including side effects, warnings, and indications. Uses live US FDA data.",
        "input_schema": {
            "type": "object",
            "properties": {
                "drug_name": {
                    "type": "string",
                    "description": "The name of the drug to look up in FDA database"
                }
            },
            "required": ["drug_name"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "brand_name": {"type": "string"},
                "generic_name": {"type": "string"},
                "warnings": {"type": "string"},
                "indications": {"type": "string"},
                "source": {"type": "string"}
            }
        }
    },

    "search_clinical_trials": {
        "name": "search_clinical_trials",
        "description": "Search for active clinical trials from ClinicalTrials.gov for a given disease or condition.",
        "input_schema": {
            "type": "object",
            "properties": {
                "condition": {
                    "type": "string",
                    "description": "The disease or condition to find clinical trials for"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of trials to return",
                    "default": 3
                }
            },
            "required": ["condition"]
        },
        "output_schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "status": {"type": "string"},
                    "phase": {"type": "string"},
                    "url": {"type": "string"}
                }
            }
        }
    }
}

# ── MCP Tool Implementations ──────────────────────────────────

@tool
def search_biology_knowledge(query: str, top_k: int = 3) -> list:
    """Search the biology knowledge base for DNA, cells, genes, proteins, CRISPR."""
    results = chroma_search(query, agent_name="biology_agent", top_k=top_k)
    return results if results else [{"answer": "No results found.", "score": 0, "source": "biology_agent"}]

@tool
def search_disease_knowledge(query: str, top_k: int = 3) -> list:
    """Search the disease knowledge base for diabetes, cancer, COVID-19, infections."""
    results = chroma_search(query, agent_name="disease_agent", top_k=top_k)
    return results if results else [{"answer": "No results found.", "score": 0, "source": "disease_agent"}]

@tool
def search_medicine_knowledge(query: str, top_k: int = 3) -> list:
    """Search the medicine knowledge base for drugs, vaccines, treatments."""
    results = chroma_search(query, agent_name="medicine_agent", top_k=top_k)
    return results if results else [{"answer": "No results found.", "score": 0, "source": "medicine_agent"}]

@tool
def search_hospital_knowledge(query: str, top_k: int = 3) -> list:
    """Search the hospital knowledge base for doctors, nurses, clinics, healthcare."""
    results = chroma_search(query, agent_name="hospital_agent", top_k=top_k)
    return results if results else [{"answer": "No results found.", "score": 0, "source": "hospital_agent"}]

@tool
def search_nutrition_knowledge(query: str, top_k: int = 3) -> list:
    """Search the nutrition knowledge base for vitamins, diet, minerals, food."""
    results = chroma_search(query, agent_name="nutrition_agent", top_k=top_k)
    return results if results else [{"answer": "No results found.", "score": 0, "source": "nutrition_agent"}]

@tool
def search_fda_drug(drug_name: str) -> dict:
    """
    Search OpenFDA database for real drug information.
    Returns warnings, indications, and side effects from US FDA.
    """
    try:
        url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{drug_name}&limit=1"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            if data.get("results"):
                result = data["results"][0]
                return {
                    "brand_name": drug_name,
                    "generic_name": str(result.get("openfda", {}).get("generic_name", ["Unknown"])[0]),
                    "warnings": str(result.get("warnings", ["No warnings found"])[0])[:500],
                    "indications": str(result.get("indications_and_usage", ["No indications found"])[0])[:500],
                    "source": "OpenFDA (US Food & Drug Administration)"
                }
        return {
            "brand_name": drug_name,
            "generic_name": "Not found",
            "warnings": "Drug not found in FDA database",
            "indications": "Drug not found in FDA database",
            "source": "OpenFDA"
        }
    except Exception as e:
        return {
            "brand_name": drug_name,
            "error": str(e),
            "source": "OpenFDA"
        }

@tool
def search_clinical_trials(condition: str, max_results: int = 3) -> list:
    """
    Search ClinicalTrials.gov for active clinical trials.
    Returns real trial data for a given disease or condition.
    """
    try:
        url = f"https://clinicaltrials.gov/api/v2/studies?query.cond={condition}&filter.overallStatus=RECRUITING&pageSize={max_results}"
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            trials = []
            for study in data.get("studies", []):
                protocol = study.get("protocolSection", {})
                id_module = protocol.get("identificationModule", {})
                status_module = protocol.get("statusModule", {})
                design_module = protocol.get("designModule", {})
                nct_id = id_module.get("nctId", "")
                trials.append({
                    "title": id_module.get("briefTitle", "Unknown")[:200],
                    "status": status_module.get("overallStatus", "Unknown"),
                    "phase": str(design_module.get("phases", ["Unknown"])),
                    "url": f"https://clinicaltrials.gov/study/{nct_id}"
                })
            return trials if trials else [{"title": "No trials found", "status": "N/A", "phase": "N/A", "url": ""}]
    except Exception as e:
        return [{"title": f"Error: {str(e)}", "status": "error", "phase": "N/A", "url": ""}]

# All MCP tools list
MCP_TOOLS = [
    search_biology_knowledge,
    search_disease_knowledge,
    search_medicine_knowledge,
    search_hospital_knowledge,
    search_nutrition_knowledge,
    search_fda_drug,
    search_clinical_trials
]

def get_mcp_schemas():
    """Returns all MCP tool schemas — useful for documentation and API."""
    return MCP_TOOL_SCHEMAS

def get_mcp_tools():
    """Returns all MCP tool implementations."""
    return MCP_TOOLS