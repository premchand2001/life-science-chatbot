# MCP and Weaviate Integration Notes

## Weaviate Setup
- Cloud cluster on us-west3.gcp.weaviate.cloud
- Collections: DiseaseAgent, MedicineAgent, HospitalAgent, BiologyAgent, NutritionAgent
- Embeddings: HuggingFace all-MiniLM-L6-v2 (local), OpenAI (cloud)

## MCP Tools
- search_biology_knowledge
- search_disease_knowledge
- search_medicine_knowledge
- search_hospital_knowledge
- search_nutrition_knowledge
- search_fda_drug (OpenFDA API)
- search_clinical_trials (ClinicalTrials.gov API)

## API Endpoints
- GET /mcp/schemas
- GET /mcp/fda/{drug_name}
- GET /mcp/trials/{condition}