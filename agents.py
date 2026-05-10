# agents.py

import difflib
from vector_store import chroma_search

ignore_words = ["what", "is", "tell", "me", "about", "do", "the", "a", "an"]

known_keywords = [
    "diabetes", "hypertension", "fever", "disease", "blood sugar",
    "corona", "coronavirus", "covid", "covid-19", "virus", "infection",
    "insulin", "aspirin", "antibiotic", "antibiotics", "vaccine", "vaccines", "medicine",
    "doctor", "doctors", "nurse", "nurses", "hospital", "hospitals", "clinic",
    "pharmacist", "dna", "cell", "protein", "photosynthesis", "gene", "genes",
    "crispr", "mutation", "mutations", "rna", "mitochondria", "enzyme", "enzymes",
    "nutrition", "carbohydrates", "vitamins", "minerals", "fat", "diet"
]

typo_map = {
    "diabitics": "diabetes",
    "diabitic": "diabetes",
    "insuline": "insulin",
    "nirse": "nurse",
    "hospitel": "hospital",
    "docter": "doctor",
    "coronaa": "corona",
    "covidd": "covid",
    "viruse": "virus"
}

def fix_typos(words):
    fixed_words = []
    for word in words:
        if word in typo_map:
            fixed_words.append(typo_map[word])
        else:
            match = difflib.get_close_matches(word, known_keywords, n=1, cutoff=0.6)
            if match:
                fixed_words.append(match[0])
            else:
                fixed_words.append(word)
    return fixed_words

def build_response(agent_name, question, results, status):
    strong_results = [r for r in results if r["score"] > 0.3]
    if strong_results:
        final_answer = " ".join([
            f"{i+1}. {r['answer']}" for i, r in enumerate(strong_results)
        ])
    else:
        final_answer = "No strong supporting result found in the knowledge base."
    return {
        "agent": agent_name,
        "question": question,
        "final_answer": final_answer,
        "results": results,
        "status": status
    }

def route_question(question):
    words = question.lower().split()
    words = [w for w in words if w not in ignore_words]
    words = fix_typos(words)
    fixed_question = " ".join(words)

    disease_keywords = [
        "diabetes", "hypertension", "fever", "disease", "blood sugar",
        "corona", "coronavirus", "covid", "covid-19", "virus", "infection",
        "cancer", "tumor", "tumour", "chemotherapy", "metastasis",
        "asthma", "alzheimer"
    ]

    medicine_keywords = [
        "insulin", "aspirin", "antibiotic", "antibiotics",
        "vaccine", "vaccines", "medicine", "paracetamol", "ibuprofen"
    ]

    hospital_keywords = [
        "doctor", "doctors", "nurse", "nurses",
        "hospital", "hospitals", "clinic",
        "pharmacist", "pharmacists", "icu", "surgeon", "radiologist"
    ]

    # Biology checked BEFORE disease to avoid wrong routing
    biology_keywords = [
        "dna", "cell", "cells", "protein", "photosynthesis",
        "gene", "genes", "crispr", "mutation", "mutations",
        "rna", "mitochondria", "enzyme", "enzymes", "ribosome",
        "mitosis", "meiosis", "genome", "chromosome", "helix"
    ]

    nutrition_keywords = [
        "nutrition", "carbohydrates", "vitamins", "minerals",
        "fat", "diet", "fiber", "calories", "omega"
    ]

    # Route to correct agent and search Chroma
    if any(word in fixed_question for word in biology_keywords):
        results = chroma_search(fixed_question, "biology_agent")
        return build_response("biology_agent", question, results, "success")

    elif any(word in fixed_question for word in disease_keywords):
        results = chroma_search(fixed_question, "disease_agent")
        return build_response("disease_agent", question, results, "success")

    elif any(word in fixed_question for word in medicine_keywords):
        results = chroma_search(fixed_question, "medicine_agent")
        return build_response("medicine_agent", question, results, "success")

    elif any(word in fixed_question for word in hospital_keywords):
        results = chroma_search(fixed_question, "hospital_agent")
        return build_response("hospital_agent", question, results, "success")

    elif any(word in fixed_question for word in nutrition_keywords):
        results = chroma_search(fixed_question, "nutrition_agent")
        return build_response("nutrition_agent", question, results, "success")

    return {
        "agent": "unknown",
        "question": question,
        "final_answer": "I don't know which agent should handle this question.",
        "results": [],
        "status": "no_match"
    }