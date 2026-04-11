import string
import difflib

ignore_words = ["what", "is", "tell", "me", "about", "do", "the", "a", "an"]

known_keywords = [
    "diabetes", "hypertension", "fever", "disease", "blood sugar",
    "insulin", "aspirin", "antibiotic", "antibiotics", "vaccine", "vaccines", "medicine",
    "doctor", "doctors", "nurse", "nurses", "hospital", "hospitals", "clinic", "pharmacist", "pharmacists"
]

typo_map = {
    "diabitics": "diabetes",
    "diabitic": "diabetes",
    "insuline": "insulin",
    "nirse": "nurse",
    "hospitel": "hospital",
    "docter": "doctor"
}

def clean_text(text):
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))

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

def search_file(filename, question):
    with open(filename, "r") as file:
        lines = file.readlines()

    question = clean_text(question)
    filtered_words = [word for word in question.split() if word not in ignore_words]
    filtered_words = fix_typos(filtered_words)

    scored_lines = []

    for line in lines:
        line_clean = clean_text(line)
        score = 0

        for word in filtered_words:
            if word in line_clean:
                score += 1

        if score > 0:
            scored_lines.append({
                "answer": line.strip(),
                "score": score,
                "source": filename
            })

    scored_lines.sort(key=lambda x: x["score"], reverse=True)
    return scored_lines[:2]

def disease_agent(question):
    return search_file("disease_data.txt", question)

def medicine_agent(question):
    return search_file("medicine_data.txt", question)

def hospital_agent(question):
    return search_file("hospital_data.txt", question)

def fallback_search(question):
    all_results = []
    all_results.extend(search_file("disease_data.txt", question))
    all_results.extend(search_file("medicine_data.txt", question))
    all_results.extend(search_file("hospital_data.txt", question))

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:2]

def build_response(agent_name, question, results, status):
    final_answer = results[0]["answer"] if results else "No answer found."
    return {
        "agent": agent_name,
        "question": question,
        "final_answer": final_answer,
        "results": results,
        "status": status
    }

def route_question(question):
    question_lower = clean_text(question)
    words = [word for word in question_lower.split() if word not in ignore_words]
    words = fix_typos(words)
    fixed_question = " ".join(words)

    if any(word in fixed_question for word in ["diabetes", "hypertension", "fever", "disease", "blood sugar"]):
        results = disease_agent(fixed_question)
        if results:
            return build_response("disease_agent", question, results, "success")

    elif any(word in fixed_question for word in ["insulin", "aspirin", "antibiotic", "antibiotics", "vaccine", "vaccines", "medicine"]):
        results = medicine_agent(fixed_question)
        if results:
            return build_response("medicine_agent", question, results, "success")

    elif any(word in fixed_question for word in ["doctor", "doctors", "nurse", "nurses", "hospital", "hospitals", "clinic", "pharmacist", "pharmacists"]):
        results = hospital_agent(fixed_question)
        if results:
            return build_response("hospital_agent", question, results, "success")

    fallback_results = fallback_search(fixed_question)

    if fallback_results:
        return build_response("fallback_search", question, fallback_results, "fallback_success")

    return {
        "agent": "unknown",
        "question": question,
        "final_answer": "I don't know which agent should handle this question.",
        "results": [],
        "status": "no_match"
    }