import string

with open("data.txt", "r") as file:
    lines = file.readlines()

ignore_words = ["what", "is", "tell", "me", "about", "do", "the", "a", "an"]

def clean_text(text):
    text = text.lower()
    return text.translate(str.maketrans("", "", string.punctuation))

def get_answer(question):
    question = clean_text(question)
    filtered_words = [word for word in question.split() if word not in ignore_words]

    scored_lines = []

    for line in lines:
        line_clean = clean_text(line)
        score = 0

        for word in filtered_words:
            if word in line_clean:
                score += 1

        if score > 0:
            scored_lines.append((line.strip(), score))

    scored_lines.sort(key=lambda x: x[1], reverse=True)

    if len(scored_lines) > 0:
        top_results = scored_lines[:2]
        return {
            "results": [
                {"answer": line, "score": score}
                for line, score in top_results
            ]
        }
    else:
        return {
            "results": [
                {"answer": "I don't know based on the data.", "score": 0}
            ]
        }