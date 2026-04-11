with open("data.txt", "r") as file:
    lines = file.readlines()

ignore_words = ["what", "is", "tell", "me", "about", "do"]

while True:
    question = input("Ask me something (type 'exit' to stop): ").lower()

    if question == "exit":
        print("Goodbye!")
        break

    filtered_words = [word for word in question.split() if word not in ignore_words]

    best_line = ""
    best_score = 0

    for line in lines:
        line_lower = line.lower()
        score = 0

        for word in filtered_words:
            if word in line_lower:
                score += 1

        if score > best_score:
            best_score = score
            best_line = line.strip()

    if best_score > 0:
        print(best_line)
    else:
        print("I don't know based on the data.")