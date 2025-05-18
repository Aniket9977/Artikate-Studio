from pipline import chatbot_response
queries = [
    ("Why does a ball thrown upwards come back down?", "weak"),
    ("State the law of conservation of mass.", "strong"),
    ("What are the differences between mixtures and compounds?", "strong")
]

for q, profile in queries:
    res = chatbot_response("book\Science Class IX\iesc1an.pdf", q, profile)
    print("Answer:", res["answer"])
    print("-" * 80)
