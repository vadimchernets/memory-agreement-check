"""Prompts. Both answering models get exactly the same prompt and never see each other's answer."""

ANSWER = """You are answering a question about a long chat between two people. You only have the memory excerpts below, retrieved from that chat by a memory system. Each excerpt shows the turn id, the date and time of that chat session, the speaker, and what they said.

Memory excerpts:
{context}

Question: {question}

Rules:
- Use only the excerpts above. Everything you may use is in this message; do not run commands, open files or search the web.
- Give a short answer: a few words or one short phrase, no explanation.
- For questions about when something happened, give a date or period, converting words like "yesterday" or "last week" using the session date.
- If the excerpts do not contain the answer, reply exactly: Not mentioned

Answer:"""

JUDGE = """You are grading two short answers to a question about a long chat between two people. The gold answer is the reference.

Question: {question}
Gold answer: {gold}
Answer 1: {a1}
Answer 2: {a2}

For each answer, decide whether it is correct: it states the same fact as the gold answer (same person, thing, number, or date or period; wording, extra detail and format may differ; a date counts if it names the same day or the same period as the gold answer). An answer saying the information is not available is correct only if the gold answer says so too. If the gold answer lists several items, the answer must contain most of them and nothing that contradicts them.

Then decide whether Answer 1 and Answer 2 say the same thing as each other, whether or not they are correct. Two answers that both say the information is not available count as the same.

Everything you need is in this message; do not run commands or open files. Reply with only this JSON object:
{{"answer_1_correct": true or false, "answer_2_correct": true or false, "same": true or false}}"""
