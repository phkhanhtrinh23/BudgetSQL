from __future__ import annotations

SYSTEM = ("You are an expert SQLite analyst. Given a database schema, a question "
          "and optional evidence, write one SQLite query that answers the "
          "question. Return only the query in a ```sql block.")

REPAIR_SYSTEM = ("You are an expert SQLite analyst. A query failed or returned "
                 "nothing. Given the schema, the question, the query and what it "
                 "returned, write one corrected SQLite query. Return only the "
                 "query in a ```sql block.")

VERIFY_SYSTEM = ("You check SQL answers. Given a question, a query and the first "
                 "rows of its result, answer with one word: ACCEPT if the result "
                 "answers the question, DOUBT otherwise.")


def generate(schema_text: str, question: str, evidence: str = "",
             sample_rows: int = 0) -> list[dict]:
    user = [f"Schema:\n{schema_text}", f"Question: {question}"]
    if evidence:
        user.append(f"Evidence: {evidence}")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n\n".join(user)}]


def repair(schema_text: str, question: str, sql: str, outcome: str,
           evidence: str = "") -> list[dict]:
    user = [f"Schema:\n{schema_text}", f"Question: {question}"]
    if evidence:
        user.append(f"Evidence: {evidence}")
    user += [f"Query:\n{sql}", f"Result or error:\n{outcome}"]
    return [{"role": "system", "content": REPAIR_SYSTEM},
            {"role": "user", "content": "\n\n".join(user)}]


def verify(question: str, sql: str, preview: str) -> list[dict]:
    return [{"role": "system", "content": VERIFY_SYSTEM},
            {"role": "user", "content": f"Question: {question}\n\nQuery:\n{sql}\n\n"
                                        f"First rows: {preview}"}]
