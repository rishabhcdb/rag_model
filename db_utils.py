import sqlite3

def log_interaction(question, answer, doc_name):
    conn = sqlite3.connect("rag_logs.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO rag_logs (question, answer, created_at, doc_name) VALUES (?, ?, datetime('now'), ?)",
        (question, answer, doc_name)
    )
    conn.commit()
    conn.close()

