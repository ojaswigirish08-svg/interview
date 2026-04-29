"""
VLSI Interview Platform — PostgreSQL Database Module
"""
import os
import time
from contextlib import contextmanager

DATABASE_URL = os.getenv("DATABASE_URL", "")

_pool = None
_db_available = False


def init_db():
    """Initialize connection pool and create tables. Called once at startup."""
    global _pool, _db_available
    if not DATABASE_URL:
        print("[DB] DATABASE_URL not set — running without database (in-memory only)")
        return

    try:
        import psycopg2
        from psycopg2 import pool
        _pool = pool.SimpleConnectionPool(1, 10, DATABASE_URL)

        # Create tables from schema.sql
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        if os.path.exists(schema_path):
            with get_conn() as conn:
                with conn.cursor() as cur:
                    with open(schema_path, "r") as f:
                        cur.execute(f.read())
                conn.commit()

        _db_available = True
        print(f"[DB] PostgreSQL connected and tables ready")
    except ImportError:
        print("[DB] psycopg2 not installed — pip install psycopg2-binary")
    except Exception as e:
        print(f"[DB] PostgreSQL connection failed: {e}")
        print("[DB] Running without database (in-memory only)")


def is_available():
    return _db_available


@contextmanager
def get_conn():
    conn = _pool.getconn()
    try:
        yield conn
    finally:
        _pool.putconn(conn)


def get_or_create_candidate(candidate_key, candidate_name, email, domain, level, education):
    """Returns candidate DB id. Creates if not exists, updates if exists."""
    if not _db_available:
        return None
    try:
        from psycopg2.extras import RealDictCursor
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    INSERT INTO candidates (candidate_key, candidate_name, email, domain, level, education)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (candidate_key) DO UPDATE SET
                        candidate_name = EXCLUDED.candidate_name,
                        domain = EXCLUDED.domain,
                        level = EXCLUDED.level,
                        education = EXCLUDED.education,
                        updated_at = NOW()
                    RETURNING id
                """, (candidate_key, candidate_name, email, domain, level, education))
                conn.commit()
                return cur.fetchone()["id"]
    except Exception as e:
        print(f"[DB] get_or_create_candidate failed: {e}")
        return None


def get_candidate_sessions(candidate_key):
    """Returns list of past session summaries for a candidate."""
    if not _db_available:
        return []
    try:
        from psycopg2.extras import RealDictCursor
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get all sessions for this candidate
                cur.execute("""
                    SELECT s.id as db_id, s.session_id, s.overall_score, s.difficulty_level,
                           s.turns_completed, s.started_at, s.grade, s.mode,
                           s.early_end_reason, s.warmup_performance
                    FROM interview_sessions s
                    JOIN candidates c ON s.candidate_id = c.id
                    WHERE c.candidate_key = %s
                    ORDER BY s.started_at ASC
                """, (candidate_key,))
                sessions = cur.fetchall()

                result = []
                for sess in sessions:
                    db_id = sess["db_id"]

                    # Get topic performance
                    cur.execute("""
                        SELECT topic, avg_score, questions_asked
                        FROM topic_performance WHERE session_id = %s
                    """, (db_id,))
                    tp = {}
                    for row in cur.fetchall():
                        tp[row["topic"]] = {"avg_score": row["avg_score"], "count": row["questions_asked"]}

                    # Get weak/strong topics
                    cur.execute("""
                        SELECT topic, category FROM session_topics WHERE session_id = %s
                    """, (db_id,))
                    weak, strong = [], []
                    for row in cur.fetchall():
                        if row["category"] == "weak":
                            weak.append(row["topic"])
                        else:
                            strong.append(row["topic"])

                    result.append({
                        "session_id": sess["session_id"],
                        "date": sess["started_at"].strftime("%Y-%m-%d %H:%M") if sess["started_at"] else "",
                        "overall_score": sess["overall_score"] or 0,
                        "difficulty_level": sess["difficulty_level"] or 1,
                        "topic_performance": tp,
                        "weak_topics": weak,
                        "strong_topics": strong,
                        "turns_completed": sess["turns_completed"] or 0,
                        "warmup_performance": sess["warmup_performance"] or "pending",
                    })

                return result
    except Exception as e:
        print(f"[DB] get_candidate_sessions failed: {e}")
        return []


def get_candidate_session_count(candidate_key):
    """Fast check: how many completed sessions does this candidate have?"""
    if not _db_available:
        return 0
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) FROM interview_sessions s
                    JOIN candidates c ON s.candidate_id = c.id
                    WHERE c.candidate_key = %s
                """, (candidate_key,))
                return cur.fetchone()[0]
    except Exception as e:
        print(f"[DB] get_candidate_session_count failed: {e}")
        return 0


def save_session(session_id, candidate_id, mode, difficulty_level, turns_completed,
                 overall_score, grade, warmup_performance, early_end_reason,
                 started_at, topic_performance, weak_topics, strong_topics,
                 warmup_turns=0, warmup_perf_str="pending", skills_asked=None):
    """Save complete session data to DB. Idempotent (safe to call multiple times)."""
    if not _db_available or not candidate_id:
        return
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                # Upsert session
                cur.execute("""
                    INSERT INTO interview_sessions
                        (session_id, candidate_id, mode, difficulty_level, turns_completed,
                         overall_score, grade, warmup_performance, early_end_reason, started_at, ended_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, to_timestamp(%s), NOW())
                    ON CONFLICT (session_id) DO UPDATE SET
                        difficulty_level = EXCLUDED.difficulty_level,
                        turns_completed = EXCLUDED.turns_completed,
                        overall_score = EXCLUDED.overall_score,
                        grade = EXCLUDED.grade,
                        warmup_performance = EXCLUDED.warmup_performance,
                        early_end_reason = EXCLUDED.early_end_reason,
                        ended_at = NOW()
                    RETURNING id
                """, (session_id, candidate_id, mode, difficulty_level, turns_completed,
                      overall_score, grade, warmup_performance, early_end_reason, started_at))
                db_id = cur.fetchone()[0]

                # Replace topic performance
                cur.execute("DELETE FROM topic_performance WHERE session_id = %s", (db_id,))
                for topic, data in (topic_performance or {}).items():
                    cur.execute("""
                        INSERT INTO topic_performance (session_id, topic, avg_score, questions_asked)
                        VALUES (%s, %s, %s, %s)
                    """, (db_id, topic, data.get("avg_score", 0), data.get("count", 0)))

                # Replace weak/strong topics
                cur.execute("DELETE FROM session_topics WHERE session_id = %s", (db_id,))
                for t in (weak_topics or []):
                    cur.execute("INSERT INTO session_topics (session_id, topic, category) VALUES (%s, %s, 'weak')", (db_id, t))
                for t in (strong_topics or []):
                    cur.execute("INSERT INTO session_topics (session_id, topic, category) VALUES (%s, %s, 'strong')", (db_id, t))

                # Warmup results
                cur.execute("DELETE FROM warmup_results WHERE session_id = %s", (db_id,))
                cur.execute("""
                    INSERT INTO warmup_results (session_id, warmup_turns, performance, skills_asked)
                    VALUES (%s, %s, %s, %s)
                """, (db_id, warmup_turns, warmup_perf_str, skills_asked or []))

                conn.commit()
                print(f"[DB] Session {session_id[:8]} saved: score={overall_score}, turns={turns_completed}")
    except Exception as e:
        print(f"[DB] save_session failed: {e}")
