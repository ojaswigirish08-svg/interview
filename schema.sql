-- VLSI Interview Platform — PostgreSQL Schema

-- Candidate profile (unique by name+email)
CREATE TABLE IF NOT EXISTS candidates (
    id              SERIAL PRIMARY KEY,
    candidate_key   TEXT UNIQUE NOT NULL,
    candidate_name  TEXT NOT NULL DEFAULT 'Candidate',
    email           TEXT NOT NULL DEFAULT '',
    domain          TEXT NOT NULL DEFAULT 'physical_design',
    level           TEXT NOT NULL DEFAULT 'unknown',
    education       TEXT NOT NULL DEFAULT '',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_candidates_key ON candidates(candidate_key);

-- Interview sessions
CREATE TABLE IF NOT EXISTS interview_sessions (
    id                  SERIAL PRIMARY KEY,
    session_id          TEXT UNIQUE NOT NULL,
    candidate_id        INTEGER NOT NULL REFERENCES candidates(id),
    mode                TEXT NOT NULL DEFAULT 'mock',
    difficulty_level    INTEGER NOT NULL DEFAULT 1,
    turns_completed     INTEGER NOT NULL DEFAULT 0,
    overall_score       INTEGER,
    grade               TEXT,
    warmup_performance  TEXT DEFAULT 'pending',
    early_end_reason    TEXT,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at            TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_sessions_candidate ON interview_sessions(candidate_id);

-- Per-topic performance within a session
CREATE TABLE IF NOT EXISTS topic_performance (
    id              SERIAL PRIMARY KEY,
    session_id      INTEGER NOT NULL REFERENCES interview_sessions(id),
    topic           TEXT NOT NULL,
    avg_score       REAL NOT NULL DEFAULT 0,
    questions_asked INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_topic_perf_session ON topic_performance(session_id);

-- Weak and strong topics per session
CREATE TABLE IF NOT EXISTS session_topics (
    id              SERIAL PRIMARY KEY,
    session_id      INTEGER NOT NULL REFERENCES interview_sessions(id),
    topic           TEXT NOT NULL,
    category        TEXT NOT NULL CHECK (category IN ('weak', 'strong'))
);

CREATE INDEX IF NOT EXISTS idx_session_topics_session ON session_topics(session_id);

-- Warmup results per session
CREATE TABLE IF NOT EXISTS warmup_results (
    id              SERIAL PRIMARY KEY,
    session_id      INTEGER NOT NULL REFERENCES interview_sessions(id),
    warmup_turns    INTEGER NOT NULL DEFAULT 0,
    performance     TEXT NOT NULL DEFAULT 'pending',
    skills_asked    TEXT[]
);
