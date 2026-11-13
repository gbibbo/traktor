-- Tabla de trabajos del Worker traktor-concept-ai (base D1 traktor-concepts). Timestamps en milisegundos Unix.
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  claimed_at INTEGER,
  attempts INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL CHECK (status IN ('queued', 'processing', 'done', 'failed')),
  concept_json TEXT NOT NULL,
  result_json TEXT,
  error TEXT
);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, created_at);
-- v6: lo que el ordenador cuenta de paso al pedir trabajo (claves "sonando" y "catalogo"), para la pantalla del telefono.
CREATE TABLE IF NOT EXISTS estado (
  clave TEXT PRIMARY KEY,
  valor TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);
