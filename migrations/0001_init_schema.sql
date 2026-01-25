-- AI Worker Database Schema
-- Stores training data, model usage, and fine-tune management

-- ============================================================================
-- TRAINING DATA COLLECTION
-- ============================================================================

-- Exercise generation training data (from edu.xaostech.io)
CREATE TABLE IF NOT EXISTS training_exercises (
  id TEXT PRIMARY KEY,
  subject TEXT NOT NULL,
  category TEXT,
  difficulty TEXT NOT NULL,
  exercise_type TEXT NOT NULL,
  topic TEXT NOT NULL,
  -- Input/Output pairs for training
  prompt_input TEXT NOT NULL,        -- The generation prompt
  model_output TEXT NOT NULL,        -- Raw model response
  parsed_exercise TEXT,              -- Parsed exercise JSON
  -- Quality signals
  user_submission_count INTEGER DEFAULT 0,
  correct_rate REAL DEFAULT 0,       -- % of correct submissions
  avg_time_seconds REAL,             -- Average completion time
  difficulty_actual REAL,            -- Calculated actual difficulty
  quality_score REAL,                -- Composite quality metric
  -- Metadata
  source_model TEXT NOT NULL,
  generation_time_ms INTEGER,
  neurons_used INTEGER,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  reviewed_at DATETIME,
  approved_for_training INTEGER DEFAULT 0
);

-- Translation training data (from lingua.xaostech.io)
CREATE TABLE IF NOT EXISTS training_translations (
  id TEXT PRIMARY KEY,
  source_lang TEXT NOT NULL,
  target_lang TEXT NOT NULL,
  source_text TEXT NOT NULL,
  -- Model outputs
  model_translation TEXT NOT NULL,
  model_used TEXT NOT NULL,
  -- Quality signals
  user_edited INTEGER DEFAULT 0,     -- Was it corrected by user?
  edited_translation TEXT,           -- User's correction if any
  rating INTEGER,                    -- User rating 1-5
  -- Metadata
  context TEXT,                      -- Sentence context if available
  domain TEXT,                       -- medical, legal, casual, etc.
  neurons_used INTEGER,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  approved_for_training INTEGER DEFAULT 0
);

-- Etymology/definition training data
CREATE TABLE IF NOT EXISTS training_definitions (
  id TEXT PRIMARY KEY,
  word TEXT NOT NULL,
  language TEXT NOT NULL,
  -- Model outputs
  model_definition TEXT NOT NULL,
  model_etymology TEXT,
  model_used TEXT NOT NULL,
  -- Quality signals (compared to dictionary)
  dictionary_match_score REAL,       -- How close to dictionary definition
  user_rating INTEGER,
  user_correction TEXT,
  -- Metadata
  neurons_used INTEGER,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  approved_for_training INTEGER DEFAULT 0
);

-- ============================================================================
-- MODEL USAGE & BUDGET TRACKING
-- ============================================================================

-- Daily neuron usage tracking
CREATE TABLE IF NOT EXISTS neuron_usage (
  date TEXT NOT NULL,                -- YYYY-MM-DD
  hour INTEGER NOT NULL,             -- 0-23
  model TEXT NOT NULL,
  task_type TEXT NOT NULL,           -- generation, translation, embedding, etc.
  request_count INTEGER DEFAULT 0,
  neurons_used INTEGER DEFAULT 0,
  avg_latency_ms REAL,
  error_count INTEGER DEFAULT 0,
  PRIMARY KEY (date, hour, model, task_type)
);

-- LoRA fine-tune registry
CREATE TABLE IF NOT EXISTS finetunes (
  id TEXT PRIMARY KEY,               -- CF finetune ID
  name TEXT NOT NULL UNIQUE,
  base_model TEXT NOT NULL,
  description TEXT,
  -- Training info
  training_data_count INTEGER,
  training_started_at DATETIME,
  training_completed_at DATETIME,
  -- Performance metrics
  eval_score REAL,
  avg_latency_ms REAL,
  neurons_per_request REAL,
  -- Status
  status TEXT DEFAULT 'pending',     -- pending, training, ready, deprecated
  active INTEGER DEFAULT 0,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Training data indexes
CREATE INDEX idx_training_exercises_subject ON training_exercises(subject, approved_for_training);
CREATE INDEX idx_training_exercises_quality ON training_exercises(quality_score DESC) WHERE approved_for_training = 1;
CREATE INDEX idx_training_translations_langs ON training_translations(source_lang, target_lang, approved_for_training);
CREATE INDEX idx_training_definitions_word ON training_definitions(word, language);

-- Usage indexes
CREATE INDEX idx_neuron_usage_date ON neuron_usage(date DESC);
CREATE INDEX idx_neuron_usage_model ON neuron_usage(model, date);

-- Finetune indexes
CREATE INDEX idx_finetunes_status ON finetunes(status, active);
CREATE INDEX idx_finetunes_model ON finetunes(base_model);
