-- Supabase Database Schema for Ruya Platform
-- Tables: Users (optional, for future auth), GeneratedQuizzes, GeneratedPodcasts, GeneratedMindMaps

-- 1. Users table (Can map to Supabase Auth auth.users, or act stand-alone)
CREATE TABLE IF NOT EXISTS public.users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email TEXT UNIQUE NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Quizzes / Question Banks
CREATE TABLE IF NOT EXISTS public.generated_quizzes (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  difficulty TEXT NOT NULL,
  num_questions INTEGER NOT NULL,
  quiz_data JSONB NOT NULL,
  type TEXT DEFAULT 'quiz', -- 'quiz' or 'question-bank'
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Podcasts
CREATE TABLE IF NOT EXISTS public.generated_podcasts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
  title TEXT NOT NULL,
  description TEXT,
  total_duration_seconds NUMERIC,
  podcast_data JSONB NOT NULL, -- contains speakers, turns with audio URLs
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. Mind Maps
CREATE TABLE IF NOT EXISTS public.generated_mindmaps (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID REFERENCES public.users(id) ON DELETE CASCADE,
  mindmap_data JSONB NOT NULL, -- the root_node and children structure
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- RLS Policies (Row Level Security) - Optional if you want to secure routes based on auth.users
-- ALTER TABLE public.generated_quizzes ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE public.generated_podcasts ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE public.generated_mindmaps ENABLE ROW LEVEL SECURITY;
