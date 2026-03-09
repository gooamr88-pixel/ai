import logging
from supabase import create_client, Client
from app.core.config import settings

logger = logging.getLogger(__name__)

supabase: Client | None = None

if settings.SUPABASE_URL and settings.SUPABASE_KEY:
    try:
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        logger.info("[INIT] ✓ Supabase client initialized")
    except Exception as e:
        logger.error(f"[INIT] ✗ Supabase initialization failed: {e}")
else:
    logger.warning("[INIT] ✗ Supabase credentials missing (SUPABASE_URL and/or SUPABASE_KEY)")
