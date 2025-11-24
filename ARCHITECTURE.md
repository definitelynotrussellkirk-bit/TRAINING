# System Architecture

**Last Updated:** 2025-11-24 (Option C Architecture)

See full architecture document for complete details.

## Quick Reference

### System Split
- **4090 (192.168.x.x):** Training, evaluation, comparison, deployment
- **3090 (192.168.x.x):** Inference server only (port 8765)

### Key Files
**4090:**
- `monitoring/deployment_orchestrator.py` - Auto-deployment
- `monitoring/model_comparison_engine.py` - Checkpoint evaluation  
- `monitoring/prediction_client.py` - API client

**3090:**
- `/home/user/llm/main.py` - FastAPI server
- `/path/to/models/deployed/` - Active checkpoint

### Deployment Flow
```
Training → Checkpoint → Comparison → Orchestrator → rsync → 3090 reload → Serving
  (4090)     (4090)       (4090)        (4090)      (8s)      (2s)      (3090)
```

**Total time:** < 15 minutes from checkpoint creation to serving

See `OPTION_C_MIGRATION_STATUS.md` for complete documentation.
