#!/usr/bin/env python3
import os
import sys
import logging


def main():
    # Avoid interfering with modules that parse CLI on import
    sys.argv = [sys.argv[0]]

    # Configure basic console logging
    logging.basicConfig(level=os.getenv("PY_LOG_LEVEL", "INFO"))
    log = logging.getLogger("rvc.api.bootstrap")

    # Prefer python-dotenv if available; otherwise skip silently
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        load_dotenv = None  # type: ignore

    if load_dotenv:
        for candidate in ("/app/.env", ".env"):
            try:
                load_dotenv(dotenv_path=candidate, override=False)
                log.info("Loaded dotenv: %s", candidate)
            except Exception:
                logging.exception("Failed to load dotenv: %s", candidate)

    # Provide sane defaults if not set, and paths exist
    defaults = {
        "weight_root": "/app/assets/weights",
        "index_root": "/app/assets/indices",
        "outside_index_root": "/app/assets/indices",
    }
    for key, path in defaults.items():
        if not os.getenv(key) and os.path.isdir(path):
            os.environ[key] = path
            log.info("Set default %s=%s", key, path)

    # Now import uvicorn and run the FastAPI app
    import uvicorn

    port_str = os.getenv("RVC_API_PORT", os.getenv("PORT", "7866"))
    try:
        port = int(port_str)
    except Exception:
        port = 7866

    log_level = os.getenv("UVICORN_LOG_LEVEL", "info")

    uvicorn.run(
        "rvc_server.app:app",
        host="0.0.0.0",
        port=port,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
