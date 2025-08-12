#!/usr/bin/env python3
import os
import sys


def main():
    # Avoid interfering with modules that parse CLI on import
    sys.argv = [sys.argv[0]]

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
