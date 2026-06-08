#!/usr/bin/env python3
"""Check DashScope / Qwen API configuration (does not call test data)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    key = os.getenv("DASHSCOPE_API_KEY", "")
    print("=== DashScope API config check ===")
    print(f".env file: {'found' if (PROJECT_ROOT / '.env').exists() else 'MISSING'}")
    print(f"DASHSCOPE_API_KEY: {'set (' + str(len(key)) + ' chars)' if key else 'NOT SET'}")

    try:
        import dashscope

        print(f"dashscope package: installed")
    except ImportError:
        print("dashscope package: NOT INSTALLED")
        print("Run: conda activate humomni && pip install dashscope")
        raise SystemExit(1)

    if not key:
        print("\nTo configure:")
        print("  cp .env.example .env")
        print("  # edit .env and set DASHSCOPE_API_KEY=sk-...")
        raise SystemExit(1)

    # Minimal live ping (no test data involved)
    try:
        from dashscope import Generation

        dashscope.api_key = key
        resp = Generation.call(
            model="qwen-turbo",
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            result_format="message",
            temperature=0,
        )
        if getattr(resp, "status_code", None) == 200:
            content = resp.output.choices[0].message.content
            print(f"\nAPI ping: SUCCESS")
            print(f"Response: {content[:80]}")
        else:
            print(f"\nAPI ping: FAILED")
            print(f"Message: {getattr(resp, 'message', resp)}")
            raise SystemExit(1)
    except Exception as exc:
        print(f"\nAPI ping: FAILED ({exc})")
        raise SystemExit(1)

    print("\nIMPORTANT (competition rule):")
    print("  Qwen API may be used for TRAINING only.")
    print("  Do NOT use API on test-set audio/text for inference or submission.")


if __name__ == "__main__":
    main()
