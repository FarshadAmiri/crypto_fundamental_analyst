"""Manual entry point inside the container: python -m app.cli run  (triggers via the API)."""
import sys

import httpx

API_URL = "http://localhost:8000"


def main():
    if len(sys.argv) < 2 or sys.argv[1] != "run":
        print("usage: python -m app.cli run")
        sys.exit(1)
    resp = httpx.post(f"{API_URL}/api/run", timeout=10)
    print(resp.json() if resp.is_success else f"error: {resp.status_code} {resp.text}")


if __name__ == "__main__":
    main()
