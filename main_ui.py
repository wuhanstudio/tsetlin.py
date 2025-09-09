
import os, sys
import streamlit.web.cli as stcli

def resolve_path(path):
    resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolved_path

if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("app.py"),
        "--server.port=8080",
        "--global.developmentMode=false",
        "--server.headless=true",
        "--",
        "--no-auth"
        # "--server.address=0.0.0.0",
    ]
    sys.exit(stcli.main())