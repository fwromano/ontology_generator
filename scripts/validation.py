import rdflib

def clean_and_validate_ttl(raw_ttl: str) -> str:
    lines = raw_ttl.splitlines()
    if len(lines) <= 1:
        return ""
    return "\n".join(lines[1:-1])
