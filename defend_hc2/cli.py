"""Small helper CLIs."""

from __future__ import annotations

from defend_hc2.session_chain import new_master_secret


def keygen() -> None:
    """Print a fresh master secret for DEFEND_HC2_MASTER_SECRET."""
    print(new_master_secret())


if __name__ == "__main__":
    keygen()
