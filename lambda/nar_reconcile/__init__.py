"""Reconcile unlinked BIBFRAME contributors to LC Name Authority File URIs."""

from .pipeline import reconcile_lccn

__all__ = ["reconcile_lccn"]
