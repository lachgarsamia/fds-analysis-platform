"""Navigable pages for FireLab's nav-rail shell (FireLab roadmap Phase 1).

Each page is a plain QWidget (see pages/base.py's Page) added to
MainWindow's QStackedWidget. LivePage hosts the app's existing, already-
built content unchanged; the rest are lazy-built placeholders reserving
their nav-rail slot until Phase 4 gives them real content.
"""
