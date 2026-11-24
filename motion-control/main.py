#!/usr/bin/env python3
"""Motion Control — point d'entrée du programme.

Usage : python main.py
"""
import sys


def main() -> int:
    try:
        from interface.main_window import lancer_application
    except ImportError as exc:
        print("Dépendances manquantes :", exc)
        print("Installez-les avec :  pip install -r requirements.txt")
        return 1
    lancer_application()
    return 0


if __name__ == "__main__":
    sys.exit(main())
