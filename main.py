"""Launcher for the NBA odds predictor web app."""

from app import create_app


if __name__ == "__main__":
    create_app().run(debug=True)
