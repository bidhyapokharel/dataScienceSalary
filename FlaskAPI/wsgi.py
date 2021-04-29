from app import app as application

try:
    from app import app as application
except ImportError:
    from __main__ import app

if __name__ == "__main__":
    application.run()