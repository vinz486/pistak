import sys

# PyInstaller / Multiprocessing support wrapper
if "--run-server" in sys.argv:
    import server
    sys.argv.remove("--run-server")
    server.main()
    sys.exit(0)

from pistak.app import PistakApp

def main():
    app = PistakApp()
    app.run()

if __name__ == "__main__":
    main()
