import os

from dotenv import load_dotenv

load_dotenv()


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    print(f"Google API Key: {api_key}")


if __name__ == "__main__":
    main()
