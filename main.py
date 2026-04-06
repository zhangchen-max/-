import asyncio
from dotenv import load_dotenv

load_dotenv()

from src.psy_nav.loop import chat_loop


def run():
    asyncio.run(chat_loop())


if __name__ == "__main__":
    run()
