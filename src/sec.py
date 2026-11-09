from dotenv import load_dotenv

load_dotenv()  # must run before `import edgar`: edgartools reads its settings at import time
import edgar  # noqa: E402,F401
