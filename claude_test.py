import json
import os
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file")

# Create client
client = anthropic.Anthropic(api_key=api_key)

# Make API call
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=200,
    temperature=0,
    messages=[
        {
            "role": "user",
            "content": (
                "Return ONLY valid JSON with keys: role_family, specialization, level, location, confidence.\n"
                "- specialization must be null unless explicitly stated (e.g., Android, Mobile, Backend).\n"
                "Allowed role_family values: ['Software Engineering','Quality Engineering','Program Management','Data Engineering']\n"
                "Map this title: 'Tester - India'\n"
                "level must be integer 1-5. confidence must be between 0 and 1."
            ),
        }
    ],
)

raw = response.content[0].text.strip()

# If the model wraps JSON in ```json fences, remove them:
raw = raw.removeprefix("```json").removesuffix("```").strip()

data = json.loads(raw) #will raise if invalid JSON
print(data)