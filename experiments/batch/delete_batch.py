from google import genai
import json

state = json.load(open(".batch_state_gemini.json"))
client = genai.Client(vertexai=True, project=state["gcp_project"], location=state.get("gcp_location",
"us-central1"))

for name in state["batch_names"]:
    client.batches.cancel(name=name)   # stop it if still running
    client.batches.delete(name=name)   # delete it
    print(f"Deleted {name}")
