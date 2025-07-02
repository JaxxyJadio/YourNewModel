from flask import Flask, request, jsonify, render_template_string
import torch
from YourNewModel.modelling_YourNewModel import create_model
from YourNewModel.tokenizer_YourNewModel import YourNewModelTokenizer
import os

# Remove hardcoded MODEL_PATH, use env or relative path
MODEL_PATH = os.environ.get('YOURNEWMODEL_PATH', os.path.join(os.path.dirname(__file__), 'YourNewModel.pt'))

app = Flask(__name__)

# Load model and tokenizer once at startup
model = create_model("small")
if torch.cuda.is_available():
    checkpoint = torch.load(MODEL_PATH)
else:
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
model.eval()

tokenizer = YourNewModelTokenizer(vocab_size=model.config.vocab_size)

def generate_response(prompt: str, max_length: int = 64) -> str:
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length)[0].tolist()
    return tokenizer.decode(output_ids)

# Tailwind CSS + simple chat UI
template = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Dense LLM Chat Panel</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-blue-50 to-gray-100 min-h-screen flex items-center justify-center">
  <div class="bg-white shadow-xl rounded-xl p-8 w-full max-w-lg flex flex-col items-center">
    <h1 class="text-3xl font-bold mb-2 text-blue-700">Dense LLM Panel</h1>
    <p class="mb-6 text-gray-500 text-center">Connect to your model and chat below.</p>
    <form id="chat-form" class="w-full flex flex-col gap-3">
      <label for="prompt" class="text-sm text-gray-600">Prompt</label>
      <textarea id="prompt" name="prompt" rows="3" class="border border-gray-300 rounded-lg p-2 w-full focus:outline-none focus:ring-2 focus:ring-blue-400" placeholder="Type your message..."></textarea>
      <button type="submit" class="bg-blue-600 text-white rounded-lg px-4 py-2 font-semibold hover:bg-blue-700 transition">Send</button>
    </form>
    <div id="response-panel" class="mt-6 w-full">
      <div class="text-xs text-gray-400 mb-1">Model Response:</div>
      <div id="response" class="p-4 bg-gray-50 rounded-lg text-gray-800 min-h-[2rem] border border-gray-200"></div>
    </div>
  </div>
  <script>
    const form = document.getElementById('chat-form');
    const responseDiv = document.getElementById('response');
    form.onsubmit = async (e) => {
      e.preventDefault();
      responseDiv.textContent = 'Connecting...';
      const prompt = document.getElementById('prompt').value;
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });
      const data = await res.json();
      responseDiv.textContent = data.response;
    };
  </script>
</body>
</html>
'''

@app.route("/")
def index():
    return render_template_string(template)

@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt.strip():
        return jsonify({"response": "Please enter a prompt."})
    response = generate_response(prompt)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
