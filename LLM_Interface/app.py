from flask import Flask, render_template, request, jsonify
import json
import sys

from model import LocalLLM
interface = LocalLLM("Qwen/Qwen3-0.6B")

def simple_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent_float = 100 * (iteration / float(total))
    percent_str = ("{0:." + str(decimals) + "f}").format(percent_float)
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent_str}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n') # New line on complete
        sys.stdout.flush()

# 1. Create a Flask application instance
app = Flask(__name__)

# 2. Define a route for the homepage
@app.route('/')
def index():
    return render_template('text_input.html')

# 3. Define a route that takes a dynamic parameter and uses a template
@app.route('/greet/<name>')
def greet(name):
    # Renders 'templates/greeting.html', passing the 'name' variable
    return render_template('greeting.html', user_name=name)

# 4. Define a route that handles form submission (GET and POST)
@app.route('/submit_form', methods=['GET', 'POST'])
def submit_form():
    if request.method == 'POST':
        # Access form data
        submitted_name = request.form['your_name']
        return f"Thanks for submitting, {submitted_name}!"
    else: # GET request
        # Show the form
        return """
            <form method="POST">
                Your Name: <input type="text" name="your_name">
                <input type="submit" value="Submit">
            </form>
        """

@app.route('/submit_text', methods=['POST'])
def handle_text_submission():
    if request.method == 'POST':
        try:
            # Get JSON data sent from the frontend
            data = request.get_json()

            if not data:
                return jsonify({"error": "No JSON data received"}), 400

            # Extract the text data (assuming it's sent with a key, e.g., 'text_data')
            submitted_text = data.get('text_data')

            if submitted_text is None: # Check if 'text_data' key exists
                return jsonify({"error": "Missing 'text_data' in JSON payload"}), 400

            print(f"Received text from frontend: '{submitted_text}'")

            # Send a success response back to the frontend
            return jsonify({"message": "Text received successfully!", "received_text": submitted_text}), 200

        except Exception as e:
            print(f"Error processing request: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        # Method Not Allowed
        return jsonify({"error": "Method not allowed. Use POST."}), 405

@app.route('/predict_s', methods=['POST'])
def predict_s():
    if request.method == 'POST':
        try:
            # Get JSON data sent from the frontend
            data = request.get_json()

            if not data:
                return jsonify({"error": "No JSON data received"}), 400

            # Extract the text data (assuming it's sent with a key, e.g., 'text_data')
            submitted_text = data.get('prompt')
            number_of_tokens = data.get('max_new_tokens')
            number_of_predictions = data.get('num_predictions')

            if submitted_text is None: # Check if 'text_data' key exists
                return jsonify({"error": "Missing 'text_data' in JSON payload"}), 400

            print(f"Received text from frontend: '{submitted_text}'")

            generated_text = submitted_text

            tokens = []
            interface.load()
            flag = False
            for i in range(number_of_tokens):
                l = interface.predict_next_token(generated_text,int(number_of_predictions))
                #print(l['predicted_next_token'])
                generated_text = generated_text + str(l['predicted_next_token'])
                tokens.append(l)
                print(str(l))
                simple_progress_bar(i,number_of_tokens-1, "token generation progress:")
                for prediction in l['top_k_predictions']:
                    if prediction['token_id'] == 26525 or prediction['token_id'] == 11162 or prediction['token_id'] == 25521 or prediction['token_id'] == 63039:
                        print(str(l))
                        flag = True
                        break
                if flag:
                    print(f"printed {i} tokens")
                    break
            if not flag:
                print(f"printed {number_of_tokens} tokens")

            interface.unload()

            formatted_predictions = []
            for prediction in tokens:
                im_formatted = []
                for token in prediction['top_k_predictions']:
                    im_formatted.append({"token": token['token'], "probability": token['probability']})
                formatted_predictions.append(im_formatted)
            #print(formatted_predictions)
            
            # Send a success response back to the frontend
            return jsonify({"message": "Text interfaced successfully!", "predicted_text": generated_text, "list_of_predictions": formatted_predictions}), 200

        except Exception as e:
            print(f"Error processing request: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        # Method Not Allowed
        return jsonify({"error": "Method not allowed. Use POST."}), 405

@app.route('/chat')
def chat_page():
    return render_template('chat.html')

@app.route('/chat_predict', methods=['POST'])
def chat_predict():
    if request.method == 'POST':
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data received"}), 400

            prompt = data.get('prompt')
            if not prompt:
                return jsonify({"error": "Missing 'prompt' in JSON payload"}), 400

            # Extract generation parameters with defaults if not provided
            temperature = data.get('temperature', 1.0)
            max_new_tokens = data.get('max_new_tokens', 50)
            top_k = data.get('top_k', 5)
            top_p = data.get('top_p', 0.95)
            # Add other parameters as needed, e.g., do_sample, num_beams, etc.
            # For now, we'll assume do_sample=True for conversational use

            # Log received data for debugging
            print(f"Received for /chat_predict: prompt='{prompt}', temp={temperature}, max_tokens={max_new_tokens}, top_k={top_k}, top_p={top_p}")

            # Use the global 'interface' to generate text
            # The generate_text method in LocalLLM handles model loading/unloading
            generated_text = interface.generate_text(
                prompt_text=prompt,
                temperature=float(temperature),
                max_new_tokens=int(max_new_tokens),
                top_k=int(top_k),
                top_p=float(top_p),
                do_sample=True # Important for more natural conversation
            )

            return jsonify({"generated_text": generated_text}), 200

        except Exception as e:
            print(f"Error processing /chat_predict request: {e}")
            # Consider logging the stack trace for more detailed debugging
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "Method not allowed. Use POST."}), 405

# 5. Run the development server (if this script is executed directly)
if __name__ == '__main__':
    # debug=True enables auto-reloading and an interactive debugger in the browser
    app.run(debug=True, port=5000) # You can specify a port
