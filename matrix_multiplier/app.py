from flask import Flask, render_template, request, jsonify
import numpy as np
import traceback # For better error logging

app = Flask(__name__)

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/multiply', methods=['POST'])
def multiply_matrices():
    """Receives two matrices, multiplies them, and returns the result."""
    try:
        data = request.get_json()

        if not data or 'matrix_a' not in data or 'matrix_b' not in data:
            return jsonify({"error": "Missing matrix data in request."}), 400

        matrix_a_list = data['matrix_a']
        matrix_b_list = data['matrix_b']

        # --- Input Validation ---
        if not (isinstance(matrix_a_list, list) and isinstance(matrix_b_list, list)):
             return jsonify({"error": "Matrices must be provided as lists."}), 400
        if len(matrix_a_list) != 3 or len(matrix_b_list) != 3:
             return jsonify({"error": "Both matrices must have 3 rows."}), 400
        if not all(len(row) == 3 for row in matrix_a_list) or \
           not all(len(row) == 3 for row in matrix_b_list):
             return jsonify({"error": "Both matrices must have 3 columns."}), 400

        # --- Convert to NumPy arrays and perform multiplication ---
        try:
            # Convert input strings/numbers to floats for calculation
            matrix_a_np = np.array(matrix_a_list, dtype=float)
            matrix_b_np = np.array(matrix_b_list, dtype=float)
        except ValueError:
             return jsonify({"error": "Invalid number format in matrices."}), 400

        # The core matrix multiplication
        result_np = np.dot(matrix_a_np, matrix_b_np)

        # Convert result back to a list of lists for JSON serialization
        result_list = result_np.tolist()

        return jsonify({"result": result_list})

    except Exception as e:
        print("An error occurred:") # Log detailed error to server console
        print(traceback.format_exc())
        # Return a generic error to the client
        return jsonify({"error": "An internal server error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True) # debug=True allows auto-reloading on code changes