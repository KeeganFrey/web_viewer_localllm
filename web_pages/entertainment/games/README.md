# Web Games Page

This directory contains a simple Flask web application for hosting web-based games.

## Structure

- `app.py`: The main Flask application file.
- `templates/`: Contains HTML templates.
  - `index.html`: The main page for the games.
- `static/`: Contains static files (CSS, JavaScript).
  - `style.css`: Basic styling for the page.
  - `game.js`: JavaScript for game logic (to be added).
- `tests/`: Contains JavaScript tests (to be added).

## How to Run

1.  **Navigate to the directory:**
    ```bash
    cd web_pages/entertainment/games
    ```

2.  **Ensure Flask is installed:**
    If you don't have Flask installed, you can install it using pip:
    ```bash
    pip install Flask
    ```

3.  **Run the Flask application:**
    ```bash
    python app.py
    ```

4.  Open your web browser and go to `http://127.0.0.1:5000/` to see the page.

## Templates

- The main game page is `templates/index.html`.
- It uses Flask's templating engine, so you can use features like `{{ url_for('static', filename='...') }}` to link static files.
- Game content should primarily be added or manipulated via JavaScript in `static/game.js` and linked HTML elements in `index.html`.

## Adding Games

1.  Develop your game logic in JavaScript, preferably in a new file within the `static/` directory, or by expanding `static/game.js`.
2.  Add necessary HTML elements to `templates/index.html` for your game.
3.  Link your new JavaScript file or ensure `game.js` is correctly linked in `index.html`.
4.  Style your game using `static/style.css` or by adding new CSS files and linking them.

## Testing

The JavaScript game logic is tested using a simple HTML-based test runner.

1.  **Navigate to the tests directory:**
    The test files are located in `web_pages/entertainment/games/tests/`.

2.  **Open the test runner in your browser:**
    Open the file `web_pages/entertainment/games/tests/test_runner.html` directly in a web browser (e.g., Chrome, Firefox).

3.  **View test results:**
    The browser will display the results of the tests on the page. It will show which tests passed and which failed.

### Test Files:
- `tests/test_runner.html`: The HTML page that executes and displays the tests.
- `tests/test_game.js`: Contains the actual test logic for `static/game.js`.

To add more tests:
1.  Modify `tests/test_game.js` to include additional test functions using the `runTest` helper.
2.  If your new game features require different HTML elements, you might need to update the simplified HTML structure within `tests/test_runner.html` or create new specific test runners.
