document.addEventListener('DOMContentLoaded', () => {
    const testResultsContainer = document.getElementById('test-results');
    const gameButton = document.getElementById('gameButton');
    let testsPassed = 0;
    let testsFailed = 0;

    function runTest(description, testFn) {
        const resultDiv = document.createElement('div');
        resultDiv.classList.add('test-case');
        try {
            testFn();
            resultDiv.textContent = `PASS: ${description}`;
            resultDiv.classList.add('pass');
            testsPassed++;
        } catch (e) {
            resultDiv.textContent = `FAIL: ${description} - ${e.message}`;
            resultDiv.classList.add('fail');
            testsFailed++;
        }
        testResultsContainer.appendChild(resultDiv);
    }

    // Test 1: Check if the button exists
    runTest('Button element should exist', () => {
        if (!gameButton) {
            throw new Error('gameButton not found');
        }
    });

    // Test 2: Check initial button text
    runTest('Button should have initial text "Click Me!"', () => {
        if (gameButton.textContent !== 'Click Me!') {
            throw new Error(`Expected "Click Me!" but got "${gameButton.textContent}"`);
        }
    });

    // Test 3: Check button text after one click
    runTest('Button text should update after one click', () => {
        gameButton.click();
        if (gameButton.textContent !== 'Clicked 1 times!') {
            throw new Error(`Expected "Clicked 1 times!" but got "${gameButton.textContent}"`);
        }
    });

    // Test 4: Check button text after multiple clicks
    runTest('Button text should update correctly after multiple clicks', () => {
        gameButton.click(); // 2nd click
        gameButton.click(); // 3rd click
        if (gameButton.textContent !== 'Clicked 3 times!') {
            throw new Error(`Expected "Clicked 3 times!" but got "${gameButton.textContent}"`);
        }
    });

    // Summary
    const summaryDiv = document.createElement('div');
    summaryDiv.innerHTML = `<h3>Test Summary: ${testsPassed} passed, ${testsFailed} failed.</h3>`;
    testResultsContainer.appendChild(summaryDiv);

    // Clean up: Reset button text for subsequent manual tests if any
    // This is tricky because the original state is 'Click Me!' but game.js changes it.
    // For a simple test runner, this is acceptable. A more robust framework would handle this better.
    // Re-initializing the button or its text:
    // gameButton.textContent = 'Click Me!'; // This would interfere if game.js re-runs or has complex state
    // For now, we'll leave it as is after the tests.
});
