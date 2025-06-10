document.addEventListener('DOMContentLoaded', () => {
    const gameButton = document.getElementById('gameButton');
    let clickCount = 0;

    if (gameButton) {
        gameButton.addEventListener('click', () => {
            clickCount++;
            gameButton.textContent = `Clicked ${clickCount} times!`;
        });
    } else {
        console.error('Element with ID "gameButton" not found.');
    }
});
