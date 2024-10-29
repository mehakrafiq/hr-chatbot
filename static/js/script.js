// Get DOM elements
const sendButton = document.getElementById('send-btn');
const userInput = document.getElementById('user-input');
const chatBox = document.getElementById('chat-box');
const typingIndicator = document.getElementById('typing-indicator');

// Function to handle sending messages
function sendMessage() {
    const messageText = userInput.value.trim(); // Get the input value
    if (messageText) {
        // Create a new message element for the user
        const userMessage = document.createElement('div');
        userMessage.textContent = messageText;
        userMessage.classList.add('message', 'user-message'); // Add classes for styling

        // Append the user message to the chat box
        chatBox.appendChild(userMessage);
        
        // Clear the input field
        userInput.value = '';

        // Scroll to the bottom of the chat box
        chatBox.scrollTop = chatBox.scrollHeight;

        // Show the typing indicator
        typingIndicator.style.display = 'block';

        // Make a POST request to the Flask API
        fetch('/ask_hr', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: messageText })
        })
        .then(response => response.json())
        .then(data => {
            // Hide the typing indicator
            typingIndicator.style.display = 'none';

            // Create a new message element for the bot response
            const botMessage = document.createElement('div');
            botMessage.textContent = `🤖: ${data.response}`; // Get the response from the API
            botMessage.classList.add('message', 'bot-message'); // Add classes for styling

            // Append the bot message to the chat box
            chatBox.appendChild(botMessage);

            // Scroll to the bottom of the chat box
            chatBox.scrollTop = chatBox.scrollHeight; // Auto-scroll to the latest message
        })
        .catch(error => {
            console.error('Error:', error);
            typingIndicator.style.display = 'none';

            // Show an error message
            const errorMessage = document.createElement('div');
            errorMessage.textContent = "🤖: An error occurred. Please try again later.";
            errorMessage.classList.add('message', 'bot-message');
            chatBox.appendChild(errorMessage);
        });
    }
}

// Event listener for the send button
sendButton.addEventListener('click', sendMessage);

// Optional: Allow pressing Enter to send the message
userInput.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
});
