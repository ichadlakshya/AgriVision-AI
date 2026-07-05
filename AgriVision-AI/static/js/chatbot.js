/* ═══════════════════════════════════════════════════════════════════
   AgriVision AI - Chatbot JavaScript
   Professional Chat Interface with Real-time Communication
   ═══════════════════════════════════════════════════════════════════ */

// Configuration
const CONFIG = {
    API_URL: '/api/chat',
    MAX_MESSAGE_LENGTH: 500,
    SCROLL_BEHAVIOR: 'smooth',
    RECONNECT_DELAY: 3000,
    MAX_RETRIES: 3
};

// State Management
const state = {
    isLoading: false,
    messageCount: 0,
    retryCount: 0,
    conversationHistory: []
};

// DOM Elements
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const loadingSpinner = document.getElementById('loadingSpinner');

// ───────────────────────────────────────────────────────────────────
// Event Listeners
// ─────────────────────────────────────────────────────────────────── 
document.addEventListener('DOMContentLoaded', () => {
    initializeChat();
    setupEventListeners();
});

function setupEventListeners() {
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage(e);
        }
    });

    messageInput.addEventListener('input', (e) => {
        e.target.style.height = 'auto';
        e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
    });

    // Auto-scroll to bottom when new messages arrive
    const observer = new MutationObserver(() => {
        scrollToBottom();
    });

    observer.observe(chatMessages, {
        childList: true,
        subtree: true
    });
}

// ───────────────────────────────────────────────────────────────────
// Initialization
// ─────────────────────────────────────────────────────────────────── 
function initializeChat() {
    chatMessages.innerHTML = `
        <div class="message message-bot">
            <div class="welcome-section">
                <div class="welcome-icon">
                    <i class="fas fa-sprout"></i>
                </div>
                <h2>Welcome to AgriVision AI</h2>
                <p>Ask me anything about crop production, yields, seasonality, regional trends, and more!</p>
                
                <div class="quick-prompts">
                    <h3>Quick Questions</h3>
                    <button class="prompt-btn" onclick="sendQuickPrompt('Which state produces the most crops?')">
                        <i class="fas fa-chart-bar"></i> Top States
                    </button>
                    <button class="prompt-btn" onclick="sendQuickPrompt('What are the most productive crops?')">
                        <i class="fas fa-wheat-awn"></i> Top Crops
                    </button>
                    <button class="prompt-btn" onclick="sendQuickPrompt('Tell me about seasonal trends')">
                        <i class="fas fa-calendar"></i> Seasons
                    </button>
                    <button class="prompt-btn" onclick="sendQuickPrompt('What are crop yield insights?')">
                        <i class="fas fa-tractor"></i> Yields
                    </button>
                </div>
            </div>
        </div>
    `;
    messageInput.focus();
}

// ───────────────────────────────────────────────────────────────────
// Message Handling
// ─────────────────────────────────────────────────────────────────── 
function sendMessage(event) {
    event.preventDefault();

    const message = messageInput.value.trim();

    if (!message) {
        messageInput.focus();
        return;
    }

    if (message.length > CONFIG.MAX_MESSAGE_LENGTH) {
        showError('Message is too long. Maximum 500 characters.');
        return;
    }

    if (state.isLoading) {
        showError('Please wait for the current message to finish.');
        return;
    }

    // Add user message to chat
    addMessage(message, 'user');

    // Clear input
    messageInput.value = '';
    messageInput.style.height = 'auto';

    // Send to API
    sendToAPI(message);
}

function sendQuickPrompt(prompt) {
    messageInput.value = prompt;
    messageInput.style.height = 'auto';
    messageInput.focus();
    
    setTimeout(() => {
        const event = new Event('submit');
        document.querySelector('.chat-input-area form').dispatchEvent(event);
    }, 100);
}

// ───────────────────────────────────────────────────────────────────
// API Communication
// ─────────────────────────────────────────────────────────────────── 
async function sendToAPI(message) {
    state.isLoading = true;
    showLoadingSpinner(true);

    try {
        const response = await fetch(CONFIG.API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            body: JSON.stringify({
                message: message
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        if (data.error) {
            showError(data.error);
        } else {
            addMessage(data.response, 'bot');
            state.retryCount = 0;
        }
    } catch (error) {
        console.error('API Error:', error);

        if (state.retryCount < CONFIG.MAX_RETRIES) {
            state.retryCount++;
            showError(`Connection error. Retrying... (Attempt ${state.retryCount}/${CONFIG.MAX_RETRIES})`);
            
            setTimeout(() => {
                sendToAPI(message);
            }, CONFIG.RECONNECT_DELAY);
        } else {
            showError('Failed to get response. Please try again.');
            state.retryCount = 0;
        }
    } finally {
        state.isLoading = false;
        showLoadingSpinner(false);
        messageInput.focus();
    }
}

// ───────────────────────────────────────────────────────────────────
// Message Display
// ─────────────────────────────────────────────────────────────────── 
function addMessage(content, type) {
    // Remove welcome section on first message
    const welcomeSection = chatMessages.querySelector('.welcome-section');
    if (welcomeSection && state.messageCount === 0) {
        welcomeSection.parentElement.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${type}`;

    const timestamp = new Date().toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        hour12: true
    });

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = formatMessage(escapeHTML(content));

    const timestampDiv = document.createElement('div');
    timestampDiv.className = 'message-timestamp';
    timestampDiv.textContent = timestamp;

    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(timestampDiv);

    chatMessages.appendChild(messageDiv);

    state.messageCount++;
    state.conversationHistory.push({
        type,
        content,
        timestamp
    });

    scrollToBottom();
}

function formatMessage(content) {
    // Format markdown-like bold text
    content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Format code blocks
    content = content.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    
    // Format inline code
    content = content.replace(/`(.*?)`/g, '<code>$1</code>');
    
    // Format line breaks
    content = content.replace(/\n/g, '<br>');
    
    // Format bullet points
    content = content.replace(/^- (.*?)$/gm, '<li>$1</li>');
    
    // Wrap list items in ul
    content = content.replace(/(<li>.*?<\/li>)/s, '<ul>$1</ul>');

    return content;
}

function escapeHTML(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ───────────────────────────────────────────────────────────────────
// UI Helpers
// ─────────────────────────────────────────────────────────────────── 
function scrollToBottom() {
    setTimeout(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }, 0);
}

function showLoadingSpinner(show) {
    if (show) {
        loadingSpinner.classList.add('active');
    } else {
        loadingSpinner.classList.remove('active');
    }
}

function showError(message) {
    addMessage(`⚠️ ${message}`, 'bot');
}

function toggleMinimize() {
    // This can be extended for minimize functionality
}

// ───────────────────────────────────────────────────────────────────
// Keyboard Shortcuts
// ─────────────────────────────────────────────────────────────────── 
document.addEventListener('keydown', (e) => {
    // Focus input with Ctrl+L or Cmd+L
    if ((e.ctrlKey || e.metaKey) && e.key === 'l') {
        e.preventDefault();
        messageInput.focus();
    }

    // Clear chat with Ctrl+Shift+C
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'C') {
        e.preventDefault();
        if (confirm('Clear chat history?')) {
            initializeChat();
            state.conversationHistory = [];
        }
    }
});

// ───────────────────────────────────────────────────────────────────
// Export conversation (bonus feature)
// ─────────────────────────────────────────────────────────────────── 
function exportConversation() {
    const text = state.conversationHistory
        .map(msg => `[${msg.type.toUpperCase()}] ${msg.timestamp}\n${msg.content}\n`)
        .join('\n');

    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `agrivision-chat-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Initialize on page load
