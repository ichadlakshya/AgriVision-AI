/**
 * AgriVision AI - Main JavaScript
 * Core functionality and interactions
 */

document.addEventListener('DOMContentLoaded', function() {
    initializeComponents();
});

/**
 * Initialize all components
 */
function initializeComponents() {
    initializeScrollAnimations();
    initializeTooltips();
    initializePageTransitions();
}

/**
 * Scroll animations for elements
 */
function initializeScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };

    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in-up');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Observe all cards and feature elements
    document.querySelectorAll('.stat-card, .feature-card, .chart-container').forEach(el => {
        observer.observe(el);
    });
}

/**
 * Initialize Bootstrap tooltips
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * Page transition animations
 */
function initializePageTransitions() {
    const links = document.querySelectorAll('a:not([target="_blank"]):not([href^="#"])');
    
    links.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            
            // Don't animate if it's an API call or local anchor
            if (href && !href.startsWith('/api') && !href.startsWith('http')) {
                e.preventDefault();
                
                // Fade out
                document.body.style.opacity = '0.9';
                
                setTimeout(() => {
                    window.location.href = href;
                }, 100);
            }
        });
    });
}

/**
 * Format number with thousand separators
 */
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/**
 * Make API request with error handling
 */
async function apiRequest(url, method = 'GET', data = null) {
    try {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            }
        };

        if (data && method !== 'GET') {
            options.body = JSON.stringify(data);
        }

        const response = await fetch(url, options);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        showNotification('An error occurred. Please try again.', 'error');
        throw error;
    }
}

/**
 * Show notification/toast
 */
function showNotification(message, type = 'info') {
    const toastContainer = document.getElementById('toastContainer');
    if (toastContainer) {
        const toastEl = document.createElement('div');
        toastEl.className = 'toast';
        toastEl.setAttribute('role', 'alert');
        toastEl.setAttribute('aria-live', 'assertive');
        toastEl.setAttribute('aria-atomic', 'true');

        const headerEl = document.createElement('div');
        headerEl.className = `toast-header bg-${type} text-white`;

        const titleEl = document.createElement('strong');
        titleEl.className = 'me-auto';
        titleEl.textContent = type.charAt(0).toUpperCase() + type.slice(1);

        const closeButton = document.createElement('button');
        closeButton.type = 'button';
        closeButton.className = 'btn-close btn-close-white';
        closeButton.setAttribute('data-bs-dismiss', 'toast');
        closeButton.setAttribute('aria-label', 'Close');

        const bodyEl = document.createElement('div');
        bodyEl.className = 'toast-body';
        bodyEl.textContent = message;

        headerEl.appendChild(titleEl);
        headerEl.appendChild(closeButton);
        toastEl.appendChild(headerEl);
        toastEl.appendChild(bodyEl);
        toastContainer.appendChild(toastEl);

        const toast = new bootstrap.Toast(toastEl);
        toast.show();
    }
}

/**
 * Debounce function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Throttle function
 */
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * Get CSRF token (if using Flask-WTF)
 */
function getCsrfToken() {
    const name = 'csrf_token=';
    const decodedCookie = decodeURIComponent(document.cookie);
    const cookieArray = decodedCookie.split(';');
    
    for(let i = 0; i < cookieArray.length; i++) {
        let cookie = cookieArray[i].trim();
        if (cookie.indexOf(name) === 0) {
            return cookie.substring(name.length, cookie.length);
        }
    }
    return '';
}

/**
 * Initialize dark mode toggle (optional)
 */
function initializeDarkMode() {
    const darkModeToggle = document.getElementById('darkModeToggle');
    if (darkModeToggle) {
        const isDarkMode = localStorage.getItem('darkMode') === 'true';
        
        if (isDarkMode) {
            document.documentElement.setAttribute('data-bs-theme', 'dark');
        }
        
        darkModeToggle.addEventListener('click', () => {
            const isCurrentlyDark = document.documentElement.getAttribute('data-bs-theme') === 'dark';
            document.documentElement.setAttribute('data-bs-theme', !isCurrentlyDark ? 'dark' : 'light');
            localStorage.setItem('darkMode', !isCurrentlyDark);
        });
    }
}

/**
 * Smooth scroll to element
 */
function smoothScrollTo(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

/**
 * Check if element is in viewport
 */
function isElementInViewport(el) {
    const rect = el.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

/**
 * Log page analytics
 */
function logPageView(pageName) {
    return pageName;
}

/**
 * Export utilities for use in other scripts
 */
window.agriVision = {
    apiRequest,
    showNotification,
    debounce,
    throttle,
    formatNumber,
    smoothScrollTo,
    isElementInViewport,
    logPageView,
    getCsrfToken
};
