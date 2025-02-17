var tl = gsap.timeline();
tl.from("#nav h3", {
    y: -10,
    opacity: 0,
    delay: 0.4,
    duration: 0.8,
    stagger: 0.3
});


tl.from("#main h1", {
    x: -100,
    opacity: 0,
    duration: 0.8,
    stagger: 0.4,
    ease: "bounce"
});


document.querySelectorAll("#main h1").forEach(h1 => {
    h1.addEventListener("mouseenter", () => {
        gsap.to(h1, { scale: 1.1, duration: 0.3, ease: "power2.inOut" });
    });
    h1.addEventListener("mouseleave", () => {
        gsap.to(h1, { scale: 1, duration: 0.3, ease: "power2.inOut" });
    });
});


tl.from("img", {
    x: 100,
    rotate: 25,
    opacity: 0,
    duration: 0.5,
    stagger: 0.5,
    onComplete: () => {
        gsap.to("img", {
            rotation: "random(-10, 10)",
            yoyo: true,
            repeat: -1,
            duration: 1,
            ease: "sine.inOut"
        });
    }
});


// Animation for chatbot toggler
gsap.to(".chatbot-toggler", {
    y: -10,
    repeat: -1,
    yoyo: true,
    duration: 1,
    ease: "sine.inOut"
});

document.querySelector(".chatbot-toggler").addEventListener("mouseenter", () => {
    gsap.to(".chatbot-toggler", { scale: 1.2, duration: 0.2 });
});
document.querySelector(".chatbot-toggler").addEventListener("mouseleave", () => {
    gsap.to(".chatbot-toggler", { scale: 1, duration: 0.2 });
});

// Chatbot functionality
const chatbotToggler = document.querySelector(".chatbot-toggler");
const closeBtn = document.querySelector(".close-btn");
const chatbox = document.querySelector(".chatbox");
const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector(".chat-input span");
const inputInitHeight = chatInput.scrollHeight;

// Function to create a chat list item
const createChatLi = (message, className) => {
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", `${className}`);
    let chatContent =
        className === "outgoing"
            ? `<p></p>`
            : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
    chatLi.innerHTML = chatContent;
    chatLi.querySelector("p").textContent = message;
    return chatLi;
};

// Function to handle structured bot responses
const createChatLiWithLink = (text, link, linkText, className) => {
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", `${className}`);
    chatLi.innerHTML = `
        <span class="material-symbols-outlined">smart_toy</span>
        <p>${text} <a href="${link}" target="_blank">${linkText}</a></p>
    `;
    return chatLi;
};

// Function to generate chatbot response by calling Flask backend
const generateResponse = async (chatElement) => {
    const messageElement = chatElement.querySelector("p");
    const userMessage = chatElement.previousElementSibling.querySelector("p").textContent;

    try {
        // Make a request to the Flask backend
        const response = await fetch(`/get?msg=${encodeURIComponent(userMessage)}`);
        if (!response.ok) {
            throw new Error("Failed to get response from the server");
        }
        const botReply = await response.text();
        messageElement.textContent = botReply;
    } catch (error) {
        console.error("Error fetching response:", error);
        messageElement.textContent = "Sorry, something went wrong. Please try again.";
    }

    // Scroll to the bottom of the chatbox
    chatbox.scrollTo(0, chatbox.scrollHeight);
};

// Function to handle user chat input
const handleChat = () => {
    const userMessage = chatInput.value.trim();
    if (!userMessage) return;
    chatInput.value = "";
    chatInput.style.height = `${inputInitHeight}px`;
    chatbox.appendChild(createChatLi(userMessage, "outgoing"));
    chatbox.scrollTo(0, chatbox.scrollHeight);

    // Simulate bot "thinking" before response
    setTimeout(() => {
        const incomingChatLi = createChatLi("Thinking...", "incoming");
        chatbox.appendChild(incomingChatLi);
        chatbox.scrollTo(0, chatbox.scrollHeight);
        generateResponse(incomingChatLi);
    }, 600);
};

// Event listener to adjust textarea height
chatInput.addEventListener("input", () => {
    chatInput.style.height = `${inputInitHeight}px`;
    chatInput.style.height = `${chatInput.scrollHeight}px`;
});

// Event listener for Enter key to send chat
chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleChat();
    }
});

// Event listener for send button
sendChatBtn.addEventListener("click", handleChat);

// Event listener for closing chatbot
closeBtn.addEventListener("click", () => document.body.classList.remove("show-chatbot"));

// Event listener for toggling chatbot
chatbotToggler.addEventListener("click", () => document.body.classList.toggle("show-chatbot"));


document.querySelectorAll("#nav h3").forEach(navItem => {
    navItem.addEventListener("mouseenter", () => {
        gsap.to(navItem, {
            scale: 1.2,
            duration: 0.2,
            ease: "power1.inOut",
            repeat: 1,
            yoyo: true 
        });
    });
});

const logo = document.querySelector("#nav h3:first-child");
logo.addEventListener("mouseenter", () => {
    gsap.to(logo, {
        scale: 1.3,
        duration: 0.3,
        ease: "power1.inOut",
        repeat: 1,
        yoyo: true 
    });
});
logo.addEventListener("mouseleave", () => {
    gsap.to(logo, {
        scale: 1,
        duration: 0.2,
        ease: "power1.inOut"
    });
});