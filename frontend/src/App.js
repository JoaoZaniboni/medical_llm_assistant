import React, { useState } from 'react';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    const newUserMessage = { sender: 'user', text: query };
    setMessages((prev) => [...prev, newUserMessage]);
    setIsLoading(true);

    try {
      const res = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      const data = await res.json();
      const newBotMessage = { sender: 'bot', text: data.response };
      setMessages((prev) => [...prev, newBotMessage]);
    } catch (error) {
      console.error(error);
      const errorMessage = { sender: 'bot', text: 'Erro ao conectar com o servidor.' };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setQuery('');
    }
  };

  return (
    <div className="app-container">
      <header className="header">Dr. Chat</header>
      <div className="chat-container">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`message ${msg.sender === 'user' ? 'user-message' : 'bot-message'}`}
          >
            {msg.text}
          </div>
        ))}
        {isLoading && <div className="message bot-message">Digitando...</div>}
      </div>
      <form className="input-container" onSubmit={handleSubmit}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Digite sua mensagem..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading || !query.trim()}>
          Enviar
        </button>
      </form>
    </div>
  );
}

export default App;
