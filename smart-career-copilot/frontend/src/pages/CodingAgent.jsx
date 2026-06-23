/**
 * CodingAgent page — coding task input with plan/code/test workflow.
 */

import React from 'react';
import ChatWindow from '../components/chat/ChatWindow';
import ChatInput from '../components/chat/ChatInput';
import useChat from '../hooks/useChat';
import './AgentPage.css';

const CodingAgent = () => {
  const { send, messages, isLoading } = useChat();

  return (
    <div className="agent-page">
      <div className="agent-page-content">
        {messages.length === 0 && (
          <div className="agent-hero animate-fade-in-up">
            <div className="hero-badge" style={{ background: 'rgba(34, 197, 94, 0.1)', color: 'var(--agent-coding)' }}>
              💻 Autonomous Coding Agent
            </div>
            <h2>Build Anything with AI</h2>
            <p>Describe your task. The agent will plan, code, test, review, and iterate automatically.</p>
            <div className="hero-suggestions stagger-children">
              {[
                'Build a REST API with FastAPI and SQLite',
                'Create a binary search tree with visualization',
                'Write a web scraper for news articles',
                'Implement a CLI todo app in Python',
              ].map((s, i) => (
                <button
                  key={i}
                  className="suggestion-pill glass-button animate-fade-in-up"
                  onClick={() => send(s, 'coding')}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        <ChatWindow messages={messages} isLoading={isLoading} />
      </div>

      <ChatInput onSend={(text, attachments) => send(text, 'coding', attachments)} isLoading={isLoading} />
    </div>
  );
};

export default CodingAgent;
