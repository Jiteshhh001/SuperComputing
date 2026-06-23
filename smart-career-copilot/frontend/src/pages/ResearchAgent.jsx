/**
 * ResearchAgent page — deep research with source cards and charts.
 */

import React from 'react';
import ChatWindow from '../components/chat/ChatWindow';
import ChatInput from '../components/chat/ChatInput';
import useChat from '../hooks/useChat';
import './AgentPage.css';

const ResearchAgent = () => {
  const { send, messages, isLoading } = useChat();

  return (
    <div className="agent-page">
      <div className="agent-page-content">
        {messages.length === 0 && (
          <div className="agent-hero animate-fade-in-up">
            <div className="hero-badge" style={{ background: 'rgba(139, 92, 246, 0.1)', color: 'var(--agent-research)' }}>
              🔍 Deep Research Agent
            </div>
            <h2>Research Any Topic</h2>
            <p>Search the web, academic papers, and get comprehensive reports with citations.</p>
            <div className="hero-suggestions stagger-children">
              {[
                'Latest breakthroughs in quantum computing',
                'Compare React vs Vue vs Angular in 2025',
                'State of AI in healthcare industry',
                'Best practices for microservices architecture',
              ].map((s, i) => (
                <button
                  key={i}
                  className="suggestion-pill glass-button animate-fade-in-up"
                  onClick={() => send(s, 'research')}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        <ChatWindow messages={messages} isLoading={isLoading} />
      </div>

      <ChatInput onSend={(text, attachments) => send(text, 'research', attachments)} isLoading={isLoading} />
    </div>
  );
};

export default ResearchAgent;
