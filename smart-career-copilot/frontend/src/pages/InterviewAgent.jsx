/**
 * InterviewAgent page — mock interview with Q&A and scoring.
 */

import React from 'react';
import ChatWindow from '../components/chat/ChatWindow';
import ChatInput from '../components/chat/ChatInput';
import useChat from '../hooks/useChat';
import './AgentPage.css';

const InterviewAgent = () => {
  const { send, messages, isLoading } = useChat();

  return (
    <div className="agent-page">
      <div className="agent-page-content">
        {messages.length === 0 && (
          <div className="agent-hero animate-fade-in-up">
            <div className="hero-badge" style={{ background: 'rgba(245, 158, 11, 0.1)', color: 'var(--agent-interview)' }}>
              🎯 Interview Coach Agent
            </div>
            <h2>Ace Your Next Interview</h2>
            <p>Practice technical and behavioral interviews with real-time AI feedback and scoring.</p>
            <div className="hero-suggestions stagger-children">
              {[
                'Start a technical Python interview',
                'Practice behavioral questions with STAR method',
                'Quiz me on system design concepts',
                'Give me feedback on my communication skills',
              ].map((s, i) => (
                <button
                  key={i}
                  className="suggestion-pill glass-button animate-fade-in-up"
                  onClick={() => send(s, 'interview')}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        <ChatWindow messages={messages} isLoading={isLoading} />
      </div>

      <ChatInput onSend={(text, attachments) => send(text, 'interview', attachments)} isLoading={isLoading} />
    </div>
  );
};

export default InterviewAgent;
