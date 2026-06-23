/**
 * TypingIndicator — animated dots shown while agent is thinking.
 */

import React from 'react';
import './TypingIndicator.css';

const TypingIndicator = () => {
  return (
    <div className="typing-indicator-wrapper animate-fade-in">
      <div className="typing-avatar">
        <span>✦</span>
      </div>
      <div className="typing-bubble glass-card">
        <div className="typing-dots">
          <span className="typing-dot" />
          <span className="typing-dot" />
          <span className="typing-dot" />
        </div>
        <span className="typing-label">Thinking...</span>
      </div>
    </div>
  );
};

export default TypingIndicator;
