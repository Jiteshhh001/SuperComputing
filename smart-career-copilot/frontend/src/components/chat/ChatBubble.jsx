/**
 * ChatBubble — individual message bubble with markdown support.
 */

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { HiOutlineClipboardDocument, HiOutlineCheck, HiOutlineSparkles, HiOutlineDocument } from 'react-icons/hi2';
import './ChatBubble.css';

const ChatBubble = ({ message }) => {
  const [copied, setCopied] = useState(false);
  const isUser = message.role === 'user';

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const agentColors = {
    resume: 'var(--agent-resume)',
    research: 'var(--agent-research)',
    coding: 'var(--agent-coding)',
    interview: 'var(--agent-interview)',
  };

  return (
    <div className={`chat-bubble-wrapper ${isUser ? 'user' : 'assistant'} animate-fade-in-up`}>
      {!isUser && (
        <div className="bubble-avatar" style={{ '--avatar-color': agentColors[message.agent_type] || 'var(--accent-primary)' }}>
          <HiOutlineSparkles size={16} />
        </div>
      )}
      <div className={`chat-bubble ${isUser ? 'user-bubble' : 'assistant-bubble'}`}>
        {!isUser && message.agent_type && (
          <span className="agent-label glass-badge" style={{ color: agentColors[message.agent_type] }}>
            {message.agent_type}
          </span>
        )}
        <div className="bubble-content">
          {isUser ? (
            <p>{message.content}</p>
          ) : (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                code({ node, inline, className, children, ...props }) {
                  return inline ? (
                    <code className={className} {...props}>{children}</code>
                  ) : (
                    <pre className="code-block">
                      <code className={className} {...props}>{children}</code>
                    </pre>
                  );
                },
              }}
            >
              {message.content}
            </ReactMarkdown>
          )}
        </div>

        {/* Attachments rendering */}
        {message.metadata?.attachments && message.metadata.attachments.length > 0 && (
          <div className="message-attachments-list">
            {message.metadata.attachments.map((file, idx) => (
              <div key={file.file_id || idx} className="message-attachment-card glass-panel">
                <HiOutlineDocument className="file-icon" size={16} />
                <div className="file-info">
                  <span className="file-name truncate" title={file.filename || file.name}>
                    {file.filename || file.name}
                  </span>
                  {file.size && (
                    <span className="file-size">
                      {(file.size / 1024).toFixed(1)} KB
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {!isUser && (
          <div className="bubble-actions">
            <button className="action-btn" onClick={handleCopy} title="Copy">
              {copied ? <HiOutlineCheck size={14} /> : <HiOutlineClipboardDocument size={14} />}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatBubble;
