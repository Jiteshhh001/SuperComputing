/**
 * ChatWindow — scrollable message area with auto-scroll.
 */

import React, { useRef, useEffect } from 'react';
import ChatBubble from './ChatBubble';
import TypingIndicator from './TypingIndicator';
import SourceCard from './SourceCard';
import './ChatWindow.css';

const ChatWindow = ({ messages, isLoading, sources }) => {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  return (
    <div className="chat-window">
      <div className="chat-messages">


        {messages.map((msg, idx) => (
          <React.Fragment key={msg.id || idx}>
            <ChatBubble message={msg} />
            {msg.sources && msg.sources.length > 0 && (
              <div className="sources-row animate-fade-in">
                {msg.sources.slice(0, 4).map((source, si) => (
                  <SourceCard key={si} source={source} />
                ))}
              </div>
            )}
          </React.Fragment>
        ))}

        {isLoading && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>
    </div>
  );
};

export default ChatWindow;
