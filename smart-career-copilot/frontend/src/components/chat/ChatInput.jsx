/**
 * ChatInput — rich input area with file attachment and send button.
 */

import React, { useState, useRef, useEffect } from 'react';
import { HiOutlinePaperAirplane, HiOutlinePaperClip, HiOutlineStopCircle, HiOutlineXMark, HiOutlineDocument } from 'react-icons/hi2';
import { uploadFile } from '../../services/api';
import './ChatInput.css';

const ChatInput = ({ onSend, isLoading, customUploadFn }) => {
  const [text, setText] = useState('');
  const [attachedFiles, setAttachedFiles] = useState([]);
  const [uploadingFile, setUploadingFile] = useState(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 200) + 'px';
    }
  }, [text]);

  const handleSend = () => {
    if ((text.trim() || attachedFiles.length > 0) && !isLoading) {
      onSend(text.trim(), attachedFiles);
      setText('');
      setAttachedFiles([]);
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploadingFile(file.name);
    try {
      const uploader = customUploadFn || uploadFile;
      const result = await uploader(file);
      if (result) {
        setAttachedFiles((prev) => [...prev, result]);
      }
    } catch (error) {
      console.error('File upload failed:', error);
    } finally {
      setUploadingFile(null);
    }
  };

  const removeAttachedFile = (fileId) => {
    setAttachedFiles((prev) => prev.filter((f) => f.file_id !== fileId));
  };

  const isSendActive = (text.trim() || attachedFiles.length > 0) && !isLoading;

  return (
    <div className="chat-input-container">
      <div className="chat-input-wrapper glass-panel">
        <div className="input-inner-layout">
          {/* Attached Files & Uploading Indicators */}
          {(attachedFiles.length > 0 || uploadingFile) && (
            <div className="input-attachments-list">
              {attachedFiles.map((file) => (
                <div key={file.file_id} className="input-attachment-pill animate-fade-in">
                  <HiOutlineDocument size={16} className="file-icon" />
                  <span className="file-name truncate" title={file.filename}>{file.filename}</span>
                  <button 
                    type="button" 
                    className="remove-file-btn" 
                    onClick={() => removeAttachedFile(file.file_id)}
                    title="Remove attachment"
                  >
                    <HiOutlineXMark size={14} />
                  </button>
                </div>
              ))}
              {uploadingFile && (
                <div className="input-attachment-pill uploading animate-pulse">
                  <span className="spinner-mini" />
                  <span className="file-name truncate">Uploading {uploadingFile}...</span>
                </div>
              )}
            </div>
          )}

          <div className="input-row">
            <button
              className="attach-btn"
              onClick={() => {
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = '.pdf,.txt,.md,.doc,.docx';
                input.onchange = handleFileChange;
                input.click();
              }}
              title="Attach file"
              disabled={isLoading || !!uploadingFile}
            >
              <HiOutlinePaperClip size={20} />
            </button>

            <textarea
              ref={textareaRef}
              className="chat-textarea"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Send a message..."
              rows={1}
              disabled={isLoading}
            />
          </div>
        </div>

        <button
          className={`send-btn ${isSendActive ? 'active' : ''}`}
          onClick={isLoading ? undefined : handleSend}
          disabled={!isSendActive}
          title={isLoading ? 'Generating...' : 'Send message'}
        >
          {isLoading ? (
            <HiOutlineStopCircle size={20} />
          ) : (
            <HiOutlinePaperAirplane size={20} />
          )}
        </button>
      </div>
      <p className="input-disclaimer">
        Smart Career Copilot may make mistakes. Verify important information.
      </p>
    </div>
  );
};

export default ChatInput;
