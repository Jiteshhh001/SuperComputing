/**
 * ResumeAgent page — resume upload, analysis, and chat.
 */

import React, { useState } from 'react';
import ChatWindow from '../components/chat/ChatWindow';
import ChatInput from '../components/chat/ChatInput';
import FileUpload from '../components/common/FileUpload';
import useChat from '../hooks/useChat';
import useFileUpload from '../hooks/useFileUpload';
import useAppStore from '../store/appStore';
import { analyzeResume } from '../services/api';
import './AgentPage.css';

const ResumeAgent = () => {
  const { send, messages, isLoading } = useChat();
  const { upload, uploading, progress } = useFileUpload('resume');
  const { resumeData, setResumeData } = useAppStore();
  const [showUpload, setShowUpload] = useState(true);

  const handleUpload = async (file) => {
    const result = await upload(file);
    if (result) {
      setResumeData(result);
      setShowUpload(false);
      
      // Async fire-and-forget to fetch ATS score and skill gap analysis
      analyzeResume(result.file_id, result.parsed_text, '')
        .then(analysis => {
          setResumeData(prev => ({ ...prev, analysis }));
        })
        .catch(console.error);

      return {
        file_id: result.file_id || 'resume_file',
        filename: result.filename,
        size: file.size,
        content_type: file.type || 'application/pdf'
      };
    }
    return null;
  };

  const handleSend = (text, attachments) => {
    const context = {};
    if (resumeData) {
      if (resumeData.analysis) {
        context.resume_analysis = resumeData.analysis;
      } else if (resumeData.parsed_text) {
        context.resume_text = resumeData.parsed_text;
      }
    }
    send(text, 'resume', attachments, context);
  };

  return (
    <div className="agent-page">
      <div className="agent-page-content">
        {showUpload && messages.length === 0 && (
          <div className="agent-hero animate-fade-in-up">
            <div className="hero-badge" style={{ background: 'rgba(6, 182, 212, 0.1)', color: 'var(--agent-resume)' }}>
              📄 Resume Tailor Agent
            </div>
            <h2>Upload Your Resume</h2>
            <p>Get your ATS score, skill gap analysis, and AI-improved bullet points.</p>
            <FileUpload onUpload={handleUpload} uploading={uploading} progress={progress} />
          </div>
        )}

        {resumeData && !showUpload && messages.length === 0 && (
          <div className="resume-info glass-panel animate-fade-in-up" style={{ padding: 'var(--space-lg)', margin: '0 auto', maxWidth: '600px' }}>
            <h3>✅ Resume Loaded: {resumeData.filename}</h3>
            <p style={{ fontSize: '0.85rem', color: 'var(--text-tertiary)' }}>
              {resumeData.page_count} pages • Ready for analysis
            </p>
          </div>
        )}

        <ChatWindow messages={messages} isLoading={isLoading} />
      </div>

      <ChatInput
        onSend={handleSend}
        isLoading={isLoading}
        customUploadFn={handleUpload}
      />
    </div>
  );
};

export default ResumeAgent;
