/**
 * FileUpload — drag-and-drop file upload zone with progress bar.
 */

import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { HiOutlineCloudArrowUp, HiOutlineDocument } from 'react-icons/hi2';
import './FileUpload.css';

const FileUpload = ({ onUpload, uploading, progress, accept = '.pdf,.txt,.md,.doc,.docx' }) => {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      onUpload(acceptedFiles[0]);
    }
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
    },
    maxFiles: 1,
    disabled: uploading,
  });

  return (
    <div className="file-upload-container">
      <div
        {...getRootProps()}
        className={`dropzone glass-card ${isDragActive ? 'active' : ''} ${uploading ? 'uploading' : ''}`}
      >
        <input {...getInputProps()} />
        {uploading ? (
          <div className="upload-progress">
            <div className="progress-bar-track">
              <div className="progress-bar-fill animate-progress" style={{ width: `${progress}%` }} />
            </div>
            <span className="progress-text">{progress}% uploaded</span>
          </div>
        ) : isDragActive ? (
          <div className="drop-active animate-pulse">
            <HiOutlineCloudArrowUp size={40} />
            <p>Drop your file here</p>
          </div>
        ) : (
          <div className="drop-idle">
            <HiOutlineDocument size={36} />
            <p><strong>Drop a file</strong> or click to upload</p>
            <span className="file-types">PDF, TXT, Markdown</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUpload;
