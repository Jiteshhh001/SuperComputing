/**
 * SourceCard — citation card showing search result with URL and snippet.
 */

import React from 'react';
import { HiOutlineGlobeAlt } from 'react-icons/hi2';
import './SourceCard.css';

const SourceCard = ({ source }) => {
  const domain = source.url ? new URL(source.url).hostname.replace('www.', '') : '';

  return (
    <a
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      className="source-card glass-card hover-lift"
    >
      <div className="source-header">
        <HiOutlineGlobeAlt size={14} />
        <span className="source-domain truncate">{domain || 'Source'}</span>
      </div>
      <div className="source-title truncate">
        {source.title || 'Untitled Source'}
      </div>
      {source.snippet && (
        <p className="source-snippet">{source.snippet.substring(0, 120)}...</p>
      )}
    </a>
  );
};

export default SourceCard;
