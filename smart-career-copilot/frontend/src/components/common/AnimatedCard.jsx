/**
 * AnimatedCard — glass card with hover animations for dashboard.
 */

import React from 'react';
import './AnimatedCard.css';

const AnimatedCard = ({ icon, title, description, color, onClick, children }) => {
  return (
    <div
      className="animated-card glass-card hover-lift gradient-border"
      onClick={onClick}
      style={{ '--card-color': color }}
    >
      {icon && <div className="card-icon">{icon}</div>}
      {title && <h3 className="card-title">{title}</h3>}
      {description && <p className="card-description">{description}</p>}
      {children}
    </div>
  );
};

export default AnimatedCard;
