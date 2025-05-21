// frontend/src/components/InputArea.jsx (versión corregida)
import { useState } from 'react';

const InputArea = ({ onSend, isLoading }) => {
  const [message, setMessage] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim() && !isLoading) {
      onSend(message);
      setMessage('');
    }
  };

  return (
    <form 
      className="input-area" 
      onSubmit={handleSubmit}
      style={{
        display: 'flex',
        padding: '10px',
        borderTop: '1px solid #e9ecef',
        backgroundColor: 'white'
      }}
    >
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Escribe tu consulta legal..."
        disabled={isLoading}
        style={{
          flex: 1,
          padding: '12px 16px',
          borderRadius: '24px',
          border: '1px solid #e9ecef',
          fontSize: '16px',
          outline: 'none'
        }}
      />
      <button 
        type="submit" 
        disabled={!message.trim() || isLoading}
        style={{
          width: '48px',
          height: '48px',
          marginLeft: '10px',
          borderRadius: '50%',
          backgroundColor: message.trim() && !isLoading ? '#4361ee' : '#e9ecef',
          color: 'white',
          border: 'none',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: message.trim() && !isLoading ? 'pointer' : 'not-allowed'
        }}
      >
        <svg 
          stroke="currentColor" 
          fill="none" 
          strokeWidth="2" 
          viewBox="0 0 24 24" 
          strokeLinecap="round" 
          strokeLinejoin="round" 
          height="20" 
          width="20" 
          xmlns="http://www.w3.org/2000/svg"
        >
          <line x1="22" y1="2" x2="11" y2="13"></line>
          <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
        </svg>
      </button>
    </form>
  );
};

export default InputArea;