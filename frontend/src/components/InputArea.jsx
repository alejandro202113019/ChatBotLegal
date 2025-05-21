import { useState } from 'react'
import { FiSend } from 'react-icons/fi'

const InputArea = ({ onSend, isLoading }) => {
  const [message, setMessage] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    if (message.trim() && !isLoading) {
      onSend(message)
      setMessage('')
    }
  }

  return (
    <form className="input-area" onSubmit={handleSubmit}>
      <input
        type="text"
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        placeholder="Escribe tu consulta legal..."
        disabled={isLoading}
      />
      <button type="submit" disabled={!message.trim() || isLoading}>
        <FiSend className="send-icon" />
      </button>
    </form>
  )
}

export default InputArea