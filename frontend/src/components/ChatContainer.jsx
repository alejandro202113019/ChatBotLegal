import { forwardRef } from 'react'
import Message from './Message'

const ChatContainer = forwardRef(({ messages, isLoading }, ref) => {
  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((message, index) => (
          <Message 
            key={index} 
            text={message.text} 
            sender={message.sender} 
          />
        ))}
        {isLoading && (
          <Message 
            text="Pensando..." 
            sender="bot" 
            isTyping 
          />
        )}
        <div ref={ref} />
      </div>
    </div>
  )
})

export default ChatContainer