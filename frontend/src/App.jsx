import { useState, useRef, useEffect } from 'react'
import ChatContainer from './components/ChatContainer'
import InputArea from './components/InputArea'
import { FiSend, FiMessageSquare } from 'react-icons/fi'

const App = () => {
  const [messages, setMessages] = useState([
    {
      text: "Hola, soy tu asistente legal. ¿En qué puedo ayudarte hoy?",
      sender: 'bot'
    }
  ])
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async (message) => {
    if (!message.trim()) return

    // Agregar mensaje del usuario
    const userMessage = { text: message, sender: 'user' }
    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      // Enviar al backend
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message })
      })

      const data = await response.json()

      // Agregar respuesta del bot
      const botMessage = { text: data.bot_response, sender: 'bot' }
      setMessages(prev => [...prev, botMessage])
    } catch (error) {
      console.error('Error:', error)
      const errorMessage = { 
        text: 'Lo siento, hubo un error al procesar tu solicitud. Por favor intenta nuevamente.', 
        sender: 'bot' 
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <FiMessageSquare className="header-icon" />
        <h1>Asistente Legal</h1>
      </header>
      
      <ChatContainer messages={messages} isLoading={isLoading} ref={messagesEndRef} />
      
      <InputArea onSend={handleSend} isLoading={isLoading} />
    </div>
  )
}

export default App