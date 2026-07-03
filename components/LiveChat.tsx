'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, MessageCircle, Minimize2, Maximize2 } from 'lucide-react'

interface ChatMessage {
  id: string
  sender: 'user' | 'support' | 'system'
  message: string
  timestamp: Date
}

export function LiveChat() {
  const [isOpen, setIsOpen] = useState(true)
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      sender: 'system',
      message: 'Welcome to Trading Signals Live Support! How can we assist you today?',
      timestamp: new Date(),
    },
  ])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return

    // Add user message
    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      sender: 'user',
      message: inputValue,
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    // Simulate support response
    setTimeout(() => {
      let response = 'Thank you for your message. Our support team is reviewing your inquiry.'

      if (inputValue.toLowerCase().includes('signal')) {
        response = 'Our trading signals are based on real-time market data and advanced technical indicators. Each signal is updated every 30 seconds for optimal accuracy.'
      } else if (inputValue.toLowerCase().includes('risk')) {
        response = 'Risk management is crucial. We recommend following the suggested stop-loss and take-profit levels. Never risk more than 2% of your account on a single trade.'
      } else if (inputValue.toLowerCase().includes('accurate')) {
        response = 'Our signals achieve 75-95% accuracy depending on market conditions. We use live market data from major exchanges and calculate indicators in real-time.'
      } else if (inputValue.toLowerCase().includes('price')) {
        response = 'Prices are updated in real-time from the market. The data displayed is current within 5 seconds of market changes.'
      }

      const supportMessage: ChatMessage = {
        id: `msg-${Date.now()}-support`,
        sender: 'support',
        message: response,
        timestamp: new Date(),
      }

      setMessages(prev => [...prev, supportMessage])
      setIsLoading(false)
    }, 800)
  }

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-4 right-4 z-40 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-full p-4 shadow-lg hover:shadow-xl transition flex items-center gap-2"
      >
        <MessageCircle size={24} />
        <span className="text-sm font-semibold">Chat</span>
      </button>
    )
  }

  return (
    <div className="fixed bottom-4 right-4 z-40 w-96 bg-white border border-slate-200 rounded-xl shadow-2xl flex flex-col h-96">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-4 rounded-t-xl flex items-center justify-between">
        <div className="flex items-center gap-2">
          <MessageCircle size={20} />
          <h3 className="font-semibold">Live Support</h3>
        </div>
        <button
          onClick={() => setIsOpen(false)}
          className="p-1 hover:bg-white/20 rounded transition"
        >
          <Minimize2 size={18} />
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 bg-slate-50">
        {messages.map(msg => (
          <div
            key={msg.id}
            className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-xs px-4 py-2 rounded-lg text-sm ${
                msg.sender === 'user'
                  ? 'bg-blue-600 text-white rounded-br-none'
                  : msg.sender === 'support'
                    ? 'bg-white border border-slate-300 text-slate-900 rounded-bl-none'
                    : 'bg-slate-200 text-slate-700 rounded-none text-xs font-medium'
              }`}
            >
              {msg.message}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white border border-slate-300 px-4 py-2 rounded-lg text-sm text-slate-600">
              <div className="flex gap-1">
                <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-slate-200 p-3 bg-white rounded-b-xl">
        <div className="flex gap-2">
          <input
            type="text"
            value={inputValue}
            onChange={e => setInputValue(e.target.value)}
            onKeyPress={e => e.key === 'Enter' && handleSendMessage()}
            placeholder="Type your message..."
            className="flex-1 bg-slate-50 border border-slate-300 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-600"
          />
          <button
            onClick={handleSendMessage}
            disabled={isLoading || !inputValue.trim()}
            className="bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white rounded-lg p-2 transition"
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  )
}
