import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import remarkGfm from 'remark-gfm'

// Componente para renderizar markdown con resaltado de código y otras características
const MarkdownRenderer = ({ children }) => {
  return (
    <ReactMarkdown
      rehypePlugins={[rehypeHighlight]}
      remarkPlugins={[remarkGfm]}
      components={{
        // Personalizar cómo se renderizan los componentes
        a: ({ node, ...props }) => (
          <a {...props} target="_blank" rel="noopener noreferrer" />
        ),
        pre: ({ node, ...props }) => (
          <pre className="code-block" {...props} />
        ),
        code: ({ node, inline, className, children, ...props }) => {
          const match = /language-(\w+)/.exec(className || '')
          return !inline && match ? (
            <code className={className} {...props}>
              {children}
            </code>
          ) : (
            <code className={inline ? 'inline-code' : ''} {...props}>
              {children}
            </code>
          )
        }
      }}
    >
      {children}
    </ReactMarkdown>
  )
}

export default MarkdownRenderer