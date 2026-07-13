import type { Metadata } from 'next'
import { Inter, JetBrains_Mono } from 'next/font/google'
import './globals.css'

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-inter',
})

const jetbrainsMono = JetBrains_Mono({ 
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
})

export const metadata: Metadata = {
  title: 'NEXUS-OMEGA | Command Center',
  description: 'God-Mode Trading Interface for NEXUS-OMEGA Bot',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body className="bg-obsidian text-gray-100 antialiased">
        {/* Scanline overlay effect */}
        <div className="scanlines" />
        
        {/* Main content */}
        <main className="relative z-10 min-h-screen">
          {children}
        </main>
        
        {/* Font loading indicator removed for production */}
      </body>
    </html>
  )
}
