'use client';

import { useState, useRef } from 'react';
import Link from 'next/link';
import { Shield, Cpu, Lock, Code, ChevronDown, BookOpen } from 'lucide-react';

type Section = 'arc' | 'proof' | 'integrate' | 'deep-dive';

interface NavigationProps {
  activeSection: Section;
  setActiveSection: (section: Section) => void;
}

interface TabDropdownProps {
  label: string;
  icon: React.ReactNode;
  isActive: boolean;
  onClick: () => void;
  children: React.ReactNode;
}

function TabDropdown({ label, icon, isActive, onClick, children }: TabDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  const handleMouseEnter = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    setIsOpen(true);
  };

  const handleMouseLeave = () => {
    timeoutRef.current = setTimeout(() => {
      setIsOpen(false);
    }, 150); // 150ms delay before closing
  };

  return (
    <div
      className="relative"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <button
        onClick={onClick}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
          isActive
            ? 'bg-purple-500/20 text-purple-400'
            : 'text-gray-400 hover:text-white hover:bg-white/5'
        }`}
      >
        {icon}
        {label}
        <ChevronDown className={`w-3 h-3 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute top-full left-0 mt-1 w-64 bg-[#0d1117] border border-gray-800 rounded-lg shadow-xl py-2 z-50">
          {children}
        </div>
      )}
    </div>
  );
}

function DropdownItem({ icon, label, description, onClick }: {
  icon: React.ReactNode;
  label: string;
  description: string;
  onClick?: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="w-full flex items-start gap-3 px-4 py-2.5 text-left hover:bg-white/5 text-gray-300 transition-colors"
    >
      <div className="mt-0.5 text-gray-500">{icon}</div>
      <div>
        <div className="text-sm font-medium">{label}</div>
        <div className="text-xs text-gray-500">{description}</div>
      </div>
    </button>
  );
}

function DropdownLink({ icon, label, description, href, external }: {
  icon: React.ReactNode;
  label: string;
  description: string;
  href: string;
  external?: boolean;
}) {
  const Component = external ? 'a' : Link;
  const props = external ? { target: '_blank', rel: 'noopener noreferrer' } : {};

  return (
    <Component
      href={href}
      {...props}
      className="flex items-start gap-3 px-4 py-2.5 text-left hover:bg-white/5 text-gray-300 transition-colors"
    >
      <div className="mt-0.5 text-gray-500">{icon}</div>
      <div>
        <div className="text-sm font-medium">{label}</div>
        <div className="text-xs text-gray-500">{description}</div>
      </div>
    </Component>
  );
}

// Dropdown for link-only navigation (no in-page sections)
function LinkDropdown({ label, icon, href, children }: {
  label: string;
  icon: React.ReactNode;
  href: string;
  children: React.ReactNode;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  const handleMouseEnter = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    setIsOpen(true);
  };

  const handleMouseLeave = () => {
    timeoutRef.current = setTimeout(() => {
      setIsOpen(false);
    }, 150);
  };

  return (
    <div
      className="relative"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <Link
        href={href}
        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all text-gray-400 hover:text-white hover:bg-white/5"
      >
        {icon}
        {label}
        <ChevronDown className={`w-3 h-3 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </Link>

      {isOpen && (
        <div className="absolute top-full left-0 mt-1 w-64 bg-[#0d1117] border border-gray-800 rounded-lg shadow-xl py-2 z-50">
          {children}
        </div>
      )}
    </div>
  );
}

// Helper to scroll to an element by ID
function scrollToSection(id: string) {
  // Small delay to ensure the section is rendered after tab switch
  setTimeout(() => {
    const element = document.getElementById(id);
    if (element) {
      const navHeight = 60; // Account for sticky nav
      const elementPosition = element.getBoundingClientRect().top + window.scrollY;
      window.scrollTo({
        top: elementPosition - navHeight,
        behavior: 'smooth'
      });
    }
  }, 50);
}

export function Navigation({ activeSection, setActiveSection }: NavigationProps) {

  // Switch tab and scroll to section
  const navigateTo = (tab: Section, sectionId: string) => {
    setActiveSection(tab);
    scrollToSection(sectionId);
  };

  return (
    <nav className="border-b border-gray-800 bg-[#0a0a0a]/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <Link href="/" className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-purple-700 rounded-lg flex items-center justify-center">
              <Shield className="w-5 h-5" />
            </div>
            <span className="font-semibold text-lg hidden sm:block">Spending Proofs</span>
          </Link>

          {/* Section Tabs with Dropdowns */}
          <div className="hidden md:flex items-center gap-1">
            {/* Arc + Agents */}
            <TabDropdown
              label="Arc + Agents"
              icon={<Cpu className="w-3.5 h-3.5" />}
              isActive={activeSection === 'arc'}
              onClick={() => navigateTo('arc', 'problem')}
            >
              <DropdownItem
                icon={<Cpu className="w-4 h-4" />}
                label="The Problem"
                description="Why agents need verifiable spending"
                onClick={() => navigateTo('arc', 'problem')}
              />
              <DropdownItem
                icon={<Shield className="w-4 h-4" />}
                label="The Solution"
                description="Cryptographic policy compliance"
                onClick={() => navigateTo('arc', 'solution')}
              />
              <DropdownItem
                icon={<Lock className="w-4 h-4" />}
                label="Why Arc"
                description="USDC gas, sub-second finality"
                onClick={() => navigateTo('arc', 'why-arc')}
              />
            </TabDropdown>

            {/* Deeper Dive */}
            <TabDropdown
              label="Deeper Dive"
              icon={<BookOpen className="w-3.5 h-3.5" />}
              isActive={activeSection === 'deep-dive'}
              onClick={() => navigateTo('deep-dive', 'use-cases')}
            >
              <DropdownItem
                icon={<Cpu className="w-4 h-4" />}
                label="Use Cases"
                description="Real agent applications"
                onClick={() => navigateTo('deep-dive', 'use-cases')}
              />
              <DropdownItem
                icon={<Shield className="w-4 h-4" />}
                label="Chain Comparison"
                description="Arc vs other L2s"
                onClick={() => navigateTo('deep-dive', 'chain-comparison')}
              />
            </TabDropdown>

            {/* Proof System */}
            <TabDropdown
              label="Proof System"
              icon={<Shield className="w-3.5 h-3.5" />}
              isActive={activeSection === 'proof'}
              onClick={() => navigateTo('proof', 'whats-proven')}
            >
              <DropdownItem
                icon={<Lock className="w-4 h-4" />}
                label="What's Proven"
                description="Public vs private signals"
                onClick={() => navigateTo('proof', 'whats-proven')}
              />
              <DropdownItem
                icon={<Cpu className="w-4 h-4" />}
                label="The Model"
                description="8 inputs, 3 outputs"
                onClick={() => navigateTo('proof', 'model')}
              />
              <DropdownItem
                icon={<Shield className="w-4 h-4" />}
                label="Jolt Atlas zkML"
                description="HyperKZG + BN254"
                onClick={() => navigateTo('proof', 'jolt-atlas')}
              />
              <div className="border-t border-gray-800 my-2" />
              <DropdownLink
                icon={<Shield className="w-4 h-4" />}
                label="Security Model"
                description="Full threat model & mitigations"
                href="/security"
              />
            </TabDropdown>

            {/* For Developers */}
            <TabDropdown
              label="For Developers"
              icon={<Code className="w-3.5 h-3.5" />}
              isActive={activeSection === 'integrate'}
              onClick={() => navigateTo('integrate', 'architecture')}
            >
              <DropdownItem
                icon={<Cpu className="w-4 h-4" />}
                label="Architecture"
                description="How the pieces fit together"
                onClick={() => navigateTo('integrate', 'architecture')}
              />
              <DropdownItem
                icon={<Code className="w-4 h-4" />}
                label="Install SDK"
                description="npm install & quickstart"
                onClick={() => navigateTo('integrate', 'install-sdk')}
              />
              <DropdownItem
                icon={<Shield className="w-4 h-4" />}
                label="Hosting Options"
                description="Hosted vs self-host prover"
                onClick={() => navigateTo('integrate', 'hosting')}
              />
              <DropdownItem
                icon={<Lock className="w-4 h-4" />}
                label="Deploy to Arc"
                description="Testnet & mainnet deployment"
                onClick={() => navigateTo('integrate', 'deploy')}
              />
              <div className="border-t border-gray-800 my-2" />
              <DropdownLink
                icon={<Code className="w-4 h-4" />}
                label="SDK Reference"
                description="Full API documentation"
                href="https://github.com/hshadab/spendingproofs/blob/main/docs/sdk-reference.md"
                external
              />
              <DropdownLink
                icon={<Shield className="w-4 h-4" />}
                label="GitHub"
                description="Source code & examples"
                href="https://github.com/hshadab/spendingproofs"
                external
              />
            </TabDropdown>

          </div>
        </div>

        <div className="flex items-center gap-3">
          <a href="https://github.com/hshadab/spendingproofs" target="_blank" rel="noopener noreferrer" className="text-sm text-gray-400 hover:text-white transition-colors hidden sm:inline">
            GitHub
          </a>
          <Link
            href="/demo"
            className="bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          >
            Try Demo
          </Link>
        </div>
      </div>
    </nav>
  );
}
