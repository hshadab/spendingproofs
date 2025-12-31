'use client';

import { useState, useRef } from 'react';
import Link from 'next/link';
import { Shield, Cpu, Lock, Code, ChevronDown, BookOpen } from 'lucide-react';

type Section = 'arc' | 'proof' | 'integrate' | 'deep-dive';

interface NavigationProps {
  activeSection: Section;
  setActiveSection: (section: Section) => void;
  activeScrollSection?: string | null;
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

function DropdownItem({ icon, label, description, onClick, isActive }: {
  icon: React.ReactNode;
  label: string;
  description: string;
  onClick?: () => void;
  isActive?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      className={`w-full flex items-start gap-3 px-4 py-2.5 text-left transition-colors ${
        isActive
          ? 'bg-purple-500/10 text-purple-400'
          : 'hover:bg-white/5 text-gray-300'
      }`}
    >
      <div className={`mt-0.5 ${isActive ? 'text-purple-400' : 'text-gray-500'}`}>{icon}</div>
      <div>
        <div className="text-sm font-medium">{label}</div>
        <div className={`text-xs ${isActive ? 'text-purple-400/70' : 'text-gray-500'}`}>{description}</div>
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

export function Navigation({ activeSection, setActiveSection, activeScrollSection }: NavigationProps) {

  // Switch tab and scroll to section
  const navigateTo = (tab: Section, sectionId: string) => {
    setActiveSection(tab);
    scrollToSection(sectionId);
  };

  // Check if a section is the active scroll section
  const isActiveScroll = (sectionId: string) => activeScrollSection === sectionId;

  return (
    <nav className="border-b border-gray-800 bg-[#0a0a0a]/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-6xl mx-auto px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <Link
            href="/"
            className="flex items-center gap-2"
            onClick={(e) => {
              e.preventDefault();
              setActiveSection('arc');
              window.scrollTo({ top: 0, behavior: 'smooth' });
              if (window.location.pathname !== '/') {
                window.location.href = '/';
              }
            }}
          >
            <img
              src="https://cdn.prod.website-files.com/65d52b07d5bc41614daa723f/665df12739c532f45b665fe7_logo-novanet.svg"
              alt="NovaNet"
              className="h-8 w-auto"
            />
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
                isActive={isActiveScroll('problem')}
              />
              <DropdownItem
                icon={<Shield className="w-4 h-4" />}
                label="The Solution"
                description="Cryptographic policy compliance"
                onClick={() => navigateTo('arc', 'solution')}
                isActive={isActiveScroll('solution')}
              />
              <DropdownItem
                icon={<Lock className="w-4 h-4" />}
                label="Why Arc"
                description="USDC gas, sub-second finality"
                onClick={() => navigateTo('arc', 'why-arc')}
                isActive={isActiveScroll('why-arc')}
              />
            </TabDropdown>

            {/* Deeper Dive */}
            <TabDropdown
              label="Deeper Dive"
              icon={<BookOpen className="w-3.5 h-3.5" />}
              isActive={activeSection === 'deep-dive'}
              onClick={() => navigateTo('deep-dive', 'agent-flow')}
            >
              <DropdownItem
                icon={<Shield className="w-4 h-4" />}
                label="Agent Flow"
                description="Complete transaction lifecycle"
                onClick={() => navigateTo('deep-dive', 'agent-flow')}
                isActive={isActiveScroll('agent-flow')}
              />
              <DropdownItem
                icon={<Cpu className="w-4 h-4" />}
                label="Use Cases"
                description="Real agent applications"
                onClick={() => navigateTo('deep-dive', 'use-cases')}
                isActive={isActiveScroll('use-cases')}
              />
              <DropdownItem
                icon={<Lock className="w-4 h-4" />}
                label="Chain Comparison"
                description="Arc vs other L2s"
                onClick={() => navigateTo('deep-dive', 'chain-comparison')}
                isActive={isActiveScroll('chain-comparison')}
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
                isActive={isActiveScroll('whats-proven')}
              />
              <DropdownItem
                icon={<Cpu className="w-4 h-4" />}
                label="The Model"
                description="8 inputs, 3 outputs"
                onClick={() => navigateTo('proof', 'model')}
                isActive={isActiveScroll('model')}
              />
              <DropdownItem
                icon={<Shield className="w-4 h-4" />}
                label="Jolt Atlas zkML"
                description="HyperKZG + BN254"
                onClick={() => navigateTo('proof', 'jolt-atlas')}
                isActive={isActiveScroll('jolt-atlas')}
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
                isActive={isActiveScroll('architecture')}
              />
              <DropdownItem
                icon={<Code className="w-4 h-4" />}
                label="Install SDK"
                description="npm install & quickstart"
                onClick={() => navigateTo('integrate', 'install-sdk')}
                isActive={isActiveScroll('install-sdk')}
              />
              <DropdownItem
                icon={<Lock className="w-4 h-4" />}
                label="Deploy to Arc"
                description="Testnet contracts & integration"
                onClick={() => navigateTo('integrate', 'deploy')}
                isActive={isActiveScroll('deploy')}
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
