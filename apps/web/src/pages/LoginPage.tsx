import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { Sparkles, Lock, Mail, ArrowRight, ShieldCheck } from 'lucide-react';

interface LoginPageProps {
  onNavigateToRegister: () => void;
}

export const LoginPage: React.FC<LoginPageProps> = ({ onNavigateToRegister }) => {
  const { login } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    try {
      await login(email, password);
    } catch (err: any) {
      console.error('Login error:', err);
      setError(err?.response?.data?.error?.message || err?.response?.data?.detail || 'Invalid email or password.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDemoLogin = async () => {
    setEmail('demo@nexus.ai');
    setPassword('nexuspassword123');
  };

  return (
    <div className="min-h-[80vh] flex items-center justify-center p-4 bg-grid-pattern">
      <div className="w-full max-w-md glass-panel p-8 rounded-3xl border-slate-700 shadow-2xl space-y-6">
        {/* Brand Header */}
        <div className="text-center space-y-2">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-nexus-accent via-nexus-purple to-indigo-500 p-[1px] shadow-glow mx-auto">
            <div className="w-full h-full bg-nexus-900 rounded-2xl flex items-center justify-center">
              <Sparkles className="w-6 h-6 text-nexus-accent" />
            </div>
          </div>
          <h2 className="text-2xl font-black tracking-tight text-white">Sign In to NEXUS</h2>
          <p className="text-xs text-slate-400">Autonomous Multi-Agent Data Intelligence Platform</p>
        </div>

        {error && (
          <div className="p-3.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs font-mono">
            {error}
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-300">Email Address</label>
            <div className="relative">
              <Mail className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="scientist@nexus.ai"
                className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-slate-950 border border-slate-700 text-xs text-white focus:outline-none focus:border-nexus-accent"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-slate-300">Password</label>
            <div className="relative">
              <Lock className="w-4 h-4 text-slate-400 absolute left-3.5 top-3" />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full pl-10 pr-4 py-2.5 rounded-xl bg-slate-950 border border-slate-700 text-xs text-white focus:outline-none focus:border-nexus-accent"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-nexus-accent to-nexus-purple text-nexus-900 font-bold text-xs shadow-glow hover:opacity-95 transition-opacity flex items-center justify-center space-x-2 disabled:opacity-50"
          >
            {isLoading ? <span>Authenticating...</span> : (
              <>
                <span>Access Command Center</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </form>

        {/* Footer info & Register link */}
        <div className="pt-4 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
          <span>Don't have an account?</span>
          <button
            type="button"
            onClick={onNavigateToRegister}
            className="text-nexus-accent font-semibold hover:underline"
          >
            Register Profile
          </button>
        </div>
      </div>
    </div>
  );
};
