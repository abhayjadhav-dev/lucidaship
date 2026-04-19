import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '@clerk/clerk-react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';

export default function LandingPage() {
  const { isLoaded, isSignedIn } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (isLoaded && isSignedIn) {
      navigate('/dashboard', { replace: true });
    }
  }, [isLoaded, isSignedIn, navigate]);

  const authEntryPath = isSignedIn ? '/dashboard' : '/login';
  const primaryPath = isSignedIn ? '/dashboard' : '/register';

  return (
    <div className="min-h-screen bg-black text-white relative">
      
      {/* ── Dynamic Ambient Glow ── */}
      <div className="bg-glow"></div>

      {/* ── Navbar ── */}
      <nav className="relative z-50 border-b border-line bg-black/80 backdrop-blur-md sticky top-0">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-24">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 border border-accent flex items-center justify-center font-mono text-[0.8rem] text-accent">
                L
              </div>
              <span className="font-serif font-semibold text-xl tracking-wide">
                Lucida<span className="text-accent">.</span>
              </span>
            </div>
            
            <div className="hidden md:flex items-center gap-10 font-mono text-[0.7rem] tracking-[0.15em] uppercase text-dim">
              <a href="#features" className="hover:text-white transition-colors">Features</a>
              <a href="#how-it-works" className="hover:text-white transition-colors">How it Works</a>
              <a href="#security" className="hover:text-white transition-colors">Security</a>
              <Link to="/academy" className="hover:text-white transition-colors">Academy</Link>
            </div>

            <div className="flex items-center gap-6">
              <Link to={authEntryPath} className="font-mono text-[0.7rem] tracking-[0.15em] uppercase text-dim hover:text-white transition-colors">
                {isSignedIn ? 'Dashboard' : 'Log in'}
              </Link>
              <Link to={primaryPath} className="btn-outline hidden sm:flex">
                {isSignedIn ? 'Open Dashboard' : 'Engage'} <ArrowRight className="w-3 h-3 ml-1" />
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* ── Hero Section ── */}
      <section className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-32 pb-40 flex flex-col items-start border-b border-line">
        <div className="font-mono text-[0.65rem] tracking-[0.25em] uppercase text-accent flex items-center gap-3 mb-8">
          <div className="w-8 h-[1px] bg-accent"></div>
          Next-Generation Analytics
        </div>
        
        <h1 className="text-5xl md:text-7xl lg:text-[6.5rem] font-serif font-light tracking-tight mb-10 leading-[0.98] w-full">
          Let Your Data Give You a <br className="hidden md:block"/>
          <em className="text-accent italic">Crystal Clear</em> View
        </h1>
        
        <p className="text-lg text-light max-w-2xl mb-14 font-light leading-[1.85]">
          Stop guessing who will convert. Upload any CSV, automatically train cutting-edge Machine Learning models, and instantly score your leads—with absolutely zero configuration.
        </p>

        <div className="flex flex-col sm:flex-row gap-5">
          <Link to={primaryPath} className="btn-primary">
            {isSignedIn ? 'Continue to Dashboard' : 'Start Scoring Leads'} <ArrowRight className="w-4 h-4" />
          </Link>
          <Link to="/academy" className="btn-outline">
            Learn the Engine
          </Link>
          <a href="#features" className="btn-outline">
            Explore Platform
          </a>
        </div>
      </section>

      {/* ── Metric Strip ── */}
      <div className="border-b border-line bg-deep py-6 overflow-hidden">
        <div className="flex gap-16 font-mono text-[0.6rem] tracking-[0.2em] uppercase text-muted items-center whitespace-nowrap animate-[marquee_32s_linear_infinite]">
          {/* Repeating for marquee effect */}
          {[...Array(3)].map((_, i) => (
            <React.Fragment key={i}>
              <span className="flex items-center gap-16"><div className="w-1 h-1 bg-accent/50 rounded-full"></div> Big Data Analytics</span>
              <span className="flex items-center gap-16"><div className="w-1 h-1 bg-accent/50 rounded-full"></div> Machine Learning</span>
              <span className="flex items-center gap-16"><div className="w-1 h-1 bg-accent/50 rounded-full"></div> Artificial Intelligence</span>
              <span className="flex items-center gap-16"><div className="w-1 h-1 bg-accent/50 rounded-full"></div> Process Automation</span>
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* ── Features Grid ── */}
      <section id="features" className="relative z-10 py-32 bg-black border-b border-line">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-end gap-6 mb-20 pb-8 border-b border-line">
            <span className="font-serif text-[5rem] font-light text-line leading-none select-none">01</span>
            <h2 className="font-serif text-4xl sm:text-5xl font-light leading-none pb-2">A Complete ML <em className="italic text-accent">Pipeline</em></h2>
            <div className="ml-auto font-mono text-[0.6rem] tracking-[0.25em] uppercase text-accent pb-3 hidden sm:block">What We Do</div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-[1px] bg-line border border-line">
            {/* Feature 1 */}
            <div className="bg-black p-12 relative overflow-hidden group hover:bg-surface transition-colors duration-500">
              <div className="absolute bottom-0 left-0 right-0 h-[2px] bg-accent scale-x-0 origin-left group-hover:scale-x-100 transition-transform duration-500"></div>
              <div className="font-serif text-5xl text-line mb-6 group-hover:text-accent transition-colors duration-500">01</div>
              <h3 className="font-serif text-[1.4rem] font-light mb-4">Upload Any Schema</h3>
              <p className="text-[0.85rem] text-dim leading-[1.85] font-light mb-6">
                We accept absolutely any CSV structure. Our adaptive engine automatically identifies date columns, categorical strings, and numeric IDs to build a comprehensive feature set without manual mapping.
              </p>
              <div className="flex gap-2">
                <span className="tag">Auto-Detect</span><span className="tag">Data Pipelines</span>
              </div>
            </div>

            {/* Feature 2 */}
            <div className="bg-black p-12 relative overflow-hidden group hover:bg-surface transition-colors duration-500">
              <div className="absolute bottom-0 left-0 right-0 h-[2px] bg-accent scale-x-0 origin-left group-hover:scale-x-100 transition-transform duration-500"></div>
              <div className="font-serif text-5xl text-line mb-6 group-hover:text-accent transition-colors duration-500">02</div>
              <h3 className="font-serif text-[1.4rem] font-light mb-4">Auto-Train Models</h3>
              <p className="text-[0.85rem] text-dim leading-[1.85] font-light mb-6">
                Powered by Scikit-Learn, LightGBM, and XGBoost. The engine imputes missing values, balances classes, and builds an accurate ensemble personalized exactly for your historic conversion data.
              </p>
              <div className="flex gap-2">
                <span className="tag border-accent text-accent bg-accent/10">Predictive</span><span className="tag">XGBoost</span>
              </div>
            </div>

            {/* Feature 3 */}
            <div className="bg-black p-12 relative overflow-hidden group hover:bg-surface transition-colors duration-500">
              <div className="absolute bottom-0 left-0 right-0 h-[2px] bg-accent scale-x-0 origin-left group-hover:scale-x-100 transition-transform duration-500"></div>
              <div className="font-serif text-5xl text-line mb-6 group-hover:text-accent transition-colors duration-500">03</div>
              <h3 className="font-serif text-[1.4rem] font-light mb-4">Score Instantly</h3>
              <p className="text-[0.85rem] text-dim leading-[1.85] font-light mb-6">
                Feed new, untested prospects into your customized pipeline. Receive a ranked list with attached probability percentages showing exactly who your sales team should call first.
              </p>
              <div className="flex gap-2">
                <span className="tag">Ranking</span><span className="tag">Output</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── CTA Footer ── */}
      <footer className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-32 text-center">
        <h2 className="text-5xl font-serif font-light mb-8">Ready to rank your <em className="italic text-accent">leads?</em></h2>
        <p className="text-dim mb-12 max-w-2xl mx-auto font-light leading-[1.85]">Join the future of high-conversion sales. Setup takes exactly zero minutes. Just bring your CSV files, and use the academy to understand every moving part of the ranking engine.</p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link to={primaryPath} className="btn-primary">
            {isSignedIn ? 'Go to Dashboard' : 'Create Your Workspace'}
          </Link>
          <Link to="/academy" className="btn-outline">
            Open Academy
          </Link>
        </div>
        <div className="mt-20 font-mono text-[0.6rem] tracking-[0.2em] uppercase text-muted border-t border-line pt-10">
          © 2024 Lucida Analytics — Clarify Through Data.
        </div>
      </footer>

    </div>
  );
}
