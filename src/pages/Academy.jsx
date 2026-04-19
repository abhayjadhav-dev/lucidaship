import React, { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  ArrowRight,
  Brain,
  CheckCircle2,
  CircleDashed,
  Database,
  FileSearch,
  GitBranch,
  Layers3,
  Medal,
  Shield,
  SlidersHorizontal,
  Sparkles,
  Target,
  Trophy,
} from 'lucide-react';

const missions = [
  {
    id: 'mission-1',
    title: 'Decode the Input',
    xp: 120,
    goal: 'Learn how the engine reads unknown CSVs without hard-coded column names.',
    lesson:
      'The DataAnalyzer inspects null ratios, uniqueness, text length, numeric patterns, and date parse success to classify columns as numeric, categorical, binary, text, temporal, id, or ignore.',
  },
  {
    id: 'mission-2',
    title: 'Find the Outcome Signal',
    xp: 160,
    goal: 'Understand how the system chooses the target column before training.',
    lesson:
      'If you do not pass a target column, Lucida searches for binary columns, encodes them safely to 0 and 1, and picks the one with the strongest average correlation to numeric predictors.',
  },
  {
    id: 'mission-3',
    title: 'Engineer the Ranking DNA',
    xp: 220,
    goal: 'See how features are built and why the ranking is not just a spreadsheet sort.',
    lesson:
      'Numeric fields are min-max scaled, low-cardinality categories get one-hot encoded, larger categories get frequency encoded, dates become recency features, and optional text embeddings are compressed with PCA.',
  },
  {
    id: 'mission-4',
    title: 'Train the Ranking Brain',
    xp: 240,
    goal: 'Trace the exact model path that produces a ranked lead list.',
    lesson:
      'The engine splits train and test data, tries SMOTE to balance the classes, trains a RandomForestClassifier, produces probabilities, multiplies by 100, and sorts descending for the final leaderboard.',
  },
];

const pipelineSteps = [
  {
    id: 'analyze',
    label: 'Analyze',
    icon: FileSearch,
    detail:
      'Column typing is inferred from the data itself. Mostly empty columns are ignored. Near-unique columns become IDs. Datelike strings become temporal features.',
    takeaway: 'This is the schema-agnostic entry point of the product.',
  },
  {
    id: 'target',
    label: 'Target',
    icon: Target,
    detail:
      'Binary columns are candidate outcomes. The engine picks the one with the strongest average relationship to numeric columns unless you override it.',
    takeaway: 'This is powerful, but it also means target choice should be audited carefully.',
  },
  {
    id: 'importance',
    label: 'Importance',
    icon: Sparkles,
    detail:
      'Numeric features use Spearman correlation, categorical features use mutual information, and temporal features use recency correlation. Relevant columns pass a threshold before modeling.',
    takeaway: 'Ranking inputs are filtered before training, which shapes the final leaderboard.',
  },
  {
    id: 'features',
    label: 'Features',
    icon: Layers3,
    detail:
      'The engine stores scalers, categorical encoders, and PCA models so the same transformations can be applied later to new leads.',
    takeaway: 'This gives scoring consistency between training data and live scoring data.',
  },
  {
    id: 'model',
    label: 'Model',
    icon: Brain,
    detail:
      'Training uses a Random Forest with 100 trees and max depth 8. If class balance is poor, SMOTE is attempted before fit.',
    takeaway: 'Today the final ranking is a tree-based probability estimate, not a hand-written formula.',
  },
  {
    id: 'rank',
    label: 'Rank',
    icon: Trophy,
    detail:
      'Each new lead is transformed with the saved pipeline, scored with predict_proba, converted into a 0 to 100 score, and sorted from highest to lowest.',
    takeaway: 'Ranking is literally probability-based ordering of scored rows.',
  },
];

const quizQuestions = [
  {
    prompt: 'What currently decides the final order of leads in scored output?',
    options: [
      'Rows are sorted alphabetically by company name',
      'Rows are sorted by Random Forest probability score descending',
      'Rows are sorted by the first numeric column only',
    ],
    correctIndex: 1,
    explanation:
      'The score method calls predict_with_explanation, then sorts the output list by score descending before returning results.',
  },
  {
    prompt: 'How does Lucida currently choose a target column if you do not specify one?',
    options: [
      'It picks the first column in the CSV',
      'It asks the user in the dashboard every time',
      'It evaluates binary columns and selects the strongest numeric relationship',
    ],
    correctIndex: 2,
    explanation:
      'The auto-detection step evaluates binary candidates and chooses the highest-scoring one based on numeric correlation heuristics.',
  },
  {
    prompt: 'Which statement best describes the current explanation layer?',
    options: [
      'Top drivers are global top feature importances, not row-specific local explanations',
      'Each lead receives a full SHAP explanation',
      'The backend stores a legal patent memo for each model',
    ],
    correctIndex: 0,
    explanation:
      'The code surfaces the top three model feature names overall, which is useful but not yet a fully local explanation per row.',
  },
];

const patentTracks = [
  {
    title: 'What is already interesting',
    points: [
      'Schema-agnostic ingestion and type inference without predefined mappings',
      'Auto target discovery from binary outcome candidates',
      'Adaptive feature engineering path chosen by observed data type and cardinality',
      'Multi-file smart merge heuristics before model training',
    ],
  },
  {
    title: 'What still feels generic',
    points: [
      'Random Forest plus SMOTE is standard ML infrastructure by itself',
      'Current top driver output is broad and not deeply explainable',
      'There is no distinctive feedback loop, domain memory, or ranking rationale ledger yet',
      'Thresholds and heuristics are useful, but they need sharper novelty framing',
    ],
  },
  {
    title: 'What could become patent-worthy product IP',
    points: [
      'A proprietary ranking rationale layer that records why each lead moved up or down versus prior runs',
      'Adaptive target confidence checks that detect weak or ambiguous conversion labels before training',
      'Cross-file entity resolution and confidence-weighted merge scoring instead of simple key guesses',
      'A closed-loop post-sale learning system that reweights feature engineering choices based on downstream outcomes',
    ],
  },
];

const initialLeads = [
  { name: 'Lead A', interactions: 14, companySize: 420, recencyDays: 3, industryFit: 82 },
  { name: 'Lead B', interactions: 8, companySize: 1200, recencyDays: 12, industryFit: 61 },
  { name: 'Lead C', interactions: 22, companySize: 180, recencyDays: 1, industryFit: 74 },
];

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function educationalScore(lead, weights) {
  const normalizedInteractions = clamp(lead.interactions / 30, 0, 1);
  const normalizedCompany = clamp(lead.companySize / 2000, 0, 1);
  const normalizedRecency = clamp(1 - lead.recencyDays / 30, 0, 1);
  const normalizedIndustry = clamp(lead.industryFit / 100, 0, 1);

  const weighted =
    normalizedInteractions * weights.interactions +
    normalizedCompany * weights.companySize +
    normalizedRecency * weights.recency +
    normalizedIndustry * weights.industryFit;

  return Math.round(weighted * 100);
}

export default function Academy() {
  const [activeStep, setActiveStep] = useState('analyze');
  const [selectedAnswers, setSelectedAnswers] = useState({});
  const [leads, setLeads] = useState(initialLeads);
  const [weights, setWeights] = useState({
    interactions: 0.32,
    companySize: 0.18,
    recency: 0.28,
    industryFit: 0.22,
  });

  const earnedXp = useMemo(() => {
    const correctAnswers = Object.entries(selectedAnswers).filter(([key, value]) => {
      const question = quizQuestions[Number(key)];
      return question && value === question.correctIndex;
    }).length;

    return missions.reduce((total, mission) => total + mission.xp, 0) + correctAnswers * 40;
  }, [selectedAnswers]);

  const rankingBoard = useMemo(() => {
    return leads
      .map((lead) => ({
        ...lead,
        teachingScore: educationalScore(lead, weights),
      }))
      .sort((a, b) => b.teachingScore - a.teachingScore);
  }, [leads, weights]);

  const activePipelineStep = pipelineSteps.find((step) => step.id === activeStep) ?? pipelineSteps[0];
  const ActivePipelineIcon = activePipelineStep.icon;

  const updateLead = (index, field, value) => {
    setLeads((current) =>
      current.map((lead, leadIndex) =>
        leadIndex === index ? { ...lead, [field]: Number(value) } : lead
      )
    );
  };

  return (
    <div className="min-h-screen bg-black text-white relative">
      <div className="bg-glow" />

      <nav className="relative z-50 border-b border-line bg-black/80 backdrop-blur-md sticky top-0">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-24">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 border border-accent flex items-center justify-center font-mono text-[0.8rem] text-accent">
                L
              </div>
              <span className="font-serif font-semibold text-xl tracking-wide">
                Lucida Academy<span className="text-accent">.</span>
              </span>
            </div>

            <div className="hidden md:flex items-center gap-10 font-mono text-[0.7rem] tracking-[0.15em] uppercase text-dim">
              <a href="#missions" className="hover:text-white transition-colors">Missions</a>
              <a href="#ranking-lab" className="hover:text-white transition-colors">Ranking Lab</a>
              <a href="#patent-board" className="hover:text-white transition-colors">Patent Board</a>
            </div>

            <div className="flex items-center gap-4">
              <Link to="/" className="font-mono text-[0.7rem] tracking-[0.15em] uppercase text-dim hover:text-white transition-colors">
                Main Site
              </Link>
              <Link to="/dashboard" className="btn-outline hidden sm:flex">
                Open Dashboard <ArrowRight className="w-3 h-3 ml-1" />
              </Link>
            </div>
          </div>
        </div>
      </nav>

      <main className="relative z-10">
        <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-16 border-b border-line">
          <div className="grid lg:grid-cols-[1.3fr,0.7fr] gap-10 items-start">
            <div>
              <div className="font-mono text-[0.65rem] tracking-[0.25em] uppercase text-accent flex items-center gap-3 mb-8">
                <div className="w-8 h-[1px] bg-accent" />
                Interactive Founder Course
              </div>
              <h1 className="text-5xl md:text-7xl font-serif font-light tracking-tight leading-[0.96] mb-8">
                Learn how your <em className="italic text-accent">ranking engine</em> actually works
              </h1>
              <p className="text-lg text-light max-w-3xl font-light leading-[1.85] mb-10">
                This course turns Lucida&apos;s real ML pipeline into a playable walkthrough so you can understand the system, explain it to investors, and spot where to push the product from useful automation into stronger defensible IP.
              </p>
              <div className="flex flex-col sm:flex-row gap-4">
                <a href="#missions" className="btn-primary">
                  Start the Course <Medal className="w-4 h-4" />
                </a>
                <a href="#ranking-lab" className="btn-outline">
                  Test the Ranking Logic
                </a>
              </div>
            </div>

            <div className="border border-line bg-surface/60 p-8">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <div className="font-mono text-[0.6rem] tracking-[0.2em] uppercase text-accent mb-2">Founder Progress</div>
                  <div className="font-serif text-3xl">{earnedXp} XP</div>
                </div>
                <div className="w-14 h-14 rounded-full border border-accent/40 bg-accent/10 flex items-center justify-center text-accent">
                  <Trophy className="w-6 h-6" />
                </div>
              </div>
              <div className="space-y-4">
                <div className="border border-line p-4 flex items-start gap-3">
                  <CheckCircle2 className="w-4 h-4 mt-0.5 text-accent" />
                  <div>
                    <div className="font-mono text-[0.65rem] tracking-[0.15em] uppercase text-white">What the rank is today</div>
                    <p className="text-sm text-light mt-2">A Random Forest probability score generated after adaptive preprocessing, then sorted descending.</p>
                  </div>
                </div>
                <div className="border border-line p-4 flex items-start gap-3">
                  <Shield className="w-4 h-4 mt-0.5 text-accent" />
                  <div>
                    <div className="font-mono text-[0.65rem] tracking-[0.15em] uppercase text-white">What investors will ask</div>
                    <p className="text-sm text-light mt-2">Why your engine is more than standard auto-ML and how its ranking decisions improve with proprietary feedback.</p>
                  </div>
                </div>
                <div className="border border-line p-4 flex items-start gap-3">
                  <GitBranch className="w-4 h-4 mt-0.5 text-accent" />
                  <div>
                    <div className="font-mono text-[0.65rem] tracking-[0.15em] uppercase text-white">What to improve next</div>
                    <p className="text-sm text-light mt-2">Versioned rationale, stronger entity resolution, and adaptive post-outcome learning would make the story sharper.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="missions" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 border-b border-line">
          <div className="flex items-end gap-6 mb-14 pb-8 border-b border-line">
            <span className="font-serif text-[5rem] font-light text-line leading-none select-none">01</span>
            <h2 className="font-serif text-4xl sm:text-5xl font-light leading-none pb-2">Mission Track</h2>
            <div className="ml-auto font-mono text-[0.6rem] tracking-[0.25em] uppercase text-accent pb-3 hidden sm:block">Playable Syllabus</div>
          </div>

          <div className="grid lg:grid-cols-2 gap-6">
            {missions.map((mission) => (
              <article key={mission.id} className="border border-line bg-surface/60 p-8 hover:border-accent/40 transition-colors duration-300">
                <div className="flex items-center justify-between mb-5">
                  <div className="font-mono text-[0.65rem] tracking-[0.18em] uppercase text-accent">{mission.id.replace('-', ' ')}</div>
                  <span className="tag border-accent/30 text-accent bg-accent/10">{mission.xp} XP</span>
                </div>
                <h3 className="font-serif text-3xl font-light mb-4">{mission.title}</h3>
                <p className="text-light text-sm leading-[1.9] mb-4">{mission.goal}</p>
                <p className="text-dim text-sm leading-[1.9]">{mission.lesson}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 border-b border-line">
          <div className="flex items-end gap-6 mb-14 pb-8 border-b border-line">
            <span className="font-serif text-[5rem] font-light text-line leading-none select-none">02</span>
            <h2 className="font-serif text-4xl sm:text-5xl font-light leading-none pb-2">Pipeline Console</h2>
          </div>

          <div className="grid lg:grid-cols-[0.8fr,1.2fr] gap-8">
            <div className="border border-line bg-surface/40 p-4">
              {pipelineSteps.map((step) => {
                const Icon = step.icon;
                const isActive = step.id === activeStep;
                return (
                  <button
                    key={step.id}
                    type="button"
                    onClick={() => setActiveStep(step.id)}
                    className={`w-full text-left border mb-3 last:mb-0 px-4 py-4 transition-colors duration-300 ${
                      isActive ? 'border-accent bg-accent/10' : 'border-line hover:border-accent/30'
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <Icon className="w-4 h-4 text-accent" />
                      <div>
                        <div className="font-mono text-[0.65rem] tracking-[0.18em] uppercase text-accent">{step.label}</div>
                        <div className="text-sm text-light mt-1">{step.takeaway}</div>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>

            <div className="border border-line bg-surface/60 p-8">
              <div className="flex items-center gap-3 mb-4">
                <ActivePipelineIcon className="w-5 h-5 text-accent" />
                <div className="font-mono text-[0.65rem] tracking-[0.18em] uppercase text-accent">
                  {activePipelineStep.label}
                </div>
              </div>
              <h3 className="font-serif text-3xl font-light mb-5">{activePipelineStep.takeaway}</h3>
              <p className="text-light leading-[1.9] mb-8">{activePipelineStep.detail}</p>
              <div className="grid sm:grid-cols-2 gap-4">
                <div className="border border-line p-5">
                  <div className="font-mono text-[0.6rem] tracking-[0.16em] uppercase text-accent mb-3">Code Reality</div>
                  <p className="text-sm text-dim leading-[1.8]">
                    This page is based on the live backend pipeline in <span className="text-white">`adaptive_scorer.py`</span> and the scoring API that returns sorted scores to the dashboard.
                  </p>
                </div>
                <div className="border border-line p-5">
                  <div className="font-mono text-[0.6rem] tracking-[0.16em] uppercase text-accent mb-3">Founder Lens</div>
                  <p className="text-sm text-dim leading-[1.8]">
                    If you can explain this step clearly, you are much closer to speaking confidently about architecture, defensibility, and why changes here matter commercially.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="ranking-lab" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 border-b border-line">
          <div className="flex items-end gap-6 mb-14 pb-8 border-b border-line">
            <span className="font-serif text-[5rem] font-light text-line leading-none select-none">03</span>
            <h2 className="font-serif text-4xl sm:text-5xl font-light leading-none pb-2">Ranking Lab</h2>
          </div>

          <div className="grid xl:grid-cols-[1.15fr,0.85fr] gap-8">
            <div className="border border-line bg-surface/60 p-8">
              <div className="flex items-center gap-3 mb-8">
                <SlidersHorizontal className="w-5 h-5 text-accent" />
                <div>
                  <div className="font-mono text-[0.65rem] tracking-[0.18em] uppercase text-accent">Teaching Sandbox</div>
                  <p className="text-sm text-dim mt-2">
                    This lab builds intuition. It is not the exact production model. Your live rank is produced by the trained Random Forest after feature engineering.
                  </p>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-6 mb-8">
                {Object.entries(weights).map(([key, value]) => (
                  <label key={key} className="border border-line p-4 block">
                    <div className="flex items-center justify-between mb-3">
                      <span className="font-mono text-[0.62rem] tracking-[0.16em] uppercase text-accent">
                        {key === 'industryFit' ? 'Industry Fit' : key}
                      </span>
                      <span className="text-sm text-white">{Math.round(value * 100)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0.05"
                      max="0.6"
                      step="0.01"
                      value={value}
                      onChange={(event) =>
                        setWeights((current) => ({ ...current, [key]: Number(event.target.value) }))
                      }
                      className="w-full accent-[#c8a96e] bg-transparent"
                    />
                  </label>
                ))}
              </div>

              <div className="space-y-5">
                {leads.map((lead, index) => (
                  <div key={lead.name} className="border border-line p-5">
                    <div className="font-serif text-2xl mb-4">{lead.name}</div>
                    <div className="grid sm:grid-cols-2 gap-4">
                      <label className="block">
                        <div className="font-mono text-[0.58rem] tracking-[0.16em] uppercase text-dim mb-2">Interactions</div>
                        <input
                          type="number"
                          min="0"
                          max="30"
                          value={lead.interactions}
                          onChange={(event) => updateLead(index, 'interactions', event.target.value)}
                          className="glass-input px-4 py-3"
                        />
                      </label>
                      <label className="block">
                        <div className="font-mono text-[0.58rem] tracking-[0.16em] uppercase text-dim mb-2">Company Size</div>
                        <input
                          type="number"
                          min="0"
                          max="2000"
                          value={lead.companySize}
                          onChange={(event) => updateLead(index, 'companySize', event.target.value)}
                          className="glass-input px-4 py-3"
                        />
                      </label>
                      <label className="block">
                        <div className="font-mono text-[0.58rem] tracking-[0.16em] uppercase text-dim mb-2">Days Since Contact</div>
                        <input
                          type="number"
                          min="0"
                          max="30"
                          value={lead.recencyDays}
                          onChange={(event) => updateLead(index, 'recencyDays', event.target.value)}
                          className="glass-input px-4 py-3"
                        />
                      </label>
                      <label className="block">
                        <div className="font-mono text-[0.58rem] tracking-[0.16em] uppercase text-dim mb-2">Industry Fit</div>
                        <input
                          type="number"
                          min="0"
                          max="100"
                          value={lead.industryFit}
                          onChange={(event) => updateLead(index, 'industryFit', event.target.value)}
                          className="glass-input px-4 py-3"
                        />
                      </label>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-6">
              <div className="border border-line bg-surface/60 p-8">
                <div className="font-mono text-[0.65rem] tracking-[0.18em] uppercase text-accent mb-4">Leaderboard</div>
                <div className="space-y-4">
                  {rankingBoard.map((lead, index) => (
                    <div key={lead.name} className="border border-line p-5 flex items-center justify-between gap-4">
                      <div className="flex items-center gap-4">
                        <div className="w-10 h-10 border border-accent/40 bg-accent/10 flex items-center justify-center font-mono text-sm text-accent">
                          #{index + 1}
                        </div>
                        <div>
                          <div className="font-serif text-2xl">{lead.name}</div>
                          <div className="text-sm text-dim">Educational score driven by your current weighting mix</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-serif text-3xl text-accent">{lead.teachingScore}</div>
                        <div className="font-mono text-[0.58rem] tracking-[0.16em] uppercase text-dim">Sandbox Rank</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="border border-line bg-surface/40 p-8">
                <div className="font-mono text-[0.65rem] tracking-[0.18em] uppercase text-accent mb-4">Production Truth</div>
                <div className="space-y-4 text-sm leading-[1.9] text-light">
                  <p>
                    In production, the backend does not use these manual weights. It creates engineered features from training data, feeds them into the trained forest, and returns a probability for each row.
                  </p>
                  <p>
                    The score you see in the dashboard is <span className="text-white">probability x 100</span>, and the backend sorts the results descending before sending them back.
                  </p>
                  <p>
                    The current explanation field shows top global feature drivers from the model, which is directionally useful but still a big opportunity for a richer proprietary rationale layer.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 border-b border-line">
          <div className="flex items-end gap-6 mb-14 pb-8 border-b border-line">
            <span className="font-serif text-[5rem] font-light text-line leading-none select-none">04</span>
            <h2 className="font-serif text-4xl sm:text-5xl font-light leading-none pb-2">Knowledge Check</h2>
          </div>

          <div className="grid lg:grid-cols-3 gap-6">
            {quizQuestions.map((question, questionIndex) => (
              <div key={question.prompt} className="border border-line bg-surface/60 p-8">
                <div className="font-mono text-[0.62rem] tracking-[0.16em] uppercase text-accent mb-4">
                  Challenge {questionIndex + 1}
                </div>
                <h3 className="font-serif text-2xl font-light mb-5">{question.prompt}</h3>
                <div className="space-y-3">
                  {question.options.map((option, optionIndex) => {
                    const selected = selectedAnswers[questionIndex] === optionIndex;
                    const correct = question.correctIndex === optionIndex;
                    return (
                      <button
                        key={option}
                        type="button"
                        onClick={() =>
                          setSelectedAnswers((current) => ({
                            ...current,
                            [questionIndex]: optionIndex,
                          }))
                        }
                        className={`w-full text-left border px-4 py-4 transition-colors duration-300 ${
                          selected
                            ? correct
                              ? 'border-accent bg-accent/10'
                              : 'border-red-400/40 bg-red-400/10'
                            : 'border-line hover:border-accent/30'
                        }`}
                      >
                        <div className="flex items-start gap-3">
                          {selected ? (
                            correct ? <CheckCircle2 className="w-4 h-4 mt-0.5 text-accent" /> : <CircleDashed className="w-4 h-4 mt-0.5 text-red-300" />
                          ) : (
                            <Database className="w-4 h-4 mt-0.5 text-dim" />
                          )}
                          <span className="text-sm leading-[1.7] text-light">{option}</span>
                        </div>
                      </button>
                    );
                  })}
                </div>
                {selectedAnswers[questionIndex] !== undefined && (
                  <p className="text-sm text-dim leading-[1.8] mt-5">{question.explanation}</p>
                )}
              </div>
            ))}
          </div>
        </section>

        <section id="patent-board" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
          <div className="flex items-end gap-6 mb-14 pb-8 border-b border-line">
            <span className="font-serif text-[5rem] font-light text-line leading-none select-none">05</span>
            <h2 className="font-serif text-4xl sm:text-5xl font-light leading-none pb-2">Patent Board</h2>
          </div>

          <div className="grid lg:grid-cols-3 gap-6 mb-10">
            {patentTracks.map((track) => (
              <div key={track.title} className="border border-line bg-surface/60 p-8">
                <h3 className="font-serif text-3xl font-light mb-6">{track.title}</h3>
                <div className="space-y-4">
                  {track.points.map((point) => (
                    <div key={point} className="flex items-start gap-3">
                      <div className="w-2 h-2 rounded-full bg-accent mt-2.5 flex-shrink-0" />
                      <p className="text-sm text-light leading-[1.8]">{point}</p>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div className="border border-accent/25 bg-accent/5 p-8">
            <div className="font-mono text-[0.62rem] tracking-[0.16em] uppercase text-accent mb-4">Practical Note</div>
            <p className="text-light leading-[1.9] max-w-4xl">
              This board is a product strategy lens, not legal advice. For a real patent path, you will want to document the precise novel mechanism, what technical problem it solves better than prior art, and how the ranking system behaves in a way that is more specific than generic auto-ML.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}
