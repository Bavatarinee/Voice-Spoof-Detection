// ─────────────────────────────────────────
//  SHARED DESIGN TOKENS
// ─────────────────────────────────────────
export const COLORS = {
  bg1:     '#0a0a1a',
  bg2:     '#0d0b1f',
  card:    'rgba(255,255,255,0.05)',
  cardHi:  'rgba(124,58,237,0.08)',
  border:  'rgba(255,255,255,0.09)',
  accent:  '#7c3aed',
  accent2: '#4f46e5',
  accentGlow: 'rgba(124,58,237,0.25)',
  real:    '#22c55e',
  realDim: 'rgba(34,197,94,0.12)',
  spoof:   '#ef4444',
  spoofDim:'rgba(239,68,68,0.12)',
  warn:    '#f59e0b',
  warnDim: 'rgba(245,158,11,0.10)',
  text:    '#e2e8f0',
  sub:     '#94a3b8',
  muted:   '#64748b',
  purple:  '#c4b5fd',
  cyan:    '#22d3ee',
};

export const GRADIENTS = {
  bg:      ['#0a0a1a', '#0d0b2a', '#0a0a1a'],
  accent:  ['#7c3aed', '#4f46e5'],
  real:    ['#16a34a', '#22c55e'],
  spoof:   ['#b91c1c', '#ef4444'],
  warn:    ['#b45309', '#f59e0b'],
  card:    ['rgba(255,255,255,0.06)', 'rgba(255,255,255,0.02)'],
  header:  ['rgba(124,58,237,0.15)', 'transparent'],
};

export const FONTS = {
  heading: { fontWeight: '800', letterSpacing: -0.5 },
  sub:     { fontWeight: '600', letterSpacing: 0.2 },
  mono:    { fontFamily: 'monospace' },
};

export const RADIUS = {
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  full: 999,
};

export const SHADOW = {
  accent: {
    shadowColor: '#7c3aed',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.4,
    shadowRadius: 12,
    elevation: 8,
  },
  card: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 8,
    elevation: 4,
  },
};
